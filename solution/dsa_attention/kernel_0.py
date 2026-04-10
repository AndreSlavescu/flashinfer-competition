import math
from functools import lru_cache

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import BFloat16, Float32, Int32


NUM_HEADS = 16
CKV_DIM = 512
KPE_DIM = 64
TOPK = 2048
WARP_SIZE = 32
WARPS_PER_BLOCK = NUM_HEADS
THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE
NOPE_VALUES_PER_LANE = CKV_DIM // WARP_SIZE
PE_VALUES_PER_LANE = KPE_DIM // WARP_SIZE
LOG2_E = math.log2(math.e)

compile_cache = {}


@cute.kernel
def _dsa_sparse_attention_device(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_flat: cute.Tensor,
    kpe_flat: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale_log2: Float32,
    output: cute.Tensor,
    lse: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    token_idx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % WARP_SIZE

    r_q_nope = cute.make_rmem_tensor(cute.make_layout(NOPE_VALUES_PER_LANE), Float32)
    r_q_pe = cute.make_rmem_tensor(cute.make_layout(PE_VALUES_PER_LANE), Float32)
    r_kc = cute.make_rmem_tensor(cute.make_layout(NOPE_VALUES_PER_LANE), Float32)
    r_out = cute.make_rmem_tensor(cute.make_layout(NOPE_VALUES_PER_LANE), Float32)

    r_out.fill(0.0)

    for i in cutlass.range_constexpr(NOPE_VALUES_PER_LANE):
        dim_idx = lane_idx + i * WARP_SIZE
        r_q_nope[i] = Float32(q_nope[token_idx, warp_idx, dim_idx])

    for i in cutlass.range_constexpr(PE_VALUES_PER_LANE):
        dim_idx = lane_idx + i * WARP_SIZE
        r_q_pe[i] = Float32(q_pe[token_idx, warp_idx, dim_idx])

    row_max = Float32(0.0)
    row_sum = Float32(0.0)
    topk = sparse_indices.shape[1]

    for idx_pos in range(topk):
        kv_idx = sparse_indices[token_idx, idx_pos]
        if kv_idx != Int32(-1):
            dot_local = Float32(0.0)

            for i in cutlass.range_constexpr(NOPE_VALUES_PER_LANE):
                dim_idx = lane_idx + i * WARP_SIZE
                kc = Float32(ckv_flat[kv_idx, dim_idx])
                r_kc[i] = kc
                dot_local += r_q_nope[i] * kc

            for i in cutlass.range_constexpr(PE_VALUES_PER_LANE):
                dim_idx = lane_idx + i * WARP_SIZE
                kp = Float32(kpe_flat[kv_idx, dim_idx])
                dot_local += r_q_pe[i] * kp

            score = cute.arch.warp_reduction_sum(dot_local) * sm_scale_log2

            if row_sum == Float32(0.0):
                for i in cutlass.range_constexpr(NOPE_VALUES_PER_LANE):
                    r_out[i] = r_kc[i]
                row_max = score
                row_sum = Float32(1.0)
            else:
                new_row_max = cute.arch.fmax(row_max, score)
                alpha = cute.math.exp2(row_max - new_row_max, fastmath=True)
                prob = cute.math.exp2(score - new_row_max, fastmath=True)
                for i in cutlass.range_constexpr(NOPE_VALUES_PER_LANE):
                    r_out[i] = r_out[i] * alpha + prob * r_kc[i]
                row_sum = row_sum * alpha + prob
                row_max = new_row_max

    if row_sum == Float32(0.0):
        for i in cutlass.range_constexpr(NOPE_VALUES_PER_LANE):
            dim_idx = lane_idx + i * WARP_SIZE
            output[token_idx, warp_idx, dim_idx] = Float32(0.0).to(BFloat16)
        if lane_idx == 0:
            lse[token_idx, warp_idx] = -Float32.inf
    else:
        inv_row_sum = Float32(1.0) / row_sum
        for i in cutlass.range_constexpr(NOPE_VALUES_PER_LANE):
            dim_idx = lane_idx + i * WARP_SIZE
            output[token_idx, warp_idx, dim_idx] = (r_out[i] * inv_row_sum).to(BFloat16)
        if lane_idx == 0:
            lse[token_idx, warp_idx] = row_max + cute.math.log2(row_sum, fastmath=True)


@cute.jit
def _dsa_sparse_attention_host(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_flat: cute.Tensor,
    kpe_flat: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale_log2: Float32,
    output: cute.Tensor,
    lse: cute.Tensor,
    stream: cuda.CUstream,
):
    _dsa_sparse_attention_device(
        q_nope,
        q_pe,
        ckv_flat,
        kpe_flat,
        sparse_indices,
        sm_scale_log2,
        output,
        lse,
    ).launch(
        grid=[q_nope.shape[0], 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream,
    )


def _make_cute_tensor(tensor: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=assumed_align).mark_layout_dynamic()


def _get_compiled_kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_flat: torch.Tensor,
    kpe_flat: torch.Tensor,
    sparse_indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    stream: cuda.CUstream,
):
    cache_key = (
        q_nope.device.index,
        q_nope.dtype,
        q_pe.dtype,
        ckv_flat.dtype,
        kpe_flat.dtype,
        sparse_indices.dtype,
        output.dtype,
        lse.dtype,
    )
    compiled = compile_cache.get(cache_key)
    if compiled is None:
        q_nope_cute = _make_cute_tensor(q_nope)
        q_pe_cute = _make_cute_tensor(q_pe)
        ckv_flat_cute = _make_cute_tensor(ckv_flat)
        kpe_flat_cute = _make_cute_tensor(kpe_flat)
        sparse_indices_cute = _make_cute_tensor(sparse_indices)
        output_cute = _make_cute_tensor(output)
        lse_cute = _make_cute_tensor(lse)
        compiled = cute.compile(
            _dsa_sparse_attention_host,
            q_nope_cute,
            q_pe_cute,
            ckv_flat_cute,
            kpe_flat_cute,
            sparse_indices_cute,
            Float32(1.0),
            output_cute,
            lse_cute,
            stream,
        )
        compile_cache[cache_key] = compiled
    return compiled


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    assert q_nope.is_cuda
    assert q_pe.is_cuda
    assert ckv_cache.is_cuda
    assert kpe_cache.is_cuda
    assert sparse_indices.is_cuda
    assert output.is_cuda
    assert lse.is_cuda

    assert q_nope.dtype == torch.bfloat16
    assert q_pe.dtype == torch.bfloat16
    assert ckv_cache.dtype == torch.bfloat16
    assert kpe_cache.dtype == torch.bfloat16
    assert sparse_indices.dtype == torch.int32
    assert output.dtype == torch.bfloat16
    assert lse.dtype == torch.float32

    assert q_nope.ndim == 3 and q_nope.shape[1] == NUM_HEADS and q_nope.shape[2] == CKV_DIM
    assert q_pe.ndim == 3 and q_pe.shape[1] == NUM_HEADS and q_pe.shape[2] == KPE_DIM
    assert ckv_cache.ndim == 3 and ckv_cache.shape[1] == 64 and ckv_cache.shape[2] == CKV_DIM
    assert kpe_cache.ndim == 3 and kpe_cache.shape[1] == 64 and kpe_cache.shape[2] == KPE_DIM
    assert sparse_indices.ndim == 2 and sparse_indices.shape[1] == TOPK
    assert output.shape == q_nope.shape
    assert lse.shape == q_nope.shape[:2]

    ckv_flat = ckv_cache.view(-1, CKV_DIM)
    kpe_flat = kpe_cache.view(-1, KPE_DIM)

    torch_stream = torch.cuda.current_stream(device=q_nope.device)
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = _get_compiled_kernel(
        q_nope,
        q_pe,
        ckv_flat,
        kpe_flat,
        sparse_indices,
        output,
        lse,
        cu_stream,
    )
    compiled(
        q_nope,
        q_pe,
        ckv_flat,
        kpe_flat,
        sparse_indices,
        float(sm_scale) * LOG2_E,
        output,
        lse,
        cu_stream,
    )
