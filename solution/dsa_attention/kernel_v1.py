"""
DSA Sparse Attention Kernel — v2 split-KV baseline with an experimental CuTeDSL
QK path.

The runtime keeps the split-KV combine in PyTorch, but removes the original
Python nested loop by vectorizing the split path. When Cutlass CuTeDSL is
available, it also attempts to offload the ckv QK matmul to a self-contained
Blackwell kernel. If that path is unavailable or fails, execution falls back to
the vectorized torch implementation.

Entry points:
    run(...) -> (output, lse)
    kernel(..., output, lse) -> writes destination tensors in place
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch


TOPK = 2048
CHUNK_SIZE = 64
NUM_Q_HEADS = 16
CKV_DIM = 512
KPE_DIM = 64
PAGE_SIZE = 64
LOG2E = 1.0 / math.log(2.0)

_CUTEDSL_QK_LAUNCH = None
_CUTEDSL_QK_ERROR = None


def _to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _maybe_zero_invalid_chunks(chunks: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if chunks.dtype.is_floating_point:
        return chunks.masked_fill(~valid_mask.unsqueeze(-1), 0)
    return chunks


def _cutedsl_runtime_enabled() -> bool:
    return os.getenv("DSA_ATTENTION_ENABLE_CUTEDSL_QK", "0") == "1"


@dataclass(frozen=True)
class DSASparseAttentionConfig:
    m_block: int = 64
    n_block: int = CHUNK_SIZE
    num_splits: int = TOPK // CHUNK_SIZE
    num_heads: int = NUM_Q_HEADS
    head_dim_ckv: int = CKV_DIM
    head_dim_kpe: int = KPE_DIM
    page_size: int = PAGE_SIZE


def _make_cutedsl_qk_launcher():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass import BFloat16, Float32, Int32
    from cutlass.cute.nvgpu import CopyUniversalOp, tcgen05
    from cutlass.cute.runtime import from_dlpack

    num_threads = 256
    layout_row_major = cutlass.utils.LayoutEnum.ROW_MAJOR
    smem_bytes = (CHUNK_SIZE * CKV_DIM * 2 * 2) + (CHUNK_SIZE * CHUNK_SIZE * 4) + 2048

    @cute.kernel
    def qk_ckv_kernel(
        mQ: cute.Tensor,  # (64, 512, batch)
        mK: cute.Tensor,  # (64, 512, batch)
        mOut: cute.Tensor,  # (64, 64, batch)
        tiled_copy_in_: cute.TiledCopy,
        tiled_copy_out_: cute.TiledCopy,
        tiled_mma_: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        batch_idx, _, _ = cute.arch.block_idx()
        mma_tiler = (CHUNK_SIZE, CHUNK_SIZE, 16)
        sQ_layout = sm100_utils.make_smem_layout_a(tiled_mma_, mma_tiler, BFloat16, 1)
        sK_layout = sm100_utils.make_smem_layout_b(tiled_mma_, mma_tiler, BFloat16, 1)

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(BFloat16, sQ_layout.outer, byte_alignment=16, swizzle=sQ_layout.inner)
        sK = smem.allocate_tensor(BFloat16, sK_layout.outer, byte_alignment=16, swizzle=sK_layout.inner)
        sO = smem.allocate_tensor(Float32, cute.make_layout((CHUNK_SIZE, CHUNK_SIZE)), byte_alignment=16)
        tmem_ptr_buf = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)
        mma_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)

        if tidx == 0:
            cute.arch.mbarrier_init(mma_mbar.iterator, 1)
        cute.arch.mbarrier_init_fence()

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)
        tmem = cutlass.utils.TmemAllocator(
            tmem_ptr_buf.iterator,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )

        gQ = mQ[None, None, batch_idx]
        gK = mK[None, None, batch_idx]
        gO = mOut[None, None, batch_idx]
        sQ_stage = sQ[None, None, None, 0]
        sK_stage = sK[None, None, None, 0]
        sQ_pi = cute.make_tensor(
            sQ_stage.iterator,
            cute.make_layout(
                (sQ_stage.shape[0][0], (sQ_stage.shape[0][1], sQ_stage.shape[2])),
                stride=(sQ_stage.stride[0][0], (sQ_stage.stride[0][1], sQ_stage.stride[2])),
            ),
        )
        sK_pi = cute.make_tensor(
            sK_stage.iterator,
            cute.make_layout(
                (sK_stage.shape[0][0], (sK_stage.shape[0][1], sK_stage.shape[2])),
                stride=(sK_stage.stride[0][0], (sK_stage.stride[0][1], sK_stage.stride[2])),
            ),
        )

        thr_copy_in = tiled_copy_in_.get_slice(tidx)
        tQgQ = thr_copy_in.partition_S(gQ)
        tQsQ = thr_copy_in.partition_D(sQ_pi)
        tKgK = thr_copy_in.partition_S(gK)
        tKsK = thr_copy_in.partition_D(sK_pi)
        tQsQ.fill(0)
        tKsK.fill(0)
        cute.arch.sync_threads()
        cute.copy(tiled_copy_in_, tQgQ, tQsQ)
        cute.copy(tiled_copy_in_, tKgK, tKsK)
        cute.arch.sync_threads()

        acc_shape = tiled_mma_.partition_shape_C((CHUNK_SIZE, CHUNK_SIZE))
        tCtAcc_fake = tiled_mma_.make_fragment_C(acc_shape)
        num_tmem_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        if warp_idx == 0:
            tmem.allocate(num_tmem_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        if warp_idx == 4:
            tCrA = tiled_mma_.make_fragment_A(sQ)
            tCrB = tiled_mma_.make_fragment_B(sK)
            tiled_mma_.set(tcgen05.Field.ACCUMULATE, False)
            for k_blk in cutlass.range_constexpr(CKV_DIM // 16):
                cute.gemm(
                    tiled_mma_,
                    tCtAcc,
                    tCrA[None, None, k_blk, 0],
                    tCrB[None, None, k_blk, 0],
                    tCtAcc,
                )
                tiled_mma_.set(tcgen05.Field.ACCUMULATE, True)
            tcgen05.commit(mma_mbar.iterator, cta_group=tcgen05.CtaGroup.ONE)

        cute.arch.mbarrier_wait(mma_mbar.iterator, 0)

        epi_tile = (CHUNK_SIZE, CHUNK_SIZE)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            mma_tiler,
            layout_row_major,
            Float32,
            Float32,
            epi_tile,
            False,
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_copy_atom, tAcc_epi[(None, None, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        cAcc = cute.make_identity_tensor((CHUNK_SIZE, CHUNK_SIZE))
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[None, None, None, 0, 0].shape, Float32)
        cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, 0, 0], tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        smem_copy_atom = cute.make_copy_atom(CopyUniversalOp(), Float32, num_bits_per_copy=32)
        tiled_copy_r2s = cute.make_tiled_copy_D(smem_copy_atom, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sO = thr_copy_r2s.partition_D(sO)
        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        cute.copy(tiled_copy_r2s, tRS_rAcc, tRS_sO)
        cute.arch.sync_threads()

        thr_copy_out = tiled_copy_out_.get_slice(tidx)
        tOsO = thr_copy_out.partition_S(sO)
        tOgO = thr_copy_out.partition_D(gO)
        cute.copy(tiled_copy_out_, tOsO, tOgO)
        cute.arch.sync_threads()

        # The minimal smoke path keeps the accumulator live until CTA exit.

    @cute.jit
    def launch(g_q, g_k, g_out, num_tokens, stream):
        mma_tiler = (CHUNK_SIZE, CHUNK_SIZE, 16)
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            cta_group,
            mma_tiler[:2],
        )

        input_elems_per_copy = 128 // BFloat16.width
        input_thread_layout = cute.make_layout(
            (num_threads // (CKV_DIM // input_elems_per_copy), CKV_DIM // input_elems_per_copy)
        )
        input_value_layout = cute.make_layout((1, input_elems_per_copy))
        tiled_copy_in = cute.make_tiled_copy_tv(
            cute.make_copy_atom(CopyUniversalOp(), BFloat16, num_bits_per_copy=128),
            input_thread_layout,
            input_value_layout,
        )

        output_elems_per_copy = 1
        output_thread_layout = cute.make_layout(
            (num_threads // (CHUNK_SIZE // output_elems_per_copy), CHUNK_SIZE // output_elems_per_copy)
        )
        output_value_layout = cute.make_layout((1, output_elems_per_copy))
        tiled_copy_out = cute.make_tiled_copy_tv(
            cute.make_copy_atom(CopyUniversalOp(), Float32, num_bits_per_copy=32),
            output_thread_layout,
            output_value_layout,
        )

        mQ = cute.make_tensor(g_q.iterator, cute.select(g_q.layout, mode=[1, 2, 0]))
        mK = cute.make_tensor(g_k.iterator, cute.select(g_k.layout, mode=[1, 2, 0]))
        mOut = cute.make_tensor(g_out.iterator, cute.select(g_out.layout, mode=[1, 2, 0]))
        qk_ckv_kernel(
            mQ,
            mK,
            mOut,
            tiled_copy_in,
            tiled_copy_out,
            tiled_mma,
        ).launch(
            grid=[g_q.shape[0], 1, 1],
            block=[num_threads, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    def run(q_bmk: torch.Tensor, k_bnk: torch.Tensor, out_bmn: torch.Tensor) -> None:
        launch(
            from_dlpack(q_bmk, assumed_align=16),
            from_dlpack(k_bnk, assumed_align=16),
            from_dlpack(out_bmn, assumed_align=16),
            q_bmk.shape[0] // (TOPK // CHUNK_SIZE),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def compile_only(q_bmk: torch.Tensor, k_bnk: torch.Tensor, out_bmn: torch.Tensor) -> None:
        from cutlass.base_dsl.compiler import CompileCallable

        compiler = CompileCallable()
        compiler(
            launch,
            from_dlpack(q_bmk, assumed_align=16),
            from_dlpack(k_bnk, assumed_align=16),
            from_dlpack(out_bmn, assumed_align=16),
            q_bmk.shape[0] // (TOPK // CHUNK_SIZE),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    run.compile_only = compile_only
    return run


def _get_cutedsl_qk_launcher():
    global _CUTEDSL_QK_LAUNCH, _CUTEDSL_QK_ERROR
    if _CUTEDSL_QK_LAUNCH is not None or _CUTEDSL_QK_ERROR is not None:
        return _CUTEDSL_QK_LAUNCH
    try:
        _CUTEDSL_QK_LAUNCH = _make_cutedsl_qk_launcher()
    except Exception as exc:  # pragma: no cover - runtime-only path
        _CUTEDSL_QK_ERROR = exc
    return _CUTEDSL_QK_LAUNCH


class DSASparseAttentionV1:
    """Split-KV implementation with a vectorized torch path and optional CuTeDSL QK."""

    def __init__(self, config: DSASparseAttentionConfig | None = None) -> None:
        self.config = config or DSASparseAttentionConfig()

    def _validate_inputs(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
    ) -> None:
        if q_nope.ndim != 3 or q_nope.shape[1:] != (self.config.num_heads, self.config.head_dim_ckv):
            raise ValueError(f"q_nope must be [num_tokens, {self.config.num_heads}, {self.config.head_dim_ckv}]")
        if q_pe.ndim != 3 or q_pe.shape[1:] != (self.config.num_heads, self.config.head_dim_kpe):
            raise ValueError(f"q_pe must be [num_tokens, {self.config.num_heads}, {self.config.head_dim_kpe}]")
        if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (self.config.page_size, self.config.head_dim_ckv):
            raise ValueError(f"ckv_cache must be [num_pages, {self.config.page_size}, {self.config.head_dim_ckv}]")
        if kpe_cache.ndim != 3 or kpe_cache.shape[1:] != (self.config.page_size, self.config.head_dim_kpe):
            raise ValueError(f"kpe_cache must be [num_pages, {self.config.page_size}, {self.config.head_dim_kpe}]")
        if sparse_indices.ndim != 2 or sparse_indices.shape[1] != self.config.num_splits * self.config.n_block:
            raise ValueError(f"sparse_indices must be [num_tokens, {self.config.num_splits * self.config.n_block}]")
        if sparse_indices.shape[0] != q_nope.shape[0]:
            raise ValueError("sparse_indices batch dimension must match q_nope")

    def _gather_kv_chunks(
        self,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        valid_mask = sparse_indices != -1
        safe_indices = sparse_indices.clamp_min(0).to(torch.long)

        kc = ckv_cache.reshape(-1, cfg.head_dim_ckv)[safe_indices]
        kp = kpe_cache.reshape(-1, cfg.head_dim_kpe)[safe_indices]

        kc = kc.view(-1, cfg.num_splits, cfg.n_block, cfg.head_dim_ckv).contiguous()
        kp = kp.view(-1, cfg.num_splits, cfg.n_block, cfg.head_dim_kpe).contiguous()
        valid_mask = valid_mask.view(-1, cfg.num_splits, cfg.n_block)

        kc = _maybe_zero_invalid_chunks(kc, valid_mask)
        kp = _maybe_zero_invalid_chunks(kp, valid_mask)
        return kc, kp, valid_mask

    def _compute_qk_ckv_cutedsl(
        self,
        q_nope: torch.Tensor,
        kc_chunks: torch.Tensor,
    ) -> torch.Tensor | None:
        if not _cutedsl_runtime_enabled():
            return None
        launcher = _get_cutedsl_qk_launcher()
        if launcher is None or not q_nope.is_cuda or not kc_chunks.is_cuda:
            return None

        cfg = self.config
        try:
            q_padded = torch.zeros(
                (q_nope.shape[0], cfg.m_block, cfg.head_dim_ckv),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            q_padded[:, : cfg.num_heads].copy_(q_nope)
            q_expanded = (
                q_padded[:, None, :, :]
                .expand(-1, cfg.num_splits, -1, -1)
                .contiguous()
                .view(-1, cfg.m_block, cfg.head_dim_ckv)
            )
            k_flat = kc_chunks.contiguous().view(-1, cfg.n_block, cfg.head_dim_ckv)
            out = torch.empty(
                (k_flat.shape[0], cfg.m_block, cfg.n_block),
                dtype=torch.float32,
                device=q_nope.device,
            )
            launcher(q_expanded, k_flat, out)
            torch.cuda.synchronize()
            return out.view(q_nope.shape[0], cfg.num_splits, cfg.m_block, cfg.n_block)[:, :, : cfg.num_heads]
        except Exception:
            return None

    def _compute_split_partials(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        kc_chunks, kp_chunks, valid_mask = self._gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)

        q_nope_f = q_nope.to(torch.float32)
        q_pe_f = q_pe.to(torch.float32)
        kc_f = kc_chunks.to(torch.float32)
        kp_f = kp_chunks.to(torch.float32)

        logits_ckv = self._compute_qk_ckv_cutedsl(q_nope, kc_chunks)
        if logits_ckv is None:
            logits_ckv = torch.einsum("thd,tsnd->tshn", q_nope_f, kc_f)

        logits_kpe = torch.einsum("thd,tsnd->tshn", q_pe_f, kp_f)
        logits = (logits_ckv + logits_kpe) * sm_scale

        mask = valid_mask[:, :, None, :]
        split_has_tokens = valid_mask.any(dim=-1, keepdim=True)
        masked_logits = logits.masked_fill(~mask, -float("inf"))
        safe_logits = torch.where(split_has_tokens[:, :, :, None], masked_logits, torch.zeros_like(masked_logits))

        attn = torch.softmax(safe_logits, dim=-1)
        attn = attn.masked_fill(~split_has_tokens[:, :, :, None], 0.0)
        partial_o = torch.einsum("tshn,tsnd->tshd", attn, kc_f)

        partial_lse = torch.full(
            (q_nope.shape[0], cfg.num_splits, cfg.num_heads),
            -float("inf"),
            dtype=torch.float32,
            device=q_nope.device,
        )
        valid_splits = split_has_tokens.expand(-1, -1, cfg.num_heads)
        partial_lse[valid_splits] = torch.logsumexp(masked_logits, dim=-1)[valid_splits] * LOG2E

        return partial_o.permute(1, 0, 2, 3).contiguous(), partial_lse.permute(1, 0, 2).contiguous()

    def _combine_splits(
        self,
        partial_o: torch.Tensor,
        partial_lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_lse = partial_lse.max(dim=0).values
        finite_mask = torch.isfinite(max_lse)
        stable_max = torch.where(finite_mask, max_lse, torch.zeros_like(max_lse))

        weights = torch.exp2(partial_lse - stable_max.unsqueeze(0))
        weight_sum = weights.sum(dim=0)
        combined = (weights.unsqueeze(-1) * partial_o).sum(dim=0)

        output = torch.zeros_like(combined)
        valid_rows = weight_sum > 0
        output[valid_rows] = combined[valid_rows] / weight_sum[valid_rows].unsqueeze(-1)

        final_lse = torch.full_like(max_lse, -float("inf"))
        final_lse[valid_rows] = stable_max[valid_rows] + torch.log2(weight_sum[valid_rows])
        return output.to(torch.bfloat16), final_lse

    @torch.no_grad()
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        sm_scale,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices)

        partial_o, partial_lse = self._compute_split_partials(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
            _to_python_float(sm_scale),
        )
        return self._combine_splits(partial_o, partial_lse)


_V1_KERNEL = DSASparseAttentionV1()


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    return _V1_KERNEL.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    out, lse_out = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output.copy_(out)
    lse.copy_(lse_out)


def _cutedsl_smoke_compile() -> None:
    launcher = _get_cutedsl_qk_launcher()
    if launcher is None:
        raise RuntimeError(f"CuTeDSL QK launcher unavailable: {_CUTEDSL_QK_ERROR!r}")

    q = torch.randn(TOPK // CHUNK_SIZE, CHUNK_SIZE, CKV_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    out = torch.empty((TOPK // CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), device="cuda", dtype=torch.float32)
    if not hasattr(launcher, "compile_only"):
        raise RuntimeError("CuTeDSL QK launcher does not expose compile_only()")
    launcher.compile_only(q, k, out)
    print("CuTeDSL QK compile PASS")


if __name__ == "__main__":
    try:
        _cutedsl_smoke_compile()
    except Exception as exc:
        print(f"CuTeDSL smoke compile FAILED: {exc}")
        raise
