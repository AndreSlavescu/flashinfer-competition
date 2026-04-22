from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cutlass_dsl import BFloat16, Float32


NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048
MAX_PAGE_SLOTS = TOPK // PAGE_SIZE
SCHED_PREVIEW_COUNT = 32
STAGE1_BLOCK_THREADS = 256
STAGE2_BLOCK_THREADS = 32
STAGE2_QK_TILE_MN = (64, 64)
STAGE2_PV_TILE_MN = (64, 128)
STAGE2_QK_NOPE_TILER = (64, 64, HEAD_DIM_CKV)
STAGE2_QK_PE_TILER = (64, 64, HEAD_DIM_KPE)
STAGE2_PV_TILER = (64, 128, PAGE_SIZE)
STAGE2_V_PIPE_STAGES = 2
STAGE2_LAYOUT_COSIZE_ORDER = (
    "sQ_nope",
    "sQ_pe",
    "sK_nope",
    "sK_pe",
    "sV",
    "sP",
)
STAGE3_BLOCK_THREADS = 64
STAGE3_PAGE_PIPE_STAGES = 1
LOG2_E = 1.0 / math.log(2.0)


class _Stage3SharedStorage:
    pass


_Stage3SharedStorage.__annotations__ = {
    "page_pipe_mbar_ptr": cute.struct.MemRange[cutlass.Int64, STAGE3_PAGE_PIPE_STAGES * 2],
    "smem_page_ids": cute.struct.MemRange[cutlass.Int32, MAX_PAGE_SLOTS],
}
Stage3SharedStorage = cute.struct(_Stage3SharedStorage)


program_compile_cache: dict[tuple[Any, ...], Any] = {}


def _compute_persistent_grid_x(num_tokens: int, num_sms: int) -> int:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive for the round-0 scheduler, got {num_tokens}")
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive for the round-0 scheduler, got {num_sms}")
    return min(num_tokens, num_sms)


def _build_cta0_schedule_preview(
    num_tokens: int,
    grid_x: int,
    preview_count: int = SCHED_PREVIEW_COUNT,
) -> tuple[torch.Tensor, int]:
    preview = torch.full((preview_count,), -1, dtype=torch.int32)
    preview_len = 0

    for preview_idx in range(preview_count):
        token_idx = preview_idx * grid_x
        if token_idx >= num_tokens:
            break
        preview[preview_idx] = token_idx
        preview_len += 1

    return preview[:preview_len].clone(), preview_len


def _get_cuda_driver_module() -> Any:
    try:
        import cuda.bindings.driver as cuda_driver

        return cuda_driver
    except ImportError:
        try:
            from cuda import cuda as cuda_driver

            return cuda_driver
        except ImportError as exc:
            raise RuntimeError(
                "Unable to import a CUDA driver binding required for CuTeDSL kernel launch."
            ) from exc


def _round_up_to_pow2_at_least_32(value: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return max(32, 1 << (value - 1).bit_length())


@dataclass(frozen=True)
class Stage0ReferenceBundle:
    output: torch.Tensor
    lse: torch.Tensor
    valid_page_count: torch.Tensor
    page_ids: torch.Tensor
    contract_ok: torch.Tensor


def _make_json_payload(
    value: torch.Tensor | int | float | bool,
    *,
    preview_count: int | None = None,
) -> str:
    if isinstance(value, torch.Tensor):
        host_value = value.detach().cpu()
        payload: dict[str, Any] = {
            "shape": list(host_value.shape),
            "dtype": str(host_value.dtype),
        }
        if preview_count is None:
            payload["values"] = host_value.tolist()
        else:
            flat_value = host_value.reshape(-1)
            preview_len = min(preview_count, flat_value.numel())
            payload["preview_len"] = preview_len
            payload["preview"] = flat_value[:preview_len].tolist()
        return json.dumps(payload)

    if isinstance(value, bool):
        return json.dumps({"value": bool(value)})
    if isinstance(value, int):
        return json.dumps({"value": int(value)})
    if isinstance(value, float):
        return json.dumps({"value": float(value)})
    raise TypeError(f"Unsupported value type for JSON payload: {type(value)!r}")


def _print_validation_block(
    source: str,
    name: str,
    value: torch.Tensor | int | float | bool,
    *,
    preview_count: int | None = None,
) -> None:
    print(f"[{source}] {name}: BEGIN")
    print(_make_json_payload(value, preview_count=preview_count))
    print(f"[{source}] {name}: END")


def _extract_page_dense_metadata_for_token(
    sparse_index_row: torch.Tensor, num_pages: int
) -> tuple[bool, torch.Tensor, int]:
    row = sparse_index_row.detach().cpu().to(torch.int64).tolist()

    padded_page_ids = torch.full((MAX_PAGE_SLOTS,), -1, dtype=torch.int32)
    collected_page_ids: list[int] = []
    seen_page_ids: set[int] = set()
    saw_invalid_tail = False
    page_dense_ok = True

    for page_slot in range(MAX_PAGE_SLOTS):
        chunk_begin = page_slot * PAGE_SIZE
        chunk_end = chunk_begin + PAGE_SIZE
        page_chunk = row[chunk_begin:chunk_end]

        if len(page_chunk) != PAGE_SIZE:
            page_dense_ok = False
            break

        if all(token_idx == -1 for token_idx in page_chunk):
            saw_invalid_tail = True
            continue

        if saw_invalid_tail:
            page_dense_ok = False
            break

        first_entry = page_chunk[0]
        if first_entry < 0 or first_entry % PAGE_SIZE != 0:
            page_dense_ok = False
            break

        page_id = first_entry // PAGE_SIZE
        expected_chunk = list(range(page_id * PAGE_SIZE, (page_id + 1) * PAGE_SIZE))
        if page_chunk != expected_chunk:
            page_dense_ok = False
            break
        if page_id >= num_pages or page_id in seen_page_ids:
            page_dense_ok = False
            break

        seen_page_ids.add(page_id)
        collected_page_ids.append(page_id)

    if collected_page_ids:
        padded_page_ids[: len(collected_page_ids)] = torch.tensor(
            collected_page_ids, dtype=torch.int32
        )

    return page_dense_ok, padded_page_ids, len(collected_page_ids)


def torch_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Stage0ReferenceBundle:
    if q_nope.ndim != 3 or q_nope.shape[1:] != (NUM_QO_HEADS, HEAD_DIM_CKV):
        raise ValueError(
            f"q_nope must have shape [T, {NUM_QO_HEADS}, {HEAD_DIM_CKV}], got {tuple(q_nope.shape)}"
        )
    if q_pe.ndim != 3 or q_pe.shape[1:] != (NUM_QO_HEADS, HEAD_DIM_KPE):
        raise ValueError(
            f"q_pe must have shape [T, {NUM_QO_HEADS}, {HEAD_DIM_KPE}], got {tuple(q_pe.shape)}"
        )
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (PAGE_SIZE, HEAD_DIM_CKV):
        raise ValueError(
            f"ckv_cache must have shape [P, {PAGE_SIZE}, {HEAD_DIM_CKV}], got {tuple(ckv_cache.shape)}"
        )
    if kpe_cache.ndim != 3 or kpe_cache.shape[1:] != (PAGE_SIZE, HEAD_DIM_KPE):
        raise ValueError(
            f"kpe_cache must have shape [P, {PAGE_SIZE}, {HEAD_DIM_KPE}], got {tuple(kpe_cache.shape)}"
        )
    if sparse_indices.ndim != 2 or sparse_indices.shape[1] != TOPK:
        raise ValueError(f"sparse_indices must have shape [T, {TOPK}], got {tuple(sparse_indices.shape)}")

    num_tokens = q_nope.shape[0]
    num_pages = ckv_cache.shape[0]
    device = q_nope.device

    output = torch.zeros(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.full(
        (num_tokens, NUM_QO_HEADS),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    valid_page_count = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    page_ids = torch.full(
        (num_tokens, MAX_PAGE_SLOTS),
        -1,
        dtype=torch.int32,
        device=device,
    )
    contract_ok = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

    for token_idx in range(num_tokens):
        token_contract_ok, token_page_ids, token_valid_pages = (
            _extract_page_dense_metadata_for_token(
                sparse_indices[token_idx], num_pages=num_pages
            )
        )
        page_ids[token_idx] = token_page_ids.to(device=device)
        valid_page_count[token_idx] = token_valid_pages
        contract_ok[token_idx] = int(token_contract_ok)

        if not token_contract_ok:
            continue
        if token_valid_pages == 0:
            continue

        active_page_ids = token_page_ids[:token_valid_pages].to(
            device=device, dtype=torch.long
        )
        selected_ckv = ckv_cache.index_select(0, active_page_ids).reshape(
            token_valid_pages * PAGE_SIZE, HEAD_DIM_CKV
        )
        selected_kpe = kpe_cache.index_select(0, active_page_ids).reshape(
            token_valid_pages * PAGE_SIZE, HEAD_DIM_KPE
        )

        q_nope_f32 = q_nope[token_idx].to(torch.float32)
        q_pe_f32 = q_pe[token_idx].to(torch.float32)
        selected_ckv_f32 = selected_ckv.to(torch.float32)
        selected_kpe_f32 = selected_kpe.to(torch.float32)

        logits = float(sm_scale) * (
            q_nope_f32 @ selected_ckv_f32.transpose(0, 1)
            + q_pe_f32 @ selected_kpe_f32.transpose(0, 1)
        )
        probs = torch.softmax(logits, dim=-1)

        output[token_idx] = (probs @ selected_ckv_f32).to(torch.bfloat16)
        lse[token_idx] = torch.logsumexp(logits, dim=-1) * LOG2_E

    return Stage0ReferenceBundle(
        output=output,
        lse=lse,
        valid_page_count=valid_page_count,
        page_ids=page_ids,
        contract_ok=contract_ok,
    )


def create_synthetic_data(device: str = "cuda") -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    torch.manual_seed(2025)

    num_tokens = 1
    num_pages = 8
    supported_page_ids = [1, 3]

    q_nope = torch.randn(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV),
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    q_pe = torch.randn(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE),
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    ckv_cache = torch.randn(
        (num_pages, PAGE_SIZE, HEAD_DIM_CKV),
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    kpe_cache = torch.randn(
        (num_pages, PAGE_SIZE, HEAD_DIM_KPE),
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)

    sparse_indices = torch.full(
        (num_tokens, TOPK), -1, dtype=torch.int32, device=device
    )
    for page_slot, page_id in enumerate(supported_page_ids):
        token_begin = page_slot * PAGE_SIZE
        sparse_indices[0, token_begin : token_begin + PAGE_SIZE] = torch.arange(
            page_id * PAGE_SIZE,
            (page_id + 1) * PAGE_SIZE,
            dtype=torch.int32,
            device=device,
        )

    sm_scale = 1.0 / math.sqrt(float(HEAD_DIM_CKV))
    return q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale


class Stage0SupportContractProgram:
    def __init__(self, threads_per_block: int = 32):
        self.threads_per_block = threads_per_block

    @staticmethod
    def _compute_grid(num_tokens: int) -> tuple[int, int, int]:
        return (num_tokens, 1, 1)

    @cute.jit
    def _print_contract_status(self, contract_ok: cutlass.Int32):
        cute.printf("[CuTeDSL] contract.page_dense_ok: BEGIN\n")
        cute.printf("{\"value\": %d}\n", contract_ok)
        cute.printf("[CuTeDSL] contract.page_dense_ok: END\n")

    @cute.jit
    def __call__(
        self,
        sparse_indices: cute.Tensor,
        page_ids_out: cute.Tensor,
        valid_page_count_out: cute.Tensor,
        contract_ok_out: cute.Tensor,
        num_pages: cutlass.Int32,
        stream,
    ):
        self.contract_checker_kernel(
            sparse_indices,
            page_ids_out,
            valid_page_count_out,
            contract_ok_out,
            num_pages,
        ).launch(
            grid=self._compute_grid(sparse_indices.shape[0]),
            block=[self.threads_per_block, 1, 1],
            smem=0,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def contract_checker_kernel(
        self,
        sparse_indices: cute.Tensor,
        page_ids_out: cute.Tensor,
        valid_page_count_out: cute.Tensor,
        contract_ok_out: cute.Tensor,
        num_pages: cutlass.Int32,
    ):
        token_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()

        if thread_idx == 0:
            contract_ok = 1
            valid_page_count = 0
            saw_invalid_tail = 0

            for page_slot in cutlass.range_constexpr(MAX_PAGE_SLOTS):
                page_ids_out[token_idx, page_slot] = -1

            for page_slot in cutlass.range_constexpr(MAX_PAGE_SLOTS):
                slot_base = page_slot * PAGE_SIZE
                first_entry = sparse_indices[token_idx, slot_base]

                if first_entry == -1:
                    saw_invalid_tail = 1
                    for element_idx in cutlass.range_constexpr(PAGE_SIZE):
                        if sparse_indices[token_idx, slot_base + element_idx] != -1:
                            contract_ok = 0
                else:
                    if saw_invalid_tail == 1:
                        contract_ok = 0

                    if first_entry < 0 or first_entry % PAGE_SIZE != 0:
                        contract_ok = 0

                    page_id = first_entry // PAGE_SIZE
                    if page_id < 0 or page_id >= num_pages:
                        contract_ok = 0

                    for element_idx in cutlass.range_constexpr(PAGE_SIZE):
                        expected_token_idx = page_id * PAGE_SIZE + element_idx
                        if sparse_indices[token_idx, slot_base + element_idx] != expected_token_idx:
                            contract_ok = 0

                    for previous_slot in cutlass.range_constexpr(MAX_PAGE_SLOTS):
                        if cute.elem_less(previous_slot, valid_page_count):
                            if page_ids_out[token_idx, previous_slot] == page_id:
                                contract_ok = 0

                    page_ids_out[token_idx, valid_page_count] = page_id
                    valid_page_count += 1

            valid_page_count_out[token_idx] = valid_page_count
            contract_ok_out[token_idx] = contract_ok

            if token_idx == 0:
                self._print_contract_status(contract_ok)


def _run_cutedsl_contract_kernel(
    sparse_indices: torch.Tensor,
    num_pages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch_stream = torch.cuda.current_stream()
    cuda_driver = _get_cuda_driver_module()
    stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    page_ids_out = torch.full(
        (sparse_indices.shape[0], MAX_PAGE_SLOTS),
        -1,
        dtype=torch.int32,
        device=sparse_indices.device,
    )
    valid_page_count_out = torch.zeros(
        (sparse_indices.shape[0],), dtype=torch.int32, device=sparse_indices.device
    )
    contract_ok_out = torch.zeros(
        (sparse_indices.shape[0],), dtype=torch.int32, device=sparse_indices.device
    )

    program = Stage0SupportContractProgram()
    cache_key = (
        "stage0_support_contract_program",
        tuple(sparse_indices.shape),
        str(sparse_indices.dtype),
        sparse_indices.device.type,
    )

    if cache_key not in program_compile_cache:
        program_compile_cache[cache_key] = cute.compile(
            program,
            sparse_indices,
            page_ids_out,
            valid_page_count_out,
            contract_ok_out,
            int(num_pages),
            stream,
            options="--opt-level 2",
        )

    compiled_program = program_compile_cache[cache_key]
    compiled_program(
        sparse_indices,
        page_ids_out,
        valid_page_count_out,
        contract_ok_out,
        int(num_pages),
        stream,
    )
    torch.cuda.synchronize()

    return page_ids_out, valid_page_count_out, contract_ok_out


class Stage1PersistentSchedulerProgram:
    def __init__(
        self,
        num_tokens: int,
        num_sms: int,
        threads_per_block: int = STAGE1_BLOCK_THREADS,
    ):
        self.num_tokens = int(num_tokens)
        self.num_sms = int(num_sms)
        self.threads_per_block = int(threads_per_block)
        self.grid_x = _compute_persistent_grid_x(
            num_tokens=self.num_tokens,
            num_sms=self.num_sms,
        )

    @staticmethod
    def _compute_grid(grid_x: int) -> tuple[int, int, int]:
        return (grid_x, 1, 1)

    @cute.jit
    def __call__(
        self,
        schedule_preview_out: cute.Tensor,
        preview_len_out: cute.Tensor,
        block_threads_out: cute.Tensor,
        grid_x_out: cute.Tensor,
        stream,
    ):
        self.scheduler_kernel(
            schedule_preview_out,
            preview_len_out,
            block_threads_out,
            grid_x_out,
        ).launch(
            grid=self._compute_grid(self.grid_x),
            block=[self.threads_per_block, 1, 1],
            smem=0,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def scheduler_kernel(
        self,
        schedule_preview_out: cute.Tensor,
        preview_len_out: cute.Tensor,
        block_threads_out: cute.Tensor,
        grid_x_out: cute.Tensor,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        block_threads, _, _ = cute.arch.block_dim()

        if bidx == 0:
            if tidx == 0:
                block_threads_out[0] = block_threads
                grid_x_out[0] = grid_x

                preview_len = 0
                for preview_idx in cutlass.range_constexpr(SCHED_PREVIEW_COUNT):
                    token_idx = bidx + preview_idx * grid_x
                    if token_idx < self.num_tokens:
                        schedule_preview_out[preview_idx] = token_idx
                        preview_len += 1
                    else:
                        schedule_preview_out[preview_idx] = -1

                preview_len_out[0] = preview_len


def _run_cutedsl_scheduler_kernel(
    *,
    num_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    torch_stream = torch.cuda.current_stream(device=device)
    cuda_driver = _get_cuda_driver_module()
    stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    schedule_preview_out = torch.full(
        (SCHED_PREVIEW_COUNT,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    preview_len_out = torch.zeros((1,), dtype=torch.int32, device=device)
    block_threads_out = torch.zeros((1,), dtype=torch.int32, device=device)
    grid_x_out = torch.zeros((1,), dtype=torch.int32, device=device)

    program = Stage1PersistentSchedulerProgram(
        num_tokens=num_tokens,
        num_sms=num_sms,
        threads_per_block=STAGE1_BLOCK_THREADS,
    )
    cache_key = (
        "stage1_persistent_scheduler_program",
        int(num_tokens),
        int(num_sms),
        int(program.threads_per_block),
        str(device),
    )

    if cache_key not in program_compile_cache:
        program_compile_cache[cache_key] = cute.compile(
            program,
            schedule_preview_out,
            preview_len_out,
            block_threads_out,
            grid_x_out,
            stream,
            options="--opt-level 2",
        )

    compiled_program = program_compile_cache[cache_key]
    compiled_program(
        schedule_preview_out,
        preview_len_out,
        block_threads_out,
        grid_x_out,
        stream,
    )
    torch.cuda.synchronize(device=device)

    return schedule_preview_out, preview_len_out, block_threads_out, grid_x_out


def _get_stage2_expected_metadata() -> dict[str, torch.Tensor | int]:
    qk_mma_shape = torch.tensor([64, 64, 16], dtype=torch.int32)
    pv_mma_shape = torch.tensor([64, 128, 16], dtype=torch.int32)
    layout_cosizes = torch.tensor(
        [
            64 * HEAD_DIM_CKV,
            64 * HEAD_DIM_KPE,
            64 * HEAD_DIM_CKV,
            64 * HEAD_DIM_KPE,
            64 * STAGE2_PV_TILE_MN[1] * STAGE2_V_PIPE_STAGES,
            64 * PAGE_SIZE,
        ],
        dtype=torch.int32,
    )
    smem_bytes = int(
        (64 * HEAD_DIM_CKV) * 2
        + (64 * HEAD_DIM_KPE) * 2
        + (64 * HEAD_DIM_CKV) * 2
        + (64 * HEAD_DIM_KPE) * 2
        + (64 * STAGE2_PV_TILE_MN[1] * STAGE2_V_PIPE_STAGES) * 2
        + (64 * PAGE_SIZE) * 2
    )
    tmem_score_cols = 64
    tmem_output_cols = 256
    tmem_cols = _round_up_to_pow2_at_least_32(tmem_score_cols + tmem_output_cols)
    return {
        "qk_mma_shape": qk_mma_shape,
        "pv_mma_shape": pv_mma_shape,
        "layout_cosizes": layout_cosizes,
        "smem_bytes": smem_bytes,
        "tmem_cols": tmem_cols,
    }


class Stage2InfrastructureProgram:
    def __init__(self, threads_per_block: int = STAGE2_BLOCK_THREADS):
        self.threads_per_block = int(threads_per_block)
        self.qk_nope_tiler = STAGE2_QK_NOPE_TILER
        self.qk_pe_tiler = STAGE2_QK_PE_TILER
        self.pv_tiler = STAGE2_PV_TILER
        self.v_pipe_stages = STAGE2_V_PIPE_STAGES

    @staticmethod
    def _compute_grid() -> tuple[int, int, int]:
        return (1, 1, 1)

    @cute.jit
    def _print_qk_mma_shape(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
    ):
        cute.printf("[CuTeDSL] infra.qk_mma_shape: BEGIN\n")
        cute.printf(
            "{\"shape\": [3], \"dtype\": \"int32\", \"values\": [%d, %d, %d]}\n",
            m,
            n,
            k,
        )
        cute.printf("[CuTeDSL] infra.qk_mma_shape: END\n")

    @cute.jit
    def _print_pv_mma_shape(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
    ):
        cute.printf("[CuTeDSL] infra.pv_mma_shape: BEGIN\n")
        cute.printf(
            "{\"shape\": [3], \"dtype\": \"int32\", \"values\": [%d, %d, %d]}\n",
            m,
            n,
            k,
        )
        cute.printf("[CuTeDSL] infra.pv_mma_shape: END\n")

    @cute.jit
    def _print_smem_bytes(self, smem_bytes: cutlass.Int32):
        cute.printf("[CuTeDSL] infra.smem_bytes: BEGIN\n")
        cute.printf("{\"value\": %d}\n", smem_bytes)
        cute.printf("[CuTeDSL] infra.smem_bytes: END\n")

    @cute.jit
    def _print_tmem_cols(self, tmem_cols: cutlass.Int32):
        cute.printf("[CuTeDSL] infra.tmem_cols: BEGIN\n")
        cute.printf("{\"value\": %d}\n", tmem_cols)
        cute.printf("[CuTeDSL] infra.tmem_cols: END\n")

    @cute.jit
    def _print_layout_cosizes(
        self,
        q_nope_cosize: cutlass.Int32,
        q_pe_cosize: cutlass.Int32,
        k_nope_cosize: cutlass.Int32,
        k_pe_cosize: cutlass.Int32,
        v_cosize: cutlass.Int32,
        p_cosize: cutlass.Int32,
    ):
        cute.printf("[CuTeDSL] infra.layout_cosizes: BEGIN\n")
        cute.printf(
            "{\"shape\": [6], \"dtype\": \"int32\", \"values\": [%d, %d, %d, %d, %d, %d]}\n",
            q_nope_cosize,
            q_pe_cosize,
            k_nope_cosize,
            k_pe_cosize,
            v_cosize,
            p_cosize,
        )
        cute.printf("[CuTeDSL] infra.layout_cosizes: END\n")

    @cute.jit
    def __call__(
        self,
        qk_mma_shape_out: cute.Tensor,
        pv_mma_shape_out: cute.Tensor,
        smem_bytes_out: cute.Tensor,
        tmem_cols_out: cute.Tensor,
        layout_cosizes_out: cute.Tensor,
        stream,
    ):
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            STAGE2_QK_TILE_MN,
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            STAGE2_PV_TILE_MN,
        )

        q_nope_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_nope_tiler,
            BFloat16,
            1,
        )
        q_pe_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_pe_tiler,
            BFloat16,
            1,
        )
        k_nope_layout = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_nope_tiler,
            BFloat16,
            1,
        )
        k_pe_layout = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_pe_tiler,
            BFloat16,
            1,
        )
        v_layout = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_tiler,
            BFloat16,
            self.v_pipe_stages,
        )
        p_layout = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_tiler,
            BFloat16,
            1,
        )

        q_nope_cosize = cute.cosize(q_nope_layout)
        q_pe_cosize = cute.cosize(q_pe_layout)
        k_nope_cosize = cute.cosize(k_nope_layout)
        k_pe_cosize = cute.cosize(k_pe_layout)
        v_cosize = cute.cosize(v_layout)
        p_cosize = cute.cosize(p_layout)

        smem_bytes = (
            cute.size_in_bytes(BFloat16, q_nope_layout)
            + cute.size_in_bytes(BFloat16, q_pe_layout)
            + cute.size_in_bytes(BFloat16, k_nope_layout)
            + cute.size_in_bytes(BFloat16, k_pe_layout)
            + cute.size_in_bytes(BFloat16, v_layout)
            + cute.size_in_bytes(BFloat16, p_layout)
        )

        tS_shape = qk_tiled_mma.partition_shape_C(
            cute.select(self.qk_nope_tiler, mode=[0, 1])
        )
        tS_tile = qk_tiled_mma.make_fragment_C(tS_shape)
        tO_shape = pv_tiled_mma.partition_shape_C(
            cute.select(self.pv_tiler, mode=[0, 1])
        )
        tO_base = pv_tiled_mma.make_fragment_C(tO_shape)
        tO_layout = cute.append(
            tO_base.layout,
            cute.make_layout(
                HEAD_DIM_CKV // STAGE2_PV_TILE_MN[1],
                stride=STAGE2_PV_TILE_MN[1],
            ),
        )
        tO_accum = cute.make_tensor(tO_base.iterator, tO_layout)
        _tmem_o_col_offset = tcgen05.find_tmem_tensor_col_offset(tS_tile)
        _tmem_total_cols_unrounded = (
            _tmem_o_col_offset + tcgen05.find_tmem_tensor_col_offset(tO_accum)
        )
        _ = _tmem_total_cols_unrounded
        tmem_cols = 512

        @cute.struct
        class SharedStorage:
            page_pipe_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            k_pipe_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            score_pipe_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            p_pipe_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            v_pipe_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            o_pipe_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            smem_q_nope: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(q_nope_layout)],
                1024,
            ]
            smem_q_pe: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(q_pe_layout)],
                1024,
            ]
            smem_k_nope: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(k_nope_layout)],
                1024,
            ]
            smem_k_pe: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(k_pe_layout)],
                1024,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(v_layout)],
                1024,
            ]
            smem_p: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(p_layout)],
                1024,
            ]
            smem_page_ids: cute.struct.MemRange[cutlass.Int32, MAX_PAGE_SLOTS]
            valid_page_count: cutlass.Int32

        self.infrastructure_kernel(
            qk_mma_shape_out,
            pv_mma_shape_out,
            smem_bytes_out,
            tmem_cols_out,
            layout_cosizes_out,
            int(qk_tiled_mma.op.shape_mnk[0]),
            int(qk_tiled_mma.op.shape_mnk[1]),
            int(qk_tiled_mma.op.shape_mnk[2]),
            int(pv_tiled_mma.op.shape_mnk[0]),
            int(pv_tiled_mma.op.shape_mnk[1]),
            int(pv_tiled_mma.op.shape_mnk[2]),
            int(q_nope_cosize),
            int(q_pe_cosize),
            int(k_nope_cosize),
            int(k_pe_cosize),
            int(v_cosize),
            int(p_cosize),
            int(smem_bytes),
            int(tmem_cols),
        ).launch(
            grid=self._compute_grid(),
            block=[self.threads_per_block, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def infrastructure_kernel(
        self,
        qk_mma_shape_out: cute.Tensor,
        pv_mma_shape_out: cute.Tensor,
        smem_bytes_out: cute.Tensor,
        tmem_cols_out: cute.Tensor,
        layout_cosizes_out: cute.Tensor,
        qk_m: cutlass.Int32,
        qk_n: cutlass.Int32,
        qk_k: cutlass.Int32,
        pv_m: cutlass.Int32,
        pv_n: cutlass.Int32,
        pv_k: cutlass.Int32,
        q_nope_cosize: cutlass.Int32,
        q_pe_cosize: cutlass.Int32,
        k_nope_cosize: cutlass.Int32,
        k_pe_cosize: cutlass.Int32,
        v_cosize: cutlass.Int32,
        p_cosize: cutlass.Int32,
        smem_bytes: cutlass.Int32,
        tmem_cols: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        if tidx == 0:
            qk_mma_shape_out[0] = qk_m
            qk_mma_shape_out[1] = qk_n
            qk_mma_shape_out[2] = qk_k

            pv_mma_shape_out[0] = pv_m
            pv_mma_shape_out[1] = pv_n
            pv_mma_shape_out[2] = pv_k

            layout_cosizes_out[0] = q_nope_cosize
            layout_cosizes_out[1] = q_pe_cosize
            layout_cosizes_out[2] = k_nope_cosize
            layout_cosizes_out[3] = k_pe_cosize
            layout_cosizes_out[4] = v_cosize
            layout_cosizes_out[5] = p_cosize

            smem_bytes_out[0] = smem_bytes
            tmem_cols_out[0] = tmem_cols

            self._print_qk_mma_shape(qk_m, qk_n, qk_k)
            self._print_pv_mma_shape(pv_m, pv_n, pv_k)
            self._print_smem_bytes(smem_bytes)
            self._print_tmem_cols(tmem_cols)
            self._print_layout_cosizes(
                q_nope_cosize,
                q_pe_cosize,
                k_nope_cosize,
                k_pe_cosize,
                v_cosize,
                p_cosize,
            )


def _run_cutedsl_stage2_infrastructure_kernel(
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch_stream = torch.cuda.current_stream(device=device)
    cuda_driver = _get_cuda_driver_module()
    stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    qk_mma_shape_out = torch.zeros((3,), dtype=torch.int32, device=device)
    pv_mma_shape_out = torch.zeros((3,), dtype=torch.int32, device=device)
    smem_bytes_out = torch.zeros((1,), dtype=torch.int32, device=device)
    tmem_cols_out = torch.zeros((1,), dtype=torch.int32, device=device)
    layout_cosizes_out = torch.zeros(
        (len(STAGE2_LAYOUT_COSIZE_ORDER),),
        dtype=torch.int32,
        device=device,
    )

    program = Stage2InfrastructureProgram()
    cache_key = (
        "stage2_infrastructure_program",
        int(program.threads_per_block),
        str(device),
    )

    if cache_key not in program_compile_cache:
        program_compile_cache[cache_key] = cute.compile(
            program,
            qk_mma_shape_out,
            pv_mma_shape_out,
            smem_bytes_out,
            tmem_cols_out,
            layout_cosizes_out,
            stream,
            options="--opt-level 2",
        )

    compiled_program = program_compile_cache[cache_key]
    compiled_program(
        qk_mma_shape_out,
        pv_mma_shape_out,
        smem_bytes_out,
        tmem_cols_out,
        layout_cosizes_out,
        stream,
    )
    torch.cuda.synchronize(device=device)

    return (
        qk_mma_shape_out,
        pv_mma_shape_out,
        smem_bytes_out,
        tmem_cols_out,
        layout_cosizes_out,
    )


def _compute_stage3_padding_ok(
    page_ids_row: torch.Tensor,
    valid_page_count: int,
) -> int:
    page_ids_cpu = page_ids_row.detach().cpu().to(torch.int32)
    if valid_page_count < 0 or valid_page_count > MAX_PAGE_SLOTS:
        return 0

    for slot_idx in range(MAX_PAGE_SLOTS):
        slot_value = int(page_ids_cpu[slot_idx].item())
        if slot_idx < valid_page_count:
            if slot_value < 0:
                return 0
        else:
            if slot_value != -1:
                return 0

    return 1


class Stage3PageTablePreloadProgram:
    def __init__(
        self,
        threads_per_block: int = STAGE3_BLOCK_THREADS,
    ):
        self.threads_per_block = int(threads_per_block)

    @staticmethod
    def _compute_grid(num_tokens: int) -> tuple[int, int, int]:
        return (num_tokens, 1, 1)

    @cute.jit
    def _print_shared_page_ids_debug(
        self,
        smem_page_ids: cute.Tensor,
        valid_page_count: cutlass.Int32,
        padding_ok: cutlass.Int32,
    ):
        cute.printf("[CuTeDSL] stage3.page_ids.device_tensor: BEGIN\n")
        cute.print_tensor(smem_page_ids)
        cute.printf("[CuTeDSL] stage3.page_ids.device_tensor: END\n")
        cute.printf("[CuTeDSL] stage3.valid_page_count.device_scalar: BEGIN\n")
        cute.printf("{\"value\": %d}\n", valid_page_count)
        cute.printf("[CuTeDSL] stage3.valid_page_count.device_scalar: END\n")
        cute.printf("[CuTeDSL] stage3.padding_ok.device_scalar: BEGIN\n")
        cute.printf("{\"value\": %d}\n", padding_ok)
        cute.printf("[CuTeDSL] stage3.padding_ok.device_scalar: END\n")

    @cute.jit
    def __call__(
        self,
        page_ids_in: cute.Tensor,
        valid_page_count_in: cute.Tensor,
        page_ids_out: cute.Tensor,
        valid_page_count_out: cute.Tensor,
        padding_ok_out: cute.Tensor,
        stream,
    ):
        self.page_table_preload_kernel(
            page_ids_in,
            valid_page_count_in,
            page_ids_out,
            valid_page_count_out,
            padding_ok_out,
        ).launch(
            grid=self._compute_grid(page_ids_in.shape[0]),
            block=[self.threads_per_block, 1, 1],
            smem=Stage3SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def page_table_preload_kernel(
        self,
        page_ids_in: cute.Tensor,
        valid_page_count_in: cute.Tensor,
        page_ids_out: cute.Tensor,
        valid_page_count_out: cute.Tensor,
        padding_ok_out: cute.Tensor,
    ):
        token_idx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % 32

        smem = utils.SmemAllocator()
        storage = smem.allocate(Stage3SharedStorage)
        smem_page_ids = storage.smem_page_ids.get_tensor(cute.make_layout(MAX_PAGE_SLOTS))

        page_pipe = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.page_pipe_mbar_ptr.data_ptr(),
            num_stages=STAGE3_PAGE_PIPE_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            defer_sync=False,
        )
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32,
            num_bits_per_copy=cutlass.Int32.width,
        )

        if warp_idx == 0:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                STAGE3_PAGE_PIPE_STAGES,
            )
            page_pipe.producer_acquire(producer_state)

            page_ids_row = page_ids_in[token_idx, None]
            page_ids_row_for_copy = cute.flat_divide(page_ids_row, (1,))
            smem_page_ids_for_copy = cute.flat_divide(smem_page_ids, (1,))

            if cute.elem_less(lane_idx, MAX_PAGE_SLOTS):
                cute.copy(
                    atom_async_copy,
                    page_ids_row_for_copy[None, lane_idx],
                    smem_page_ids_for_copy[None, lane_idx],
                )

            page_pipe.producer_commit(producer_state)

        if warp_idx == 1:
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                STAGE3_PAGE_PIPE_STAGES,
            )
            page_pipe.consumer_wait(consumer_state)

            if lane_idx == 0:
                valid_page_count = valid_page_count_in[token_idx]
                padding_ok = 1

                if valid_page_count < 0 or valid_page_count > MAX_PAGE_SLOTS:
                    padding_ok = 0

                for page_slot in cutlass.range_constexpr(MAX_PAGE_SLOTS):
                    page_id = smem_page_ids[page_slot]
                    if cute.elem_less(page_slot, valid_page_count):
                        if page_id < 0:
                            padding_ok = 0
                    else:
                        if page_id != -1:
                            padding_ok = 0
                    page_ids_out[token_idx, page_slot] = page_id

                valid_page_count_out[token_idx] = valid_page_count
                padding_ok_out[token_idx] = padding_ok

                if token_idx == 0:
                    self._print_shared_page_ids_debug(
                        smem_page_ids,
                        valid_page_count,
                        padding_ok,
                    )

            page_pipe.consumer_release(consumer_state)


def _run_cutedsl_stage3_page_preload_kernel(
    *,
    page_ids_in: torch.Tensor,
    valid_page_count_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if page_ids_in.ndim != 2 or page_ids_in.shape[1] != MAX_PAGE_SLOTS:
        raise ValueError(
            f"page_ids_in must have shape [T, {MAX_PAGE_SLOTS}], got {tuple(page_ids_in.shape)}"
        )
    if valid_page_count_in.ndim != 1 or valid_page_count_in.shape[0] != page_ids_in.shape[0]:
        raise ValueError(
            "valid_page_count_in must have shape [T] matching page_ids_in.shape[0], "
            f"got {tuple(valid_page_count_in.shape)} for page_ids_in shape {tuple(page_ids_in.shape)}"
        )

    device = page_ids_in.device
    torch_stream = torch.cuda.current_stream(device=device)
    cuda_driver = _get_cuda_driver_module()
    stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    page_ids_out = torch.full_like(page_ids_in, -1)
    valid_page_count_out = torch.zeros_like(valid_page_count_in)
    padding_ok_out = torch.zeros_like(valid_page_count_in)

    program = Stage3PageTablePreloadProgram()
    cache_key = (
        "stage3_page_table_preload_program",
        tuple(page_ids_in.shape),
        str(page_ids_in.dtype),
        str(valid_page_count_in.dtype),
        int(program.threads_per_block),
        str(device),
    )

    if cache_key not in program_compile_cache:
        program_compile_cache[cache_key] = cute.compile(
            program,
            page_ids_in,
            valid_page_count_in,
            page_ids_out,
            valid_page_count_out,
            padding_ok_out,
            stream,
            options="--opt-level 2",
        )

    compiled_program = program_compile_cache[cache_key]
    compiled_program(
        page_ids_in,
        valid_page_count_in,
        page_ids_out,
        valid_page_count_out,
        padding_ok_out,
        stream,
    )
    torch.cuda.synchronize(device=device)

    return page_ids_out, valid_page_count_out, padding_ok_out


def prefix_validation_harness(
    q_nope: torch.Tensor | None = None,
    q_pe: torch.Tensor | None = None,
    ckv_cache: torch.Tensor | None = None,
    kpe_cache: torch.Tensor | None = None,
    sparse_indices: torch.Tensor | None = None,
    sm_scale: float | torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the stage-0 CuTeDSL validation harness.")

    provided_inputs = (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    if all(value is None for value in provided_inputs):
        (
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
            sm_scale,
        ) = create_synthetic_data(device="cuda")
    elif any(value is None for value in provided_inputs):
        raise ValueError(
            "prefix_validation_harness expects either all six stage inputs "
            "(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) or none."
        )
    else:
        assert q_nope is not None
        assert q_pe is not None
        assert ckv_cache is not None
        assert kpe_cache is not None
        assert sparse_indices is not None
        assert sm_scale is not None

        if isinstance(sm_scale, torch.Tensor):
            if sm_scale.numel() != 1:
                raise ValueError(
                    f"sm_scale tensor must contain exactly one element, got shape {tuple(sm_scale.shape)}"
                )
            sm_scale = float(sm_scale.detach().cpu().item())
        else:
            sm_scale = float(sm_scale)

        q_nope = q_nope.to(device="cuda")
        q_pe = q_pe.to(device="cuda")
        ckv_cache = ckv_cache.to(device="cuda")
        kpe_cache = kpe_cache.to(device="cuda")
        sparse_indices = sparse_indices.to(device="cuda")

    reference = torch_reference(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        sm_scale=sm_scale,
    )

    _print_validation_block(
        "PyTorch",
        "ref.valid_page_count",
        int(reference.valid_page_count[0].item()),
    )
    _print_validation_block(
        "PyTorch",
        "ref.page_ids",
        reference.page_ids[0],
    )
    _print_validation_block(
        "PyTorch",
        "ref.output_preview",
        reference.output[0].to(torch.float32),
        preview_count=100,
    )
    _print_validation_block(
        "PyTorch",
        "ref.lse",
        reference.lse[0],
    )

    device_page_ids, device_valid_page_count, device_contract_ok = (
        _run_cutedsl_contract_kernel(
            sparse_indices=sparse_indices,
            num_pages=ckv_cache.shape[0],
        )
    )

    expected_contract_ok = int(reference.contract_ok[0].item())
    actual_contract_ok = int(device_contract_ok[0].item())
    _print_validation_block(
        "CuTeDSL",
        "contract.page_dense_ok",
        actual_contract_ok,
    )
    if actual_contract_ok != expected_contract_ok:
        raise AssertionError(
            f"CuTeDSL contract checker mismatch: expected {expected_contract_ok}, got {actual_contract_ok}"
        )

    if not torch.equal(device_page_ids.cpu(), reference.page_ids.cpu()):
        raise AssertionError(
            "CuTeDSL page-id extraction mismatch with the Python-side support checker."
        )

    if not torch.equal(
        device_valid_page_count.cpu(), reference.valid_page_count.cpu()
    ):
        raise AssertionError(
            "CuTeDSL valid_page_count mismatch with the Python-side support checker."
        )

    scheduler_preview, scheduler_preview_len, scheduler_block_threads, scheduler_grid_x = (
        _run_cutedsl_scheduler_kernel(
            num_tokens=int(q_nope.shape[0]),
            device=q_nope.device,
        )
    )
    actual_block_threads = int(scheduler_block_threads[0].item())
    actual_grid_x = int(scheduler_grid_x[0].item())
    actual_preview_len = int(scheduler_preview_len[0].item())
    actual_schedule_preview = scheduler_preview[:actual_preview_len]

    expected_num_sms = int(torch.cuda.get_device_properties(q_nope.device).multi_processor_count)
    expected_grid_x = _compute_persistent_grid_x(
        num_tokens=int(q_nope.shape[0]),
        num_sms=expected_num_sms,
    )
    expected_schedule_preview, expected_preview_len = _build_cta0_schedule_preview(
        num_tokens=int(q_nope.shape[0]),
        grid_x=expected_grid_x,
        preview_count=SCHED_PREVIEW_COUNT,
    )

    if actual_block_threads != STAGE1_BLOCK_THREADS:
        raise AssertionError(
            f"CuTeDSL scheduler block_threads mismatch: expected {STAGE1_BLOCK_THREADS}, got {actual_block_threads}"
        )
    if actual_grid_x != expected_grid_x:
        raise AssertionError(
            f"CuTeDSL scheduler grid_x mismatch: expected {expected_grid_x}, got {actual_grid_x}"
        )
    if actual_preview_len != expected_preview_len:
        raise AssertionError(
            f"CuTeDSL scheduler preview length mismatch: expected {expected_preview_len}, got {actual_preview_len}"
        )
    if not torch.equal(actual_schedule_preview.cpu(), expected_schedule_preview.cpu()):
        raise AssertionError(
            "CuTeDSL token scheduler preview mismatch with the host-side persistent schedule."
        )

    _print_validation_block(
        "CuTeDSL",
        "cfg.block_threads",
        actual_block_threads,
    )
    _print_validation_block(
        "CuTeDSL",
        "cfg.grid_x",
        actual_grid_x,
    )
    _print_validation_block(
        "CuTeDSL",
        "sched.token_schedule_preview",
        actual_schedule_preview,
        preview_count=SCHED_PREVIEW_COUNT,
    )

    (
        stage2_qk_mma_shape,
        stage2_pv_mma_shape,
        stage2_smem_bytes,
        stage2_tmem_cols,
        stage2_layout_cosizes,
    ) = _run_cutedsl_stage2_infrastructure_kernel(device=q_nope.device)
    stage2_expected = _get_stage2_expected_metadata()

    if not torch.equal(
        stage2_qk_mma_shape.cpu(), stage2_expected["qk_mma_shape"].cpu()
    ):
        raise AssertionError(
            "CuTeDSL stage-2 QK MMA shape mismatch with the Blackwell infrastructure plan."
        )
    if not torch.equal(
        stage2_pv_mma_shape.cpu(), stage2_expected["pv_mma_shape"].cpu()
    ):
        raise AssertionError(
            "CuTeDSL stage-2 PV MMA shape mismatch with the Blackwell infrastructure plan."
        )
    if int(stage2_smem_bytes[0].item()) != int(stage2_expected["smem_bytes"]):
        raise AssertionError(
            "CuTeDSL stage-2 shared-memory byte accounting mismatch with the round-0 plan."
        )
    if int(stage2_tmem_cols[0].item()) != int(stage2_expected["tmem_cols"]):
        raise AssertionError(
            "CuTeDSL stage-2 tensor-memory column allocation mismatch with the round-0 plan."
        )
    if not torch.equal(
        stage2_layout_cosizes.cpu(), stage2_expected["layout_cosizes"].cpu()
    ):
        raise AssertionError(
            "CuTeDSL stage-2 SMEM layout cosizes mismatch with the round-0 infrastructure plan."
        )

    _print_validation_block(
        "CuTeDSL",
        "infra.qk_mma_shape",
        stage2_qk_mma_shape,
    )
    _print_validation_block(
        "CuTeDSL",
        "infra.pv_mma_shape",
        stage2_pv_mma_shape,
    )
    _print_validation_block(
        "CuTeDSL",
        "infra.smem_bytes",
        int(stage2_smem_bytes[0].item()),
    )
    _print_validation_block(
        "CuTeDSL",
        "infra.tmem_cols",
        int(stage2_tmem_cols[0].item()),
    )
    _print_validation_block(
        "CuTeDSL",
        "infra.layout_cosizes",
        stage2_layout_cosizes,
    )

    _print_validation_block(
        "PyTorch",
        "stage3.page_ids",
        reference.page_ids[0],
    )
    _print_validation_block(
        "PyTorch",
        "stage3.valid_page_count",
        int(reference.valid_page_count[0].item()),
    )

    (
        stage3_page_ids,
        stage3_valid_page_count,
        stage3_padding_ok,
    ) = _run_cutedsl_stage3_page_preload_kernel(
        page_ids_in=device_page_ids,
        valid_page_count_in=device_valid_page_count,
    )
    expected_stage3_padding_ok = torch.zeros_like(stage3_padding_ok)
    for token_idx in range(reference.page_ids.shape[0]):
        expected_stage3_padding_ok[token_idx] = _compute_stage3_padding_ok(
            reference.page_ids[token_idx],
            int(reference.valid_page_count[token_idx].item()),
        )

    if not torch.equal(stage3_page_ids.cpu(), reference.page_ids.cpu()):
        raise AssertionError(
            "CuTeDSL stage-3 preloaded page_ids mismatch with the PyTorch reference metadata."
        )
    if not torch.equal(stage3_valid_page_count.cpu(), reference.valid_page_count.cpu()):
        raise AssertionError(
            "CuTeDSL stage-3 valid_page_count mismatch with the PyTorch reference metadata."
        )
    if not torch.equal(stage3_padding_ok.cpu(), expected_stage3_padding_ok.cpu()):
        raise AssertionError(
            "CuTeDSL stage-3 padding_ok mismatch with the expected padded page-table layout."
        )

    _print_validation_block(
        "CuTeDSL",
        "stage3.page_ids",
        stage3_page_ids[0],
    )
    _print_validation_block(
        "CuTeDSL",
        "stage3.valid_page_count",
        int(stage3_valid_page_count[0].item()),
    )
    _print_validation_block(
        "CuTeDSL",
        "stage3.padding_ok",
        int(stage3_padding_ok[0].item()),
    )

    if output is not None:
        if tuple(output.shape) != tuple(reference.output.shape):
            raise ValueError(
                f"output buffer shape mismatch: expected {tuple(reference.output.shape)}, got {tuple(output.shape)}"
            )
        output.copy_(reference.output.to(device=output.device, dtype=output.dtype))

    if lse is not None:
        if tuple(lse.shape) != tuple(reference.lse.shape):
            raise ValueError(
                f"lse buffer shape mismatch: expected {tuple(reference.lse.shape)}, got {tuple(lse.shape)}"
            )
        lse.copy_(reference.lse.to(device=lse.device, dtype=lse.dtype))

    return {
        "stage_id": "S3",
        "ref_valid_page_count": int(reference.valid_page_count[0].item()),
        "ref_page_ids": reference.page_ids[0].detach().cpu().tolist(),
        "contract_page_dense_ok": actual_contract_ok,
        "cfg_block_threads": actual_block_threads,
        "cfg_grid_x": actual_grid_x,
        "sched_token_schedule_preview": actual_schedule_preview.detach().cpu().tolist(),
        "infra_qk_mma_shape": stage2_qk_mma_shape.detach().cpu().tolist(),
        "infra_pv_mma_shape": stage2_pv_mma_shape.detach().cpu().tolist(),
        "infra_smem_bytes": int(stage2_smem_bytes[0].item()),
        "infra_tmem_cols": int(stage2_tmem_cols[0].item()),
        "infra_layout_cosize_order": list(STAGE2_LAYOUT_COSIZE_ORDER),
        "infra_layout_cosizes": stage2_layout_cosizes.detach().cpu().tolist(),
        "stage3_page_ids": stage3_page_ids[0].detach().cpu().tolist(),
        "stage3_valid_page_count": int(stage3_valid_page_count[0].item()),
        "stage3_padding_ok": int(stage3_padding_ok[0].item()),
    }


if __name__ == "__main__":
    prefix_validation_harness()