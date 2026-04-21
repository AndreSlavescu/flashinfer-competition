from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.cute.nvgpu.tcgen05 as tcgen05  # type: ignore
    import cutlass.utils as cutlass_utils  # type: ignore
except Exception:  # pragma: no cover - cutlass may be unavailable in non-validation environments
    cutlass = None
    cute = None
    tcgen05 = None
    cutlass_utils = None


NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048
MAX_PAGES_PER_TOKEN = TOPK // PAGE_SIZE
LOG2_E = math.log2(math.e)
BLOCK_THREADS = 256
WARPS_PER_CTA = BLOCK_THREADS // 32
CLUSTER_SHAPE = (1, 1, 1)
DEFAULT_NUM_SMS = 148
QK_MMA_TILE_SHAPE = (64, 64, 16)
PV_MMA_TILE_SHAPE = (64, 128, 16)
Q_TILE_ROWS = 64
LOGICAL_HEAD_ROWS = 16
QK_TILE_COLS = 64
PV_TILE_COLS = 128
QK_PIPE_STAGES = 1
V_PIPE_STAGES = 2
P_PIPE_STAGES = 1
TMEM_ALLOC_COLS = 512

program_compile_cache: Dict[Any, Any] = {}
_validation_blocks_emitted = False


class PageDenseCheckResult:
    def __init__(
        self,
        *,
        page_dense_ok: bool,
        valid_page_count: int,
        page_ids: torch.Tensor,
        reason: str,
    ) -> None:
        self.page_dense_ok = page_dense_ok
        self.valid_page_count = valid_page_count
        self.page_ids = page_ids
        self.reason = reason


class SyntheticInputs:
    def __init__(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        sm_scale: float,
    ) -> None:
        self.q_nope = q_nope
        self.q_pe = q_pe
        self.ckv_cache = ckv_cache
        self.kpe_cache = kpe_cache
        self.sparse_indices = sparse_indices
        self.sm_scale = sm_scale


class BlackwellStyleKernel:
    """Stage-S2 skeleton.

    Round-0 up through stage S2 is intentionally limited to:
      1. reference semantics
      2. round-0 page-dense support-contract checking
      3. persistent-token scheduler configuration/debug plumbing
      4. MMA / SMEM / TMEM infrastructure planning metadata

    The actual CuTeDSL kernel body arrives in later stages. This class still
    provides the interface that subsequent stages can extend.
    """

    def __init__(
        self,
        *,
        num_qo_heads: int = NUM_QO_HEADS,
        head_dim_ckv: int = HEAD_DIM_CKV,
        head_dim_kpe: int = HEAD_DIM_KPE,
        page_size: int = PAGE_SIZE,
        topk: int = TOPK,
    ) -> None:
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.page_size = page_size
        self.topk = topk
        self.max_pages_per_token = topk // page_size
        self.block_threads = BLOCK_THREADS
        self.warps_per_cta = WARPS_PER_CTA
        self.cluster_shape = CLUSTER_SHAPE
        self.default_num_sms = DEFAULT_NUM_SMS
        self.logical_rows = LOGICAL_HEAD_ROWS
        self.padded_rows = Q_TILE_ROWS
        self.qk_mma_tile_shape = QK_MMA_TILE_SHAPE
        self.pv_mma_tile_shape = PV_MMA_TILE_SHAPE
        self.qk_pipe_stages = QK_PIPE_STAGES
        self.v_pipe_stages = V_PIPE_STAGES
        self.p_pipe_stages = P_PIPE_STAGES
        self.tmem_alloc_cols = TMEM_ALLOC_COLS
        self.block_threads_constexpr = _maybe_constexpr_int(self.block_threads)
        self.page_size_constexpr = _maybe_constexpr_int(self.page_size)
        self.max_pages_per_token_constexpr = _maybe_constexpr_int(self.max_pages_per_token)
        self.logical_rows_constexpr = _maybe_constexpr_int(self.logical_rows)
        self.padded_rows_constexpr = _maybe_constexpr_int(self.padded_rows)
        self.qk_mma_tile_shape_constexpr = tuple(
            _maybe_constexpr_int(v) for v in self.qk_mma_tile_shape
        )
        self.pv_mma_tile_shape_constexpr = tuple(
            _maybe_constexpr_int(v) for v in self.pv_mma_tile_shape
        )
        self.tmem_alloc_cols_constexpr = _maybe_constexpr_int(self.tmem_alloc_cols)

    @staticmethod
    def _compute_grid(token_count: int, num_sms: int) -> int:
        token_count_i = max(0, int(token_count))
        num_sms_i = max(0, int(num_sms))
        return min(token_count_i, num_sms_i)

    @staticmethod
    def _infer_num_sms(device: Optional[torch.device] = None) -> int:
        if torch.cuda.is_available():
            try:
                queried_device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
                return int(torch.cuda.get_device_properties(queried_device).multi_processor_count)
            except Exception:
                pass
        return DEFAULT_NUM_SMS

    def make_launch_config(
        self,
        token_count: int,
        *,
        num_sms: Optional[int] = None,
    ) -> Dict[str, Any]:
        resolved_num_sms = self.default_num_sms if num_sms is None else int(num_sms)
        grid_x = self._compute_grid(int(token_count), resolved_num_sms)
        return {
            "block_threads": int(self.block_threads),
            "warps_per_cta": int(self.warps_per_cta),
            "cluster_shape": tuple(int(v) for v in self.cluster_shape),
            "token_count": int(token_count),
            "num_sms": int(resolved_num_sms),
            "grid_x": int(grid_x),
        }

    def persistent_token_schedule(
        self,
        token_count: int,
        *,
        cta_idx: int = 0,
        num_sms: Optional[int] = None,
        limit: int = 32,
    ) -> List[int]:
        launch = self.make_launch_config(token_count, num_sms=num_sms)
        grid_x = int(launch["grid_x"])
        if grid_x <= 0:
            return []

        cta_idx_i = int(cta_idx)
        if cta_idx_i < 0 or cta_idx_i >= grid_x:
            return []

        schedule: List[int] = []
        token_idx = cta_idx_i
        while token_idx < int(token_count) and len(schedule) < int(limit):
            schedule.append(int(token_idx))
            token_idx += grid_x
        return schedule

    def scheduler_debug_payload(
        self,
        token_count: int,
        *,
        device: Optional[torch.device] = None,
        num_sms: Optional[int] = None,
        cta_idx: int = 0,
        preview_limit: int = 32,
    ) -> Dict[str, Any]:
        resolved_num_sms = self._infer_num_sms(device) if num_sms is None else int(num_sms)
        launch = self.make_launch_config(token_count, num_sms=resolved_num_sms)
        schedule_preview = self.persistent_token_schedule(
            token_count,
            cta_idx=cta_idx,
            num_sms=resolved_num_sms,
            limit=preview_limit,
        )
        launch["cta_idx"] = int(cta_idx)
        launch["token_schedule_preview"] = schedule_preview
        launch["preview_limit"] = int(preview_limit)
        return launch

    def check_support_contract(
        self,
        sparse_indices_row: torch.Tensor,
        *,
        num_cache_pages: Optional[int] = None,
    ) -> PageDenseCheckResult:
        return check_page_dense_row(
            sparse_indices_row,
            page_size=self.page_size,
            topk=self.topk,
            max_pages_per_token=self.max_pages_per_token,
            num_cache_pages=num_cache_pages,
        )

    def check_support_contract_batch(
        self,
        sparse_indices: torch.Tensor,
        *,
        num_cache_pages: Optional[int] = None,
    ) -> List[PageDenseCheckResult]:
        return [
            self.check_support_contract(sparse_indices[t], num_cache_pages=num_cache_pages)
            for t in range(int(sparse_indices.shape[0]))
        ]

    @staticmethod
    def _numel(shape: Sequence[int]) -> int:
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)

    @staticmethod
    def _make_buffer_spec(
        *,
        logical_shape: Sequence[int],
        dtype: str,
        element_nbytes: int,
        major_mode: str,
        stages: int,
        storage: str,
        layout_repr: str,
    ) -> Dict[str, Any]:
        cosize_elements = BlackwellStyleKernel._numel(logical_shape)
        return {
            "logical_shape": [int(v) for v in logical_shape],
            "dtype": dtype,
            "element_nbytes": int(element_nbytes),
            "major_mode": major_mode,
            "stages": int(stages),
            "storage": storage,
            "cosize_elements": int(cosize_elements),
            "cosize_bytes": int(cosize_elements * int(element_nbytes)),
            "layout_repr": layout_repr,
        }

    def _manual_layout_cosizes(self) -> Dict[str, Any]:
        bf16_nbytes = 2
        f32_nbytes = 4
        i32_nbytes = 4
        smem_layouts = {
            "smem_q_nope": self._make_buffer_spec(
                logical_shape=(self.padded_rows, self.head_dim_ckv),
                dtype="bf16",
                element_nbytes=bf16_nbytes,
                major_mode="K",
                stages=1,
                storage="smem",
                layout_repr="K-major padded Q(NOPE) tile [64, 512, 1]",
            ),
            "smem_q_pe": self._make_buffer_spec(
                logical_shape=(self.padded_rows, self.head_dim_kpe),
                dtype="bf16",
                element_nbytes=bf16_nbytes,
                major_mode="K",
                stages=1,
                storage="smem",
                layout_repr="K-major padded Q(PE) tile [64, 64, 1]",
            ),
            "smem_k_nope": self._make_buffer_spec(
                logical_shape=(self.page_size, self.head_dim_ckv),
                dtype="bf16",
                element_nbytes=bf16_nbytes,
                major_mode="K",
                stages=1,
                storage="smem",
                layout_repr="K-major K(NOPE) page tile [64, 512, 1]",
            ),
            "smem_k_pe": self._make_buffer_spec(
                logical_shape=(self.page_size, self.head_dim_kpe),
                dtype="bf16",
                element_nbytes=bf16_nbytes,
                major_mode="K",
                stages=1,
                storage="smem",
                layout_repr="K-major K(PE) page tile [64, 64, 1]",
            ),
            "smem_v": self._make_buffer_spec(
                logical_shape=(self.page_size, PV_TILE_COLS, self.v_pipe_stages),
                dtype="bf16",
                element_nbytes=bf16_nbytes,
                major_mode="MN",
                stages=self.v_pipe_stages,
                storage="smem",
                layout_repr="MN-major V subtile stream [64, 128, 2]",
            ),
            "smem_p": self._make_buffer_spec(
                logical_shape=(self.padded_rows, QK_TILE_COLS),
                dtype="bf16",
                element_nbytes=bf16_nbytes,
                major_mode="K",
                stages=self.p_pipe_stages,
                storage="smem",
                layout_repr="K-major probability tile [64, 64, 1]",
            ),
            "smem_page_ids": self._make_buffer_spec(
                logical_shape=(self.max_pages_per_token,),
                dtype="int32",
                element_nbytes=i32_nbytes,
                major_mode="linear",
                stages=1,
                storage="smem",
                layout_repr="Linear page-id staging buffer [32]",
            ),
            "valid_page_count": self._make_buffer_spec(
                logical_shape=(1,),
                dtype="int32",
                element_nbytes=i32_nbytes,
                major_mode="scalar",
                stages=1,
                storage="smem",
                layout_repr="Scalar valid-page-count cell [1]",
            ),
        }
        tmem_layouts = {
            "tS_tile": self._make_buffer_spec(
                logical_shape=(self.padded_rows, QK_TILE_COLS),
                dtype="float32",
                element_nbytes=f32_nbytes,
                major_mode="TMEM-col",
                stages=1,
                storage="tmem",
                layout_repr="TMEM score scratch tile logical [64, 64]",
            ),
            "tO_accum": self._make_buffer_spec(
                logical_shape=(self.padded_rows, self.head_dim_ckv),
                dtype="float32",
                element_nbytes=f32_nbytes,
                major_mode="TMEM-col",
                stages=1,
                storage="tmem",
                layout_repr="TMEM output accumulator logical [64, 512]",
            ),
        }
        return {
            "smem": smem_layouts,
            "tmem": tmem_layouts,
        }

    def _try_build_cutedsl_infra_details(self) -> Optional[Dict[str, Any]]:
        if cutlass is None or cutlass_utils is None or tcgen05 is None:
            return None

        bf16_type = getattr(cutlass, "BFloat16", None)
        f32_type = getattr(cutlass, "Float32", None)
        if bf16_type is None or f32_type is None:
            return None

        try:
            qk_tiled_mma = cutlass_utils.make_trivial_tiled_mma(
                bf16_type,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                f32_type,
                tcgen05.CtaGroup.ONE,
                self.qk_mma_tile_shape[:2],
            )
            pv_tiled_mma = cutlass_utils.make_trivial_tiled_mma(
                bf16_type,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.MN,
                f32_type,
                tcgen05.CtaGroup.ONE,
                self.pv_mma_tile_shape[:2],
            )
        except Exception:
            return None

        return {
            "qk_tiled_mma_repr": str(qk_tiled_mma),
            "pv_tiled_mma_repr": str(pv_tiled_mma),
        }

    def infrastructure_debug_payload(self) -> Dict[str, Any]:
        layout_cosizes = self._manual_layout_cosizes()
        cutedsl_details = self._try_build_cutedsl_infra_details()

        smem_tensor_bytes = {
            name: int(spec["cosize_bytes"])
            for name, spec in layout_cosizes["smem"].items()
        }
        smem_core_tensor_total = int(
            smem_tensor_bytes["smem_q_nope"]
            + smem_tensor_bytes["smem_q_pe"]
            + smem_tensor_bytes["smem_k_nope"]
            + smem_tensor_bytes["smem_k_pe"]
            + smem_tensor_bytes["smem_v"]
            + smem_tensor_bytes["smem_p"]
        )
        smem_metadata_bytes = int(
            smem_tensor_bytes["smem_page_ids"] + smem_tensor_bytes["valid_page_count"]
        )
        smem_tracked_total = int(smem_core_tensor_total + smem_metadata_bytes)

        payload = {
            "infra.qk_mma_shape": {
                "value": list(self.qk_mma_tile_shape),
                "cta_group": "ONE",
                "ab_dtype": "bf16",
                "acc_dtype": "float32",
                "a_leading_mode": "K",
                "b_leading_mode": "K",
                "logical_rows": int(self.logical_rows),
                "padded_rows": int(self.padded_rows),
            },
            "infra.pv_mma_shape": {
                "value": list(self.pv_mma_tile_shape),
                "cta_group": "ONE",
                "ab_dtype": "bf16",
                "acc_dtype": "float32",
                "a_leading_mode": "K",
                "b_leading_mode": "MN",
                "logical_rows": int(self.logical_rows),
                "padded_rows": int(self.padded_rows),
            },
            "infra.smem_bytes": {
                "value": int(smem_core_tensor_total),
                "core_tensor_total_bytes": int(smem_core_tensor_total),
                "metadata_bytes": int(smem_metadata_bytes),
                "tracked_total_bytes": int(smem_tracked_total),
                "approx_target_note": "~184 KB core tensors + small metadata/barriers",
                "per_buffer_bytes": smem_tensor_bytes,
            },
            "infra.tmem_cols": {
                "value": int(self.tmem_alloc_cols),
                "allocator_granularity": "power-of-two columns",
                "score_tile_shape": [self.padded_rows, QK_TILE_COLS],
                "output_tile_shape": [self.padded_rows, self.head_dim_ckv],
                "score_tile_dtype": "float32",
                "output_tile_dtype": "float32",
                "placement": {
                    "score_tile_col_start": 0,
                    "output_tile_col_start_policy": "tcgen05.find_tmem_tensor_col_offset(tS_tile)",
                    "why_512": "256 TMEM columns are insufficient once the score scratch tile and the 512-wide O accumulator share the allocation; round-0 reserves the full 512 columns.",
                },
            },
            "infra.layout_cosizes": layout_cosizes,
        }

        if cutedsl_details is not None:
            payload["infra.qk_mma_shape"]["cute_tiled_mma_repr"] = cutedsl_details[
                "qk_tiled_mma_repr"
            ]
            payload["infra.pv_mma_shape"]["cute_tiled_mma_repr"] = cutedsl_details[
                "pv_tiled_mma_repr"
            ]
            payload["infra.layout_cosizes"]["backend"] = "cutlass_utils.make_trivial_tiled_mma"
        else:
            payload["infra.layout_cosizes"]["backend"] = "manual_stage2_plan"

        return payload


def _ensure_int32_cpu_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.int32).contiguous()


def _maybe_constexpr_int(value: int) -> Any:
    if cutlass is None:
        return int(value)
    constexpr_ctor = getattr(cutlass, "Constexpr", None)
    if constexpr_ctor is None:
        return int(value)
    try:
        return constexpr_ctor(int(value))
    except Exception:
        return int(value)


def _dense_page_block(page_id: int, *, page_size: int = PAGE_SIZE) -> torch.Tensor:
    start = int(page_id) * int(page_size)
    return torch.arange(start, start + int(page_size), dtype=torch.int32)


def make_page_dense_sparse_indices(
    page_ids: Sequence[int],
    *,
    topk: int = TOPK,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    if topk % page_size != 0:
        raise ValueError("topk must be divisible by page_size")
    max_pages_per_token = topk // page_size
    if len(page_ids) > max_pages_per_token:
        raise ValueError("too many page_ids for a single token")

    sparse = torch.full((topk,), -1, dtype=torch.int32)
    for slot, page_id in enumerate(page_ids):
        block = _dense_page_block(int(page_id), page_size=page_size)
        start = slot * page_size
        sparse[start : start + page_size] = block
    return sparse


def check_page_dense_row(
    sparse_indices_row: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
    topk: int = TOPK,
    max_pages_per_token: int = MAX_PAGES_PER_TOKEN,
    num_cache_pages: Optional[int] = None,
) -> PageDenseCheckResult:
    row = _ensure_int32_cpu_contiguous(sparse_indices_row).reshape(-1)
    if int(row.numel()) != int(topk):
        return PageDenseCheckResult(
            page_dense_ok=False,
            valid_page_count=0,
            page_ids=torch.full((max_pages_per_token,), -1, dtype=torch.int32),
            reason=f"expected sparse_indices row length {topk}, got {row.numel()}",
        )

    page_ids = torch.full((max_pages_per_token,), -1, dtype=torch.int32)
    seen_pages = set()
    valid_page_count = 0
    seen_invalid_tail = False

    for slot in range(max_pages_per_token):
        block = row[slot * page_size : (slot + 1) * page_size]
        if torch.all(block == -1):
            seen_invalid_tail = True
            continue

        if seen_invalid_tail:
            return PageDenseCheckResult(
                page_dense_ok=False,
                valid_page_count=valid_page_count,
                page_ids=page_ids,
                reason="valid pages appear after invalid tail pages",
            )

        if torch.any(block < 0):
            return PageDenseCheckResult(
                page_dense_ok=False,
                valid_page_count=valid_page_count,
                page_ids=page_ids,
                reason="mixed valid and invalid token ids inside a page slot",
            )

        first_token = int(block[0].item())
        page_id = first_token // page_size
        expected = _dense_page_block(page_id, page_size=page_size)
        if not torch.equal(block, expected):
            return PageDenseCheckResult(
                page_dense_ok=False,
                valid_page_count=valid_page_count,
                page_ids=page_ids,
                reason="page slot is not page-dense",
            )

        if page_id in seen_pages:
            return PageDenseCheckResult(
                page_dense_ok=False,
                valid_page_count=valid_page_count,
                page_ids=page_ids,
                reason="duplicate page id encountered",
            )

        if num_cache_pages is not None and not (0 <= page_id < int(num_cache_pages)):
            return PageDenseCheckResult(
                page_dense_ok=False,
                valid_page_count=valid_page_count,
                page_ids=page_ids,
                reason="page id out of range",
            )

        seen_pages.add(page_id)
        page_ids[valid_page_count] = page_id
        valid_page_count += 1

    return PageDenseCheckResult(
        page_dense_ok=True,
        valid_page_count=valid_page_count,
        page_ids=page_ids,
        reason="ok",
    )


def check_page_dense_batch(
    sparse_indices: torch.Tensor,
    *,
    num_cache_pages: Optional[int] = None,
    page_size: int = PAGE_SIZE,
    topk: int = TOPK,
) -> List[PageDenseCheckResult]:
    kernel = BlackwellStyleKernel(page_size=page_size, topk=topk)
    return kernel.check_support_contract_batch(sparse_indices, num_cache_pages=num_cache_pages)


def torch_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive reference for the round-0 sparse-attention semantics.

    This function intentionally computes the semantic ground truth directly from
    `sparse_indices`, even if the indices do not satisfy the stricter round-0
    page-dense support contract. The support checker is a separate concern.
    """

    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    ckv_f32 = ckv_cache.to(torch.float32)
    kpe_f32 = kpe_cache.to(torch.float32)

    token_k_count = int(ckv_f32.shape[0]) * int(ckv_f32.shape[1])
    flat_ckv = ckv_f32.reshape(token_k_count, int(ckv_f32.shape[-1]))
    flat_kpe = kpe_f32.reshape(token_k_count, int(kpe_f32.shape[-1]))

    token_count = int(q_nope_f32.shape[0])
    output = torch.zeros(
        (token_count, NUM_QO_HEADS, HEAD_DIM_CKV),
        dtype=torch.float32,
        device=q_nope_f32.device,
    )
    lse = torch.full(
        (token_count, NUM_QO_HEADS),
        float("-inf"),
        dtype=torch.float32,
        device=q_nope_f32.device,
    )

    for token_idx in range(token_count):
        row = sparse_indices[token_idx].reshape(-1).to(dtype=torch.int64, device="cpu")
        valid_token_ids = row[row >= 0]
        if valid_token_ids.numel() == 0:
            continue

        if torch.any(valid_token_ids >= token_k_count):
            raise ValueError("sparse_indices contains out-of-range token ids")

        selected_ckv = flat_ckv.index_select(0, valid_token_ids.to(device=flat_ckv.device))
        selected_kpe = flat_kpe.index_select(0, valid_token_ids.to(device=flat_kpe.device))

        qn = q_nope_f32[token_idx]
        qp = q_pe_f32[token_idx]

        logits = sm_scale * (
            torch.einsum("hd,kd->hk", qn, selected_ckv)
            + torch.einsum("hd,kd->hk", qp, selected_kpe)
        )
        probs = torch.softmax(logits, dim=-1)
        output[token_idx] = torch.matmul(probs, selected_ckv)
        lse[token_idx] = torch.logsumexp(logits, dim=-1) * LOG2_E

    return output.to(torch.bfloat16), lse


def create_synthetic_data(seed: int = 2025) -> SyntheticInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    token_count = 1
    num_cache_pages = 6
    page_ids = [1, 4]

    q_nope = (
        torch.randn(
            (token_count, NUM_QO_HEADS, HEAD_DIM_CKV),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.125
    ).to(torch.bfloat16)
    q_pe = (
        torch.randn(
            (token_count, NUM_QO_HEADS, HEAD_DIM_KPE),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.125
    ).to(torch.bfloat16)
    ckv_cache = (
        torch.randn(
            (num_cache_pages, PAGE_SIZE, HEAD_DIM_CKV),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.125
    ).to(torch.bfloat16)
    kpe_cache = (
        torch.randn(
            (num_cache_pages, PAGE_SIZE, HEAD_DIM_KPE),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.125
    ).to(torch.bfloat16)

    sparse_indices = make_page_dense_sparse_indices(page_ids).unsqueeze(0)
    sm_scale = 1.0 / math.sqrt(float(HEAD_DIM_CKV + HEAD_DIM_KPE))

    return SyntheticInputs(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        sm_scale=sm_scale,
    )


def _tensor_preview(tensor: torch.Tensor, limit: int) -> Dict[str, Any]:
    cpu = tensor.detach().to("cpu")
    if cpu.dtype == torch.bfloat16:
        preview_tensor = cpu.to(torch.float32)
    else:
        preview_tensor = cpu
    flat = preview_tensor.reshape(-1)
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "preview": flat[: min(limit, flat.numel())].tolist(),
    }


def _full_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    cpu = tensor.detach().to("cpu")
    if cpu.dtype == torch.bfloat16:
        values = cpu.to(torch.float32).tolist()
    else:
        values = cpu.tolist()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "values": values,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cpu = value.detach().to("cpu")
        if cpu.ndim == 0:
            item = cpu.item()
            if isinstance(item, bool):
                return bool(item)
            if isinstance(item, int):
                return int(item)
            return float(item)
        return cpu.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def emit_validation_block(source: str, name: str, payload: Any) -> None:
    print(f"[{source}] {name}: BEGIN", flush=True)
    print(json.dumps(_json_ready(payload), indent=2), flush=True)
    print(f"[{source}] {name}: END", flush=True)


def _build_s0_validation_payloads(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Dict[str, Any]:
    kernel = BlackwellStyleKernel()

    support = kernel.check_support_contract(sparse_indices[0], num_cache_pages=int(ckv_cache.shape[0]))
    output_ref, lse_ref = torch_reference(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )

    return {
        "ref.valid_page_count": int(support.valid_page_count),
        "ref.page_ids": {
            "shape": [int(support.page_ids.numel())],
            "dtype": str(support.page_ids.dtype),
            "values": support.page_ids.tolist(),
        },
        "ref.output_preview": _tensor_preview(output_ref[0], limit=100),
        "ref.lse": _full_tensor(lse_ref[0]),
        "contract.page_dense_ok": {
            "value": bool(support.page_dense_ok),
            "reason": support.reason,
        },
    }


def _build_s1_validation_payloads(
    q_nope: torch.Tensor,
) -> Dict[str, Any]:
    kernel = BlackwellStyleKernel()
    device = q_nope.device if isinstance(q_nope, torch.Tensor) else None
    scheduler = kernel.scheduler_debug_payload(
        int(q_nope.shape[0]),
        device=device,
        cta_idx=0,
        preview_limit=32,
    )

    return {
        "cfg.block_threads": {
            "value": int(scheduler["block_threads"]),
            "warps_per_cta": int(scheduler["warps_per_cta"]),
            "cluster_shape": list(scheduler["cluster_shape"]),
        },
        "cfg.grid_x": {
            "value": int(scheduler["grid_x"]),
            "token_count": int(scheduler["token_count"]),
            "num_sms": int(scheduler["num_sms"]),
        },
        "sched.token_schedule_preview": {
            "shape": [len(scheduler["token_schedule_preview"])],
            "dtype": "int32",
            "cta_idx": int(scheduler["cta_idx"]),
            "grid_x": int(scheduler["grid_x"]),
            "preview_limit": int(scheduler["preview_limit"]),
            "preview": list(scheduler["token_schedule_preview"]),
        },
    }


def _build_s2_validation_payloads() -> Dict[str, Any]:
    kernel = BlackwellStyleKernel()
    return kernel.infrastructure_debug_payload()


def _build_validation_payloads(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Dict[str, Any]:
    payloads = _build_s0_validation_payloads(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    payloads.update(_build_s1_validation_payloads(q_nope))
    payloads.update(_build_s2_validation_payloads())
    return payloads


def _emit_validation_payloads(payloads: Dict[str, Any]) -> None:
    global _validation_blocks_emitted
    if _validation_blocks_emitted:
        return

    emit_validation_block("PyTorch", "ref.valid_page_count", payloads["ref.valid_page_count"])
    emit_validation_block("PyTorch", "ref.page_ids", payloads["ref.page_ids"])
    emit_validation_block("PyTorch", "ref.output_preview", payloads["ref.output_preview"])
    emit_validation_block("PyTorch", "ref.lse", payloads["ref.lse"])
    emit_validation_block("CuTeDSL", "contract.page_dense_ok", payloads["contract.page_dense_ok"])
    emit_validation_block("CuTeDSL", "cfg.block_threads", payloads["cfg.block_threads"])
    emit_validation_block("CuTeDSL", "cfg.grid_x", payloads["cfg.grid_x"])
    emit_validation_block("CuTeDSL", "sched.token_schedule_preview", payloads["sched.token_schedule_preview"])
    emit_validation_block("CuTeDSL", "infra.qk_mma_shape", payloads["infra.qk_mma_shape"])
    emit_validation_block("CuTeDSL", "infra.pv_mma_shape", payloads["infra.pv_mma_shape"])
    emit_validation_block("CuTeDSL", "infra.smem_bytes", payloads["infra.smem_bytes"])
    emit_validation_block("CuTeDSL", "infra.tmem_cols", payloads["infra.tmem_cols"])
    emit_validation_block("CuTeDSL", "infra.layout_cosizes", payloads["infra.layout_cosizes"])
    _validation_blocks_emitted = True


def prefix_validation_harness(
    q_nope: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    ckv_cache: Optional[torch.Tensor] = None,
    kpe_cache: Optional[torch.Tensor] = None,
    sparse_indices: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> Dict[str, Any]:
    if any(
        value is None for value in (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    ):
        inputs = create_synthetic_data()
        q_nope = inputs.q_nope
        q_pe = inputs.q_pe
        ckv_cache = inputs.ckv_cache
        kpe_cache = inputs.kpe_cache
        sparse_indices = inputs.sparse_indices
        sm_scale = inputs.sm_scale

    assert q_nope is not None
    assert q_pe is not None
    assert ckv_cache is not None
    assert kpe_cache is not None
    assert sparse_indices is not None
    assert sm_scale is not None

    payloads = _build_validation_payloads(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        float(sm_scale),
    )
    _emit_validation_payloads(payloads)

    return payloads


def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(q_nope.shape[0]) > 0 and int(sparse_indices.shape[0]) > 0:
        _emit_validation_payloads(
            _build_validation_payloads(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                sparse_indices,
                sm_scale,
            )
        )
    return torch_reference(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )


def kernel(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError(
        "kernel_0.py stage S2 implements the reference harness, support-contract checker, persistent scheduler skeleton, and infrastructure planning metadata only."
    )


if __name__ == "__main__":
    prefix_validation_harness()