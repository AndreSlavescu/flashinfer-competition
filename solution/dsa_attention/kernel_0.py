import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

try:
    import cuda.bindings.driver as cuda
    CudaStream = cuda.CUstream
except Exception:  # pragma: no cover - defensive fallback for stripped runtimes
    cuda = None
    CudaStream = Any

import cutlass
import cutlass.cute as cute
try:
    import cutlass.pipeline as pipeline
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    pipeline = None
try:
    import cutlass.utils as cutlass_utils
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    cutlass_utils = None
try:
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    tcgen05 = None

try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except Exception:  # pragma: no cover - stage scaffolding must remain importable
    sm100_utils = None

def _load_static_persistent_scheduler_api():
    top_level_import_error: Optional[BaseException] = None
    try:
        from cutlass.utils import (  # type: ignore[attr-defined]
            PersistentTileSchedulerParams as persistent_tile_scheduler_params,
        )
        from cutlass.utils import (  # type: ignore[attr-defined]
            StaticPersistentTileScheduler as static_persistent_tile_scheduler,
        )

        return (
            persistent_tile_scheduler_params,
            static_persistent_tile_scheduler,
            None,
        )
    except (ImportError, AttributeError) as exc:
        top_level_import_error = exc

    module_path = None
    module_import_error: Optional[BaseException] = None
    try:
        cutlass_package_file = getattr(cutlass, "__file__", None)
        if cutlass_package_file is not None:
            module_path = os.path.join(
                os.path.dirname(cutlass_package_file),
                "utils",
                "static_persistent_tile_scheduler.py",
            )
        if module_path is None or not os.path.exists(module_path):
            raise FileNotFoundError(
                f"Unable to locate cutlass.utils.static_persistent_tile_scheduler at {module_path}"
            )

        module_spec = importlib.util.spec_from_file_location(
            "_kernel_0_static_persistent_tile_scheduler",
            module_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                f"Unable to create an import spec for CUTLASS scheduler module {module_path}"
            )

        scheduler_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(scheduler_module)
        return (
            scheduler_module.PersistentTileSchedulerParams,
            scheduler_module.StaticPersistentTileScheduler,
            top_level_import_error,
        )
    except Exception as exc:  # pragma: no cover - fallback diagnostics only
        module_import_error = exc

    class PersistentTileSchedulerParams:  # type: ignore[no-redef]
        pass

    return (
        PersistentTileSchedulerParams,
        None,
        module_import_error or top_level_import_error,
    )


(
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
    _STATIC_SCHEDULER_IMPORT_ERROR,
) = _load_static_persistent_scheduler_api()


LOG2_E = 1.4426950408889634
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048
PAGE_BITS = 6
PAGES_IN_FIXTURE = 64
TOKENS_IN_FIXTURE = 3
MAX_ACTIVE_CLUSTERS = 148
CLUSTER_SHAPE_MNK = (1, 1, 1)
SPARSE_TILE_N = 32
NUM_SPARSE_TILES = TOPK // SPARSE_TILE_N
QK_MMA_SHAPE = (64, 32, 16)
PV_MMA_SHAPE = (64, 128, 16)
PV_OUT_TILE = 128
PADDED_Q_HEAD_ROWS = QK_MMA_SHAPE[0]
QK_PIPE_STAGES = 2
P_PIPE_STAGES = 2
V_PIPE_STAGES = 2
O_PIPE_STAGES = 1
INT32_BYTES = 4
INT64_BYTES = 8
BF16_BYTES = cutlass.BFloat16.width // 8
FP32_BYTES = cutlass.Float32.width // 8
TMEM_COLUMN_BYTES_SM100 = 512


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


S3_Q_CKV_SMEM_BYTES = PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV * BF16_BYTES
S3_Q_KPE_SMEM_BYTES = PADDED_Q_HEAD_ROWS * HEAD_DIM_KPE * BF16_BYTES
S3_Q_SMEM_BYTES = S3_Q_CKV_SMEM_BYTES + S3_Q_KPE_SMEM_BYTES
S3_KC_QK_RING_BYTES = QK_PIPE_STAGES * SPARSE_TILE_N * HEAD_DIM_CKV * BF16_BYTES
S3_KPE_QK_RING_BYTES = QK_PIPE_STAGES * SPARSE_TILE_N * HEAD_DIM_KPE * BF16_BYTES
S3_QK_RING_BYTES = S3_KC_QK_RING_BYTES + S3_KPE_QK_RING_BYTES
S3_P_RING_BYTES = P_PIPE_STAGES * PADDED_Q_HEAD_ROWS * SPARSE_TILE_N * BF16_BYTES
S3_V_RING_BYTES = V_PIPE_STAGES * SPARSE_TILE_N * PV_OUT_TILE * BF16_BYTES
S3_TMEM_S_BYTES = PADDED_Q_HEAD_ROWS * SPARSE_TILE_N * QK_PIPE_STAGES * FP32_BYTES
S3_TMEM_O_BYTES = PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV * FP32_BYTES
S3_TMEM_LOGICAL_COLS = (S3_TMEM_S_BYTES + S3_TMEM_O_BYTES) // TMEM_COLUMN_BYTES_SM100
S3_TMEM_ALLOC_COLS = max(32, 1 << (S3_TMEM_LOGICAL_COLS - 1).bit_length())
S3_BARRIER_STORAGE_BYTES = (4 + 4 + 4 + 4 + 4 + 4 + 2) * INT64_BYTES
S3_SPARSE_META_BYTES = (
    QK_PIPE_STAGES * SPARSE_TILE_N * INT32_BYTES
    + QK_PIPE_STAGES * SPARSE_TILE_N * INT32_BYTES
    + QK_PIPE_STAGES * INT32_BYTES
)
S3_CORR_META_BYTES = (
    QK_PIPE_STAGES * NUM_QO_HEADS * FP32_BYTES
    + QK_PIPE_STAGES * NUM_QO_HEADS * FP32_BYTES
    + QK_PIPE_STAGES * NUM_QO_HEADS * FP32_BYTES
    + QK_PIPE_STAGES * INT32_BYTES
)
S3_PRE_OPERAND_BYTES = (
    S3_BARRIER_STORAGE_BYTES + INT32_BYTES + S3_SPARSE_META_BYTES + S3_CORR_META_BYTES
)


@cute.struct
class Stage0SharedStorage:
    """Minimal shared-storage scaffold for the staged round-0 kernel path."""

    placeholder: cutlass.Int32


@cute.struct
class Stage3SharedStorage:
    """Round-0 S3 shared-storage scaffold matching the architecture plan."""

    # Pipeline barriers.
    page_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
    qk_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
    s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
    p_mbar_ptr: cute.struct.MemRange[cutlass.Int64, P_PIPE_STAGES * 2]
    corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
    v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, V_PIPE_STAGES * 2]
    o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, O_PIPE_STAGES * 2]

    # TMEM allocator handoff.
    tmem_holding_buf: cutlass.Int32

    # Sparse metadata ring.
    smem_page_id: cute.struct.MemRange[cutlass.Int32, QK_PIPE_STAGES * SPARSE_TILE_N]
    smem_page_off: cute.struct.MemRange[cutlass.Int32, QK_PIPE_STAGES * SPARSE_TILE_N]
    smem_valid_mask: cute.struct.MemRange[cutlass.Int32, QK_PIPE_STAGES]

    # Correction metadata ring.
    smem_corr_alpha: cute.struct.MemRange[cutlass.Float32, QK_PIPE_STAGES * NUM_QO_HEADS]
    smem_corr_row_sum: cute.struct.MemRange[cutlass.Float32, QK_PIPE_STAGES * NUM_QO_HEADS]
    smem_corr_row_max: cute.struct.MemRange[cutlass.Float32, QK_PIPE_STAGES * NUM_QO_HEADS]
    smem_corr_flags: cute.struct.MemRange[cutlass.Int32, QK_PIPE_STAGES]

    # Main operand storage aligned for UMMA-facing SMEM.
    smem_q_ckv: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, PADDED_Q_HEAD_ROWS * HEAD_DIM_CKV],
        1024,
    ]
    smem_q_kpe: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, PADDED_Q_HEAD_ROWS * HEAD_DIM_KPE],
        1024,
    ]
    smem_kc_qk: cute.struct.Align[
        cute.struct.MemRange[
            cutlass.BFloat16, QK_PIPE_STAGES * SPARSE_TILE_N * HEAD_DIM_CKV
        ],
        1024,
    ]
    smem_kpe_qk: cute.struct.Align[
        cute.struct.MemRange[
            cutlass.BFloat16, QK_PIPE_STAGES * SPARSE_TILE_N * HEAD_DIM_KPE
        ],
        1024,
    ]
    smem_p: cute.struct.Align[
        cute.struct.MemRange[
            cutlass.BFloat16, P_PIPE_STAGES * PADDED_Q_HEAD_ROWS * SPARSE_TILE_N
        ],
        1024,
    ]
    smem_v: cute.struct.Align[
        cute.struct.MemRange[
            cutlass.BFloat16, V_PIPE_STAGES * SPARSE_TILE_N * PV_OUT_TILE
        ],
        1024,
    ]


S3_SHARED_STORAGE_BYTES = Stage3SharedStorage.size_in_bytes()


@dataclass
class SchedulerDebugResult:
    grid_shape: torch.Tensor
    debug_token_order: torch.Tensor
    debug_token_done_count: torch.Tensor


def _enum_to_int(value: Any) -> int:
    """Best-effort normalization for CUTLASS / CuTe enum-like values."""

    raw_value = getattr(value, "value", None)
    if raw_value is not None:
        try:
            return int(raw_value)
        except Exception:
            pass

    try:
        return int(value)
    except Exception:
        pass

    raw_name = getattr(value, "name", None)
    if raw_name is None:
        raw_name = str(value).rsplit(".", 1)[-1]
    normalized = str(raw_name).upper()
    if normalized == "MN":
        return 0
    if normalized == "K":
        return 1
    if normalized == "ONE":
        return 1
    if normalized == "TWO":
        return 2
    if normalized == "FOUR":
        return 4
    raise TypeError(f"Unable to normalize enum-like value {value!r} into an int")


class BlackwellStyleKernel:
    """Round-0 stage scaffold.

    Stage S0 is host-reference only, but we still keep a CuTeDSL-shaped shell
    so later stages can extend the same program structure rather than replacing
    it outright.
    """

    def __init__(
        self,
        num_qo_heads: int = NUM_QO_HEADS,
        head_dim_ckv: int = HEAD_DIM_CKV,
        head_dim_kpe: int = HEAD_DIM_KPE,
        page_size: int = PAGE_SIZE,
        topk: int = TOPK,
        cluster_shape_mnk: Tuple[int, int, int] = CLUSTER_SHAPE_MNK,
        max_active_clusters: int = MAX_ACTIVE_CLUSTERS,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ) -> None:
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.page_size = page_size
        self.topk = topk
        self.cluster_shape_mnk = cluster_shape_mnk
        self.max_active_clusters = max_active_clusters
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.threads_per_cta = 256
        self.q_major_mode = getattr(getattr(tcgen05, "OperandMajorMode", None), "K", "K")
        self.k_major_mode = getattr(getattr(tcgen05, "OperandMajorMode", None), "K", "K")
        self.p_major_mode = getattr(getattr(tcgen05, "OperandMajorMode", None), "K", "K")
        self.v_major_mode = getattr(getattr(tcgen05, "OperandMajorMode", None), "MN", "MN")
        self.q_major_mode_int = _enum_to_int(self.q_major_mode)
        self.k_major_mode_int = _enum_to_int(self.k_major_mode)
        self.p_major_mode_int = _enum_to_int(self.p_major_mode)
        self.v_major_mode_int = _enum_to_int(self.v_major_mode)
        self.qk_mma_shape = QK_MMA_SHAPE
        self.pv_mma_shape = PV_MMA_SHAPE
        self.qk_ckv_iters = self.head_dim_ckv // self.qk_mma_shape[2]
        self.qk_kpe_iters = self.head_dim_kpe // self.qk_mma_shape[2]
        self.pv_k_iters = SPARSE_TILE_N // self.pv_mma_shape[2]
        self.pv_out_iters = self.head_dim_ckv // self.pv_mma_shape[1]
        self.enable_stage2_mma_probe = False
        self.enable_stage3_storage_probe = False
        self.enable_stage4_q_prologue_probe = False
        self.qk_tiled_mma = None
        self.pv_tiled_mma = None
        self._stage2_mma_cache: Optional[Dict[str, Any]] = None
        self._stage3_storage_cache: Optional[Dict[str, Any]] = None
        self._last_scheduler_params_error: Optional[BaseException] = None
        self._host_launch_grid: Optional[Tuple[int, int, int]] = None
        self.shared_storage_bytes = S3_SHARED_STORAGE_BYTES
        self.tmem_alloc_cols = S3_TMEM_ALLOC_COLS
        self.q_smem_bytes = S3_Q_SMEM_BYTES
        self.qk_ring_bytes = S3_QK_RING_BYTES
        self.p_ring_bytes = S3_P_RING_BYTES
        self.v_ring_bytes = S3_V_RING_BYTES
        self.stage3_q_ckv_smem_layout_staged = None
        self.stage3_q_kpe_smem_layout_staged = None
        self.stage3_kc_qk_smem_layout_staged = None
        self.stage3_kpe_qk_smem_layout_staged = None
        self.stage3_p_smem_layout_staged = None
        self.stage3_v_smem_layout_staged = None
        self.stage3_tmem_s_staged_fake = None
        self.stage3_tmem_o_staged_fake = None
        self.stage3_q_ckv_smem_elements = None
        self.stage3_q_kpe_smem_elements = None
        self.stage3_kc_qk_smem_elements = None
        self.stage3_kpe_qk_smem_elements = None
        self.stage3_p_smem_elements = None
        self.stage3_v_smem_elements = None
        self.stage3_tmem_s_cols = None
        self.stage3_tmem_o_cols = None

    @staticmethod
    def _compute_grid(
        num_tokens: int,
        *,
        max_active_clusters: int = MAX_ACTIVE_CLUSTERS,
    ) -> Tuple[int, int, int]:
        return (1, 1, max(1, min(num_tokens, max_active_clusters)))

    @staticmethod
    def _compute_num_scheduler_iters(num_tokens: int, num_launched_ctas: int) -> int:
        if num_launched_ctas <= 0:
            raise ValueError(f"num_launched_ctas must be positive, got {num_launched_ctas}")
        if num_tokens <= 0:
            return 1
        return max(1, (num_tokens + num_launched_ctas - 1) // num_launched_ctas)

    def _require_host_launch_grid(self) -> Tuple[int, int, int]:
        """Resolve the host-computed launch grid before entering the CuTe JIT path."""

        grid = self._host_launch_grid
        if grid is None:
            raise RuntimeError(
                "Stage S1 launch grid must be computed on the host before cute.compile(...) "
                "and stored in program._host_launch_grid; do not derive it from CuTe tensor "
                "shapes inside the @cute.jit launch path."
            )
        return grid

    @cute.jit
    def _make_scheduler_params(self, num_tokens: cutlass.Int32):
        return PersistentTileSchedulerParams(
            problem_shape_ntile_mnl=(num_tokens, 1, 1),
            cluster_shape_mnk=self.cluster_shape_mnk,
            swizzle_size=self.swizzle_size,
            raster_along_m=self.raster_along_m,
        )

    @cute.jit
    def _make_stage2_tiled_mma_atoms_jit(self):
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            self.q_major_mode,
            self.k_major_mode,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.qk_mma_shape[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            self.p_major_mode,
            self.v_major_mode,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.pv_mma_shape[:2],
        )
        return qk_tiled_mma, pv_tiled_mma

    @cute.jit
    def _ensure_stage2_tiled_mma_atoms_jit(self):
        """Construct and persist the S2 MMA atoms on the program object.

        Keep this helper free of Python-object conditionals so CuTe never tries
        to lower dynamic control flow over ``self`` state during S2 probing.
        """

        qk_tiled_mma, pv_tiled_mma = self._make_stage2_tiled_mma_atoms_jit()
        self.qk_tiled_mma = qk_tiled_mma
        self.pv_tiled_mma = pv_tiled_mma
        return qk_tiled_mma, pv_tiled_mma

    def _stage2_atoms_persisted_host(self) -> Tuple[int, int]:
        qk_op = getattr(self.qk_tiled_mma, "op", None)
        pv_op = getattr(self.pv_tiled_mma, "op", None)
        qk_persisted = int(getattr(qk_op, "shape_mnk", None) is not None)
        pv_persisted = int(getattr(pv_op, "shape_mnk", None) is not None)
        return qk_persisted, pv_persisted

    def _require_stage2_mma_atoms(self) -> Dict[str, Any]:
        """Return the S2 MMA metadata only after the JIT/device probe populated it."""

        if self._stage2_mma_cache is not None:
            return self._stage2_mma_cache

        raise RuntimeError(
            "Stage S2 MMA metadata is populated only from the @cute.jit / @cute.kernel "
            "probe path. Call stage2_mma_debug_reference(device=...) so the live CuTe "
            "launch path constructs the MMA atoms inside a valid MLIR context."
        )

    @cute.jit
    def _ensure_stage3_storage_artifacts_jit(self, qk_tiled_mma, pv_tiled_mma):
        """Construct the real S3 CuTe layout artifacts needed by later stages.

        S3 is not only an arithmetic budget check: later stages S4-S12 consume the
        actual staged SMEM/TMEM layouts. Build and persist those layout objects here
        from the official helper APIs so the launch path already matches the
        Blackwell reference style.
        """

        q_ckv_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_shape,
            cutlass.BFloat16,
            self.qk_ckv_iters,
        )
        q_ckv_smem_layout_staged = cute.logical_divide(
            q_ckv_smem_layout_staged,
            (None, None, None, self.qk_ckv_iters),
        )

        q_kpe_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_shape,
            cutlass.BFloat16,
            self.qk_kpe_iters,
        )
        q_kpe_smem_layout_staged = cute.logical_divide(
            q_kpe_smem_layout_staged,
            (None, None, None, self.qk_kpe_iters),
        )

        kc_qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_shape,
            cutlass.BFloat16,
            self.qk_ckv_iters * QK_PIPE_STAGES,
        )
        kc_qk_smem_layout_staged = cute.logical_divide(
            kc_qk_smem_layout_staged,
            (None, None, None, self.qk_ckv_iters),
        )

        kpe_qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_shape,
            cutlass.BFloat16,
            self.qk_kpe_iters * QK_PIPE_STAGES,
        )
        kpe_qk_smem_layout_staged = cute.logical_divide(
            kpe_qk_smem_layout_staged,
            (None, None, None, self.qk_kpe_iters),
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_shape,
            cutlass.BFloat16,
            self.pv_k_iters * P_PIPE_STAGES,
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged,
            (None, None, None, self.pv_k_iters),
        )

        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_shape,
            cutlass.BFloat16,
            self.pv_k_iters * V_PIPE_STAGES,
        )
        v_smem_layout_staged = cute.logical_divide(
            v_smem_layout_staged,
            (None, None, None, self.pv_k_iters),
        )

        tmem_s_shape = qk_tiled_mma.partition_shape_C(self.qk_mma_shape[:2])
        tmem_s_staged_fake = qk_tiled_mma.make_fragment_C(
            cute.append(tmem_s_shape, QK_PIPE_STAGES)
        )
        tmem_o_tile_shape = pv_tiled_mma.partition_shape_C(self.pv_mma_shape[:2])
        tmem_o_tile_fake = pv_tiled_mma.make_fragment_C(tmem_o_tile_shape)
        # Model O as the single logical [64, 512] TMEM tensor planned for S3, not
        # as four independent [64, 128] slices. Follow the Blackwell MLA style and
        # append the output-slice mode onto the partitioned C fragment layout.
        tmem_o_staged_layout = cute.append(
            tmem_o_tile_fake.layout,
            cute.make_layout(self.pv_out_iters, stride=self.pv_mma_shape[1] // 2),
        )
        tmem_o_staged_fake = cute.make_tensor(
            tmem_o_tile_fake.iterator,
            tmem_o_staged_layout,
        )

        q_ckv_smem_elements = cute.cosize(q_ckv_smem_layout_staged)
        q_kpe_smem_elements = cute.cosize(q_kpe_smem_layout_staged)
        kc_qk_smem_elements = cute.cosize(kc_qk_smem_layout_staged)
        kpe_qk_smem_elements = cute.cosize(kpe_qk_smem_layout_staged)
        p_smem_elements = cute.cosize(p_smem_layout_staged)
        v_smem_elements = cute.cosize(v_smem_layout_staged)

        q_smem_bytes = (q_ckv_smem_elements + q_kpe_smem_elements) * BF16_BYTES
        qk_ring_bytes = (kc_qk_smem_elements + kpe_qk_smem_elements) * BF16_BYTES
        p_ring_bytes = p_smem_elements * BF16_BYTES
        v_ring_bytes = v_smem_elements * BF16_BYTES

        tmem_s_cols = cutlass_utils.get_num_tmem_alloc_cols(
            tmem_s_staged_fake,
            rounding=False,
            arch="sm_100",
        )
        tmem_o_cols = cutlass_utils.get_num_tmem_alloc_cols(
            tmem_o_staged_fake,
            rounding=False,
            arch="sm_100",
        )
        tmem_alloc_cols = cutlass_utils.get_num_tmem_alloc_cols(
            [
                tmem_s_staged_fake,
                tmem_o_staged_fake,
            ],
            rounding=True,
            arch="sm_100",
        )

        self.stage3_q_ckv_smem_layout_staged = q_ckv_smem_layout_staged
        self.stage3_q_kpe_smem_layout_staged = q_kpe_smem_layout_staged
        self.stage3_kc_qk_smem_layout_staged = kc_qk_smem_layout_staged
        self.stage3_kpe_qk_smem_layout_staged = kpe_qk_smem_layout_staged
        self.stage3_p_smem_layout_staged = p_smem_layout_staged
        self.stage3_v_smem_layout_staged = v_smem_layout_staged
        self.stage3_tmem_s_staged_fake = tmem_s_staged_fake
        self.stage3_tmem_o_staged_fake = tmem_o_staged_fake
        self.stage3_q_ckv_smem_elements = q_ckv_smem_elements
        self.stage3_q_kpe_smem_elements = q_kpe_smem_elements
        self.stage3_kc_qk_smem_elements = kc_qk_smem_elements
        self.stage3_kpe_qk_smem_elements = kpe_qk_smem_elements
        self.stage3_p_smem_elements = p_smem_elements
        self.stage3_v_smem_elements = v_smem_elements
        self.stage3_tmem_s_cols = tmem_s_cols
        self.stage3_tmem_o_cols = tmem_o_cols
        self.q_smem_bytes = q_smem_bytes
        self.qk_ring_bytes = qk_ring_bytes
        self.p_ring_bytes = p_ring_bytes
        self.v_ring_bytes = v_ring_bytes
        self.tmem_alloc_cols = tmem_alloc_cols

        return (
            q_ckv_smem_layout_staged,
            q_kpe_smem_layout_staged,
            kc_qk_smem_layout_staged,
            kpe_qk_smem_layout_staged,
            p_smem_layout_staged,
            v_smem_layout_staged,
            tmem_s_staged_fake,
            tmem_o_staged_fake,
            q_smem_bytes,
            qk_ring_bytes,
            p_ring_bytes,
            v_ring_bytes,
            tmem_alloc_cols,
            tmem_s_cols,
            tmem_o_cols,
        )

    def _require_stage3_storage_artifacts(self) -> None:
        required_fields = (
            "stage3_q_ckv_smem_layout_staged",
            "stage3_q_kpe_smem_layout_staged",
            "stage3_kc_qk_smem_layout_staged",
            "stage3_kpe_qk_smem_layout_staged",
            "stage3_p_smem_layout_staged",
            "stage3_v_smem_layout_staged",
            "stage3_tmem_s_staged_fake",
            "stage3_tmem_o_staged_fake",
            "stage3_q_ckv_smem_elements",
            "stage3_q_kpe_smem_elements",
            "stage3_kc_qk_smem_elements",
            "stage3_kpe_qk_smem_elements",
            "stage3_p_smem_elements",
            "stage3_v_smem_elements",
            "stage3_tmem_s_cols",
            "stage3_tmem_o_cols",
        )
        missing = [field_name for field_name in required_fields if getattr(self, field_name) is None]
        if missing:
            raise RuntimeError(
                "Stage S3 probe did not persist the planned CuTe layout/TMEM artifacts needed "
                f"by later stages; missing fields: {missing}"
            )

    def stage3_storage_host_reference(
        self,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return {
            "shared_storage_bytes": torch.tensor(
                [self.shared_storage_bytes], dtype=torch.int32, device=device
            ).contiguous(),
            "tmem_alloc_cols": torch.tensor(
                [self.tmem_alloc_cols], dtype=torch.int32, device=device
            ).contiguous(),
            "q_smem_bytes": torch.tensor(
                [self.q_smem_bytes], dtype=torch.int32, device=device
            ).contiguous(),
            "qk_ring_bytes": torch.tensor(
                [self.qk_ring_bytes], dtype=torch.int32, device=device
            ).contiguous(),
            "p_ring_bytes": torch.tensor(
                [self.p_ring_bytes], dtype=torch.int32, device=device
            ).contiguous(),
            "v_ring_bytes": torch.tensor(
                [self.v_ring_bytes], dtype=torch.int32, device=device
            ).contiguous(),
        }

    def _stage3_storage_result_from_meta(
        self,
        storage_meta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "shared_storage_bytes": storage_meta[0:1].contiguous(),
            "tmem_alloc_cols": storage_meta[1:2].contiguous(),
            "q_smem_bytes": storage_meta[2:3].contiguous(),
            "qk_ring_bytes": storage_meta[3:4].contiguous(),
            "p_ring_bytes": storage_meta[4:5].contiguous(),
            "v_ring_bytes": storage_meta[5:6].contiguous(),
        }

    def stage3_storage_debug_reference(
        self,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Device-captured S3 storage outputs emitted by the CuTe launch path."""

        if self._stage3_storage_cache is not None:
            self._require_stage3_storage_artifacts()
            return self._stage3_storage_cache

        if sm100_utils is None or tcgen05 is None or cutlass_utils is None:
            raise RuntimeError(
                "Stage S3 requires cutlass.utils, tcgen05, and blackwell_helpers so the real "
                "SMEM/TMEM layout artifacts can be constructed through the CuTe kernel path."
            )

        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Stage S3 requires a CUDA device so the SharedStorage metadata is "
                    "captured from the CuTe JIT/kernel path rather than synthesized "
                    "purely on the host."
                )
            device = torch.device("cuda")
        if device.type != "cuda":
            raise RuntimeError(
                f"Stage S3 requires a CUDA device for the live storage probe, got device={device}"
            )

        previous_stage3_probe_state = self.enable_stage3_storage_probe
        try:
            self.enable_stage3_storage_probe = True
            probe_result = _launch_s1_scheduler_debug_kernel(
                self,
                num_tokens=1,
                device=device,
            )
        finally:
            self.enable_stage3_storage_probe = previous_stage3_probe_state

        self._require_stage3_storage_artifacts()
        result = self._stage3_storage_result_from_meta(
            probe_result["debug_stage3_storage_meta"].to(dtype=torch.int32, device=device)
        )
        host_reference = self.stage3_storage_host_reference(device=device)
        for field_name in result.keys():
            if not torch.equal(result[field_name], host_reference[field_name]):
                raise AssertionError(
                    f"Stage S3 probe metadata disagreed with the host storage model for {field_name}"
                )

        self._stage3_storage_cache = result
        return result

    def stage2_mma_debug_reference(
        self,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Device-captured S2 outputs describing the live QK/PV MMA atoms."""

        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Stage S2 requires a CUDA device so the MMA atoms are constructed "
                    "inside the CuTe JIT/kernel path rather than on the host."
                )
            device = torch.device("cuda")
        if device.type != "cuda":
            raise RuntimeError(
                f"Stage S2 requires a CUDA device for the live MMA probe, got device={device}"
            )

        previous_probe_state = self.enable_stage2_mma_probe
        self.enable_stage2_mma_probe = True
        try:
            probe_result = _launch_s1_scheduler_debug_kernel(
                self,
                num_tokens=1,
                device=device,
            )
        finally:
            self.enable_stage2_mma_probe = previous_probe_state

        mma_meta = probe_result["debug_stage2_mma_meta"].to(dtype=torch.int32, device=device)
        qk_atom_persisted_host, pv_atom_persisted_host = self._stage2_atoms_persisted_host()
        qk_atom_persisted = torch.tensor(
            [qk_atom_persisted_host], dtype=torch.int32, device=device
        ).contiguous()
        pv_atom_persisted = torch.tensor(
            [pv_atom_persisted_host], dtype=torch.int32, device=device
        ).contiguous()
        if int(mma_meta[16].item()) != qk_atom_persisted_host or int(mma_meta[17].item()) != pv_atom_persisted_host:
            raise AssertionError(
                "Stage S2 probe metadata reported persisted MMA flags that disagree with the live "
                "program object state after the CuTe launch path completed"
            )
        result = {
            "qk_mma_shape": mma_meta[0:3].contiguous(),
            "pv_mma_shape": mma_meta[3:6].contiguous(),
            "qk_ckv_iters": mma_meta[6:7].contiguous(),
            "qk_kpe_iters": mma_meta[7:8].contiguous(),
            "pv_k_iters": mma_meta[8:9].contiguous(),
            "pv_out_iters": mma_meta[9:10].contiguous(),
            "qk_a_major_mode": mma_meta[10:11].contiguous(),
            "qk_b_major_mode": mma_meta[11:12].contiguous(),
            "pv_a_major_mode": mma_meta[12:13].contiguous(),
            "pv_b_major_mode": mma_meta[13:14].contiguous(),
            "qk_helper_constructed": mma_meta[14:15].contiguous(),
            "pv_helper_constructed": mma_meta[15:16].contiguous(),
            "qk_atom_persisted": qk_atom_persisted,
            "pv_atom_persisted": pv_atom_persisted,
        }
        self._stage2_mma_cache = result
        return result

    def scheduler_debug_reference(
        self,
        num_tokens: int,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Host emulation of the S1 persistent token scheduler debug buffers.

        The kernel slice for S1 is scheduler-only, so we mirror exactly the
        token ownership that the future CuTeDSL persistent kernel will use:

          token_idx = cta_linear_z + scheduler_iter * grid_z

        with one CTA per token until `grid_z` is saturated at 148 CTAs.
        """

        grid_shape_tuple = self._compute_grid(
            num_tokens, max_active_clusters=self.max_active_clusters
        )
        num_launched_ctas = grid_shape_tuple[2]
        num_scheduler_iters = self._compute_num_scheduler_iters(
            num_tokens, num_launched_ctas
        )

        grid_shape = torch.tensor(
            grid_shape_tuple,
            dtype=torch.int32,
            device=device,
        ).contiguous()
        debug_token_order = torch.full(
            (num_launched_ctas, num_scheduler_iters),
            -1,
            dtype=torch.int32,
            device=device,
        )
        debug_token_done_count = torch.zeros(
            (num_tokens,),
            dtype=torch.int32,
            device=device,
        )

        for cta_idx in range(num_launched_ctas):
            for scheduler_iter in range(num_scheduler_iters):
                token_idx = cta_idx + scheduler_iter * num_launched_ctas
                if token_idx >= num_tokens:
                    continue
                debug_token_order[cta_idx, scheduler_iter] = token_idx
                debug_token_done_count[token_idx] += 1

        return {
            "grid_shape": grid_shape,
            "debug_token_order": debug_token_order.contiguous(),
            "debug_token_done_count": debug_token_done_count.contiguous(),
        }

    @cute.jit
    def _emit_stage_probe_metadata(
        self,
        debug_stage2_mma_meta: cute.Tensor,
        debug_stage3_storage_meta: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        """Emit S2/S3 probe metadata outside any dynamic thread predicate.

        CuTe cannot lower dynamic ``if tidx == 0`` regions that capture
        user-defined Python / CuTe struct objects such as layout-backed
        ``SharedStorage`` or tiled MMA descriptors. Keep those references in a
        constexpr-only helper, and let the scheduler-only single-thread region
        below handle just the dynamic token-order bookkeeping.
        """

        if cutlass.const_expr(self.enable_stage2_mma_probe):
            qk_tiled_mma, pv_tiled_mma = self._ensure_stage2_tiled_mma_atoms_jit()
            debug_stage2_mma_meta[0] = qk_tiled_mma.op.shape_mnk[0]
            debug_stage2_mma_meta[1] = qk_tiled_mma.op.shape_mnk[1]
            debug_stage2_mma_meta[2] = qk_tiled_mma.op.shape_mnk[2]
            debug_stage2_mma_meta[3] = pv_tiled_mma.op.shape_mnk[0]
            debug_stage2_mma_meta[4] = pv_tiled_mma.op.shape_mnk[1]
            debug_stage2_mma_meta[5] = pv_tiled_mma.op.shape_mnk[2]
            debug_stage2_mma_meta[6] = cutlass.Int32(self.qk_ckv_iters)
            debug_stage2_mma_meta[7] = cutlass.Int32(self.qk_kpe_iters)
            debug_stage2_mma_meta[8] = cutlass.Int32(self.pv_k_iters)
            debug_stage2_mma_meta[9] = cutlass.Int32(self.pv_out_iters)
            debug_stage2_mma_meta[10] = cutlass.Int32(self.q_major_mode_int)
            debug_stage2_mma_meta[11] = cutlass.Int32(self.k_major_mode_int)
            debug_stage2_mma_meta[12] = cutlass.Int32(self.p_major_mode_int)
            debug_stage2_mma_meta[13] = cutlass.Int32(self.v_major_mode_int)
            debug_stage2_mma_meta[14] = cutlass.Int32(1)
            debug_stage2_mma_meta[15] = cutlass.Int32(1)
            debug_stage2_mma_meta[16] = cutlass.Int32(1 if self.qk_tiled_mma is not None else 0)
            debug_stage2_mma_meta[17] = cutlass.Int32(1 if self.pv_tiled_mma is not None else 0)

        if cutlass.const_expr(self.enable_stage3_storage_probe):
            debug_stage3_storage_meta[0] = cutlass.Int32(SharedStorage.size_in_bytes())
            debug_stage3_storage_meta[1] = cutlass.Int32(self.tmem_alloc_cols)
            debug_stage3_storage_meta[2] = cutlass.Int32(self.q_smem_bytes)
            debug_stage3_storage_meta[3] = cutlass.Int32(self.qk_ring_bytes)
            debug_stage3_storage_meta[4] = cutlass.Int32(self.p_ring_bytes)
            debug_stage3_storage_meta[5] = cutlass.Int32(self.v_ring_bytes)
        return

    @cute.jit
    def _run_scheduler_single_thread(
        self,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        tile_sched_params: PersistentTileSchedulerParams,
        num_tokens: cutlass.Int32,
    ):
        """Keep the dynamic thread-predicated region scheduler-only."""

        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        if tidx == 0:
            cute.printf(
                "S1 scheduler kernel enter: block_idx=({}, {}, {}), grid_dim={}, num_tokens={}, cluster_shape_mnk={}",
                bidx,
                bidy,
                bidz,
                cute.arch.grid_dim(),
                num_tokens,
                self.cluster_shape_mnk,
            )
            cute.printf(
                "S2 mma config: qk_shape=({}, {}, {}), pv_shape=({}, {}, {}), qk_ckv_iters={}, qk_kpe_iters={}, pv_k_iters={}, pv_out_iters={}",
                self.qk_mma_shape[0],
                self.qk_mma_shape[1],
                self.qk_mma_shape[2],
                self.pv_mma_shape[0],
                self.pv_mma_shape[1],
                self.pv_mma_shape[2],
                self.qk_ckv_iters,
                self.qk_kpe_iters,
                self.pv_k_iters,
                self.pv_out_iters,
            )
            scheduler = StaticPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
            )
            work_tile = scheduler.initial_work_tile_info()
            scheduler_iter = cutlass.Int32(0)
            cute.printf(
                "S1 scheduler initial work: valid={}, tile_idx={}",
                work_tile.is_valid_tile,
                work_tile.tile_idx,
            )
            while work_tile.is_valid_tile:
                token_idx = work_tile.tile_idx[0]
                cute.printf(
                    "S1 scheduler iter: cta_z={}, iter={}, token_idx={}, tile_idx={}",
                    bidz,
                    scheduler_iter,
                    token_idx,
                    work_tile.tile_idx,
                )
                debug_token_order[bidz, scheduler_iter] = token_idx
                token_done_count = debug_token_done_count[token_idx]
                debug_token_done_count[token_idx] = token_done_count + cutlass.Int32(1)
                scheduler.advance_to_next_work()
                scheduler_iter = scheduler_iter + cutlass.Int32(1)
                work_tile = scheduler.get_current_work()
            cute.printf(
                "S1 scheduler kernel exit: cta_z={}, scheduled_iters={}",
                bidz,
                scheduler_iter,
            )
        return

    @cute.jit
    def _run_stage4_q_prologue_single_thread(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        debug_q_ckv_padded: cute.Tensor,
        debug_q_kpe_padded: cute.Tensor,
        sQ_ckv_staged: cute.Tensor,
        sQ_kpe_staged: cute.Tensor,
        tile_sched_params: PersistentTileSchedulerParams,
        num_tokens: cutlass.Int32,
    ):
        if cutlass.const_expr(self.enable_stage4_q_prologue_probe):
            _, _, bidz = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            if bidz == 0:
                if tidx == 0:
                    bf16_zero_pad = cutlass.BFloat16(0.0)
                    scheduler = StaticPersistentTileScheduler.create(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        cute.arch.grid_dim(),
                    )
                    work_tile = scheduler.initial_work_tile_info()
                    token_idx = cutlass.Int32(0)
                    has_valid_token = work_tile.is_valid_tile
                    if has_valid_token:
                        token_idx = work_tile.tile_idx[0]

                    cute.printf(
                        "S4 q prologue enter: cta_z={}, has_valid_token={}, token_idx={}, logical_heads={}, padded_rows={}, head_dim_ckv={}, head_dim_kpe={}",
                        bidz,
                        has_valid_token,
                        token_idx,
                        self.num_qo_heads,
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_ckv,
                        self.head_dim_kpe,
                    )
                    cute.printf(
                        "S4 staged q tensors: real Stage-3 swizzled SMEM tensors are allocated, but this single-thread probe mirrors the logical padded Q payload directly into GMEM debug tensors to avoid illegal dense remapping of the swizzled backing storage; padded_q_ckv_shape=({}, {}), padded_q_kpe_shape=({}, {}), qk_ckv_iters={}, qk_kpe_iters={}",
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_ckv,
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_kpe,
                        self.qk_ckv_iters,
                        self.qk_kpe_iters,
                    )
                    cute.printf(
                        "S4 q prologue note: this probe validates the logical 16->64 row padding contract and preserves the Stage-3 SMEM allocation objects without fabricating incompatible dense 2D views over the swizzled layouts"
                    )
                    cute.printf(
                        "S4 swizzled q tensors kept opaque for this probe: logical_debug_q_ckv_shape=({}, {}), logical_debug_q_kpe_shape=({}, {})",
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_ckv,
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_kpe,
                    )

                    for row_idx in cutlass.range(PADDED_Q_HEAD_ROWS):
                        for col_idx in cutlass.range(self.head_dim_ckv):
                            q_value = bf16_zero_pad
                            if has_valid_token and row_idx < self.num_qo_heads:
                                q_value = q_nope[token_idx, row_idx, col_idx]
                            debug_q_ckv_padded[row_idx, col_idx] = q_value

                    for row_idx in cutlass.range(PADDED_Q_HEAD_ROWS):
                        for col_idx in cutlass.range(self.head_dim_kpe):
                            q_value = bf16_zero_pad
                            if has_valid_token and row_idx < self.num_qo_heads:
                                q_value = q_pe[token_idx, row_idx, col_idx]
                            debug_q_kpe_padded[row_idx, col_idx] = q_value

                    cute.printf(
                        "S4 logical padded q mirror ready: q_ckv_shape=({}, {}), q_kpe_shape=({}, {}), qk_ckv_iters={}, qk_kpe_iters={}",
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_ckv,
                        PADDED_Q_HEAD_ROWS,
                        self.head_dim_kpe,
                        self.qk_ckv_iters,
                        self.qk_kpe_iters,
                    )

                    cute.printf(
                        "S4 q prologue logical/debug mirror ready: token_idx={}, debug_q_ckv[0,0]={}, debug_q_ckv[15,511]={}, debug_q_ckv[16,0]={}, debug_q_kpe[0,0]={}, debug_q_kpe[15,63]={}, debug_q_kpe[16,0]={}",
                        token_idx,
                        debug_q_ckv_padded[0, 0],
                        debug_q_ckv_padded[
                            self.num_qo_heads - 1, self.head_dim_ckv - 1
                        ],
                        debug_q_ckv_padded[self.num_qo_heads, 0],
                        debug_q_kpe_padded[0, 0],
                        debug_q_kpe_padded[
                            self.num_qo_heads - 1, self.head_dim_kpe - 1
                        ],
                        debug_q_kpe_padded[self.num_qo_heads, 0],
                    )
                    cute.print_tensor(debug_q_ckv_padded[(0, None)])
                    cute.print_tensor(debug_q_kpe_padded[(0, None)])

                    cute.printf(
                        "S4 q prologue exit: token_idx={}, q_ckv[0,0]={}, q_ckv[15,511]={}, q_ckv[16,0]={}, q_kpe[0,0]={}, q_kpe[15,63]={}, q_kpe[16,0]={}",
                        token_idx,
                        debug_q_ckv_padded[0, 0],
                        debug_q_ckv_padded[self.num_qo_heads - 1, self.head_dim_ckv - 1],
                        debug_q_ckv_padded[self.num_qo_heads, 0],
                        debug_q_kpe_padded[0, 0],
                        debug_q_kpe_padded[self.num_qo_heads - 1, self.head_dim_kpe - 1],
                        debug_q_kpe_padded[self.num_qo_heads, 0],
                    )

        if cutlass.const_expr(pipeline is not None):
            pipeline.agent_sync(pipeline.Agent.ThreadBlock)
        return

    @cute.jit
    def persistent_token_scheduler_debug(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        debug_stage2_mma_meta: cute.Tensor,
        debug_stage3_storage_meta: cute.Tensor,
        debug_q_ckv_padded: cute.Tensor,
        debug_q_kpe_padded: cute.Tensor,
        tile_sched_params: PersistentTileSchedulerParams,
        num_tokens: cutlass.Int32,
        SharedStorage: cutlass.Constexpr,
    ):
        """Stage-S1 CuTeDSL scheduler scaffold.

        Round-0 S1 validates only the launch/scheduler contract. We keep the
        device-path shell here, using the official persistent scheduler APIs, so
        later stages can directly extend this method into the real token loop.
        """

        self._emit_stage_probe_metadata(
            debug_stage2_mma_meta,
            debug_stage3_storage_meta,
            SharedStorage,
        )
        self._run_scheduler_single_thread(
            debug_token_order,
            debug_token_done_count,
            tile_sched_params,
            num_tokens,
        )
        if cutlass.const_expr(self.enable_stage4_q_prologue_probe):
            qk_tiled_mma, _ = self._make_stage2_tiled_mma_atoms_jit()
            q_ckv_smem_layout_staged = sm100_utils.make_smem_layout_a(
                qk_tiled_mma,
                self.qk_mma_shape,
                cutlass.BFloat16,
                self.qk_ckv_iters,
            )
            q_ckv_smem_layout_staged = cute.logical_divide(
                q_ckv_smem_layout_staged,
                (None, None, None, self.qk_ckv_iters),
            )
            q_kpe_smem_layout_staged = sm100_utils.make_smem_layout_a(
                qk_tiled_mma,
                self.qk_mma_shape,
                cutlass.BFloat16,
                self.qk_kpe_iters,
            )
            q_kpe_smem_layout_staged = cute.logical_divide(
                q_kpe_smem_layout_staged,
                (None, None, None, self.qk_kpe_iters),
            )
            cute.printf(
                "S4 rebuilt kernel-local q staged layouts: q_ckv_cosize={}, q_kpe_cosize={}, qk_ckv_iters={}, qk_kpe_iters={}",
                cute.cosize(q_ckv_smem_layout_staged),
                cute.cosize(q_kpe_smem_layout_staged),
                self.qk_ckv_iters,
                self.qk_kpe_iters,
            )
            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ_ckv_staged = storage.smem_q_ckv.get_tensor(
                q_ckv_smem_layout_staged.outer,
                swizzle=q_ckv_smem_layout_staged.inner,
            )
            sQ_kpe_staged = storage.smem_q_kpe.get_tensor(
                q_kpe_smem_layout_staged.outer,
                swizzle=q_kpe_smem_layout_staged.inner,
            )
            self._run_stage4_q_prologue_single_thread(
                q_nope,
                q_pe,
                debug_q_ckv_padded,
                debug_q_kpe_padded,
                sQ_ckv_staged,
                sQ_kpe_staged,
                tile_sched_params,
                num_tokens,
            )
        return

    @cute.jit
    def __call__(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        debug_stage2_mma_meta: cute.Tensor,
        debug_stage3_storage_meta: cute.Tensor,
        debug_q_ckv_padded: cute.Tensor,
        debug_q_kpe_padded: cute.Tensor,
        num_tokens: cutlass.Int32,
        stream: CudaStream,
    ):
        """Stage-S1 launch-time orchestration for the persistent scheduler slice."""

        tile_sched_params = self._make_scheduler_params(num_tokens)
        qk_tiled_mma, pv_tiled_mma = self._ensure_stage2_tiled_mma_atoms_jit()
        (
            q_ckv_smem_layout_staged,
            q_kpe_smem_layout_staged,
            kc_qk_smem_layout_staged,
            kpe_qk_smem_layout_staged,
            p_smem_layout_staged,
            v_smem_layout_staged,
            _tmem_s_staged_fake,
            _tmem_o_staged_fake,
            q_smem_bytes,
            qk_ring_bytes,
            p_ring_bytes,
            v_ring_bytes,
            tmem_alloc_cols,
            _tmem_s_cols,
            _tmem_o_cols,
        ) = self._ensure_stage3_storage_artifacts_jit(qk_tiled_mma, pv_tiled_mma)

        @cute.struct
        class Stage3LayoutBackedSharedStorage:
            # Pipeline barriers.
            page_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
            qk_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
            s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
            p_mbar_ptr: cute.struct.MemRange[cutlass.Int64, P_PIPE_STAGES * 2]
            corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, QK_PIPE_STAGES * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, V_PIPE_STAGES * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, O_PIPE_STAGES * 2]

            # TMEM allocator handoff.
            tmem_holding_buf: cutlass.Int32

            # Sparse metadata ring.
            smem_page_id: cute.struct.MemRange[
                cutlass.Int32, QK_PIPE_STAGES * SPARSE_TILE_N
            ]
            smem_page_off: cute.struct.MemRange[
                cutlass.Int32, QK_PIPE_STAGES * SPARSE_TILE_N
            ]
            smem_valid_mask: cute.struct.MemRange[cutlass.Int32, QK_PIPE_STAGES]

            # Correction metadata ring.
            smem_corr_alpha: cute.struct.MemRange[
                cutlass.Float32, QK_PIPE_STAGES * NUM_QO_HEADS
            ]
            smem_corr_row_sum: cute.struct.MemRange[
                cutlass.Float32, QK_PIPE_STAGES * NUM_QO_HEADS
            ]
            smem_corr_row_max: cute.struct.MemRange[
                cutlass.Float32, QK_PIPE_STAGES * NUM_QO_HEADS
            ]
            smem_corr_flags: cute.struct.MemRange[cutlass.Int32, QK_PIPE_STAGES]

            # Main operand storage aligned for UMMA-facing SMEM.
            smem_q_ckv: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(q_ckv_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_kpe: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(q_kpe_smem_layout_staged)
                ],
                1024,
            ]
            smem_kc_qk: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(kc_qk_smem_layout_staged)
                ],
                1024,
            ]
            smem_kpe_qk: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(kpe_qk_smem_layout_staged)
                ],
                1024,
            ]
            smem_p: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(p_smem_layout_staged)
                ],
                1024,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(v_smem_layout_staged)
                ],
                1024,
            ]

        SharedStorage = Stage3LayoutBackedSharedStorage
        self.shared_storage_bytes = SharedStorage.size_in_bytes()
        self.q_smem_bytes = q_smem_bytes
        self.qk_ring_bytes = qk_ring_bytes
        self.p_ring_bytes = p_ring_bytes
        self.v_ring_bytes = v_ring_bytes
        self.tmem_alloc_cols = tmem_alloc_cols
        self.kernel1_impl(
            q_nope,
            q_pe,
            debug_token_order,
            debug_token_done_count,
            debug_stage2_mma_meta,
            debug_stage3_storage_meta,
            debug_q_ckv_padded,
            debug_q_kpe_padded,
            tile_sched_params,
            num_tokens,
            SharedStorage,
        ).launch(
            grid=self._host_launch_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel1_impl(
        self,
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        debug_stage2_mma_meta: cute.Tensor,
        debug_stage3_storage_meta: cute.Tensor,
        debug_q_ckv_padded: cute.Tensor,
        debug_q_kpe_padded: cute.Tensor,
        tile_sched_params: PersistentTileSchedulerParams,
        num_tokens: cutlass.Int32,
        SharedStorage: cutlass.Constexpr,
    ):
        """Deferred device entry for the round-0 staged kernel."""

        self.persistent_token_scheduler_debug(
            q_nope,
            q_pe,
            debug_token_order,
            debug_token_done_count,
            debug_stage2_mma_meta,
            debug_stage3_storage_meta,
            debug_q_ckv_padded,
            debug_q_kpe_padded,
            tile_sched_params,
            num_tokens,
            SharedStorage,
        )
        return


program_compile_cache: Dict[Any, Any] = dict()


@dataclass
class SyntheticFixture:
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    ckv_cache: torch.Tensor
    kpe_cache: torch.Tensor
    sparse_indices: torch.Tensor
    sm_scale: torch.Tensor


def _deterministic_bf16(
    shape: Tuple[int, ...],
    modulus: int,
    center: int,
    scale: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    values = torch.arange(math.prod(shape), dtype=torch.float32, device=device).reshape(shape)
    values = ((values % modulus) - center) / scale
    return values.to(torch.bfloat16).contiguous()


def _build_sparse_fixture(
    num_tokens: int,
    topk: int,
    max_flat_token_idx: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    sparse_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)

    # Token 0: fully valid, dense-in-order sparse list.
    sparse_indices[0] = torch.arange(topk, dtype=torch.int32, device=device)

    # Token 1: mixed valid / invalid with explicit repeats.
    mixed = torch.full((topk,), -1, dtype=torch.int32, device=device)
    for j in range(topk):
        pattern = j % 6
        if pattern == 0:
            mixed[j] = int((j * 17 + 5) % max_flat_token_idx)
        elif pattern == 1:
            mixed[j] = mixed[j - 1]
        elif pattern == 2:
            mixed[j] = -1
        elif pattern == 3:
            mixed[j] = int((j * 29 + 11) % max_flat_token_idx)
        elif pattern == 4:
            mixed[j] = -1
        else:
            mixed[j] = int((j * 7 + 3) % max_flat_token_idx)
    sparse_indices[1] = mixed

    # Token 2: all invalid edge case.
    sparse_indices[2] = -1
    return sparse_indices.contiguous()


def create_synthetic_data(
    *,
    num_tokens: int = TOKENS_IN_FIXTURE,
    num_pages: int = PAGES_IN_FIXTURE,
    num_qo_heads: int = NUM_QO_HEADS,
    head_dim_ckv: int = HEAD_DIM_CKV,
    head_dim_kpe: int = HEAD_DIM_KPE,
    page_size: int = PAGE_SIZE,
    topk: int = TOPK,
    device: Optional[torch.device] = None,
) -> SyntheticFixture:
    """Builds the deterministic S0 host fixture.

    The fixture deliberately includes:
    - token 0: normal all-valid sparse neighborhood
    - token 1: mixed valid and `-1` indices, plus repeated indices
    - token 2: all invalid sparse neighborhood
    """

    q_nope = _deterministic_bf16(
        (num_tokens, num_qo_heads, head_dim_ckv),
        modulus=29,
        center=14,
        scale=96.0,
        device=device,
    )
    q_pe = _deterministic_bf16(
        (num_tokens, num_qo_heads, head_dim_kpe),
        modulus=17,
        center=8,
        scale=64.0,
        device=device,
    )
    ckv_cache = _deterministic_bf16(
        (num_pages, page_size, head_dim_ckv),
        modulus=31,
        center=15,
        scale=128.0,
        device=device,
    )
    kpe_cache = _deterministic_bf16(
        (num_pages, page_size, head_dim_kpe),
        modulus=23,
        center=11,
        scale=96.0,
        device=device,
    )
    sparse_indices = _build_sparse_fixture(num_tokens, topk, num_pages * page_size, device=device)
    sm_scale = torch.tensor(
        1.0 / math.sqrt(head_dim_ckv + head_dim_kpe),
        dtype=torch.float32,
        device=device,
    )

    return SyntheticFixture(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        sm_scale=sm_scale,
    )


def _flatten_caches(ckv_cache: torch.Tensor, kpe_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_ckv = ckv_cache.reshape(-1, ckv_cache.shape[-1]).contiguous()
    flat_kpe = kpe_cache.reshape(-1, kpe_cache.shape[-1]).contiguous()
    return flat_ckv, flat_kpe


def torch_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Vectorized PyTorch reference for the stage-0 golden fixture."""

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    device = q_nope.device
    sm_scale_f32 = float(sm_scale.item()) if isinstance(sm_scale, torch.Tensor) else float(sm_scale)

    flat_ckv, flat_kpe = _flatten_caches(ckv_cache, kpe_cache)

    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    output_f32 = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse_f32 = torch.full((num_tokens, num_qo_heads), float("-inf"), dtype=torch.float32, device=device)

    for token_idx in range(num_tokens):
        valid_mask = sparse_indices[token_idx] >= 0
        if not torch.any(valid_mask):
            continue

        token_sparse = sparse_indices[token_idx, valid_mask].to(torch.long)
        kc = flat_ckv.index_select(0, token_sparse).to(torch.float32)
        kp = flat_kpe.index_select(0, token_sparse).to(torch.float32)

        logits = q_nope_f32[token_idx] @ kc.transpose(0, 1)
        logits = logits + q_pe_f32[token_idx] @ kp.transpose(0, 1)
        scaled = logits * sm_scale_f32

        probs = torch.softmax(scaled, dim=-1)
        output_f32[token_idx] = probs @ kc
        lse_f32[token_idx] = torch.logsumexp(scaled, dim=-1) / math.log(2.0)

    return {
        "ref_output": output_f32.to(torch.bfloat16).contiguous(),
        "ref_lse": lse_f32.contiguous(),
        "fixture_sparse_indices": sparse_indices.to(torch.int32).contiguous(),
    }


def torch_reference_naive(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
) -> Dict[str, torch.Tensor]:
    """Naive PyTorch baseline used by the S0 prefix validation harness."""

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    device = q_nope.device
    output_f32 = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse_f32 = torch.full((num_tokens, num_qo_heads), float("-inf"), dtype=torch.float32, device=device)

    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)
    ckv_f32 = ckv_cache.to(torch.float32)
    kpe_f32 = kpe_cache.to(torch.float32)
    sm_scale_f32 = float(sm_scale.item()) if isinstance(sm_scale, torch.Tensor) else float(sm_scale)

    for token_idx in range(num_tokens):
        for head_idx in range(num_qo_heads):
            logits = []
            gathered_values = []

            qn = q_nope_f32[token_idx, head_idx]
            qp = q_pe_f32[token_idx, head_idx]

            for sparse_col in range(sparse_indices.shape[1]):
                flat_tok_idx = int(sparse_indices[token_idx, sparse_col].item())
                if flat_tok_idx < 0:
                    continue

                page_idx = flat_tok_idx >> PAGE_BITS
                slot_idx = flat_tok_idx & (page_size - 1)

                kc = ckv_f32[page_idx, slot_idx]
                kp = kpe_f32[page_idx, slot_idx]
                logit = torch.dot(qn, kc) + torch.dot(qp, kp)
                logits.append(logit)
                gathered_values.append(kc)

            if not logits:
                continue

            logits_tensor = torch.stack(logits)
            scaled = logits_tensor * sm_scale_f32
            row_max = torch.max(scaled)
            numerators = torch.exp(scaled - row_max)
            row_sum = torch.sum(numerators)
            probs = numerators / row_sum

            row_out = torch.zeros((head_dim_ckv,), dtype=torch.float32, device=device)
            for prob, value in zip(probs, gathered_values):
                row_out = row_out + prob * value

            output_f32[token_idx, head_idx] = row_out
            lse_f32[token_idx, head_idx] = (row_max + torch.log(row_sum)) / math.log(2.0)

    return {
        "ref_output": output_f32.to(torch.bfloat16).contiguous(),
        "ref_lse": lse_f32.contiguous(),
        "fixture_sparse_indices": sparse_indices.to(torch.int32).contiguous(),
    }


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    finite_mask = torch.isfinite(tensor) if tensor.dtype.is_floating_point else None
    if finite_mask is None or not torch.any(finite_mask):
        return f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}"

    finite_values = tensor[finite_mask]
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={finite_values.min().item():.6f}, max={finite_values.max().item():.6f}"
    )


def _format_int_tensor(tensor: torch.Tensor) -> str:
    return str(tensor.detach().cpu().tolist())


def _stage0_prefix_validation_harness(
    vectorized_result: Dict[str, torch.Tensor],
    naive_result: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compares the stage-0 host outputs against an independent naive baseline."""

    exact_match = torch.equal(
        vectorized_result["fixture_sparse_indices"], naive_result["fixture_sparse_indices"]
    )
    output_match = torch.allclose(
        vectorized_result["ref_output"].to(torch.float32),
        naive_result["ref_output"].to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    lse_match = torch.allclose(
        vectorized_result["ref_lse"],
        naive_result["ref_lse"],
        rtol=1e-4,
        atol=1e-4,
    )

    num_tokens = vectorized_result["fixture_sparse_indices"].shape[0]
    has_expected_fixture_tokens = num_tokens == TOKENS_IN_FIXTURE
    token2_zero = False
    token2_neg_inf = False
    token1_has_repeat = False
    token1_has_valid = False
    token1_has_invalid = False

    if has_expected_fixture_tokens:
        token2_zero = torch.equal(
            vectorized_result["ref_output"][2],
            torch.zeros_like(vectorized_result["ref_output"][2]),
        )
        token2_neg_inf = torch.equal(
            vectorized_result["ref_lse"][2],
            torch.full_like(vectorized_result["ref_lse"][2], float("-inf")),
        )
        token1_has_repeat = bool(
            torch.any(
                vectorized_result["fixture_sparse_indices"][1, 1:]
                == vectorized_result["fixture_sparse_indices"][1, :-1]
            ).item()
        )
        token1_has_valid = bool(
            torch.any(vectorized_result["fixture_sparse_indices"][1] >= 0).item()
        )
        token1_has_invalid = bool(
            torch.any(vectorized_result["fixture_sparse_indices"][1] < 0).item()
        )

    matched = all(
        [
            exact_match,
            output_match,
            lse_match,
            has_expected_fixture_tokens,
            token2_zero,
            token2_neg_inf,
            token1_has_valid,
            token1_has_invalid,
            token1_has_repeat,
        ]
    )

    report_lines = [
        "Stage S0 prefix validation:",
        f"  fixture_sparse_indices exact match: {exact_match}",
        f"  ref_output allclose: {output_match}",
        f"  ref_lse allclose: {lse_match}",
        f"  deterministic fixture token count is {TOKENS_IN_FIXTURE}: {has_expected_fixture_tokens} (observed={num_tokens})",
        f"  all-invalid token output exactly zero: {token2_zero} (checked={has_expected_fixture_tokens})",
        f"  all-invalid token lse exactly -inf: {token2_neg_inf} (checked={has_expected_fixture_tokens})",
        f"  mixed token contains at least one valid sparse index: {token1_has_valid} (checked={has_expected_fixture_tokens})",
        f"  mixed token contains at least one invalid sparse index: {token1_has_invalid} (checked={has_expected_fixture_tokens})",
        f"  mixed token contains repeated sparse indices: {token1_has_repeat} (checked={has_expected_fixture_tokens})",
        f"  {_summarize_tensor('ref_output', vectorized_result['ref_output'].to(torch.float32))}",
        f"  {_summarize_tensor('ref_lse', vectorized_result['ref_lse'])}",
    ]

    return {
        "matched": matched,
        "report": "\n".join(report_lines),
        "stage_outputs": {
            "ref_output": vectorized_result["ref_output"],
            "ref_lse": vectorized_result["ref_lse"],
            "fixture_sparse_indices": vectorized_result["fixture_sparse_indices"],
        },
    }


def torch_reference_s1_scheduler(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
    *,
    max_active_clusters: int = MAX_ACTIVE_CLUSTERS,
) -> Dict[str, torch.Tensor]:
    """Vectorized host mirror of the S1 persistent token scheduler."""

    del q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
    num_tokens = q_nope.shape[0]
    device = q_nope.device
    grid_shape_tuple = BlackwellStyleKernel._compute_grid(
        num_tokens,
        max_active_clusters=max_active_clusters,
    )
    num_launched_ctas = grid_shape_tuple[2]
    num_scheduler_iters = BlackwellStyleKernel._compute_num_scheduler_iters(
        num_tokens,
        num_launched_ctas,
    )

    grid_shape = torch.tensor(grid_shape_tuple, dtype=torch.int32, device=device)
    scheduler_iter_offsets = (
        torch.arange(num_scheduler_iters, dtype=torch.int32, device=device).reshape(1, num_scheduler_iters)
        * num_launched_ctas
    )
    cta_offsets = torch.arange(num_launched_ctas, dtype=torch.int32, device=device).reshape(
        num_launched_ctas, 1
    )
    debug_token_order = scheduler_iter_offsets + cta_offsets
    valid_mask = debug_token_order < num_tokens
    debug_token_order = torch.where(
        valid_mask,
        debug_token_order,
        torch.full_like(debug_token_order, -1),
    ).contiguous()

    debug_token_done_count = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    if num_tokens > 0:
        valid_tokens = debug_token_order[debug_token_order >= 0].to(torch.long)
        if valid_tokens.numel() > 0:
            debug_token_done_count.scatter_add_(
                0,
                valid_tokens,
                torch.ones_like(valid_tokens, dtype=torch.int32),
            )

    return {
        "grid_shape": grid_shape.contiguous(),
        "debug_token_order": debug_token_order,
        "debug_token_done_count": debug_token_done_count.contiguous(),
    }


def torch_reference_s1_scheduler_naive(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
    *,
    max_active_clusters: int = MAX_ACTIVE_CLUSTERS,
) -> Dict[str, torch.Tensor]:
    """Naive independent S1 scheduler reference with explicit per-CTA iteration."""

    del q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
    num_tokens = q_nope.shape[0]
    device = q_nope.device
    grid_shape_tuple = (1, 1, max(1, min(num_tokens, max_active_clusters)))
    num_launched_ctas = grid_shape_tuple[2]
    num_scheduler_iters = BlackwellStyleKernel._compute_num_scheduler_iters(
        num_tokens,
        num_launched_ctas,
    )

    grid_shape = torch.tensor(grid_shape_tuple, dtype=torch.int32, device=device)
    debug_token_order = torch.full(
        (num_launched_ctas, num_scheduler_iters),
        -1,
        dtype=torch.int32,
        device=device,
    )
    debug_token_done_count = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

    for cta_idx in range(num_launched_ctas):
        token_idx = cta_idx
        scheduler_iter = 0
        while scheduler_iter < num_scheduler_iters:
            if token_idx < num_tokens:
                debug_token_order[cta_idx, scheduler_iter] = token_idx
                debug_token_done_count[token_idx] += 1
            scheduler_iter += 1
            token_idx += num_launched_ctas

    return {
        "grid_shape": grid_shape.contiguous(),
        "debug_token_order": debug_token_order.contiguous(),
        "debug_token_done_count": debug_token_done_count.contiguous(),
    }


def _stage1_prefix_validation_harness(
    cute_result: Dict[str, torch.Tensor],
    naive_result: Dict[str, torch.Tensor],
    *,
    dependency_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Validates the S1 persistent token scheduler launch contract."""

    grid_shape_match = torch.equal(cute_result["grid_shape"], naive_result["grid_shape"])
    token_order_match = torch.equal(
        cute_result["debug_token_order"], naive_result["debug_token_order"]
    )
    token_done_match = torch.equal(
        cute_result["debug_token_done_count"],
        naive_result["debug_token_done_count"],
    )

    num_tokens = int(cute_result["debug_token_done_count"].numel())
    launched_ctas = int(cute_result["grid_shape"][2].item())
    expected_launched_ctas = max(1, min(num_tokens, MAX_ACTIVE_CLUSTERS))
    launched_ctas_match = launched_ctas == expected_launched_ctas

    num_scheduler_iters = int(cute_result["debug_token_order"].shape[1])
    expected_scheduler_iters = BlackwellStyleKernel._compute_num_scheduler_iters(
        num_tokens,
        launched_ctas,
    )
    scheduler_iter_match = num_scheduler_iters == expected_scheduler_iters

    expected_done_count = torch.ones_like(cute_result["debug_token_done_count"])
    scheduled_once = torch.equal(cute_result["debug_token_done_count"], expected_done_count)

    token_order_stride_ok = True
    if num_scheduler_iters > 1:
        token_order_cpu = cute_result["debug_token_order"].detach().cpu()
        for cta_idx in range(token_order_cpu.shape[0]):
            valid_entries = token_order_cpu[cta_idx][token_order_cpu[cta_idx] >= 0]
            if valid_entries.numel() <= 1:
                continue
            diffs = valid_entries[1:] - valid_entries[:-1]
            if not torch.equal(
                diffs,
                torch.full_like(diffs, launched_ctas),
            ):
                token_order_stride_ok = False
                break

    matched = all(
        [
            dependency_validation["matched"],
            grid_shape_match,
            token_order_match,
            token_done_match,
            launched_ctas_match,
            scheduler_iter_match,
            scheduled_once,
            token_order_stride_ok,
        ]
    )

    report_lines = [
        "Stage S1 prefix validation:",
        f"  dependency S0 validation preserved: {dependency_validation['matched']}",
        f"  grid_shape exact match: {grid_shape_match}",
        f"  debug_token_order exact match: {token_order_match}",
        f"  debug_token_done_count exact match: {token_done_match}",
        f"  launched CTA count equals min(T, 148): {launched_ctas_match} (launched={launched_ctas}, expected={expected_launched_ctas})",
        f"  scheduler iterations match ceil_div(T, launched_ctas): {scheduler_iter_match} (observed={num_scheduler_iters}, expected={expected_scheduler_iters})",
        f"  every token is scheduled exactly once: {scheduled_once}",
        f"  per-CTA token stride equals launched_ctas: {token_order_stride_ok}",
        f"  grid_shape={_format_int_tensor(cute_result['grid_shape'])}",
        f"  debug_token_order={_format_int_tensor(cute_result['debug_token_order'])}",
        f"  debug_token_done_count={_format_int_tensor(cute_result['debug_token_done_count'])}",
    ]

    return {
        "matched": matched,
        "report": "\n".join(report_lines),
        "stage_outputs": {
            "grid_shape": cute_result["grid_shape"],
            "debug_token_order": cute_result["debug_token_order"],
            "debug_token_done_count": cute_result["debug_token_done_count"],
        },
    }


def torch_reference_s2_mma_config(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Host mirror of the S2 Blackwell QK/PV MMA configuration."""

    del ckv_cache, kpe_cache, sparse_indices, sm_scale
    program = BlackwellStyleKernel(
        num_qo_heads=q_nope.shape[1],
        head_dim_ckv=q_nope.shape[2],
        head_dim_kpe=q_pe.shape[2],
    )
    return program.stage2_mma_debug_reference(device=q_nope.device)


def torch_reference_s2_mma_config_naive(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Independent arithmetic-only S2 reference for the MMA atom metadata."""

    del ckv_cache, kpe_cache, sparse_indices, sm_scale
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    device = q_nope.device
    qk_mma_shape = torch.tensor(QK_MMA_SHAPE, dtype=torch.int32, device=device).contiguous()
    pv_mma_shape = torch.tensor(PV_MMA_SHAPE, dtype=torch.int32, device=device).contiguous()
    q_mode = _enum_to_int(tcgen05.OperandMajorMode.K if tcgen05 is not None else "K")
    mn_mode = _enum_to_int(tcgen05.OperandMajorMode.MN if tcgen05 is not None else "MN")

    return {
        "qk_mma_shape": qk_mma_shape,
        "pv_mma_shape": pv_mma_shape,
        "qk_ckv_iters": torch.tensor(
            [head_dim_ckv // QK_MMA_SHAPE[2]], dtype=torch.int32, device=device
        ).contiguous(),
        "qk_kpe_iters": torch.tensor(
            [head_dim_kpe // QK_MMA_SHAPE[2]], dtype=torch.int32, device=device
        ).contiguous(),
        "pv_k_iters": torch.tensor(
            [SPARSE_TILE_N // PV_MMA_SHAPE[2]], dtype=torch.int32, device=device
        ).contiguous(),
        "pv_out_iters": torch.tensor(
            [head_dim_ckv // PV_MMA_SHAPE[1]], dtype=torch.int32, device=device
        ).contiguous(),
        "qk_a_major_mode": torch.tensor([q_mode], dtype=torch.int32, device=device).contiguous(),
        "qk_b_major_mode": torch.tensor([q_mode], dtype=torch.int32, device=device).contiguous(),
        "pv_a_major_mode": torch.tensor([q_mode], dtype=torch.int32, device=device).contiguous(),
        "pv_b_major_mode": torch.tensor([mn_mode], dtype=torch.int32, device=device).contiguous(),
        "qk_helper_constructed": torch.ones((1,), dtype=torch.int32, device=device).contiguous(),
        "pv_helper_constructed": torch.ones((1,), dtype=torch.int32, device=device).contiguous(),
        "qk_atom_persisted": torch.ones((1,), dtype=torch.int32, device=device).contiguous(),
        "pv_atom_persisted": torch.ones((1,), dtype=torch.int32, device=device).contiguous(),
    }


def _stage2_prefix_validation_harness(
    cute_result: Dict[str, torch.Tensor],
    naive_result: Dict[str, torch.Tensor],
    *,
    dependency_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Validates the S2 Blackwell trivial-MMA atom configuration."""

    qk_shape_match = torch.equal(cute_result["qk_mma_shape"], naive_result["qk_mma_shape"])
    pv_shape_match = torch.equal(cute_result["pv_mma_shape"], naive_result["pv_mma_shape"])
    qk_ckv_iters_match = torch.equal(
        cute_result["qk_ckv_iters"], naive_result["qk_ckv_iters"]
    )
    qk_kpe_iters_match = torch.equal(
        cute_result["qk_kpe_iters"], naive_result["qk_kpe_iters"]
    )
    pv_k_iters_match = torch.equal(cute_result["pv_k_iters"], naive_result["pv_k_iters"])
    pv_out_iters_match = torch.equal(
        cute_result["pv_out_iters"], naive_result["pv_out_iters"]
    )
    qk_a_major_mode_match = torch.equal(
        cute_result["qk_a_major_mode"], naive_result["qk_a_major_mode"]
    )
    qk_b_major_mode_match = torch.equal(
        cute_result["qk_b_major_mode"], naive_result["qk_b_major_mode"]
    )
    pv_a_major_mode_match = torch.equal(
        cute_result["pv_a_major_mode"], naive_result["pv_a_major_mode"]
    )
    pv_b_major_mode_match = torch.equal(
        cute_result["pv_b_major_mode"], naive_result["pv_b_major_mode"]
    )
    qk_helper_constructed_match = torch.equal(
        cute_result["qk_helper_constructed"], naive_result["qk_helper_constructed"]
    )
    pv_helper_constructed_match = torch.equal(
        cute_result["pv_helper_constructed"], naive_result["pv_helper_constructed"]
    )
    qk_atom_persisted_match = torch.equal(
        cute_result["qk_atom_persisted"], naive_result["qk_atom_persisted"]
    )
    pv_atom_persisted_match = torch.equal(
        cute_result["pv_atom_persisted"], naive_result["pv_atom_persisted"]
    )

    qk_shape = cute_result["qk_mma_shape"].detach().cpu()
    pv_shape = cute_result["pv_mma_shape"].detach().cpu()
    qk_shape_expected = tuple(qk_shape.tolist()) == QK_MMA_SHAPE
    pv_shape_expected = tuple(pv_shape.tolist()) == PV_MMA_SHAPE
    qk_ckv_iters_derived = int(cute_result["qk_ckv_iters"][0].item()) == HEAD_DIM_CKV // int(
        qk_shape[2].item()
    )
    qk_kpe_iters_derived = int(cute_result["qk_kpe_iters"][0].item()) == HEAD_DIM_KPE // int(
        qk_shape[2].item()
    )
    pv_k_iters_derived = int(cute_result["pv_k_iters"][0].item()) == SPARSE_TILE_N // int(
        pv_shape[2].item()
    )
    pv_out_iters_derived = int(cute_result["pv_out_iters"][0].item()) == HEAD_DIM_CKV // int(
        pv_shape[1].item()
    )
    qk_k_atom_bf16 = int(qk_shape[2].item()) == 16
    pv_k_atom_bf16 = int(pv_shape[2].item()) == 16

    matched = all(
        [
            dependency_validation["matched"],
            qk_shape_match,
            pv_shape_match,
            qk_ckv_iters_match,
            qk_kpe_iters_match,
            pv_k_iters_match,
            pv_out_iters_match,
            qk_a_major_mode_match,
            qk_b_major_mode_match,
            pv_a_major_mode_match,
            pv_b_major_mode_match,
            qk_helper_constructed_match,
            pv_helper_constructed_match,
            qk_atom_persisted_match,
            pv_atom_persisted_match,
            qk_shape_expected,
            pv_shape_expected,
            qk_ckv_iters_derived,
            qk_kpe_iters_derived,
            pv_k_iters_derived,
            pv_out_iters_derived,
            qk_k_atom_bf16,
            pv_k_atom_bf16,
        ]
    )

    report_lines = [
        "Stage S2 prefix validation:",
        f"  dependency S1 validation preserved: {dependency_validation['matched']}",
        f"  qk_mma_shape exact match: {qk_shape_match}",
        f"  pv_mma_shape exact match: {pv_shape_match}",
        f"  qk_ckv_iters exact match: {qk_ckv_iters_match}",
        f"  qk_kpe_iters exact match: {qk_kpe_iters_match}",
        f"  pv_k_iters exact match: {pv_k_iters_match}",
        f"  pv_out_iters exact match: {pv_out_iters_match}",
        f"  qk A major mode exact match (expected K-major): {qk_a_major_mode_match}",
        f"  qk B major mode exact match (expected K-major): {qk_b_major_mode_match}",
        f"  pv A major mode exact match (expected K-major): {pv_a_major_mode_match}",
        f"  pv B major mode exact match (expected MN-major): {pv_b_major_mode_match}",
        f"  qk helper atom constructed without fallback: {qk_helper_constructed_match}",
        f"  pv helper atom constructed without fallback: {pv_helper_constructed_match}",
        f"  qk tiled MMA object persisted on program for S3/S7 reuse: {qk_atom_persisted_match}",
        f"  pv tiled MMA object persisted on program for S3/S10 reuse: {pv_atom_persisted_match}",
        f"  qk_mma_shape equals planned (64, 32, 16): {qk_shape_expected}",
        f"  pv_mma_shape equals planned (64, 128, 16): {pv_shape_expected}",
        f"  qk_ckv_iters equals HEAD_DIM_CKV / K_atom: {qk_ckv_iters_derived}",
        f"  qk_kpe_iters equals HEAD_DIM_KPE / K_atom: {qk_kpe_iters_derived}",
        f"  pv_k_iters equals sparse_tile_n / K_atom: {pv_k_iters_derived}",
        f"  pv_out_iters equals HEAD_DIM_CKV / pv_n_tile: {pv_out_iters_derived}",
        f"  qk K atom uses BF16 UMMA K=16: {qk_k_atom_bf16}",
        f"  pv K atom uses BF16 UMMA K=16: {pv_k_atom_bf16}",
        f"  qk_mma_shape={_format_int_tensor(cute_result['qk_mma_shape'])}",
        f"  pv_mma_shape={_format_int_tensor(cute_result['pv_mma_shape'])}",
        f"  qk_ckv_iters={_format_int_tensor(cute_result['qk_ckv_iters'])}",
        f"  qk_kpe_iters={_format_int_tensor(cute_result['qk_kpe_iters'])}",
        f"  pv_k_iters={_format_int_tensor(cute_result['pv_k_iters'])}",
        f"  pv_out_iters={_format_int_tensor(cute_result['pv_out_iters'])}",
        f"  qk_a_major_mode={_format_int_tensor(cute_result['qk_a_major_mode'])}",
        f"  qk_b_major_mode={_format_int_tensor(cute_result['qk_b_major_mode'])}",
        f"  pv_a_major_mode={_format_int_tensor(cute_result['pv_a_major_mode'])}",
        f"  pv_b_major_mode={_format_int_tensor(cute_result['pv_b_major_mode'])}",
        f"  qk_helper_constructed={_format_int_tensor(cute_result['qk_helper_constructed'])}",
        f"  pv_helper_constructed={_format_int_tensor(cute_result['pv_helper_constructed'])}",
        f"  qk_atom_persisted={_format_int_tensor(cute_result['qk_atom_persisted'])}",
        f"  pv_atom_persisted={_format_int_tensor(cute_result['pv_atom_persisted'])}",
    ]

    return {
        "matched": matched,
        "report": "\n".join(report_lines),
        "stage_outputs": {
            "qk_mma_shape": cute_result["qk_mma_shape"],
            "pv_mma_shape": cute_result["pv_mma_shape"],
            "qk_ckv_iters": cute_result["qk_ckv_iters"],
            "qk_kpe_iters": cute_result["qk_kpe_iters"],
            "pv_k_iters": cute_result["pv_k_iters"],
            "pv_out_iters": cute_result["pv_out_iters"],
            "qk_a_major_mode": cute_result["qk_a_major_mode"],
            "qk_b_major_mode": cute_result["qk_b_major_mode"],
            "pv_a_major_mode": cute_result["pv_a_major_mode"],
            "pv_b_major_mode": cute_result["pv_b_major_mode"],
            "qk_helper_constructed": cute_result["qk_helper_constructed"],
            "pv_helper_constructed": cute_result["pv_helper_constructed"],
            "qk_atom_persisted": cute_result["qk_atom_persisted"],
            "pv_atom_persisted": cute_result["pv_atom_persisted"],
        },
    }


def torch_reference_s3_storage_config(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Host mirror of the S3 storage-layout / shared-storage configuration."""

    device = q_nope.device
    del q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
    return BlackwellStyleKernel().stage3_storage_host_reference(device=device)


def torch_reference_s3_storage_config_naive(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Independent arithmetic-only S3 reference for storage sizes and TMEM columns."""

    del q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
    device = q_nope.device

    padded_rows = QK_MMA_SHAPE[0]
    q_smem_bytes = padded_rows * (HEAD_DIM_CKV + HEAD_DIM_KPE) * BF16_BYTES
    qk_ring_bytes = QK_PIPE_STAGES * SPARSE_TILE_N * (HEAD_DIM_CKV + HEAD_DIM_KPE) * BF16_BYTES
    p_ring_bytes = P_PIPE_STAGES * padded_rows * SPARSE_TILE_N * BF16_BYTES
    v_ring_bytes = V_PIPE_STAGES * SPARSE_TILE_N * PV_OUT_TILE * BF16_BYTES

    barrier_bytes = (6 * (QK_PIPE_STAGES * 2) + (O_PIPE_STAGES * 2)) * INT64_BYTES
    sparse_meta_bytes = (
        QK_PIPE_STAGES * SPARSE_TILE_N * INT32_BYTES
        + QK_PIPE_STAGES * SPARSE_TILE_N * INT32_BYTES
        + QK_PIPE_STAGES * INT32_BYTES
    )
    corr_meta_bytes = (
        QK_PIPE_STAGES * NUM_QO_HEADS * FP32_BYTES * 3
        + QK_PIPE_STAGES * INT32_BYTES
    )
    pre_operand_bytes = barrier_bytes + INT32_BYTES + sparse_meta_bytes + corr_meta_bytes
    shared_storage_bytes = _align_up(pre_operand_bytes, 1024)
    shared_storage_bytes += q_smem_bytes
    shared_storage_bytes = _align_up(shared_storage_bytes, 1024)
    shared_storage_bytes += qk_ring_bytes
    shared_storage_bytes = _align_up(shared_storage_bytes, 1024)
    shared_storage_bytes += p_ring_bytes
    shared_storage_bytes = _align_up(shared_storage_bytes, 1024)
    shared_storage_bytes += v_ring_bytes

    tmem_s_bytes = padded_rows * SPARSE_TILE_N * QK_PIPE_STAGES * FP32_BYTES
    tmem_o_bytes = padded_rows * HEAD_DIM_CKV * FP32_BYTES
    tmem_logical_cols = (tmem_s_bytes + tmem_o_bytes) // TMEM_COLUMN_BYTES_SM100
    tmem_alloc_cols = max(32, 1 << (tmem_logical_cols - 1).bit_length())

    return {
        "shared_storage_bytes": torch.tensor(
            [shared_storage_bytes], dtype=torch.int32, device=device
        ).contiguous(),
        "tmem_alloc_cols": torch.tensor(
            [tmem_alloc_cols], dtype=torch.int32, device=device
        ).contiguous(),
        "q_smem_bytes": torch.tensor([q_smem_bytes], dtype=torch.int32, device=device).contiguous(),
        "qk_ring_bytes": torch.tensor(
            [qk_ring_bytes], dtype=torch.int32, device=device
        ).contiguous(),
        "p_ring_bytes": torch.tensor([p_ring_bytes], dtype=torch.int32, device=device).contiguous(),
        "v_ring_bytes": torch.tensor([v_ring_bytes], dtype=torch.int32, device=device).contiguous(),
    }


def torch_reference_s4_q_prologue(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Vectorized host mirror of the S4 Q-prologue load and 16->64 row padding."""

    del ckv_cache, kpe_cache, sparse_indices, sm_scale
    device = q_nope.device
    debug_q_ckv_padded = torch.zeros(
        (PADDED_Q_HEAD_ROWS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
        device=device,
    )
    debug_q_kpe_padded = torch.zeros(
        (PADDED_Q_HEAD_ROWS, HEAD_DIM_KPE),
        dtype=torch.bfloat16,
        device=device,
    )

    if q_nope.shape[0] > 0:
        debug_q_ckv_padded[: q_nope.shape[1]].copy_(q_nope[0])
        debug_q_kpe_padded[: q_pe.shape[1]].copy_(q_pe[0])

    return {
        "debug_q_ckv_padded": debug_q_ckv_padded.contiguous(),
        "debug_q_kpe_padded": debug_q_kpe_padded.contiguous(),
    }


def torch_reference_s4_q_prologue_naive(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Independent naive S4 reference with explicit row-padding loops."""

    del ckv_cache, kpe_cache, sparse_indices, sm_scale
    device = q_nope.device
    debug_q_ckv_padded = torch.zeros(
        (PADDED_Q_HEAD_ROWS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
        device=device,
    )
    debug_q_kpe_padded = torch.zeros(
        (PADDED_Q_HEAD_ROWS, HEAD_DIM_KPE),
        dtype=torch.bfloat16,
        device=device,
    )

    if q_nope.shape[0] > 0:
        for row_idx in range(PADDED_Q_HEAD_ROWS):
            if row_idx < q_nope.shape[1]:
                for col_idx in range(HEAD_DIM_CKV):
                    debug_q_ckv_padded[row_idx, col_idx] = q_nope[0, row_idx, col_idx]
                for col_idx in range(HEAD_DIM_KPE):
                    debug_q_kpe_padded[row_idx, col_idx] = q_pe[0, row_idx, col_idx]
            else:
                for col_idx in range(HEAD_DIM_CKV):
                    debug_q_ckv_padded[row_idx, col_idx] = 0
                for col_idx in range(HEAD_DIM_KPE):
                    debug_q_kpe_padded[row_idx, col_idx] = 0

    return {
        "debug_q_ckv_padded": debug_q_ckv_padded.contiguous(),
        "debug_q_kpe_padded": debug_q_kpe_padded.contiguous(),
    }


def _stage3_prefix_validation_harness(
    cute_result: Dict[str, torch.Tensor],
    naive_result: Dict[str, torch.Tensor],
    *,
    dependency_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Validates the S3 SMEM/TMEM layout budget and SharedStorage contract."""

    shared_storage_match = torch.equal(
        cute_result["shared_storage_bytes"], naive_result["shared_storage_bytes"]
    )
    tmem_alloc_cols_match = torch.equal(
        cute_result["tmem_alloc_cols"], naive_result["tmem_alloc_cols"]
    )
    q_smem_bytes_match = torch.equal(cute_result["q_smem_bytes"], naive_result["q_smem_bytes"])
    qk_ring_bytes_match = torch.equal(
        cute_result["qk_ring_bytes"], naive_result["qk_ring_bytes"]
    )
    p_ring_bytes_match = torch.equal(cute_result["p_ring_bytes"], naive_result["p_ring_bytes"])
    v_ring_bytes_match = torch.equal(cute_result["v_ring_bytes"], naive_result["v_ring_bytes"])

    shared_storage_expected = int(cute_result["shared_storage_bytes"][0].item()) == S3_SHARED_STORAGE_BYTES
    tmem_cols_expected = int(cute_result["tmem_alloc_cols"][0].item()) == S3_TMEM_ALLOC_COLS
    q_budget_expected = int(cute_result["q_smem_bytes"][0].item()) == S3_Q_SMEM_BYTES
    qk_budget_expected = int(cute_result["qk_ring_bytes"][0].item()) == S3_QK_RING_BYTES
    p_budget_expected = int(cute_result["p_ring_bytes"][0].item()) == S3_P_RING_BYTES
    v_budget_expected = int(cute_result["v_ring_bytes"][0].item()) == S3_V_RING_BYTES
    shared_storage_within_budget = int(cute_result["shared_storage_bytes"][0].item()) <= (228 - 1) * 1024
    tmem_cols_power_of_two = int(cute_result["tmem_alloc_cols"][0].item()) & (
        int(cute_result["tmem_alloc_cols"][0].item()) - 1
    ) == 0
    tmem_cols_multiple_of_32 = int(cute_result["tmem_alloc_cols"][0].item()) % 32 == 0

    matched = all(
        [
            dependency_validation["matched"],
            shared_storage_match,
            tmem_alloc_cols_match,
            q_smem_bytes_match,
            qk_ring_bytes_match,
            p_ring_bytes_match,
            v_ring_bytes_match,
            shared_storage_expected,
            tmem_cols_expected,
            q_budget_expected,
            qk_budget_expected,
            p_budget_expected,
            v_budget_expected,
            shared_storage_within_budget,
            tmem_cols_power_of_two,
            tmem_cols_multiple_of_32,
        ]
    )

    report_lines = [
        "Stage S3 prefix validation:",
        f"  dependency S2 validation preserved: {dependency_validation['matched']}",
        f"  shared_storage_bytes exact match: {shared_storage_match}",
        f"  tmem_alloc_cols exact match: {tmem_alloc_cols_match}",
        f"  q_smem_bytes exact match: {q_smem_bytes_match}",
        f"  qk_ring_bytes exact match: {qk_ring_bytes_match}",
        f"  p_ring_bytes exact match: {p_ring_bytes_match}",
        f"  v_ring_bytes exact match: {v_ring_bytes_match}",
        f"  shared_storage_bytes equals Stage3SharedStorage.size_in_bytes(): {shared_storage_expected}",
        f"  tmem_alloc_cols equals planned rounded allocation 512: {tmem_cols_expected}",
        f"  q_smem_bytes equals padded Q budget (64x512 + 64x64 bf16): {q_budget_expected}",
        f"  qk_ring_bytes equals double-buffered Kc/Kpe ring budget: {qk_budget_expected}",
        f"  p_ring_bytes equals double-buffered P ring budget: {p_budget_expected}",
        f"  v_ring_bytes equals double-buffered V-slice ring budget: {v_budget_expected}",
        f"  shared_storage_bytes fits within SM100 dynamic SMEM budget: {shared_storage_within_budget}",
        f"  tmem_alloc_cols is a power of two: {tmem_cols_power_of_two}",
        f"  tmem_alloc_cols is a multiple of 32: {tmem_cols_multiple_of_32}",
        f"  shared_storage_bytes={_format_int_tensor(cute_result['shared_storage_bytes'])}",
        f"  tmem_alloc_cols={_format_int_tensor(cute_result['tmem_alloc_cols'])}",
        f"  q_smem_bytes={_format_int_tensor(cute_result['q_smem_bytes'])}",
        f"  qk_ring_bytes={_format_int_tensor(cute_result['qk_ring_bytes'])}",
        f"  p_ring_bytes={_format_int_tensor(cute_result['p_ring_bytes'])}",
        f"  v_ring_bytes={_format_int_tensor(cute_result['v_ring_bytes'])}",
    ]

    return {
        "matched": matched,
        "report": "\n".join(report_lines),
        "stage_outputs": {
            "shared_storage_bytes": cute_result["shared_storage_bytes"],
            "tmem_alloc_cols": cute_result["tmem_alloc_cols"],
            "q_smem_bytes": cute_result["q_smem_bytes"],
            "qk_ring_bytes": cute_result["qk_ring_bytes"],
            "p_ring_bytes": cute_result["p_ring_bytes"],
            "v_ring_bytes": cute_result["v_ring_bytes"],
        },
    }


def _stage4_prefix_validation_harness(
    cute_result: Dict[str, torch.Tensor],
    naive_result: Dict[str, torch.Tensor],
    *,
    dependency_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Validates the S4 Q-prologue load and 16->64 row-padding contract."""

    q_ckv_match = torch.equal(
        cute_result["debug_q_ckv_padded"],
        naive_result["debug_q_ckv_padded"],
    )
    q_kpe_match = torch.equal(
        cute_result["debug_q_kpe_padded"],
        naive_result["debug_q_kpe_padded"],
    )
    q_ckv_shape_match = tuple(cute_result["debug_q_ckv_padded"].shape) == (
        PADDED_Q_HEAD_ROWS,
        HEAD_DIM_CKV,
    )
    q_kpe_shape_match = tuple(cute_result["debug_q_kpe_padded"].shape) == (
        PADDED_Q_HEAD_ROWS,
        HEAD_DIM_KPE,
    )
    q_ckv_dtype_match = cute_result["debug_q_ckv_padded"].dtype == torch.bfloat16
    q_kpe_dtype_match = cute_result["debug_q_kpe_padded"].dtype == torch.bfloat16
    q_ckv_padded_zero = torch.equal(
        cute_result["debug_q_ckv_padded"][NUM_QO_HEADS:],
        torch.zeros_like(cute_result["debug_q_ckv_padded"][NUM_QO_HEADS:]),
    )
    q_kpe_padded_zero = torch.equal(
        cute_result["debug_q_kpe_padded"][NUM_QO_HEADS:],
        torch.zeros_like(cute_result["debug_q_kpe_padded"][NUM_QO_HEADS:]),
    )
    q_ckv_live_rows_nonzero = bool(
        torch.count_nonzero(cute_result["debug_q_ckv_padded"][:NUM_QO_HEADS]).item() > 0
    )
    q_kpe_live_rows_nonzero = bool(
        torch.count_nonzero(cute_result["debug_q_kpe_padded"][:NUM_QO_HEADS]).item() > 0
    )

    matched = all(
        [
            dependency_validation["matched"],
            q_ckv_match,
            q_kpe_match,
            q_ckv_shape_match,
            q_kpe_shape_match,
            q_ckv_dtype_match,
            q_kpe_dtype_match,
            q_ckv_padded_zero,
            q_kpe_padded_zero,
            q_ckv_live_rows_nonzero,
            q_kpe_live_rows_nonzero,
        ]
    )

    report_lines = [
        "Stage S4 prefix validation:",
        f"  dependency S3 validation preserved: {dependency_validation['matched']}",
        f"  debug_q_ckv_padded exact match: {q_ckv_match}",
        f"  debug_q_kpe_padded exact match: {q_kpe_match}",
        f"  debug_q_ckv_padded shape is (64, 512): {q_ckv_shape_match}",
        f"  debug_q_kpe_padded shape is (64, 64): {q_kpe_shape_match}",
        f"  debug_q_ckv_padded dtype is bf16: {q_ckv_dtype_match}",
        f"  debug_q_kpe_padded dtype is bf16: {q_kpe_dtype_match}",
        f"  padded q_ckv rows 16..63 are exactly zero: {q_ckv_padded_zero}",
        f"  padded q_kpe rows 16..63 are exactly zero: {q_kpe_padded_zero}",
        f"  live q_ckv rows 0..15 contain non-zero signal: {q_ckv_live_rows_nonzero}",
        f"  live q_kpe rows 0..15 contain non-zero signal: {q_kpe_live_rows_nonzero}",
        f"  {_summarize_tensor('debug_q_ckv_padded', cute_result['debug_q_ckv_padded'].to(torch.float32))}",
        f"  {_summarize_tensor('debug_q_kpe_padded', cute_result['debug_q_kpe_padded'].to(torch.float32))}",
    ]

    return {
        "matched": matched,
        "report": "\n".join(report_lines),
        "stage_outputs": {
            "debug_q_ckv_padded": cute_result["debug_q_ckv_padded"],
            "debug_q_kpe_padded": cute_result["debug_q_kpe_padded"],
        },
    }


def prefix_validation_harness(
    vectorized_result: Dict[str, torch.Tensor],
    naive_result: Dict[str, torch.Tensor],
    *,
    stage_id: str = "S0",
    dependency_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if stage_id == "S0":
        return _stage0_prefix_validation_harness(vectorized_result, naive_result)
    if stage_id == "S1":
        if dependency_validation is None:
            raise ValueError("S1 validation requires dependency_validation from S0")
        return _stage1_prefix_validation_harness(
            vectorized_result,
            naive_result,
            dependency_validation=dependency_validation,
        )
    if stage_id == "S2":
        if dependency_validation is None:
            raise ValueError("S2 validation requires dependency_validation from S1")
        return _stage2_prefix_validation_harness(
            vectorized_result,
            naive_result,
            dependency_validation=dependency_validation,
        )
    if stage_id == "S3":
        if dependency_validation is None:
            raise ValueError("S3 validation requires dependency_validation from S2")
        return _stage3_prefix_validation_harness(
            vectorized_result,
            naive_result,
            dependency_validation=dependency_validation,
        )
    if stage_id == "S4":
        if dependency_validation is None:
            raise ValueError("S4 validation requires dependency_validation from S3")
        return _stage4_prefix_validation_harness(
            vectorized_result,
            naive_result,
            dependency_validation=dependency_validation,
        )
    raise ValueError(f"Unsupported stage_id for prefix validation: {stage_id}")


def _launch_s1_scheduler_debug_kernel(
    program: BlackwellStyleKernel,
    num_tokens: int,
    *,
    device: torch.device,
    q_nope: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if device.type != "cuda":
        raise RuntimeError(
            f"Stage S1 requires CUDA tensors so the scheduler artifacts come from the CuTe device path, got device={device}"
        )
    if cuda is None:
        raise RuntimeError("Stage S1 requires cuda.bindings.driver for CuTe kernel launch")
    if StaticPersistentTileScheduler is None:
        scheduler_import_detail = ""
        if _STATIC_SCHEDULER_IMPORT_ERROR is not None:
            scheduler_import_detail = (
                f" import_error={type(_STATIC_SCHEDULER_IMPORT_ERROR).__name__}:"
                f" {_STATIC_SCHEDULER_IMPORT_ERROR}"
            )
        raise RuntimeError(
            "Stage S1 requires StaticPersistentTileScheduler support before cute.compile()."
            f"{scheduler_import_detail}"
        )

    grid_shape_tuple = program._compute_grid(
        num_tokens,
        max_active_clusters=program.max_active_clusters,
    )
    program._host_launch_grid = grid_shape_tuple
    program._require_host_launch_grid()
    num_launched_ctas = grid_shape_tuple[2]
    num_scheduler_iters = BlackwellStyleKernel._compute_num_scheduler_iters(
        num_tokens,
        num_launched_ctas,
    )

    if q_nope is None:
        q_nope = torch.zeros(
            (max(1, num_tokens), NUM_QO_HEADS, HEAD_DIM_CKV),
            dtype=torch.bfloat16,
            device=device,
        )
    if q_pe is None:
        q_pe = torch.zeros(
            (max(1, num_tokens), NUM_QO_HEADS, HEAD_DIM_KPE),
            dtype=torch.bfloat16,
            device=device,
        )
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    if q_nope.device != device or q_pe.device != device:
        raise RuntimeError(
            f"Stage S1/S4 launch tensors must live on device={device}, got q_nope.device={q_nope.device}, q_pe.device={q_pe.device}"
        )

    debug_token_order = torch.full(
        (num_launched_ctas, num_scheduler_iters),
        -1,
        dtype=torch.int32,
        device=device,
    )
    debug_token_done_count = torch.zeros(
        (num_tokens,),
        dtype=torch.int32,
        device=device,
    )
    debug_stage2_mma_meta = torch.full(
        (18,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    debug_stage3_storage_meta = torch.full(
        (6,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    debug_q_ckv_padded = torch.zeros(
        (PADDED_Q_HEAD_ROWS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
        device=device,
    )
    debug_q_kpe_padded = torch.zeros(
        (PADDED_Q_HEAD_ROWS, HEAD_DIM_KPE),
        dtype=torch.bfloat16,
        device=device,
    )
    grid_shape = torch.tensor(
        grid_shape_tuple,
        dtype=torch.int32,
        device=device,
    ).contiguous()

    torch_stream = torch.cuda.current_stream(device=device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compile_stream = cute.runtime.make_fake_stream()

    # CuTe JIT signatures annotated with cute.Tensor expect CuTe runtime tensor
    # objects at both compile time and launch time. Keep the underlying storage in
    # the PyTorch debug buffers, but pass DLPack-backed CuTe tensor views into the
    # DSL entry so S1 produces the required GMEM scheduler artifacts on-device.
    q_nope_cute = cute.runtime.from_dlpack(q_nope).mark_layout_dynamic(
        leading_dim=q_nope.ndim - 1
    )
    q_pe_cute = cute.runtime.from_dlpack(q_pe).mark_layout_dynamic(
        leading_dim=q_pe.ndim - 1
    )
    debug_token_order_cute = cute.runtime.from_dlpack(
        debug_token_order
    ).mark_layout_dynamic(leading_dim=debug_token_order.ndim - 1)
    debug_token_done_count_cute = (
        cute.runtime.from_dlpack(debug_token_done_count).mark_layout_dynamic(
            leading_dim=debug_token_done_count.ndim - 1
        )
    )
    debug_stage2_mma_meta_cute = (
        cute.runtime.from_dlpack(debug_stage2_mma_meta).mark_layout_dynamic(
            leading_dim=debug_stage2_mma_meta.ndim - 1
        )
    )
    debug_stage3_storage_meta_cute = (
        cute.runtime.from_dlpack(debug_stage3_storage_meta).mark_layout_dynamic(
            leading_dim=debug_stage3_storage_meta.ndim - 1
        )
    )
    debug_q_ckv_padded_cute = (
        cute.runtime.from_dlpack(debug_q_ckv_padded).mark_layout_dynamic(
            leading_dim=debug_q_ckv_padded.ndim - 1
        )
    )
    debug_q_kpe_padded_cute = (
        cute.runtime.from_dlpack(debug_q_kpe_padded).mark_layout_dynamic(
            leading_dim=debug_q_kpe_padded.ndim - 1
        )
    )

    cache_key = (
        "S1_scheduler_debug",
        id(program),
        num_tokens,
        tuple(q_nope.shape),
        tuple(q_pe.shape),
        tuple(debug_token_order.shape),
        tuple(debug_token_done_count.shape),
        tuple(debug_stage2_mma_meta.shape),
        tuple(debug_stage3_storage_meta.shape),
        tuple(debug_q_ckv_padded.shape),
        tuple(debug_q_kpe_padded.shape),
        str(device),
        grid_shape_tuple,
        program.cluster_shape_mnk,
        program.max_active_clusters,
        program.swizzle_size,
        program.raster_along_m,
        program.threads_per_cta,
        program.enable_stage2_mma_probe,
        program.enable_stage3_storage_probe,
        program.enable_stage4_q_prologue_probe,
    )
    if cache_key not in program_compile_cache:
        program_compile_cache[cache_key] = cute.compile(
            program,
            q_nope_cute,
            q_pe_cute,
            debug_token_order_cute,
            debug_token_done_count_cute,
            debug_stage2_mma_meta_cute,
            debug_stage3_storage_meta_cute,
            debug_q_ckv_padded_cute,
            debug_q_kpe_padded_cute,
            num_tokens,
            compile_stream,
        )

    compiled_program = program_compile_cache[cache_key]
    compiled_program(
        q_nope_cute,
        q_pe_cute,
        debug_token_order_cute,
        debug_token_done_count_cute,
        debug_stage2_mma_meta_cute,
        debug_stage3_storage_meta_cute,
        debug_q_ckv_padded_cute,
        debug_q_kpe_padded_cute,
        num_tokens,
        stream,
    )
    torch.cuda.synchronize(device=device)

    return {
        "grid_shape": grid_shape,
        "debug_token_order": debug_token_order.contiguous(),
        "debug_token_done_count": debug_token_done_count.contiguous(),
        "debug_stage2_mma_meta": debug_stage2_mma_meta.contiguous(),
        "debug_stage3_storage_meta": debug_stage3_storage_meta.contiguous(),
        "debug_q_ckv_padded": debug_q_ckv_padded.contiguous(),
        "debug_q_kpe_padded": debug_q_kpe_padded.contiguous(),
    }


def run(*args, stage_id: str = "S4") -> Dict[str, Any]:
    """Computes the staged round-0 prefix outputs for the canonical fixture.

    Accepted call patterns:
      - run()
      - run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
      - run(..., stage_id="S0", "S1", "S2", "S3", or "S4")

    The 6-tensor form is accepted only to match the generic stage-runner call
    signature. The staged prefix harness still always rebuilds and validates the
    canonical deterministic 3-token fixture required by the architecture plan.
    """

    if stage_id not in {"S0", "S1", "S2", "S3", "S4"}:
        raise ValueError(f"Unsupported stage_id for this module: {stage_id}")

    if len(args) == 1 and isinstance(args[0], str):
        stage_id = args[0]
        if stage_id not in {"S0", "S1", "S2", "S3", "S4"}:
            raise ValueError(f"Unsupported stage_id for this module: {stage_id}")
        args = ()

    fixture_device = None
    if len(args) == 0:
        if stage_id in {"S1", "S2", "S3", "S4"} and torch.cuda.is_available():
            fixture_device = torch.device("cuda")
    elif len(args) == 6:
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale = args
        if not all(isinstance(arg, torch.Tensor) for arg in args):
            raise TypeError(
                "run() expects tensor arguments when 6 positional arguments are provided: "
                "(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)"
            )
        # Preserve the S0 contract: always rebuild the canonical deterministic
        # fixture, but mirror the runner's device placement when tensors are
        # provided so the returned references stay on the expected device.
        fixture_device = q_nope.device
    else:
        raise TypeError(
            "run() expects either no positional arguments or exactly 6 positional arguments: "
            "(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)"
        )

    fixture = create_synthetic_data(device=fixture_device)
    q_nope = fixture.q_nope
    q_pe = fixture.q_pe
    ckv_cache = fixture.ckv_cache
    kpe_cache = fixture.kpe_cache
    sparse_indices = fixture.sparse_indices
    sm_scale = fixture.sm_scale

    vectorized_result = torch_reference(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    naive_result = torch_reference_naive(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    s0_validation = prefix_validation_harness(
        vectorized_result,
        naive_result,
        stage_id="S0",
    )

    if stage_id == "S0":
        _ = BlackwellStyleKernel()
        return {
            "stage_id": stage_id,
            "matched": s0_validation["matched"],
            "report": s0_validation["report"],
            "ref_output": s0_validation["stage_outputs"]["ref_output"],
            "ref_lse": s0_validation["stage_outputs"]["ref_lse"],
            "fixture_sparse_indices": s0_validation["stage_outputs"]["fixture_sparse_indices"],
            "stage_outputs": s0_validation["stage_outputs"],
        }

    program = BlackwellStyleKernel()
    program.enable_stage2_mma_probe = stage_id in {"S2", "S3", "S4"}
    program.enable_stage3_storage_probe = stage_id in {"S3", "S4"}
    program.enable_stage4_q_prologue_probe = stage_id == "S4"
    cute_scheduler_result = _launch_s1_scheduler_debug_kernel(
        program,
        q_nope.shape[0],
        device=q_nope.device,
        q_nope=q_nope,
        q_pe=q_pe,
    )
    if program.enable_stage3_storage_probe:
        program._require_stage3_storage_artifacts()
        program._stage3_storage_cache = program._stage3_storage_result_from_meta(
            cute_scheduler_result["debug_stage3_storage_meta"].to(
                dtype=torch.int32, device=q_nope.device
            )
        )
    cute_scheduler_host_mirror = program.scheduler_debug_reference(
        q_nope.shape[0],
        device=q_nope.device,
    )
    naive_scheduler_result = torch_reference_s1_scheduler_naive(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    vectorized_scheduler_result = torch_reference_s1_scheduler(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    if not torch.equal(
        cute_scheduler_result["grid_shape"], vectorized_scheduler_result["grid_shape"]
    ) or not torch.equal(
        cute_scheduler_result["debug_token_order"],
        vectorized_scheduler_result["debug_token_order"],
    ) or not torch.equal(
        cute_scheduler_result["debug_token_done_count"],
        vectorized_scheduler_result["debug_token_done_count"],
    ):
        raise AssertionError("S1 device scheduler outputs diverged from the vectorized schedule mirror")
    if not torch.equal(
        cute_scheduler_result["grid_shape"], cute_scheduler_host_mirror["grid_shape"]
    ) or not torch.equal(
        cute_scheduler_result["debug_token_order"],
        cute_scheduler_host_mirror["debug_token_order"],
    ) or not torch.equal(
        cute_scheduler_result["debug_token_done_count"],
        cute_scheduler_host_mirror["debug_token_done_count"],
    ):
        raise AssertionError("S1 device scheduler outputs diverged from the host scheduler mirror")

    validation = prefix_validation_harness(
        cute_scheduler_result,
        naive_scheduler_result,
        stage_id="S1",
        dependency_validation=s0_validation,
    )
    s1_stage_outputs = {
        **s0_validation["stage_outputs"],
        **validation["stage_outputs"],
    }

    if stage_id == "S1":
        return {
            "stage_id": stage_id,
            "matched": validation["matched"],
            "report": validation["report"],
            "ref_output": s0_validation["stage_outputs"]["ref_output"],
            "ref_lse": s0_validation["stage_outputs"]["ref_lse"],
            "fixture_sparse_indices": s0_validation["stage_outputs"]["fixture_sparse_indices"],
            "grid_shape": validation["stage_outputs"]["grid_shape"],
            "debug_token_order": validation["stage_outputs"]["debug_token_order"],
            "debug_token_done_count": validation["stage_outputs"]["debug_token_done_count"],
            "stage_outputs": s1_stage_outputs,
        }

    cute_mma_result = program.stage2_mma_debug_reference(device=q_nope.device)
    vectorized_mma_result = torch_reference_s2_mma_config(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    naive_mma_result = torch_reference_s2_mma_config_naive(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    for field_name in (
        "qk_mma_shape",
        "pv_mma_shape",
        "qk_ckv_iters",
        "qk_kpe_iters",
        "pv_k_iters",
        "pv_out_iters",
        "qk_a_major_mode",
        "qk_b_major_mode",
        "pv_a_major_mode",
        "pv_b_major_mode",
        "qk_helper_constructed",
        "pv_helper_constructed",
        "qk_atom_persisted",
        "pv_atom_persisted",
    ):
        if not torch.equal(cute_mma_result[field_name], vectorized_mma_result[field_name]):
            raise AssertionError(
                f"S2 host MMA metadata diverged between program reference and vectorized mirror for {field_name}"
            )

    s2_validation = prefix_validation_harness(
        cute_mma_result,
        naive_mma_result,
        stage_id="S2",
        dependency_validation=validation,
    )
    stage_outputs = {
        **s1_stage_outputs,
        **s2_validation["stage_outputs"],
    }

    if stage_id == "S2":
        return {
            "stage_id": stage_id,
            "matched": s2_validation["matched"],
            "report": s2_validation["report"],
            "ref_output": s0_validation["stage_outputs"]["ref_output"],
            "ref_lse": s0_validation["stage_outputs"]["ref_lse"],
            "fixture_sparse_indices": s0_validation["stage_outputs"]["fixture_sparse_indices"],
            "grid_shape": validation["stage_outputs"]["grid_shape"],
            "debug_token_order": validation["stage_outputs"]["debug_token_order"],
            "debug_token_done_count": validation["stage_outputs"]["debug_token_done_count"],
            "qk_mma_shape": s2_validation["stage_outputs"]["qk_mma_shape"],
            "pv_mma_shape": s2_validation["stage_outputs"]["pv_mma_shape"],
            "qk_ckv_iters": s2_validation["stage_outputs"]["qk_ckv_iters"],
            "qk_kpe_iters": s2_validation["stage_outputs"]["qk_kpe_iters"],
            "pv_k_iters": s2_validation["stage_outputs"]["pv_k_iters"],
            "pv_out_iters": s2_validation["stage_outputs"]["pv_out_iters"],
            "qk_a_major_mode": s2_validation["stage_outputs"]["qk_a_major_mode"],
            "qk_b_major_mode": s2_validation["stage_outputs"]["qk_b_major_mode"],
            "pv_a_major_mode": s2_validation["stage_outputs"]["pv_a_major_mode"],
            "pv_b_major_mode": s2_validation["stage_outputs"]["pv_b_major_mode"],
            "qk_helper_constructed": s2_validation["stage_outputs"]["qk_helper_constructed"],
            "pv_helper_constructed": s2_validation["stage_outputs"]["pv_helper_constructed"],
            "qk_atom_persisted": s2_validation["stage_outputs"]["qk_atom_persisted"],
            "pv_atom_persisted": s2_validation["stage_outputs"]["pv_atom_persisted"],
            "stage_outputs": stage_outputs,
        }

    cute_storage_result = program.stage3_storage_debug_reference(device=q_nope.device)
    vectorized_storage_result = torch_reference_s3_storage_config(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    naive_storage_result = torch_reference_s3_storage_config_naive(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    for field_name in (
        "shared_storage_bytes",
        "tmem_alloc_cols",
        "q_smem_bytes",
        "qk_ring_bytes",
        "p_ring_bytes",
        "v_ring_bytes",
    ):
        if not torch.equal(cute_storage_result[field_name], vectorized_storage_result[field_name]):
            raise AssertionError(
                f"S3 storage metadata diverged between the CuTe probe and the host storage mirror for {field_name}"
            )

    s3_validation = prefix_validation_harness(
        cute_storage_result,
        naive_storage_result,
        stage_id="S3",
        dependency_validation=s2_validation,
    )
    stage_outputs = {
        **stage_outputs,
        **s3_validation["stage_outputs"],
    }

    if stage_id == "S3":
        return {
            "stage_id": stage_id,
            "matched": s3_validation["matched"],
            "report": s3_validation["report"],
            "ref_output": s0_validation["stage_outputs"]["ref_output"],
            "ref_lse": s0_validation["stage_outputs"]["ref_lse"],
            "fixture_sparse_indices": s0_validation["stage_outputs"]["fixture_sparse_indices"],
            "grid_shape": validation["stage_outputs"]["grid_shape"],
            "debug_token_order": validation["stage_outputs"]["debug_token_order"],
            "debug_token_done_count": validation["stage_outputs"]["debug_token_done_count"],
            "qk_mma_shape": s2_validation["stage_outputs"]["qk_mma_shape"],
            "pv_mma_shape": s2_validation["stage_outputs"]["pv_mma_shape"],
            "qk_ckv_iters": s2_validation["stage_outputs"]["qk_ckv_iters"],
            "qk_kpe_iters": s2_validation["stage_outputs"]["qk_kpe_iters"],
            "pv_k_iters": s2_validation["stage_outputs"]["pv_k_iters"],
            "pv_out_iters": s2_validation["stage_outputs"]["pv_out_iters"],
            "qk_a_major_mode": s2_validation["stage_outputs"]["qk_a_major_mode"],
            "qk_b_major_mode": s2_validation["stage_outputs"]["qk_b_major_mode"],
            "pv_a_major_mode": s2_validation["stage_outputs"]["pv_a_major_mode"],
            "pv_b_major_mode": s2_validation["stage_outputs"]["pv_b_major_mode"],
            "qk_helper_constructed": s2_validation["stage_outputs"]["qk_helper_constructed"],
            "pv_helper_constructed": s2_validation["stage_outputs"]["pv_helper_constructed"],
            "qk_atom_persisted": s2_validation["stage_outputs"]["qk_atom_persisted"],
            "pv_atom_persisted": s2_validation["stage_outputs"]["pv_atom_persisted"],
            "shared_storage_bytes": s3_validation["stage_outputs"]["shared_storage_bytes"],
            "tmem_alloc_cols": s3_validation["stage_outputs"]["tmem_alloc_cols"],
            "q_smem_bytes": s3_validation["stage_outputs"]["q_smem_bytes"],
            "qk_ring_bytes": s3_validation["stage_outputs"]["qk_ring_bytes"],
            "p_ring_bytes": s3_validation["stage_outputs"]["p_ring_bytes"],
            "v_ring_bytes": s3_validation["stage_outputs"]["v_ring_bytes"],
            "stage_outputs": stage_outputs,
        }

    cute_q_prologue_result = {
        "debug_q_ckv_padded": cute_scheduler_result["debug_q_ckv_padded"],
        "debug_q_kpe_padded": cute_scheduler_result["debug_q_kpe_padded"],
    }
    vectorized_q_prologue_result = torch_reference_s4_q_prologue(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    naive_q_prologue_result = torch_reference_s4_q_prologue_naive(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )
    for field_name in ("debug_q_ckv_padded", "debug_q_kpe_padded"):
        if not torch.equal(cute_q_prologue_result[field_name], vectorized_q_prologue_result[field_name]):
            raise AssertionError(
                f"S4 Q-prologue outputs diverged between the CuTe device path and the vectorized host mirror for {field_name}"
            )

    s4_validation = prefix_validation_harness(
        cute_q_prologue_result,
        naive_q_prologue_result,
        stage_id="S4",
        dependency_validation=s3_validation,
    )
    stage_outputs = {
        **stage_outputs,
        **s4_validation["stage_outputs"],
    }

    return {
        "stage_id": stage_id,
        "matched": s4_validation["matched"],
        "report": s4_validation["report"],
        "ref_output": s0_validation["stage_outputs"]["ref_output"],
        "ref_lse": s0_validation["stage_outputs"]["ref_lse"],
        "fixture_sparse_indices": s0_validation["stage_outputs"]["fixture_sparse_indices"],
        "grid_shape": validation["stage_outputs"]["grid_shape"],
        "debug_token_order": validation["stage_outputs"]["debug_token_order"],
        "debug_token_done_count": validation["stage_outputs"]["debug_token_done_count"],
        "qk_mma_shape": s2_validation["stage_outputs"]["qk_mma_shape"],
        "pv_mma_shape": s2_validation["stage_outputs"]["pv_mma_shape"],
        "qk_ckv_iters": s2_validation["stage_outputs"]["qk_ckv_iters"],
        "qk_kpe_iters": s2_validation["stage_outputs"]["qk_kpe_iters"],
        "pv_k_iters": s2_validation["stage_outputs"]["pv_k_iters"],
        "pv_out_iters": s2_validation["stage_outputs"]["pv_out_iters"],
        "qk_a_major_mode": s2_validation["stage_outputs"]["qk_a_major_mode"],
        "qk_b_major_mode": s2_validation["stage_outputs"]["qk_b_major_mode"],
        "pv_a_major_mode": s2_validation["stage_outputs"]["pv_a_major_mode"],
        "pv_b_major_mode": s2_validation["stage_outputs"]["pv_b_major_mode"],
        "qk_helper_constructed": s2_validation["stage_outputs"]["qk_helper_constructed"],
        "pv_helper_constructed": s2_validation["stage_outputs"]["pv_helper_constructed"],
        "qk_atom_persisted": s2_validation["stage_outputs"]["qk_atom_persisted"],
        "pv_atom_persisted": s2_validation["stage_outputs"]["pv_atom_persisted"],
        "shared_storage_bytes": s3_validation["stage_outputs"]["shared_storage_bytes"],
        "tmem_alloc_cols": s3_validation["stage_outputs"]["tmem_alloc_cols"],
        "q_smem_bytes": s3_validation["stage_outputs"]["q_smem_bytes"],
        "qk_ring_bytes": s3_validation["stage_outputs"]["qk_ring_bytes"],
        "p_ring_bytes": s3_validation["stage_outputs"]["p_ring_bytes"],
        "v_ring_bytes": s3_validation["stage_outputs"]["v_ring_bytes"],
        "debug_q_ckv_padded": s4_validation["stage_outputs"]["debug_q_ckv_padded"],
        "debug_q_kpe_padded": s4_validation["stage_outputs"]["debug_q_kpe_padded"],
        "stage_outputs": stage_outputs,
    }


def kernel(*args, stage_id: str = "S4") -> Dict[str, Any]:
    """Default entry point used by the stage tooling."""

    return run(*args, stage_id=stage_id)


if __name__ == "__main__":
    result = run(stage_id="S4" if torch.cuda.is_available() else "S0")
    print(result["report"])