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


@cute.struct
class Stage0SharedStorage:
    """Minimal shared-storage scaffold for the staged round-0 kernel path."""

    placeholder: cutlass.Int32


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
        self.qk_tiled_mma = None
        self.pv_tiled_mma = None
        self._stage2_mma_cache: Optional[Dict[str, Any]] = None
        self._last_scheduler_params_error: Optional[BaseException] = None
        self._host_launch_grid: Optional[Tuple[int, int, int]] = None

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
    def persistent_token_scheduler_debug(
        self,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        debug_stage2_mma_meta: cute.Tensor,
        tile_sched_params: PersistentTileSchedulerParams,
        num_tokens: cutlass.Int32,
        SharedStorage: cutlass.Constexpr,
    ):
        """Stage-S1 CuTeDSL scheduler scaffold.

        Round-0 S1 validates only the launch/scheduler contract. We keep the
        device-path shell here, using the official persistent scheduler APIs, so
        later stages can directly extend this method into the real token loop.
        """

        del SharedStorage

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
            if cutlass.const_expr(self.enable_stage2_mma_probe):
                qk_tiled_mma, pv_tiled_mma = self._ensure_stage2_tiled_mma_atoms_jit()
                cute.printf(
                    "S2 live qk_tiled_mma built for requested major modes (K, K): shape_mnk=({}, {}, {})",
                    qk_tiled_mma.op.shape_mnk[0],
                    qk_tiled_mma.op.shape_mnk[1],
                    qk_tiled_mma.op.shape_mnk[2],
                )
                cute.printf(
                    "S2 live pv_tiled_mma built for requested major modes (K, MN): shape_mnk=({}, {}, {})",
                    pv_tiled_mma.op.shape_mnk[0],
                    pv_tiled_mma.op.shape_mnk[1],
                    pv_tiled_mma.op.shape_mnk[2],
                )
                cute.printf(
                    "S2 persisted program MMA objects for downstream reuse: qk_shape=({}, {}, {}), pv_shape=({}, {}, {})",
                    self.qk_tiled_mma.op.shape_mnk[0],
                    self.qk_tiled_mma.op.shape_mnk[1],
                    self.qk_tiled_mma.op.shape_mnk[2],
                    self.pv_tiled_mma.op.shape_mnk[0],
                    self.pv_tiled_mma.op.shape_mnk[1],
                    self.pv_tiled_mma.op.shape_mnk[2],
                )
                if bidz == 0:
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
                    debug_stage2_mma_meta[16] = cutlass.Int32(
                        1 if self.qk_tiled_mma is not None else 0
                    )
                    debug_stage2_mma_meta[17] = cutlass.Int32(
                        1 if self.pv_tiled_mma is not None else 0
                    )
                    cute.printf(
                        "S2 probe metadata written: qk_shape=({}, {}, {}), pv_shape=({}, {}, {}), q_modes=({}, {}), pv_modes=({}, {})",
                        debug_stage2_mma_meta[0],
                        debug_stage2_mma_meta[1],
                        debug_stage2_mma_meta[2],
                        debug_stage2_mma_meta[3],
                        debug_stage2_mma_meta[4],
                        debug_stage2_mma_meta[5],
                        debug_stage2_mma_meta[10],
                        debug_stage2_mma_meta[11],
                        debug_stage2_mma_meta[12],
                        debug_stage2_mma_meta[13],
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
    def __call__(
        self,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        debug_stage2_mma_meta: cute.Tensor,
        num_tokens: cutlass.Int32,
        stream: CudaStream,
    ):
        """Stage-S1 launch-time orchestration for the persistent scheduler slice."""

        tile_sched_params = self._make_scheduler_params(num_tokens)
        if cutlass.const_expr(self.enable_stage2_mma_probe):
            self._ensure_stage2_tiled_mma_atoms_jit()
        self.kernel1_impl(
            debug_token_order,
            debug_token_done_count,
            debug_stage2_mma_meta,
            tile_sched_params,
            num_tokens,
            Stage0SharedStorage,
        ).launch(
            grid=self._host_launch_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=Stage0SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel1_impl(
        self,
        debug_token_order: cute.Tensor,
        debug_token_done_count: cute.Tensor,
        debug_stage2_mma_meta: cute.Tensor,
        tile_sched_params: PersistentTileSchedulerParams,
        num_tokens: cutlass.Int32,
        SharedStorage: cutlass.Constexpr,
    ):
        """Deferred device entry for the round-0 staged kernel."""

        self.persistent_token_scheduler_debug(
            debug_token_order,
            debug_token_done_count,
            debug_stage2_mma_meta,
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
    raise ValueError(f"Unsupported stage_id for prefix validation: {stage_id}")


def _launch_s1_scheduler_debug_kernel(
    program: BlackwellStyleKernel,
    num_tokens: int,
    *,
    device: torch.device,
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

    cache_key = (
        "S1_scheduler_debug",
        id(program),
        num_tokens,
        tuple(debug_token_order.shape),
        tuple(debug_token_done_count.shape),
        tuple(debug_stage2_mma_meta.shape),
        str(device),
        grid_shape_tuple,
        program.cluster_shape_mnk,
        program.max_active_clusters,
        program.swizzle_size,
        program.raster_along_m,
        program.threads_per_cta,
        program.enable_stage2_mma_probe,
    )
    if cache_key not in program_compile_cache:
        program_compile_cache[cache_key] = cute.compile(
            program,
            debug_token_order_cute,
            debug_token_done_count_cute,
            debug_stage2_mma_meta_cute,
            num_tokens,
            compile_stream,
        )

    compiled_program = program_compile_cache[cache_key]
    compiled_program(
        debug_token_order_cute,
        debug_token_done_count_cute,
        debug_stage2_mma_meta_cute,
        num_tokens,
        stream,
    )
    torch.cuda.synchronize(device=device)

    return {
        "grid_shape": grid_shape,
        "debug_token_order": debug_token_order.contiguous(),
        "debug_token_done_count": debug_token_done_count.contiguous(),
        "debug_stage2_mma_meta": debug_stage2_mma_meta.contiguous(),
    }


def run(*args, stage_id: str = "S2") -> Dict[str, Any]:
    """Computes the staged round-0 prefix outputs for the canonical fixture.

    Accepted call patterns:
      - run()
      - run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
      - run(..., stage_id="S0", "S1", or "S2")

    The 6-tensor form is accepted only to match the generic stage-runner call
    signature. The staged prefix harness still always rebuilds and validates the
    canonical deterministic 3-token fixture required by the architecture plan.
    """

    if stage_id not in {"S0", "S1", "S2"}:
        raise ValueError(f"Unsupported stage_id for this module: {stage_id}")

    if len(args) == 1 and isinstance(args[0], str):
        stage_id = args[0]
        if stage_id not in {"S0", "S1", "S2"}:
            raise ValueError(f"Unsupported stage_id for this module: {stage_id}")
        args = ()

    fixture_device = None
    if len(args) == 0:
        if stage_id in {"S1", "S2"} and torch.cuda.is_available():
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
    program.enable_stage2_mma_probe = stage_id == "S2"
    cute_scheduler_result = _launch_s1_scheduler_debug_kernel(
        program,
        q_nope.shape[0],
        device=q_nope.device,
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


def kernel(*args, stage_id: str = "S2") -> Dict[str, Any]:
    """Default entry point used by the stage tooling."""

    return run(*args, stage_id=stage_id)


if __name__ == "__main__":
    result = run(stage_id="S2" if torch.cuda.is_available() else "S0")
    print(result["report"])