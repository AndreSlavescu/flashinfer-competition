import sys
from dataclasses import dataclass
from pathlib import Path
from numbers import Number
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUTLASS_PYTHON_ROOT = _REPO_ROOT / "references" / "cutlass" / "python" / "CuTeDSL"
if _CUTLASS_PYTHON_ROOT.exists():
    cutlass_python_root = str(_CUTLASS_PYTHON_ROOT)
    if cutlass_python_root not in sys.path:
        sys.path.insert(0, cutlass_python_root)


try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as cutlass_utils
    from cutlass.cute.runtime import from_dlpack

    _HAS_CUTLASS = True
except Exception:  # pragma: no cover - exercised only on hosts without CuTeDSL
    cutlass = None
    cute = None
    pipeline = None
    cutlass_utils = None
    from_dlpack = None
    _HAS_CUTLASS = False


compile_cache: dict[tuple[Any, ...], Any] = {}
debug_s0_launch_meta: Optional[torch.Tensor] = None
debug_s0_token_block_map: Optional[torch.Tensor] = None
debug_s1_indices_tile0: Optional[torch.Tensor] = None
debug_s1_valid_mask_tile0: Optional[torch.Tensor] = None
debug_s1_ckv_tile0: Optional[torch.Tensor] = None
debug_s1_kpe_tile0: Optional[torch.Tensor] = None
debug_s1_v_tile0: Optional[torch.Tensor] = None
debug_s2_logits_tile0: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class KernelConfig:
    threads_per_cta: int = 224
    warp_count: int = 7
    owner_warp_index: int = 3
    num_heads: int = 16
    head_dim_ckv: int = 512
    head_dim_kpe: int = 64
    page_size: int = 64
    topk: int = 2048
    qk_tile_shape: tuple[int, int, int] = (64, 32, 16)
    pv_tile_shape: tuple[int, int, int] = (64, 128, 16)
    dynamic_smem_bytes: int = 154 * 1024
    reserved_tmem_columns: int = 512
    launch_cluster_shape: tuple[int, int, int] = (1, 1, 1)
    tmem_alloc_barrier_id: int = 7
    ctas_per_sm: int = 1

    @classmethod
    def from_inputs(
        cls,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
    ) -> "KernelConfig":
        config = cls()
        config.validate_inputs(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            sparse_indices=sparse_indices,
            output=output,
            lse=lse,
        )
        return config

    def validate_inputs(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        if q_nope.ndim != 3 or q_nope.shape[1:] != (self.num_heads, self.head_dim_ckv):
            raise ValueError(
                f"q_nope must have shape [T,{self.num_heads},{self.head_dim_ckv}], got {tuple(q_nope.shape)}"
            )
        if q_pe.ndim != 3 or q_pe.shape != (q_nope.shape[0], self.num_heads, self.head_dim_kpe):
            raise ValueError(
                f"q_pe must have shape [T,{self.num_heads},{self.head_dim_kpe}], got {tuple(q_pe.shape)}"
            )
        if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (self.page_size, self.head_dim_ckv):
            raise ValueError(
                f"ckv_cache must have shape [P,{self.page_size},{self.head_dim_ckv}], got {tuple(ckv_cache.shape)}"
            )
        if kpe_cache.ndim != 3 or kpe_cache.shape != (
            ckv_cache.shape[0],
            self.page_size,
            self.head_dim_kpe,
        ):
            raise ValueError(
                f"kpe_cache must have shape [P,{self.page_size},{self.head_dim_kpe}], got {tuple(kpe_cache.shape)}"
            )
        if sparse_indices.ndim != 2 or sparse_indices.shape != (q_nope.shape[0], self.topk):
            raise ValueError(
                f"sparse_indices must have shape [T,{self.topk}], got {tuple(sparse_indices.shape)}"
            )
        if output.shape != q_nope.shape:
            raise ValueError(f"output must have shape {tuple(q_nope.shape)}, got {tuple(output.shape)}")
        if lse.shape != (q_nope.shape[0], self.num_heads):
            raise ValueError(
                f"lse must have shape [T,{self.num_heads}], got {tuple(lse.shape)}"
            )
        if not q_nope.dtype.is_floating_point:
            raise TypeError(f"q_nope must be floating-point, got {q_nope.dtype}")
        if not q_pe.dtype.is_floating_point:
            raise TypeError(f"q_pe must be floating-point, got {q_pe.dtype}")
        if not ckv_cache.dtype.is_floating_point:
            raise TypeError(f"ckv_cache must be floating-point, got {ckv_cache.dtype}")
        if not kpe_cache.dtype.is_floating_point:
            raise TypeError(f"kpe_cache must be floating-point, got {kpe_cache.dtype}")
        if not output.dtype.is_floating_point:
            raise TypeError(f"output must be floating-point, got {output.dtype}")
        if not lse.dtype.is_floating_point:
            raise TypeError(f"lse must be floating-point, got {lse.dtype}")
        if sparse_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"sparse_indices must be int32 or int64, got {sparse_indices.dtype}")

        expected_device = q_nope.device
        for name, tensor in (
            ("q_pe", q_pe),
            ("ckv_cache", ckv_cache),
            ("kpe_cache", kpe_cache),
            ("sparse_indices", sparse_indices),
            ("output", output),
            ("lse", lse),
        ):
            if tensor.device != expected_device:
                raise ValueError(
                    f"{name} must be on device {expected_device}, got {tensor.device}"
                )

    def grid_shape(self, num_tokens: int) -> tuple[int, int, int]:
        return (int(num_tokens), 1, 1)

    def block_shape(self) -> tuple[int, int, int]:
        return (self.threads_per_cta, 1, 1)

    def token_for_block(self, block_idx_x: int) -> int:
        return int(block_idx_x)

    @property
    def dynamic_smem_kib(self) -> float:
        return float(self.dynamic_smem_bytes) / 1024.0

    @property
    def sparse_tile_size(self) -> int:
        return int(self.qk_tile_shape[1])

    @staticmethod
    def stage0_debug_exports() -> tuple[str, str]:
        return ("debug_s0_launch_meta", "debug_s0_token_block_map")

    @staticmethod
    def stage1_debug_exports() -> tuple[str, str, str, str, str]:
        return (
            "debug_s1_indices_tile0",
            "debug_s1_valid_mask_tile0",
            "debug_s1_ckv_tile0",
            "debug_s1_kpe_tile0",
            "debug_s1_v_tile0",
        )

    @staticmethod
    def stage2_debug_exports() -> tuple[str, ...]:
        return ("debug_s2_logits_tile0",)


def _default_debug_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _normalize_sm_scale(sm_scale: Any) -> float:
    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError(f"sm_scale tensor must be scalar, got shape {tuple(sm_scale.shape)}")
        return float(sm_scale.item())
    if isinstance(sm_scale, Number):
        return float(sm_scale)
    raise TypeError(f"sm_scale must be a scalar number or scalar tensor, got {type(sm_scale)!r}")


def emit_launch_metadata(
    config: KernelConfig,
    num_tokens: int,
    *,
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    debug_device = _default_debug_device(device)
    launch_meta = torch.empty((num_tokens, 4), dtype=torch.int64, device=debug_device)
    launch_meta[:, 0] = config.block_shape()[0]
    launch_meta[:, 1] = config.grid_shape(num_tokens)[0]
    launch_meta[:, 2] = config.dynamic_smem_bytes
    launch_meta[:, 3] = config.reserved_tmem_columns

    token_block_map = torch.empty((num_tokens, 4), dtype=torch.int64, device=debug_device)
    token_ids = torch.arange(num_tokens, dtype=torch.int64, device=debug_device)
    token_block_map[:, 0] = token_ids
    token_block_map[:, 1] = token_ids
    token_block_map[:, 2] = 0
    token_block_map[:, 3] = config.num_heads
    return {
        "debug_s0_launch_meta": launch_meta,
        "debug_s0_token_block_map": token_block_map,
    }


def _select_debug_exports(
    exports: Mapping[str, torch.Tensor],
    requested_names: Optional[Iterable[str]],
) -> dict[str, torch.Tensor]:
    if requested_names is None:
        return {}
    requested = list(requested_names)
    missing = [name for name in requested if name not in exports]
    if missing:
        raise KeyError(f"Unknown debug export names requested: {missing}")
    return {name: exports[name] for name in requested}


def _eager_full_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    reference_output = torch.zeros_like(q_nope)
    reference_lse = torch.full(
        (num_tokens, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=q_nope.device,
    )

    for token_idx in range(num_tokens):
        token_sparse_indices = sparse_indices[token_idx]
        valid_mask = token_sparse_indices != -1
        valid_indices = token_sparse_indices[valid_mask].to(torch.long)
        if valid_indices.numel() == 0:
            reference_output[token_idx].zero_()
            continue

        kc = kc_all[valid_indices]
        kp = kp_all[valid_indices]
        qn = q_nope[token_idx].to(torch.float32)
        qp = q_pe[token_idx].to(torch.float32)

        logits = (qn @ kc.T) + (qp @ kp.T)
        logits_scaled = logits * sm_scale
        reference_lse[token_idx] = torch.logsumexp(logits_scaled, dim=-1) / torch.log(
            torch.tensor(2.0, dtype=torch.float32, device=q_nope.device)
        )
        attention = torch.softmax(logits_scaled, dim=-1)
        reference_output[token_idx] = (attention @ kc).to(torch.bfloat16)

    return reference_output, reference_lse


def build_sparse_index_pipeline(
    config: KernelConfig,
    sparse_indices: torch.Tensor,
    *,
    tile_index: int = 0,
    num_pages: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    tile_size = config.sparse_tile_size
    tile_start = int(tile_index) * tile_size
    tile_stop = tile_start + tile_size
    indices_tile = sparse_indices[:, tile_start:tile_stop].to(torch.int32)

    valid_mask = indices_tile >= 0
    if num_pages is not None:
        total_kv_tokens = int(num_pages) * config.page_size
        valid_mask = valid_mask & (indices_tile < total_kv_tokens)

    zero_i32 = torch.zeros_like(indices_tile, dtype=torch.int32)
    page_indices = torch.where(
        valid_mask,
        torch.div(indices_tile, config.page_size, rounding_mode="floor"),
        zero_i32,
    )
    slot_indices = torch.where(
        valid_mask,
        torch.remainder(indices_tile, config.page_size),
        zero_i32,
    )
    return {
        "indices_tile": indices_tile,
        "valid_mask": valid_mask.to(torch.int32),
        "page_indices": page_indices.to(torch.int32),
        "slot_indices": slot_indices.to(torch.int32),
        "tile_start": torch.tensor(tile_start, dtype=torch.int32, device=sparse_indices.device),
    }


def _eager_decode_sparse_tile_reference(
    config: KernelConfig,
    sparse_indices: torch.Tensor,
    *,
    tile_index: int = 0,
    num_pages: int,
) -> dict[str, torch.Tensor]:
    tile_size = config.sparse_tile_size
    tile_start = int(tile_index) * tile_size
    tile_stop = tile_start + tile_size
    indices_tile = sparse_indices[:, tile_start:tile_stop].to(torch.int32).clone()

    total_kv_tokens = int(num_pages) * config.page_size
    valid_mask = (indices_tile >= 0) & (indices_tile < total_kv_tokens)
    safe_indices = indices_tile.to(torch.int64).clone()
    safe_indices.masked_fill_(~valid_mask, 0)

    page_indices = torch.div(safe_indices, config.page_size, rounding_mode="floor").to(torch.int32)
    slot_indices = torch.remainder(safe_indices, config.page_size).to(torch.int32)
    return {
        "indices_tile": indices_tile,
        "valid_mask": valid_mask.to(torch.int32),
        "page_indices": page_indices,
        "slot_indices": slot_indices,
        "safe_indices": safe_indices,
        "tile_start": torch.tensor(tile_start, dtype=torch.int32, device=sparse_indices.device),
    }


def _derive_pv_view_from_ckv(
    config: KernelConfig,
    ckv_tile: torch.Tensor,
) -> torch.Tensor:
    num_tokens, tile_size, head_dim = ckv_tile.shape
    pv_width = int(config.pv_tile_shape[1])
    if head_dim % pv_width != 0:
        raise ValueError(f"CKV head dim {head_dim} must be divisible by PV width {pv_width}.")
    pv_slices = head_dim // pv_width
    pv_view = (
        ckv_tile.reshape(num_tokens, tile_size, pv_slices, pv_width)
        .permute(0, 2, 1, 3)
        .contiguous()
        .permute(0, 2, 1, 3)
        .reshape(num_tokens, tile_size, head_dim)
    )
    return pv_view


def gather_sparse_tile(
    config: KernelConfig,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_tile_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    indices_tile = sparse_tile_state["indices_tile"]
    valid_mask = sparse_tile_state["valid_mask"].to(torch.bool)
    page_indices = sparse_tile_state["page_indices"]
    slot_indices = sparse_tile_state["slot_indices"]

    num_tokens, tile_size = indices_tile.shape
    ckv_tile = torch.zeros(
        (num_tokens, tile_size, config.head_dim_ckv),
        dtype=ckv_cache.dtype,
        device=ckv_cache.device,
    )
    kpe_tile = torch.zeros(
        (num_tokens, tile_size, config.head_dim_kpe),
        dtype=kpe_cache.dtype,
        device=kpe_cache.device,
    )

    for token_idx in range(num_tokens):
        for sparse_lane in range(tile_size):
            if not bool(valid_mask[token_idx, sparse_lane].item()):
                continue
            page_idx = int(page_indices[token_idx, sparse_lane].item())
            slot_idx = int(slot_indices[token_idx, sparse_lane].item())
            ckv_tile[token_idx, sparse_lane].copy_(ckv_cache[page_idx, slot_idx])
            kpe_tile[token_idx, sparse_lane].copy_(kpe_cache[page_idx, slot_idx])

    v_tile = _derive_pv_view_from_ckv(config, ckv_tile)
    return {
        "ckv_tile": ckv_tile,
        "kpe_tile": kpe_tile,
        "v_tile": v_tile,
    }


def _eager_gather_sparse_tile_reference(
    config: KernelConfig,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_tile_reference: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    indices_tile = sparse_tile_reference["indices_tile"]
    valid_mask = sparse_tile_reference["valid_mask"].to(torch.bool)
    safe_indices = sparse_tile_reference["safe_indices"].to(torch.long)

    num_tokens, tile_size = indices_tile.shape
    flat_ckv = ckv_cache.reshape(-1, config.head_dim_ckv)
    flat_kpe = kpe_cache.reshape(-1, config.head_dim_kpe)

    ckv_tile = torch.zeros(
        (num_tokens, tile_size, config.head_dim_ckv),
        dtype=ckv_cache.dtype,
        device=ckv_cache.device,
    )
    kpe_tile = torch.zeros(
        (num_tokens, tile_size, config.head_dim_kpe),
        dtype=kpe_cache.dtype,
        device=kpe_cache.device,
    )

    for token_idx in range(num_tokens):
        for sparse_lane in range(tile_size):
            if not bool(valid_mask[token_idx, sparse_lane].item()):
                continue
            row_idx = int(safe_indices[token_idx, sparse_lane].item())
            ckv_tile[token_idx, sparse_lane].copy_(flat_ckv[row_idx])
            kpe_tile[token_idx, sparse_lane].copy_(flat_kpe[row_idx])

    v_tile = ckv_tile.clone()
    return {
        "ckv_tile": ckv_tile,
        "kpe_tile": kpe_tile,
        "v_tile": v_tile,
    }


def export_debug_stage_s1(
    config: KernelConfig,
    *,
    sparse_indices: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sparse_tile_state = build_sparse_index_pipeline(
        config,
        sparse_indices,
        tile_index=0,
        num_pages=ckv_cache.shape[0],
    )
    gathered_tile = gather_sparse_tile(
        config,
        ckv_cache,
        kpe_cache,
        sparse_tile_state,
    )
    return {
        "debug_s1_indices_tile0": sparse_tile_state["indices_tile"],
        "debug_s1_valid_mask_tile0": sparse_tile_state["valid_mask"],
        "debug_s1_ckv_tile0": gathered_tile["ckv_tile"],
        "debug_s1_kpe_tile0": gathered_tile["kpe_tile"],
        "debug_s1_v_tile0": gathered_tile["v_tile"],
    }


def load_q_phase_slice(
    config: KernelConfig,
    q_tensor: torch.Tensor,
    *,
    slice_start: int,
    slice_width: Optional[int] = None,
) -> torch.Tensor:
    if q_tensor.ndim != 3:
        raise ValueError(f"q_tensor must have shape [T,H,K], got {tuple(q_tensor.shape)}")

    live_rows = config.num_heads
    padded_rows = config.qk_tile_shape[0]
    k_extent = int(config.qk_tile_shape[2] if slice_width is None else slice_width)
    slice_stop = int(slice_start) + k_extent
    if int(slice_start) < 0 or slice_stop > q_tensor.shape[-1]:
        raise ValueError(
            f"Requested Q slice [{slice_start}:{slice_stop}) is out of bounds for shape {tuple(q_tensor.shape)}"
        )
    if q_tensor.shape[1] != live_rows:
        raise ValueError(f"Expected {live_rows} live Q rows, got {q_tensor.shape[1]}")

    q_phase_slice = torch.zeros(
        (q_tensor.shape[0], padded_rows, k_extent),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    )
    q_phase_slice[:, :live_rows, :].copy_(q_tensor[:, :, int(slice_start):slice_stop])
    return q_phase_slice


def qk_mainloop(
    config: KernelConfig,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    gathered_tile: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    padded_rows, tile_cols, k_extent = config.qk_tile_shape
    logits_tile = torch.zeros(
        (q_nope.shape[0], padded_rows, tile_cols),
        dtype=torch.float32,
        device=q_nope.device,
    )

    ckv_tile = gathered_tile["ckv_tile"].to(torch.float32)
    kpe_tile = gathered_tile["kpe_tile"].to(torch.float32)

    for slice_start in range(0, config.head_dim_ckv, k_extent):
        q_slice = load_q_phase_slice(config, q_nope, slice_start=slice_start, slice_width=k_extent).to(torch.float32)
        k_slice = ckv_tile[:, :, slice_start : slice_start + k_extent]
        logits_tile.add_((q_slice.unsqueeze(2) * k_slice.unsqueeze(1)).sum(dim=-1))

    for slice_start in range(0, config.head_dim_kpe, k_extent):
        q_slice = load_q_phase_slice(config, q_pe, slice_start=slice_start, slice_width=k_extent).to(torch.float32)
        k_slice = kpe_tile[:, :, slice_start : slice_start + k_extent]
        logits_tile.add_((q_slice.unsqueeze(2) * k_slice.unsqueeze(1)).sum(dim=-1))

    if padded_rows > config.num_heads:
        logits_tile[:, config.num_heads :, :].zero_()
    return logits_tile


def export_debug_stage_s2(
    config: KernelConfig,
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    sparse_indices: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sparse_tile_state = build_sparse_index_pipeline(
        config,
        sparse_indices,
        tile_index=0,
        num_pages=ckv_cache.shape[0],
    )
    gathered_tile = gather_sparse_tile(
        config,
        ckv_cache,
        kpe_cache,
        sparse_tile_state,
    )
    return {
        "debug_s2_logits_tile0": qk_mainloop(
            config,
            q_nope,
            q_pe,
            gathered_tile,
        )
    }


def _eager_qk_logits_reference(
    config: KernelConfig,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    gathered_tile_reference: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    logits_tile = torch.zeros(
        (q_nope.shape[0], config.qk_tile_shape[0], config.qk_tile_shape[1]),
        dtype=torch.float32,
        device=q_nope.device,
    )
    live_logits = torch.matmul(
        q_nope.to(torch.float32),
        gathered_tile_reference["ckv_tile"].to(torch.float32).transpose(1, 2),
    ) + torch.matmul(
        q_pe.to(torch.float32),
        gathered_tile_reference["kpe_tile"].to(torch.float32).transpose(1, 2),
    )
    logits_tile[:, : config.num_heads, :].copy_(live_logits)
    return logits_tile


if _HAS_CUTLASS:

    @cute.struct
    class _ShellSharedStorage:
        tmem_dealloc_mbar_ptr: cutlass.Int64
        tmem_holding_buf: cutlass.Int32


    @cute.kernel()
    def _dsa_sparse_attention_shell_kernel(
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        ckv_cache: cute.Tensor,
        kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor,
        output: cute.Tensor,
        lse: cute.Tensor,
        num_tokens: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_ShellSharedStorage)
        _shell_smem_bytes = KernelConfig.dynamic_smem_bytes - _ShellSharedStorage.size_in_bytes()
        if _shell_smem_bytes > 0:
            smem.allocate(_shell_smem_bytes, byte_alignment=128)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=KernelConfig.tmem_alloc_barrier_id,
            num_threads=KernelConfig.threads_per_cta,
        )
        tmem = cutlass_utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=KernelConfig.owner_warp_index,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        if tidx == 0:
            cute.printf(
                "S0 shell: blockIdx.x={} num_tokens={} blockDim.x={} smem={} tmem_cols={}",
                bidx,
                num_tokens,
                KernelConfig.threads_per_cta,
                KernelConfig.dynamic_smem_bytes,
                KernelConfig.reserved_tmem_columns,
            )
            cute.printf(
                "S1 shell: sparse_tile={} page_size={} p0_stages={} p1_stages={}",
                KernelConfig.qk_tile_shape[1],
                KernelConfig.page_size,
                2,
                2,
            )
            cute.printf(
                "S2 shell: owner_warp={} qk_tile_mnk=({}, {}, {}) live_rows={} p2_stages={}",
                KernelConfig.owner_warp_index,
                KernelConfig.qk_tile_shape[0],
                KernelConfig.qk_tile_shape[1],
                KernelConfig.qk_tile_shape[2],
                KernelConfig.num_heads,
                1,
            )
        if warp_idx == KernelConfig.owner_warp_index:
            tmem.allocate(KernelConfig.reserved_tmem_columns)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        if tidx == 0:
            cute.printf("S0/S2 tmem ptr token={} ptr={}", bidx, tmem_ptr)
        if warp_idx == KernelConfig.owner_warp_index:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


    @cute.jit
    def _dsa_sparse_attention_shell_host(
        q_nope: cute.Tensor,
        q_pe: cute.Tensor,
        ckv_cache: cute.Tensor,
        kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor,
        output: cute.Tensor,
        lse: cute.Tensor,
        num_tokens: cutlass.Int32,
    ):
        _dsa_sparse_attention_shell_kernel(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
            output,
            lse,
            num_tokens,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(KernelConfig.threads_per_cta, 1, 1),
            cluster=KernelConfig.launch_cluster_shape,
            smem=KernelConfig.dynamic_smem_bytes,
            min_blocks_per_mp=KernelConfig.ctas_per_sm,
        )


def _get_compiled_kernel(
    config: KernelConfig,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    stream: Any,
):
    if not _HAS_CUTLASS:
        raise RuntimeError("CuTeDSL is not available on this host.")

    cache_key = (
        tuple(q_nope.shape),
        tuple(q_pe.shape),
        tuple(ckv_cache.shape),
        tuple(kpe_cache.shape),
        tuple(sparse_indices.shape),
        config.threads_per_cta,
        config.dynamic_smem_bytes,
        config.reserved_tmem_columns,
    )
    compiled = compile_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            _dsa_sparse_attention_shell_host,
            from_dlpack(q_nope),
            from_dlpack(q_pe),
            from_dlpack(ckv_cache),
            from_dlpack(kpe_cache),
            from_dlpack(sparse_indices),
            from_dlpack(output),
            from_dlpack(lse),
            cutlass.Int32(q_nope.shape[0]),
            stream,
        )
        compile_cache[cache_key] = compiled
    return compiled


def _maybe_launch_cute_shell(
    config: KernelConfig,
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    if not _HAS_CUTLASS or not q_nope.is_cuda:
        return

    if not hasattr(cutlass, "torch"):
        return

    current_stream = cutlass.torch.default_stream()
    compiled = _get_compiled_kernel(
        config,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        lse,
        current_stream,
    )
    compiled(
        from_dlpack(q_nope),
        from_dlpack(q_pe),
        from_dlpack(ckv_cache),
        from_dlpack(kpe_cache),
        from_dlpack(sparse_indices),
        from_dlpack(output),
        from_dlpack(lse),
        cutlass.Int32(q_nope.shape[0]),
        current_stream,
    )


def dsa_sparse_attention_fwd(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    debug_exports: Optional[Sequence[str]] = None,
    enable_cute_shell: bool = False,
) -> dict[str, torch.Tensor]:
    _normalize_sm_scale(sm_scale)

    config = KernelConfig.from_inputs(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices,
        output=output,
        lse=lse,
    )

    if enable_cute_shell:
        _maybe_launch_cute_shell(
            config,
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            sparse_indices=sparse_indices,
            output=output,
            lse=lse,
        )

    exports = emit_launch_metadata(config, q_nope.shape[0], device=output.device)
    requested_debug_exports = tuple(debug_exports) if debug_exports is not None else ()
    if any(name in config.stage1_debug_exports() for name in requested_debug_exports):
        exports.update(
            export_debug_stage_s1(
                config,
                sparse_indices=sparse_indices,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
            )
        )
    if any(name in config.stage2_debug_exports() for name in requested_debug_exports):
        exports.update(
            export_debug_stage_s2(
                config,
                q_nope=q_nope,
                q_pe=q_pe,
                sparse_indices=sparse_indices,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
            )
        )

    global debug_s0_launch_meta, debug_s0_token_block_map
    global debug_s1_indices_tile0, debug_s1_valid_mask_tile0
    global debug_s1_ckv_tile0, debug_s1_kpe_tile0, debug_s1_v_tile0
    global debug_s2_logits_tile0
    debug_s0_launch_meta = exports["debug_s0_launch_meta"]
    debug_s0_token_block_map = exports["debug_s0_token_block_map"]
    debug_s1_indices_tile0 = exports.get("debug_s1_indices_tile0")
    debug_s1_valid_mask_tile0 = exports.get("debug_s1_valid_mask_tile0")
    debug_s1_ckv_tile0 = exports.get("debug_s1_ckv_tile0")
    debug_s1_kpe_tile0 = exports.get("debug_s1_kpe_tile0")
    debug_s1_v_tile0 = exports.get("debug_s1_v_tile0")
    debug_s2_logits_tile0 = exports.get("debug_s2_logits_tile0")
    return _select_debug_exports(exports, debug_exports)


def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    dsa_sparse_attention_fwd(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
        output,
        lse,
    )
    return None


def _allocate_validation_io(
    *,
    num_tokens: int = 5,
    num_pages: int = 8,
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    alloc_device = _default_debug_device(device)
    q_nope = torch.zeros(
        (num_tokens, 16, 512),
        dtype=torch.bfloat16,
        device=alloc_device,
    )
    q_pe = torch.zeros(
        (num_tokens, 16, 64),
        dtype=torch.bfloat16,
        device=alloc_device,
    )
    ckv_cache = torch.zeros(
        (num_pages, 64, 512),
        dtype=torch.bfloat16,
        device=alloc_device,
    )
    kpe_cache = torch.zeros(
        (num_pages, 64, 64),
        dtype=torch.bfloat16,
        device=alloc_device,
    )
    sparse_indices = torch.full(
        (num_tokens, 2048),
        -1,
        dtype=torch.int32,
        device=alloc_device,
    )
    output = torch.zeros_like(q_nope)
    lse = torch.zeros((num_tokens, 16), dtype=torch.float32, device=alloc_device)
    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "output": output,
        "lse": lse,
    }


def _eager_prefix_reference_s0(
    config: KernelConfig,
    *,
    num_tokens: int,
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    return emit_launch_metadata(config, num_tokens, device=device)


def _eager_prefix_reference_s1(
    config: KernelConfig,
    *,
    sparse_indices: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    exports = emit_launch_metadata(config, sparse_indices.shape[0], device=device)
    sparse_tile_reference = _eager_decode_sparse_tile_reference(
        config,
        sparse_indices,
        tile_index=0,
        num_pages=ckv_cache.shape[0],
    )
    gathered_tile_reference = _eager_gather_sparse_tile_reference(
        config,
        ckv_cache,
        kpe_cache,
        sparse_tile_reference,
    )
    exports.update(
        {
            "debug_s1_indices_tile0": sparse_tile_reference["indices_tile"],
            "debug_s1_valid_mask_tile0": sparse_tile_reference["valid_mask"],
            "debug_s1_ckv_tile0": gathered_tile_reference["ckv_tile"],
            "debug_s1_kpe_tile0": gathered_tile_reference["kpe_tile"],
            "debug_s1_v_tile0": gathered_tile_reference["v_tile"],
        }
    )
    return exports


def _eager_prefix_reference_s2(
    config: KernelConfig,
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    sparse_indices: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    exports = emit_launch_metadata(config, sparse_indices.shape[0], device=device)
    sparse_tile_reference = _eager_decode_sparse_tile_reference(
        config,
        sparse_indices,
        tile_index=0,
        num_pages=ckv_cache.shape[0],
    )
    gathered_tile_reference = _eager_gather_sparse_tile_reference(
        config,
        ckv_cache,
        kpe_cache,
        sparse_tile_reference,
    )
    exports.update(
        {
            "debug_s1_indices_tile0": sparse_tile_reference["indices_tile"],
            "debug_s1_valid_mask_tile0": sparse_tile_reference["valid_mask"],
            "debug_s1_ckv_tile0": gathered_tile_reference["ckv_tile"],
            "debug_s1_kpe_tile0": gathered_tile_reference["kpe_tile"],
            "debug_s1_v_tile0": gathered_tile_reference["v_tile"],
            "debug_s2_logits_tile0": _eager_qk_logits_reference(
                config,
                q_nope,
                q_pe,
                gathered_tile_reference,
            ),
        }
    )
    return exports


def _resolve_validation_inputs(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], float, bool]:
    if len(args) == 1 and isinstance(args[0], Mapping):
        case = dict(args[0])
        if "q_nope" in case:
            return (
                {
                    "q_nope": case["q_nope"],
                    "q_pe": case["q_pe"],
                    "ckv_cache": case["ckv_cache"],
                    "kpe_cache": case["kpe_cache"],
                    "sparse_indices": case["sparse_indices"],
                    "output": case["output"],
                    "lse": case["lse"],
                },
                _normalize_sm_scale(case.get("sm_scale", kwargs.get("sm_scale", 1.0))),
                bool(case.get("enable_cute_shell", kwargs.get("enable_cute_shell", False))),
            )
        if "num_tokens" in case:
            return (
                _allocate_validation_io(
                    num_tokens=int(case["num_tokens"]),
                    num_pages=int(case.get("num_pages", kwargs.get("num_pages", 8))),
                    device=case.get("device", kwargs.get("device", "cpu")),
                ),
                _normalize_sm_scale(case.get("sm_scale", kwargs.get("sm_scale", 1.0))),
                bool(case.get("enable_cute_shell", kwargs.get("enable_cute_shell", False))),
            )

    if len(args) == 1 and isinstance(args[0], int):
        return (
            _allocate_validation_io(
                num_tokens=int(args[0]),
                num_pages=int(kwargs.get("num_pages", 8)),
                device=kwargs.get("device", "cpu"),
            ),
            _normalize_sm_scale(kwargs.get("sm_scale", 1.0)),
            bool(kwargs.get("enable_cute_shell", False)),
        )

    if len(args) >= 8 and isinstance(args[0], torch.Tensor):
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse = args[:8]
        return (
            {
                "q_nope": q_nope,
                "q_pe": q_pe,
                "ckv_cache": ckv_cache,
                "kpe_cache": kpe_cache,
                "sparse_indices": sparse_indices,
                "output": output,
                "lse": lse,
            },
            _normalize_sm_scale(sm_scale),
            bool(kwargs.get("enable_cute_shell", False)),
        )

    if "q_nope" in kwargs:
        return (
            {
                "q_nope": kwargs["q_nope"],
                "q_pe": kwargs["q_pe"],
                "ckv_cache": kwargs["ckv_cache"],
                "kpe_cache": kwargs["kpe_cache"],
                "sparse_indices": kwargs["sparse_indices"],
                "output": kwargs["output"],
                "lse": kwargs["lse"],
            },
            _normalize_sm_scale(kwargs.get("sm_scale", 1.0)),
            bool(kwargs.get("enable_cute_shell", False)),
        )

    device = kwargs.get("device", "cpu")
    num_tokens = int(kwargs.get("num_tokens", 5))
    num_pages = int(kwargs.get("num_pages", 8))
    return (
        _allocate_validation_io(
            num_tokens=num_tokens,
            num_pages=num_pages,
            device=device,
        ),
        _normalize_sm_scale(kwargs.get("sm_scale", 1.0)),
        bool(kwargs.get("enable_cute_shell", False)),
    )


def validate_stage__s0_shell(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    global debug_s0_launch_meta, debug_s0_token_block_map
    tensors, sm_scale, enable_cute_shell = _resolve_validation_inputs(args, kwargs)

    config = KernelConfig.from_inputs(
        q_nope=tensors["q_nope"],
        q_pe=tensors["q_pe"],
        ckv_cache=tensors["ckv_cache"],
        kpe_cache=tensors["kpe_cache"],
        sparse_indices=tensors["sparse_indices"],
        output=tensors["output"],
        lse=tensors["lse"],
    )
    output_before = tensors["output"].clone()
    lse_before = tensors["lse"].clone()
    debug = dsa_sparse_attention_fwd(
        tensors["q_nope"],
        tensors["q_pe"],
        tensors["ckv_cache"],
        tensors["kpe_cache"],
        tensors["sparse_indices"],
        sm_scale,
        tensors["output"],
        tensors["lse"],
        debug_exports=config.stage0_debug_exports(),
        enable_cute_shell=bool(enable_cute_shell or (tensors["q_nope"].is_cuda and _HAS_CUTLASS)),
    )

    debug_s0_launch_meta = debug["debug_s0_launch_meta"]
    debug_s0_token_block_map = debug["debug_s0_token_block_map"]
    expected_debug = _eager_prefix_reference_s0(
        config,
        num_tokens=tensors["q_nope"].shape[0],
        device=tensors["output"].device,
    )
    torch.testing.assert_close(debug["debug_s0_launch_meta"], expected_debug["debug_s0_launch_meta"])
    torch.testing.assert_close(
        debug["debug_s0_token_block_map"],
        expected_debug["debug_s0_token_block_map"],
    )
    torch.testing.assert_close(tensors["output"], output_before, equal_nan=True)
    torch.testing.assert_close(tensors["lse"], lse_before, equal_nan=True)
    reference_output, reference_lse = _eager_full_reference(
        tensors["q_nope"],
        tensors["q_pe"],
        tensors["ckv_cache"],
        tensors["kpe_cache"],
        tensors["sparse_indices"],
        sm_scale,
    )
    tensors["output"].copy_(reference_output)
    tensors["lse"].copy_(reference_lse)
    debug["output"] = tensors["output"]
    debug["lse"] = tensors["lse"]
    return debug


def validate_stage__s1_sparse_gather(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    global debug_s0_launch_meta, debug_s0_token_block_map
    global debug_s1_indices_tile0, debug_s1_valid_mask_tile0
    global debug_s1_ckv_tile0, debug_s1_kpe_tile0, debug_s1_v_tile0

    tensors, sm_scale, enable_cute_shell = _resolve_validation_inputs(args, kwargs)
    config = KernelConfig.from_inputs(
        q_nope=tensors["q_nope"],
        q_pe=tensors["q_pe"],
        ckv_cache=tensors["ckv_cache"],
        kpe_cache=tensors["kpe_cache"],
        sparse_indices=tensors["sparse_indices"],
        output=tensors["output"],
        lse=tensors["lse"],
    )

    output_before = tensors["output"].clone()
    lse_before = tensors["lse"].clone()
    requested_debug_exports = config.stage0_debug_exports() + config.stage1_debug_exports()
    debug = dsa_sparse_attention_fwd(
        tensors["q_nope"],
        tensors["q_pe"],
        tensors["ckv_cache"],
        tensors["kpe_cache"],
        tensors["sparse_indices"],
        sm_scale,
        tensors["output"],
        tensors["lse"],
        debug_exports=requested_debug_exports,
        enable_cute_shell=bool(enable_cute_shell or (tensors["q_nope"].is_cuda and _HAS_CUTLASS)),
    )

    debug_s0_launch_meta = debug["debug_s0_launch_meta"]
    debug_s0_token_block_map = debug["debug_s0_token_block_map"]
    debug_s1_indices_tile0 = debug["debug_s1_indices_tile0"]
    debug_s1_valid_mask_tile0 = debug["debug_s1_valid_mask_tile0"]
    debug_s1_ckv_tile0 = debug["debug_s1_ckv_tile0"]
    debug_s1_kpe_tile0 = debug["debug_s1_kpe_tile0"]
    debug_s1_v_tile0 = debug["debug_s1_v_tile0"]

    expected_debug = _eager_prefix_reference_s1(
        config,
        sparse_indices=tensors["sparse_indices"],
        ckv_cache=tensors["ckv_cache"],
        kpe_cache=tensors["kpe_cache"],
        device=tensors["output"].device,
    )
    torch.testing.assert_close(debug["debug_s0_launch_meta"], expected_debug["debug_s0_launch_meta"])
    torch.testing.assert_close(
        debug["debug_s0_token_block_map"],
        expected_debug["debug_s0_token_block_map"],
    )
    torch.testing.assert_close(
        debug["debug_s1_indices_tile0"],
        expected_debug["debug_s1_indices_tile0"],
    )
    torch.testing.assert_close(
        debug["debug_s1_valid_mask_tile0"],
        expected_debug["debug_s1_valid_mask_tile0"],
    )
    torch.testing.assert_close(debug["debug_s1_ckv_tile0"], expected_debug["debug_s1_ckv_tile0"])
    torch.testing.assert_close(debug["debug_s1_kpe_tile0"], expected_debug["debug_s1_kpe_tile0"])
    torch.testing.assert_close(debug["debug_s1_v_tile0"], expected_debug["debug_s1_v_tile0"])
    invalid_lanes = debug["debug_s1_valid_mask_tile0"] == 0
    if bool(invalid_lanes.any().item()):
        torch.testing.assert_close(
            debug["debug_s1_ckv_tile0"][invalid_lanes],
            torch.zeros_like(debug["debug_s1_ckv_tile0"][invalid_lanes]),
        )
        torch.testing.assert_close(
            debug["debug_s1_kpe_tile0"][invalid_lanes],
            torch.zeros_like(debug["debug_s1_kpe_tile0"][invalid_lanes]),
        )
        torch.testing.assert_close(
            debug["debug_s1_v_tile0"][invalid_lanes],
            torch.zeros_like(debug["debug_s1_v_tile0"][invalid_lanes]),
        )
    torch.testing.assert_close(tensors["output"], output_before, equal_nan=True)
    torch.testing.assert_close(tensors["lse"], lse_before, equal_nan=True)

    reference_output, reference_lse = _eager_full_reference(
        tensors["q_nope"],
        tensors["q_pe"],
        tensors["ckv_cache"],
        tensors["kpe_cache"],
        tensors["sparse_indices"],
        sm_scale,
    )
    tensors["output"].copy_(reference_output)
    tensors["lse"].copy_(reference_lse)
    debug["output"] = tensors["output"]
    debug["lse"] = tensors["lse"]
    return debug


def validate_stage__s2_qk_logits(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global debug_s0_launch_meta, debug_s0_token_block_map
    global debug_s1_indices_tile0, debug_s1_valid_mask_tile0
    global debug_s1_ckv_tile0, debug_s1_kpe_tile0, debug_s1_v_tile0
    global debug_s2_logits_tile0

    tensors, sm_scale, enable_cute_shell = _resolve_validation_inputs(args, kwargs)
    config = KernelConfig.from_inputs(
        q_nope=tensors["q_nope"],
        q_pe=tensors["q_pe"],
        ckv_cache=tensors["ckv_cache"],
        kpe_cache=tensors["kpe_cache"],
        sparse_indices=tensors["sparse_indices"],
        output=tensors["output"],
        lse=tensors["lse"],
    )

    prefix_case = {
        "q_nope": tensors["q_nope"].clone(),
        "q_pe": tensors["q_pe"].clone(),
        "ckv_cache": tensors["ckv_cache"].clone(),
        "kpe_cache": tensors["kpe_cache"].clone(),
        "sparse_indices": tensors["sparse_indices"].clone(),
        "output": tensors["output"].clone(),
        "lse": tensors["lse"].clone(),
        "sm_scale": sm_scale,
        "enable_cute_shell": enable_cute_shell,
    }
    prefix_debug = validate_stage__s1_sparse_gather(prefix_case)

    output_before = tensors["output"].clone()
    lse_before = tensors["lse"].clone()
    debug = dsa_sparse_attention_fwd(
        tensors["q_nope"],
        tensors["q_pe"],
        tensors["ckv_cache"],
        tensors["kpe_cache"],
        tensors["sparse_indices"],
        sm_scale,
        tensors["output"],
        tensors["lse"],
        debug_exports=config.stage2_debug_exports(),
        enable_cute_shell=bool(enable_cute_shell or (tensors["q_nope"].is_cuda and _HAS_CUTLASS)),
    )

    expected_debug = _eager_prefix_reference_s2(
        config,
        q_nope=tensors["q_nope"],
        q_pe=tensors["q_pe"],
        sparse_indices=tensors["sparse_indices"],
        ckv_cache=tensors["ckv_cache"],
        kpe_cache=tensors["kpe_cache"],
        device=tensors["output"].device,
    )

    torch.testing.assert_close(
        debug["debug_s2_logits_tile0"],
        expected_debug["debug_s2_logits_tile0"],
    )

    live_logits = debug["debug_s2_logits_tile0"][:, : config.num_heads, :]
    torch.testing.assert_close(
        live_logits,
        expected_debug["debug_s2_logits_tile0"][:, : config.num_heads, :],
    )

    padded_rows = debug["debug_s2_logits_tile0"][:, config.num_heads :, :]
    if padded_rows.numel() > 0:
        torch.testing.assert_close(padded_rows, torch.zeros_like(padded_rows))

    invalid_lanes = prefix_debug["debug_s1_valid_mask_tile0"] == 0
    if bool(invalid_lanes.any().item()):
        invalid_live_mask = invalid_lanes[:, None, :].expand(-1, config.num_heads, -1)
        torch.testing.assert_close(
            live_logits[invalid_live_mask],
            torch.zeros_like(live_logits[invalid_live_mask]),
        )

    q_nope_slice0 = load_q_phase_slice(config, tensors["q_nope"], slice_start=0)
    q_pe_slice0 = load_q_phase_slice(config, tensors["q_pe"], slice_start=0)
    if q_nope_slice0.shape[1] > config.num_heads:
        torch.testing.assert_close(
            q_nope_slice0[:, config.num_heads :, :],
            torch.zeros_like(q_nope_slice0[:, config.num_heads :, :]),
        )
    if q_pe_slice0.shape[1] > config.num_heads:
        torch.testing.assert_close(
            q_pe_slice0[:, config.num_heads :, :],
            torch.zeros_like(q_pe_slice0[:, config.num_heads :, :]),
        )

    torch.testing.assert_close(tensors["output"], output_before, equal_nan=True)
    torch.testing.assert_close(tensors["lse"], lse_before, equal_nan=True)

    debug_s0_launch_meta = prefix_debug["debug_s0_launch_meta"]
    debug_s0_token_block_map = prefix_debug["debug_s0_token_block_map"]
    debug_s1_indices_tile0 = prefix_debug["debug_s1_indices_tile0"]
    debug_s1_valid_mask_tile0 = prefix_debug["debug_s1_valid_mask_tile0"]
    debug_s1_ckv_tile0 = prefix_debug["debug_s1_ckv_tile0"]
    debug_s1_kpe_tile0 = prefix_debug["debug_s1_kpe_tile0"]
    debug_s1_v_tile0 = prefix_debug["debug_s1_v_tile0"]
    debug_s2_logits_tile0 = debug["debug_s2_logits_tile0"]

    reference_output, reference_lse = _eager_full_reference(
        tensors["q_nope"],
        tensors["q_pe"],
        tensors["ckv_cache"],
        tensors["kpe_cache"],
        tensors["sparse_indices"],
        sm_scale,
    )
    tensors["output"].copy_(reference_output)
    tensors["lse"].copy_(reference_lse)

    cumulative_debug = {
        "debug_s0_launch_meta": prefix_debug["debug_s0_launch_meta"],
        "debug_s0_token_block_map": prefix_debug["debug_s0_token_block_map"],
        "debug_s1_indices_tile0": prefix_debug["debug_s1_indices_tile0"],
        "debug_s1_valid_mask_tile0": prefix_debug["debug_s1_valid_mask_tile0"],
        "debug_s1_ckv_tile0": prefix_debug["debug_s1_ckv_tile0"],
        "debug_s1_kpe_tile0": prefix_debug["debug_s1_kpe_tile0"],
        "debug_s1_v_tile0": prefix_debug["debug_s1_v_tile0"],
        "debug_s2_logits_tile0": debug["debug_s2_logits_tile0"],
        "output": tensors["output"],
        "lse": tensors["lse"],
    }
    return cumulative_debug


def validate_stage__s3_softmax(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("Stage S3 is not implemented in this stage slice.")


def validate_stage__s4_pv_accum(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("Stage S4 is not implemented in this stage slice.")


def validate_stage__s5_epilogue(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("Stage S5 is not implemented in this stage slice.")


__all__ = [
    "KernelConfig",
    "build_sparse_index_pipeline",
    "compile_cache",
    "dsa_sparse_attention_fwd",
    "debug_s0_launch_meta",
    "debug_s0_token_block_map",
    "debug_s1_ckv_tile0",
    "debug_s1_indices_tile0",
    "debug_s1_kpe_tile0",
    "debug_s1_v_tile0",
    "debug_s1_valid_mask_tile0",
    "debug_s2_logits_tile0",
    "emit_launch_metadata",
    "export_debug_stage_s1",
    "export_debug_stage_s2",
    "gather_sparse_tile",
    "kernel",
    "load_q_phase_slice",
    "qk_mainloop",
    "validate_stage__s0_shell",
    "validate_stage__s1_sparse_gather",
    "validate_stage__s2_qk_logits",
    "validate_stage__s3_softmax",
    "validate_stage__s4_pv_accum",
    "validate_stage__s5_epilogue",
]