import json
import math
from typing import Any, Final, Sequence

import torch


NUM_QO_HEADS: Final[int] = 16
HEAD_DIM_CKV: Final[int] = 512
HEAD_DIM_KPE: Final[int] = 64
PAGE_SIZE: Final[int] = 64
TOPK: Final[int] = 2048
MAX_PAGES_PER_TOKEN: Final[int] = TOPK // PAGE_SIZE
DEFAULT_SM_SCALE: Final[float] = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
BLOCK_THREADS: Final[int] = 256
WARP_SIZE: Final[int] = 32
WARPS_PER_CTA: Final[int] = BLOCK_THREADS // WARP_SIZE
B200_NUM_SMS: Final[int] = 148
ROUND0_CLUSTER_SHAPE: Final[tuple[int, int, int]] = (1, 1, 1)


class PageDenseSupportResult:
    __slots__ = ("page_dense_ok", "valid_page_count", "page_ids", "error")

    def __init__(
        self,
        page_dense_ok: bool,
        valid_page_count: int,
        page_ids: tuple[int, ...],
        error: str | None = None,
    ) -> None:
        self.page_dense_ok = page_dense_ok
        self.valid_page_count = valid_page_count
        self.page_ids = page_ids
        self.error = error

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "page_dense_ok": self.page_dense_ok,
            "valid_page_count": self.valid_page_count,
            "page_ids": list(self.page_ids),
            "error": self.error,
        }


class ReferenceOutputs:
    __slots__ = ("output", "lse", "valid_page_count", "page_ids")

    def __init__(
        self,
        output: torch.Tensor,
        lse: torch.Tensor,
        valid_page_count: int,
        page_ids: tuple[int, ...],
    ) -> None:
        self.output = output
        self.lse = lse
        self.valid_page_count = valid_page_count
        self.page_ids = page_ids


class PersistentLaunchConfig:
    __slots__ = ("block_threads", "grid_x", "num_sms", "cluster_shape")

    def __init__(
        self,
        *,
        block_threads: int,
        grid_x: int,
        num_sms: int,
        cluster_shape: tuple[int, int, int],
    ) -> None:
        self.block_threads = block_threads
        self.grid_x = grid_x
        self.num_sms = num_sms
        self.cluster_shape = cluster_shape

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "block_threads": self.block_threads,
            "grid_x": self.grid_x,
            "num_sms": self.num_sms,
            "cluster_shape": list(self.cluster_shape),
            "warps_per_cta": WARPS_PER_CTA,
        }


def _tensor_preview(
    tensor: torch.Tensor,
    *,
    limit: int,
    logical_layout: str,
) -> dict[str, Any]:
    flat = tensor.reshape(-1)
    preview = flat[:limit].to(torch.float32).cpu().tolist()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "layout": logical_layout,
        "preview": preview,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _emit_validation_block(source: str, name: str, payload: Any) -> None:
    print(f"[{source}] {name}: BEGIN")
    print(json.dumps(payload, indent=2, default=_json_default, allow_nan=True))
    print(f"[{source}] {name}: END")


def make_persistent_launch_config(
    num_tokens: int,
    *,
    num_sms: int = B200_NUM_SMS,
) -> PersistentLaunchConfig:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, but received {num_tokens}.")
    grid_x = min(num_tokens, num_sms)
    return PersistentLaunchConfig(
        block_threads=BLOCK_THREADS,
        grid_x=grid_x,
        num_sms=num_sms,
        cluster_shape=ROUND0_CLUSTER_SHAPE,
    )


def build_persistent_token_schedule(
    num_tokens: int,
    *,
    cta_idx: int,
    grid_x: int,
) -> list[int]:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, but received {num_tokens}.")
    if grid_x <= 0:
        return []
    if not (0 <= cta_idx < grid_x):
        return []
    return list(range(cta_idx, num_tokens, grid_x))


def make_stage1_scheduler_debug(
    num_tokens: int,
    *,
    num_sms: int = B200_NUM_SMS,
    preview_limit: int = 32,
) -> tuple[PersistentLaunchConfig, list[int]]:
    launch_cfg = make_persistent_launch_config(num_tokens, num_sms=num_sms)
    token_schedule = build_persistent_token_schedule(
        num_tokens,
        cta_idx=0,
        grid_x=launch_cfg.grid_x,
    )
    return launch_cfg, token_schedule[:preview_limit]


def build_page_dense_sparse_indices(page_ids: Sequence[int]) -> torch.Tensor:
    if len(page_ids) > MAX_PAGES_PER_TOKEN:
        raise ValueError(
            f"Received {len(page_ids)} page ids, but round-0 supports at most "
            f"{MAX_PAGES_PER_TOKEN} full pages per token."
        )

    indices = torch.full((TOPK,), -1, dtype=torch.int32)
    for slot, page_id in enumerate(page_ids):
        if page_id < 0:
            raise ValueError("Page ids must be non-negative for valid pages.")
        base_token = page_id * PAGE_SIZE
        token_ids = torch.arange(
            base_token,
            base_token + PAGE_SIZE,
            dtype=torch.int32,
        )
        start = slot * PAGE_SIZE
        end = start + PAGE_SIZE
        indices[start:end] = token_ids
    return indices


def check_round0_page_dense_support(
    sparse_indices_row: torch.Tensor,
    *,
    num_pages: int | None = None,
) -> PageDenseSupportResult:
    row = torch.as_tensor(sparse_indices_row, dtype=torch.int64).reshape(-1)

    if row.numel() != TOPK:
        return PageDenseSupportResult(
            page_dense_ok=False,
            valid_page_count=0,
            page_ids=(),
            error=(
                f"Expected sparse_indices row of length {TOPK}, "
                f"but received {row.numel()} elements."
            ),
        )

    page_ids: list[int] = []
    seen_page_ids: set[int] = set()
    seen_invalid_tail = False

    for page_slot in range(MAX_PAGES_PER_TOKEN):
        start = page_slot * PAGE_SIZE
        end = start + PAGE_SIZE
        page_block = row[start:end]

        if torch.all(page_block.eq(-1)):
            seen_invalid_tail = True
            continue

        if bool(torch.any(page_block.eq(-1)).item()):
            return PageDenseSupportResult(
                page_dense_ok=False,
                valid_page_count=len(page_ids),
                page_ids=tuple(page_ids),
                error=(
                    f"Page slot {page_slot} contains a partial invalid tail. "
                    "Round-0 only supports whole valid pages or whole -1 pages."
                ),
            )

        if seen_invalid_tail:
            return PageDenseSupportResult(
                page_dense_ok=False,
                valid_page_count=len(page_ids),
                page_ids=tuple(page_ids),
                error=(
                    f"Encountered a valid page after the invalid tail at page slot "
                    f"{page_slot}. Valid pages must precede all invalid pages."
                ),
            )

        first_token = int(page_block[0].item())
        if first_token < 0:
            return PageDenseSupportResult(
                page_dense_ok=False,
                valid_page_count=len(page_ids),
                page_ids=tuple(page_ids),
                error=f"Page slot {page_slot} starts with a negative token id.",
            )

        page_id = first_token // PAGE_SIZE
        if num_pages is not None and not (0 <= page_id < num_pages):
            return PageDenseSupportResult(
                page_dense_ok=False,
                valid_page_count=len(page_ids),
                page_ids=tuple(page_ids),
                error=(
                    f"Page id {page_id} at page slot {page_slot} is out of range for "
                    f"num_pages={num_pages}."
                ),
            )

        expected_block = torch.arange(
            page_id * PAGE_SIZE,
            (page_id + 1) * PAGE_SIZE,
            dtype=torch.int64,
            device=row.device,
        )
        if not torch.equal(page_block, expected_block):
            return PageDenseSupportResult(
                page_dense_ok=False,
                valid_page_count=len(page_ids),
                page_ids=tuple(page_ids),
                error=(
                    f"Page slot {page_slot} is not page-dense. Expected token ids "
                    f"{int(expected_block[0].item())}..{int(expected_block[-1].item())}."
                ),
            )

        if page_id in seen_page_ids:
            return PageDenseSupportResult(
                page_dense_ok=False,
                valid_page_count=len(page_ids),
                page_ids=tuple(page_ids),
                error=f"Duplicate page id {page_id} encountered at page slot {page_slot}.",
            )

        seen_page_ids.add(page_id)
        page_ids.append(page_id)

    return PageDenseSupportResult(
        page_dense_ok=True,
        valid_page_count=len(page_ids),
        page_ids=tuple(page_ids),
        error=None,
    )


def _make_pattern_tensor(
    shape: Sequence[int],
    *,
    modulus: int,
    scale: float,
    offset: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    numel = math.prod(shape)
    base = torch.arange(numel, dtype=torch.float32)
    centered = (torch.remainder(base, modulus) - (modulus // 2)) * scale + offset
    return centered.reshape(*shape).to(dtype)


def make_prefix_validation_inputs() -> dict[str, Any]:
    token_page_ids = (1, 3)
    num_pages = 5

    q_nope = _make_pattern_tensor(
        (1, NUM_QO_HEADS, HEAD_DIM_CKV),
        modulus=23,
        scale=1.0 / 128.0,
    )
    q_pe = _make_pattern_tensor(
        (1, NUM_QO_HEADS, HEAD_DIM_KPE),
        modulus=19,
        scale=1.0 / 96.0,
        offset=1.0 / 192.0,
    )
    ckv_cache = _make_pattern_tensor(
        (num_pages, PAGE_SIZE, HEAD_DIM_CKV),
        modulus=29,
        scale=1.0 / 256.0,
        offset=-1.0 / 512.0,
    )
    kpe_cache = _make_pattern_tensor(
        (num_pages, PAGE_SIZE, HEAD_DIM_KPE),
        modulus=17,
        scale=1.0 / 160.0,
        offset=1.0 / 320.0,
    )
    sparse_indices = build_page_dense_sparse_indices(token_page_ids).reshape(1, TOPK)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": torch.tensor(DEFAULT_SM_SCALE, dtype=torch.float32),
    }


def _reference_single_token(
    q_nope_token: torch.Tensor,
    q_pe_token: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices_row: torch.Tensor,
    sm_scale: float,
) -> ReferenceOutputs:
    support = check_round0_page_dense_support(
        sparse_indices_row,
        num_pages=int(ckv_cache.shape[0]),
    )
    if not support.page_dense_ok:
        raise ValueError(
            "Round-0 reference path only accepts page-dense sparse indices: "
            f"{support.error}"
        )

    if support.valid_page_count == 0:
        output = torch.zeros(
            (NUM_QO_HEADS, HEAD_DIM_CKV),
            dtype=torch.bfloat16,
        )
        lse = torch.full((NUM_QO_HEADS,), float("-inf"), dtype=torch.float32)
        return ReferenceOutputs(
            output=output,
            lse=lse,
            valid_page_count=0,
            page_ids=support.page_ids,
        )

    valid_token_count = support.valid_page_count * PAGE_SIZE
    token_ids = sparse_indices_row[:valid_token_count].to(torch.int64)

    flat_ckv = ckv_cache.reshape(-1, HEAD_DIM_CKV).to(torch.float32)
    flat_kpe = kpe_cache.reshape(-1, HEAD_DIM_KPE).to(torch.float32)
    selected_ckv = flat_ckv.index_select(0, token_ids)
    selected_kpe = flat_kpe.index_select(0, token_ids)

    q_nope_f32 = q_nope_token.to(torch.float32)
    q_pe_f32 = q_pe_token.to(torch.float32)
    logits = sm_scale * (
        torch.matmul(q_nope_f32, selected_ckv.transpose(0, 1))
        + torch.matmul(q_pe_f32, selected_kpe.transpose(0, 1))
    )
    probs = torch.softmax(logits, dim=-1)
    output = torch.matmul(probs, selected_ckv).to(torch.bfloat16)
    lse = torch.logsumexp(logits, dim=-1) / math.log(2.0)

    return ReferenceOutputs(
        output=output,
        lse=lse.to(torch.float32),
        valid_page_count=support.valid_page_count,
        page_ids=support.page_ids,
    )


def pytorch_reference_sparse_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, list[ReferenceOutputs]]:
    q_nope = torch.as_tensor(q_nope)
    q_pe = torch.as_tensor(q_pe)
    ckv_cache = torch.as_tensor(ckv_cache)
    kpe_cache = torch.as_tensor(kpe_cache)
    sparse_indices = torch.as_tensor(sparse_indices)
    sm_scale_value = float(torch.as_tensor(sm_scale, dtype=torch.float32).item())

    token_outputs: list[ReferenceOutputs] = []
    output_rows: list[torch.Tensor] = []
    lse_rows: list[torch.Tensor] = []
    for token_idx in range(int(q_nope.shape[0])):
        ref = _reference_single_token(
            q_nope[token_idx],
            q_pe[token_idx],
            ckv_cache,
            kpe_cache,
            sparse_indices[token_idx],
            sm_scale_value,
        )
        token_outputs.append(ref)
        output_rows.append(ref.output)
        lse_rows.append(ref.lse)

    output = torch.stack(output_rows, dim=0)
    lse = torch.stack(lse_rows, dim=0)
    return output, lse, token_outputs


def persistent_reference_sparse_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
    *,
    num_sms: int = B200_NUM_SMS,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[ReferenceOutputs],
    PersistentLaunchConfig,
    list[int],
]:
    q_nope = torch.as_tensor(q_nope)
    q_pe = torch.as_tensor(q_pe)
    ckv_cache = torch.as_tensor(ckv_cache)
    kpe_cache = torch.as_tensor(kpe_cache)
    sparse_indices = torch.as_tensor(sparse_indices)
    sm_scale_value = float(torch.as_tensor(sm_scale, dtype=torch.float32).item())

    num_tokens = int(q_nope.shape[0])
    launch_cfg, cta0_schedule_preview = make_stage1_scheduler_debug(
        num_tokens,
        num_sms=num_sms,
    )

    output = torch.empty(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
    )
    lse = torch.empty((num_tokens, NUM_QO_HEADS), dtype=torch.float32)
    per_token_outputs: dict[int, ReferenceOutputs] = {}

    for cta_idx in range(launch_cfg.grid_x):
        token_schedule = build_persistent_token_schedule(
            num_tokens,
            cta_idx=cta_idx,
            grid_x=launch_cfg.grid_x,
        )
        for token_idx in token_schedule:
            ref = _reference_single_token(
                q_nope[token_idx],
                q_pe[token_idx],
                ckv_cache,
                kpe_cache,
                sparse_indices[token_idx],
                sm_scale_value,
            )
            per_token_outputs[token_idx] = ref
            output[token_idx].copy_(ref.output)
            lse[token_idx].copy_(ref.lse)

    token_outputs = [per_token_outputs[token_idx] for token_idx in range(num_tokens)]
    return output, lse, token_outputs, launch_cfg, cta0_schedule_preview


def round0_dispatch_supported(
    sparse_indices: torch.Tensor,
    *,
    num_pages: int | None = None,
) -> bool:
    sparse_indices = torch.as_tensor(sparse_indices)
    if sparse_indices.ndim == 1:
        sparse_indices = sparse_indices.reshape(1, -1)

    for token_idx in range(int(sparse_indices.shape[0])):
        support = check_round0_page_dense_support(
            sparse_indices[token_idx],
            num_pages=num_pages,
        )
        if not support.page_dense_ok:
            return False
    return True


def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages = int(torch.as_tensor(ckv_cache).shape[0])
    sparse_indices_tensor = torch.as_tensor(sparse_indices)
    if not round0_dispatch_supported(sparse_indices_tensor, num_pages=num_pages):
        raise ValueError(
            "kernel_0 round-0 only supports page-dense sparse_indices with whole-page "
            "contiguous token ids and a trailing all--1 tail."
        )

    output, lse, _, _, _ = persistent_reference_sparse_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices_tensor,
        sm_scale=sm_scale,
    )
    return output, lse


def prefix_validation_harness(
    q_nope: torch.Tensor | None = None,
    q_pe: torch.Tensor | None = None,
    ckv_cache: torch.Tensor | None = None,
    kpe_cache: torch.Tensor | None = None,
    sparse_indices: torch.Tensor | None = None,
    sm_scale: torch.Tensor | float | None = None,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> dict[str, Any]:
    if q_nope is None:
        inputs = make_prefix_validation_inputs()
    else:
        if any(
            value is None
            for value in (q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
        ):
            raise ValueError(
                "prefix_validation_harness expected either no arguments for the "
                "built-in debug case or the full kernel-style input tuple."
            )
        inputs = {
            "q_nope": torch.as_tensor(q_nope),
            "q_pe": torch.as_tensor(q_pe),
            "ckv_cache": torch.as_tensor(ckv_cache),
            "kpe_cache": torch.as_tensor(kpe_cache),
            "sparse_indices": torch.as_tensor(sparse_indices),
            "sm_scale": torch.as_tensor(sm_scale, dtype=torch.float32),
        }

    sparse_row = inputs["sparse_indices"][0]
    support = check_round0_page_dense_support(
        sparse_row,
        num_pages=int(inputs["ckv_cache"].shape[0]),
    )
    ref_output, ref_lse, token_outputs = pytorch_reference_sparse_attention(**inputs)
    kernel_output, kernel_lse, _, launch_cfg, cta0_schedule_preview = (
        persistent_reference_sparse_attention(**inputs)
    )
    if output is not None:
        output.copy_(kernel_output)
    if lse is not None:
        lse.copy_(kernel_lse)
    token0 = token_outputs[0]

    ref_valid_page_count = token0.valid_page_count
    ref_page_ids = list(token0.page_ids)
    ref_output_preview = _tensor_preview(
        ref_output,
        limit=100,
        logical_layout="row-major [token, head, dim]",
    )
    ref_lse_token0 = ref_lse[0].to(torch.float32).cpu().tolist()
    contract_page_dense_ok = support.page_dense_ok

    _emit_validation_block("PyTorch", "ref.valid_page_count", ref_valid_page_count)
    _emit_validation_block("PyTorch", "ref.page_ids", ref_page_ids)
    _emit_validation_block("PyTorch", "ref.output_preview", ref_output_preview)
    _emit_validation_block("PyTorch", "ref.lse", ref_lse_token0)
    _emit_validation_block("CuTeDSL", "contract.page_dense_ok", contract_page_dense_ok)
    _emit_validation_block("CuTeDSL", "cfg.block_threads", launch_cfg.block_threads)
    _emit_validation_block("CuTeDSL", "cfg.grid_x", launch_cfg.grid_x)
    _emit_validation_block(
        "CuTeDSL",
        "sched.token_schedule_preview",
        cta0_schedule_preview,
    )

    return {
        "inputs": {
            "q_nope_shape": list(inputs["q_nope"].shape),
            "q_pe_shape": list(inputs["q_pe"].shape),
            "ckv_cache_shape": list(inputs["ckv_cache"].shape),
            "kpe_cache_shape": list(inputs["kpe_cache"].shape),
            "sparse_indices_shape": list(inputs["sparse_indices"].shape),
            "sm_scale": float(inputs["sm_scale"].item()),
        },
        "ref": {
            "valid_page_count": ref_valid_page_count,
            "page_ids": ref_page_ids,
            "output_preview": ref_output_preview,
            "lse": ref_lse_token0,
        },
        "contract": contract_page_dense_ok,
        "contract_details": support.to_debug_dict(),
        "cfg": launch_cfg.to_debug_dict(),
        "sched": {
            "token_schedule_preview": cta0_schedule_preview,
        },
    }


if __name__ == "__main__":
    prefix_validation_harness()