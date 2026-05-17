"""
DSA Sparse Attention Kernel — fused 3-warpgroup CuTeDSL implementation.

Uses CuTeDSL tcgen05.mma (Blackwell UTCMMA) with three cooperating warpgroups:
    WG0 (Load):    TMA gather4 for Q, CKV, KPE, V staging into operand SMEM
    WG1 (MMA):     QK GEMM (S = Q_nope @ Kc^T + Q_pe @ Kp^T), then PV MMA
    WG2 (Softmax): Score TMEM -> softmax -> P staging, then output epilogue

Entry points:
    run(...) -> (output, lse)
    kernel(..., output, lse) -> writes destination tensors in place

Env vars:
    FLASHMLA_DSA_VALIDATE_FUSED=assert  — compare against dense fallback
    FLASHMLA_DSA_USE_FALLBACK=1         — override with dense fallback output
"""

import math
import os
from types import SimpleNamespace
import torch

# Constants
TOPK = 2048
CHUNK_SIZE = 64
NUM_Q_HEADS = 16
CKV_DIM = 512
KPE_DIM = 64
PAGE_SIZE = 64
LOG2E = 1.0 / math.log(2.0)
NUM_SPLITS = TOPK // CHUNK_SIZE  # 32
M_BLOCK = 64  # padded query dim (16 real -> 64 for UTCMMA minimum)
N_BLOCK = 64  # KV chunk size
WARP_SIZE = 32
WARPGROUP_THREADS = 128
WARPS_PER_WARPGROUP = WARPGROUP_THREADS // WARP_SIZE
REBUILD_THREADS_PER_CTA = 3 * WARPGROUP_THREADS  # WG0(load) + WG1(mma) + WG2(softmax)
NUM_CKV_K_TILES = CKV_DIM // N_BLOCK  # 8 — K-dimension tiles for Q_nope @ CKV^T
OPERAND_PIPE_STAGES = 2
V_PIPE_STAGES = 2
# Split-shell QK debug gate:
# limit CKV accumulation to a prefix of tiles so we can isolate whether the
# bug shows up on tile 0 or only when advancing across stages.
_DEBUG_SPLIT_QK_TILE_LIMIT = NUM_CKV_K_TILES
_DEBUG_SPLIT_INCLUDE_KPE = True
_DEBUG_SPLIT_CKV_TILE_COUNT = max(1, min(NUM_CKV_K_TILES, _DEBUG_SPLIT_QK_TILE_LIMIT))

# CuTeDSL imports (lazy, only on GPU)
_cutedsl_kernel = None
_split_shell_launcher = None
_sm100_dsl_helpers = None
_split_validation_emitted = False
_compiled_split_shell_cache = {}
_fused_output_scratch_cache = {}


def _get_sm100_dsl_helpers():
    """Create and cache reusable SM100 CuTeDSL helper ops for this module."""
    global _sm100_dsl_helpers
    if _sm100_dsl_helpers is not None:
        return _sm100_dsl_helpers

    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
    from cutlass.cute.nvgpu import cpasync
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.pipeline as pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass import Float32, BFloat16, Int32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import dsl_user_op
    import cuda.bindings.driver as cuda

    num_threads = 128

    @dsl_user_op
    def _load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
        dst = cute.make_rmem_tensor_like(src, src.element_type, loc=loc, ip=ip)
        cute.autovec_copy(src, dst, loc=loc, ip=ip)
        return dst

    @dsl_user_op
    def _get_tma_desc_addr(
        tma_atom: cute.CopyAtom, *, loc=None, ip=None
    ) -> cute.Pointer:
        exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(
            tma_atom._trait.value, loc=loc, ip=ip
        )
        tma_desc_ptr_type = ir.Type.parse(
            "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>"
        )
        return _cute_nvgpu_ir.get_tma_desc_addr(
            tma_desc_ptr_type, exec_atom, loc=loc, ip=ip
        )

    @dsl_user_op
    def _tma_gather4_load(
        tma_desc_ptr: cute.Pointer,
        dst_smem_ptr: cute.Pointer,
        mbarrier_ptr: cute.Pointer,
        col_idx: Int32,
        row0: Int32,
        row1: Int32,
        row2: Int32,
        row3: Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        desc_addr = tma_desc_ptr.toint(loc=loc, ip=ip).ir_value()
        dst_addr = dst_smem_ptr.toint(loc=loc, ip=ip).ir_value()
        mbar_addr = mbarrier_ptr.toint(loc=loc, ip=ip).ir_value()
        ptx = (
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
            "mbarrier::complete_tx::bytes.cta_group::1 "
            "[$0], [$1, {$2, $3, $4, $5, $6}], [$7];"
        )
        llvm.inline_asm(
            None,
            [
                dst_addr,
                desc_addr,
                Int32(col_idx).ir_value(),
                Int32(row0).ir_value(),
                Int32(row1).ir_value(),
                Int32(row2).ir_value(),
                Int32(row3).ir_value(),
                mbar_addr,
            ],
            ptx,
            "r,l,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def _make_swizzled_mn_view(s_dst: cute.Tensor, dst_swizzle) -> cute.Tensor:
        dst_layout = cute.make_layout(
            (
                (s_dst.shape[0][0], s_dst.shape[1]),
                (s_dst.shape[0][1], s_dst.shape[2]),
            ),
            stride=(
                (s_dst.stride[0][0], s_dst.stride[1]),
                (s_dst.stride[0][1], s_dst.stride[2]),
            ),
        )
        return cute.make_tensor(
            cute.recast_ptr(s_dst.iterator, swizzle_=None),
            cute.make_composed_layout(dst_swizzle, 0, dst_layout),
        )

    @cute.jit
    def _make_rowmajor_mn_view(
        s_src_row: cute.Tensor, s_dst: cute.Tensor
    ) -> cute.Tensor:
        s_src = s_src_row[None, None, 0]
        src_layout = cute.make_layout(
            (
                (s_dst.shape[0][0], s_dst.shape[1]),
                (s_dst.shape[0][1], s_dst.shape[2]),
            ),
            stride=(
                (
                    s_dst.shape[0][1] * s_dst.shape[2],
                    s_dst.shape[0][0] * s_dst.shape[0][1] * s_dst.shape[2],
                ),
                (1, s_dst.shape[0][1]),
            ),
        )
        return cute.make_tensor(s_src.iterator, src_layout)

    @cute.jit
    def _make_rowmajor_operand_view(
        s_src_row: cute.Tensor, s_dst: cute.Tensor
    ) -> cute.Tensor:
        s_src = s_src_row[None, None, 0]
        src_layout = cute.make_layout(
            (s_dst.shape[0], s_dst.shape[1], s_dst.shape[2]),
            stride=(
                (s_dst.shape[0][1] * s_dst.shape[2], 1),
                s_dst.shape[0][0] * s_dst.shape[0][1] * s_dst.shape[2],
                s_dst.shape[0][1],
            ),
        )
        return cute.make_tensor(s_src.iterator, src_layout)

    @cute.jit
    def _make_gather_rowgroup_view(s_src: cute.Tensor) -> cute.Tensor:
        grouped_layout = cute.make_layout(
            ((4, CHUNK_SIZE // 4), N_BLOCK, 1),
            # Gather4 writes each 4-row group as four contiguous row-major rows:
            # [row0_col0..63, row1_col0..63, row2_col0..63, row3_col0..63].
            stride=((N_BLOCK, 4 * N_BLOCK), 1, CHUNK_SIZE * N_BLOCK),
        )
        return cute.make_tensor(s_src.iterator, grouped_layout)

    @cute.jit
    def _unpack_gather4_stage(
        s_src_packed: cute.Tensor,
        s_dst_rowmajor: cute.Tensor,
        tidx: Int32,
    ) -> None:
        for elem in cutlass.range(
            tidx, CHUNK_SIZE * N_BLOCK, WARPGROUP_THREADS, unroll=1
        ):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            s_dst_rowmajor[row, col, 0] = s_src_packed[row, col, 0]

    @cute.jit
    def _mask_invalid_gather_rows(
        s_stage: cute.Tensor,
        s_valid: cute.Tensor,
        tidx: Int32,
    ) -> None:
        for elem in cutlass.range(
            tidx, CHUNK_SIZE * N_BLOCK, WARPGROUP_THREADS, unroll=1
        ):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            if s_valid[row] == Int32(0):
                s_stage[row, col, 0] = BFloat16(0.0)

    @cute.jit
    def _transpose_gather4_stage_masked(
        s_src_packed: cute.Tensor,
        s_dst_rowmajor: cute.Tensor,
        s_valid: cute.Tensor,
        tidx: Int32,
    ) -> None:
        for elem in cutlass.range(
            tidx, CHUNK_SIZE * N_BLOCK, WARPGROUP_THREADS, unroll=1
        ):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            value = BFloat16(0.0)
            if s_valid[col] != Int32(0):
                value = s_src_packed[col, row, 0]
            s_dst_rowmajor[row, col, 0] = value

    @cute.jit
    def _copy_rowmajor_stage_to_operand_a(
        s_src_row: cute.Tensor,
        s_dst: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
    ) -> None:
        _ = tiled_mma
        smem_copy_atom_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            BFloat16,
            num_bits_per_copy=16,
        )
        smem_tiled_copy = cute.make_cotiled_copy(
            smem_copy_atom_bf16,
            cute.make_ordered_layout((num_threads, 1), (1, 0)),
            s_dst.layout,
        )
        thr_smem_copy = smem_tiled_copy.get_slice(tidx)
        s_src = _make_rowmajor_operand_view(s_src_row, s_dst)
        tSs = thr_smem_copy.partition_S(s_src)
        tDs = thr_smem_copy.partition_D(s_dst)
        cute.copy(smem_tiled_copy, tSs, tDs)

    @cute.jit
    def _copy_rowmajor_stage_to_operand_b(
        s_src_row: cute.Tensor,
        s_dst: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
    ) -> None:
        _ = tiled_mma
        smem_copy_atom_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            BFloat16,
            num_bits_per_copy=16,
        )
        smem_tiled_copy = cute.make_cotiled_copy(
            smem_copy_atom_bf16,
            cute.make_ordered_layout((num_threads, 1), (1, 0)),
            s_dst.layout,
        )
        thr_smem_copy = smem_tiled_copy.get_slice(tidx)
        s_src = _make_rowmajor_operand_view(s_src_row, s_dst)
        tSs = thr_smem_copy.partition_S(s_src)
        tDs = thr_smem_copy.partition_D(s_dst)
        cute.copy(smem_tiled_copy, tSs, tDs)

    @cute.jit
    def _copy_rowmajor_stage_to_operand_qk_b(
        s_src_row: cute.Tensor,
        s_dst: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
    ) -> None:
        _copy_rowmajor_stage_to_operand_b(s_src_row, s_dst, tiled_mma, tidx)

    @cute.jit
    def _make_gather4_copy_fn(
        tma_atom: cute.CopyAtom,
        s_dst: cute.Tensor,
        s_idx: cute.Tensor,
        s_group_mask: cute.Tensor,
        warp_idx: Int32,
    ):
        tile_m = cute.size(s_idx, mode=[0])
        tile_k = cute.size(s_dst[None, None, 0]) // tile_m
        copy_idx = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128
            ),
            cute.make_layout(1),
            cute.make_layout(4),
        )
        warp_copy_idx = copy_idx.get_slice(warp_idx)
        tSR_sIdx = warp_copy_idx.partition_S(s_idx)
        tSR_sDst = warp_copy_idx.partition_S(_make_gather_rowgroup_view(s_dst))
        tma_desc_ptr = _get_tma_desc_addr(tma_atom)

        def copy_fn(src_idx, dst_idx, tma_bar_ptr: cute.Pointer):
            # Reload sparse row indices at issue time; sIdx is populated after this
            # closure is created and remains live across CKV/KPE/V gather phases.
            tSR_rIdx = _load_s2r(tSR_sIdx)
            col_idx = tile_k * src_idx
            for m in cutlass.range(cute.size(tSR_rIdx, mode=[1]), unroll_full=True):
                if s_group_mask[m] != Int32(0):
                    with cute.arch.elect_one():
                        _tma_gather4_load(
                            tma_desc_ptr,
                            tSR_sDst[None, m, None, dst_idx].iterator,
                            tma_bar_ptr,
                            Int32(col_idx),
                            tSR_rIdx[0, m],
                            tSR_rIdx[1, m],
                            tSR_rIdx[2, m],
                            tSR_rIdx[3, m],
                        )

        return copy_fn

    _sm100_dsl_helpers = SimpleNamespace(
        cutlass=cutlass,
        cute=cute,
        tcgen05=tcgen05,
        cpasync=cpasync,
        utils=utils,
        sm100_utils=sm100_utils,
        pipeline=pipeline,
        pipeline_init_arrive=pipeline_init_arrive,
        pipeline_init_wait=pipeline_init_wait,
        Float32=Float32,
        BFloat16=BFloat16,
        Int32=Int32,
        cuda=cuda,
        load_s2r=_load_s2r,
        get_tma_desc_addr=_get_tma_desc_addr,
        tma_gather4_load=_tma_gather4_load,
        make_swizzled_mn_view=_make_swizzled_mn_view,
        make_rowmajor_mn_view=_make_rowmajor_mn_view,
        make_gather_rowgroup_view=_make_gather_rowgroup_view,
        unpack_gather4_stage=_unpack_gather4_stage,
        mask_invalid_gather_rows=_mask_invalid_gather_rows,
        transpose_gather4_stage_masked=_transpose_gather4_stage_masked,
        copy_rowmajor_stage_to_operand_a=_copy_rowmajor_stage_to_operand_a,
        copy_rowmajor_stage_to_operand_b=_copy_rowmajor_stage_to_operand_b,
        copy_rowmajor_stage_to_operand_qk_b=_copy_rowmajor_stage_to_operand_qk_b,
        make_gather4_copy_fn=_make_gather4_copy_fn,
    )
    return _sm100_dsl_helpers


def _get_cutedsl_kernel():
    """Lazily initialize the CuTeDSL QK GEMM kernel."""
    global _cutedsl_kernel
    if _cutedsl_kernel is not None:
        return _cutedsl_kernel

    helpers = _get_sm100_dsl_helpers()
    cutlass = helpers.cutlass
    cute = helpers.cute
    tcgen05 = helpers.tcgen05
    cpasync = helpers.cpasync
    utils = helpers.utils
    sm100_utils = helpers.sm100_utils
    pipeline = helpers.pipeline
    pipeline_init_arrive = helpers.pipeline_init_arrive
    pipeline_init_wait = helpers.pipeline_init_wait
    Float32 = helpers.Float32
    BFloat16 = helpers.BFloat16
    cuda = helpers.cuda

    class BatchedQKKernel:
        """Batched QK GEMM: S[batch, M, N] = Q_nope @ Kc^T + Q_pe @ Kp^T."""

        def __init__(self):
            self.threads_per_cta = 128
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (M_BLOCK, N_BLOCK)

        @cute.jit
        def __call__(
            self,
            mQ_nope: cute.Tensor,
            mQ_pe: cute.Tensor,
            mKc: cute.Tensor,
            mKp: cute.Tensor,
            mS: cute.Tensor,
            stream: cuda.CUstream,
        ):
            a_dtype = mQ_nope.element_type
            b_dtype = mKc.element_type
            a_major = utils.LayoutEnum.from_tensor(mQ_nope).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mKc).mma_major_mode()
            self.c_layout = utils.LayoutEnum.from_tensor(mS)

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype, a_major, b_major, Float32, self.cta_group, self.mma_tiler_mn,
            )

            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            k_tile = mma_inst_shape_k * 4
            self.mma_tiler = (M_BLOCK, N_BLOCK, k_tile)
            self.cta_tile_shape_mnk = (
                M_BLOCK // cute.size(tiled_mma.thr_id.shape), N_BLOCK, k_tile,
            )
            self.epi_tile = self.cta_tile_shape_mnk[:2]

            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,),
            )

            self.num_ab_stage = 2
            self.num_acc_stage = 1

            self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, a_dtype, self.num_ab_stage,
            )
            self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, b_dtype, self.num_ab_stage,
            )
            self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

            atom_thr_size = cute.size(tiled_mma.thr_id.shape)
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

            tma_atom_qn, tma_Qn = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mQ_nope, a_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_kc, tma_Kc = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mKc, b_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_qp, tma_Qp = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mQ_pe, a_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_kp, tma_Kp = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mKp, b_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

            a_copy = cute.size_in_bytes(a_dtype, a_smem)
            b_copy = cute.size_in_bytes(b_dtype, b_smem)
            self.num_tma_load_bytes = (a_copy + b_copy) * atom_thr_size

            batch_size = mS.shape[2]
            grid = (1, 1, batch_size)

            self.kernel(
                tiled_mma,
                tma_atom_qn, tma_Qn, tma_atom_kc, tma_Kc,
                tma_atom_qp, tma_Qp, tma_atom_kp, tma_Kp,
                mS,
                self.cluster_layout_vmnk,
                self.a_smem_layout_staged, self.b_smem_layout_staged,
                self.epi_tile,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_qn: cute.CopyAtom, mQ_nope: cute.Tensor,
            tma_atom_kc: cute.CopyAtom, mKc: cute.Tensor,
            tma_atom_qp: cute.CopyAtom, mQ_pe: cute.Tensor,
            tma_atom_kp: cute.CopyAtom, mKp: cute.Tensor,
            mS: cute.Tensor,
            cluster_layout_vmnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_tile: cute.Tile,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, bidz = cute.arch.block_idx()

            use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
            mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
            is_leader_cta = mma_tile_coord_v == 0
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            mma_tile_coord_mnl = (
                bidx // cute.size(tiled_mma.thr_id.shape), bidy, bidz,
            )

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_qn)
                cpasync.prefetch_descriptor(tma_atom_kc)
                cpasync.prefetch_descriptor(tma_atom_qp)
                cpasync.prefetch_descriptor(tma_atom_kp)

            @cute.struct
            class SharedStorage:
                ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
                acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            ab_pg = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            ab_cg = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_mbar.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pg, consumer_group=ab_cg,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            ).make_participants()

            acc_pg = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            acc_cg = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
            acc_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_mbar.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_pg, consumer_group=acc_cg,
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            acc_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            acc_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            tmem_bar = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
            tmem = utils.TmemAllocator(
                storage.tmem_buf, barrier_for_retrieve=tmem_bar,
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
            )

            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            sA = smem.allocate_tensor(
                element_type=BFloat16, layout=a_smem_layout_staged.outer,
                byte_alignment=128, swizzle=a_smem_layout_staged.inner,
            )
            sB = smem.allocate_tensor(
                element_type=BFloat16, layout=b_smem_layout_staged.outer,
                byte_alignment=128, swizzle=b_smem_layout_staged.inner,
            )

            # Partition CKV globals
            gQn = cute.local_tile(
                mQ_nope, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gKc = cute.local_tile(
                mKc, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            k_ckv = cute.size(gQn, mode=[3])

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            tCgQn = thr_mma.partition_A(gQn)
            tCgKc = thr_mma.partition_B(gKc)

            acl = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            bcl = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

            tAsQn, tAgQn = cpasync.tma_partition(
                tma_atom_qn, block_in_cluster_coord_vmnk[2], acl,
                cute.group_modes(sA, 0, 3), cute.group_modes(tCgQn, 0, 3),
            )
            tBsKc, tBgKc = cpasync.tma_partition(
                tma_atom_kc, block_in_cluster_coord_vmnk[1], bcl,
                cute.group_modes(sB, 0, 3), cute.group_modes(tCgKc, 0, 3),
            )

            # Partition KPE globals
            gQp = cute.local_tile(
                mQ_pe, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gKp = cute.local_tile(
                mKp, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            k_kpe = cute.size(gQp, mode=[3])

            tCgQp = thr_mma.partition_A(gQp)
            tCgKp = thr_mma.partition_B(gKp)

            tAsQp, tAgQp = cpasync.tma_partition(
                tma_atom_qp, block_in_cluster_coord_vmnk[2], acl,
                cute.group_modes(sA, 0, 3), cute.group_modes(tCgQp, 0, 3),
            )
            tBsKp, tBgKp = cpasync.tma_partition(
                tma_atom_kp, block_in_cluster_coord_vmnk[1], bcl,
                cute.group_modes(sB, 0, 3), cute.group_modes(tCgKp, 0, 3),
            )

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mn)
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            gS = cute.local_tile(
                mS, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
            tCgS = thr_mma.partition_C(gS)

            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tAgQn = tAgQn[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgKc = tBgKc[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
            tAgQp = tAgQp[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgKp = tBgKp[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

            # === CKV GEMM ===
            if warp_idx == 0:
                pf_cnt = cutlass.min(self.num_ab_stage - 2, k_ckv)
                for k in cutlass.range(pf_cnt, unroll=1):
                    ph = ab_prod.acquire_and_advance()
                    cute.copy(tma_atom_qn, tAgQn[(None, k)],
                              tAsQn[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_atom_kc, tBgKc[(None, k)],
                              tBsKc[(None, ph.index)], tma_bar_ptr=ph.barrier)

                pf = cutlass.Boolean(False)
                if is_leader_cta:
                    pf = ab_cons.try_wait()
                pe = ab_prod.try_acquire()

                for k in cutlass.range(k_ckv):
                    if k < k_ckv - pf_cnt:
                        ph = ab_prod.acquire_and_advance(pe)
                        cute.copy(tma_atom_qn, tAgQn[(None, ph.count)],
                                  tAsQn[(None, ph.index)], tma_bar_ptr=ph.barrier)
                        cute.copy(tma_atom_kc, tBgKc[(None, ph.count)],
                                  tBsKc[(None, ph.index)], tma_bar_ptr=ph.barrier)

                    if is_leader_cta:
                        ch = ab_cons.wait_and_advance(pf)
                        for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            cute.gemm(tiled_mma, tCtAcc,
                                      tCrA[(None, None, kb, ch.index)],
                                      tCrB[(None, None, kb, ch.index)], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ch.release()

                    if k + 1 < k_ckv - pf_cnt:
                        pe = ab_prod.try_acquire()
                    if k + 1 < k_ckv and is_leader_cta:
                        pf = ab_cons.try_wait()

                # === KPE GEMM (accumulate) ===
                pf_kpe = cutlass.min(self.num_ab_stage - 2, k_kpe)
                for k in cutlass.range(pf_kpe, unroll=1):
                    ph = ab_prod.acquire_and_advance()
                    cute.copy(tma_atom_qp, tAgQp[(None, k)],
                              tAsQp[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_atom_kp, tBgKp[(None, k)],
                              tBsKp[(None, ph.index)], tma_bar_ptr=ph.barrier)

                pfk = cutlass.Boolean(False)
                if is_leader_cta:
                    pfk = ab_cons.try_wait()
                pek = ab_prod.try_acquire()

                for k in cutlass.range(k_kpe):
                    if k < k_kpe - pf_kpe:
                        ph = ab_prod.acquire_and_advance(pek)
                        cute.copy(tma_atom_qp, tAgQp[(None, ph.count)],
                                  tAsQp[(None, ph.index)], tma_bar_ptr=ph.barrier)
                        cute.copy(tma_atom_kp, tBgKp[(None, ph.count)],
                                  tBsKp[(None, ph.index)], tma_bar_ptr=ph.barrier)

                    if is_leader_cta:
                        ch = ab_cons.wait_and_advance(pfk)
                        for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            cute.gemm(tiled_mma, tCtAcc,
                                      tCrA[(None, None, kb, ch.index)],
                                      tCrB[(None, None, kb, ch.index)], tCtAcc)
                        ch.release()

                    if k + 1 < k_kpe - pf_kpe:
                        pek = ab_prod.try_acquire()
                    if k + 1 < k_kpe and is_leader_cta:
                        pfk = ab_cons.try_wait()

                if is_leader_cta:
                    acc_pipe.producer_commit(acc_ps)

            # === Epilogue: TMEM -> global S ===
            tmem.relinquish_alloc_permit()
            acc_pipe.consumer_wait(acc_cs)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, mS.element_type,
                Float32, epi_tile, use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
            )

            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            tCgS_epi = cute.flat_divide(
                tCgS[((None, None), 0, 0, None, None, None)], epi_tile
            )
            tTR_gS = thr_t2r.partition_D(tCgS_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gS[(None, None, None, 0, 0, 0, 0, 0)].shape, Float32
            )

            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mS.element_type)
            tTR_rS = cute.make_rmem_tensor(
                tTR_gS[(None, None, None, 0, 0, 0, 0, 0)].shape, mS.element_type
            )

            tTR_gS = tTR_gS[(None, None, None, None, None, *mma_tile_coord_mnl)]
            tTR_gS = cute.group_modes(tTR_gS, 3, cute.rank(tTR_gS))

            for si in cutlass.range(cute.size(tTR_tAcc.shape, mode=[3])):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, si)], tTR_rAcc)
                v = tTR_rAcc.load()
                v = v.to(mS.element_type)
                tTR_rS.store(v)
                cute.copy(simt_atom, tTR_rS, tTR_gS[(None, None, None, si)])

            pipeline.sync(barrier_id=1)
            tmem.free(tmem_ptr)
            if warp_idx == 0:
                ab_prod.tail()

    _cutedsl_kernel = BatchedQKKernel()
    return _cutedsl_kernel


def _make_split_shell_launcher():
    helpers = _get_sm100_dsl_helpers()
    cutlass = helpers.cutlass
    cute = helpers.cute
    tcgen05 = helpers.tcgen05
    cpasync = helpers.cpasync
    utils = helpers.utils
    sm100_utils = helpers.sm100_utils
    pipeline = helpers.pipeline
    Float32 = helpers.Float32
    BFloat16 = helpers.BFloat16
    Int32 = helpers.Int32
    cuda = helpers.cuda
    copy_rowmajor_stage_to_operand_a = helpers.copy_rowmajor_stage_to_operand_a
    copy_rowmajor_stage_to_operand_b = helpers.copy_rowmajor_stage_to_operand_b
    copy_rowmajor_stage_to_operand_qk_b = helpers.copy_rowmajor_stage_to_operand_qk_b
    mask_invalid_gather_rows = helpers.mask_invalid_gather_rows
    transpose_gather4_stage_masked = helpers.transpose_gather4_stage_masked
    make_gather4_copy_fn = helpers.make_gather4_copy_fn

    row_major = utils.LayoutEnum.ROW_MAJOR
    col_major = utils.LayoutEnum.COL_MAJOR

    def make_thread_cooperative_group(size: int):
        return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)

    @cute.jit
    def _warpgroup_thread_idx(tidx: Int32) -> Int32:
        return tidx % Int32(WARPGROUP_THREADS)

    @cute.jit
    def _load_split_index_groups(
        m_idx: cute.Tensor,
        s_idx: cute.Tensor,
        s_valid: cute.Tensor,
        s_group_mask: cute.Tensor,
        tidx: Int32,
        token_idx: Int32,
        split_base: Int32,
        total_kv: Int32,
    ) -> None:
        for group_idx in cutlass.range(tidx, CHUNK_SIZE // 4, 128, unroll=1):
            row_base = Int32(4) * group_idx
            sparse_base = split_base + row_base
            group_mask = Int32(0)

            idx0 = m_idx[token_idx, sparse_base + 0]
            idx1 = m_idx[token_idx, sparse_base + 1]
            idx2 = m_idx[token_idx, sparse_base + 2]
            idx3 = m_idx[token_idx, sparse_base + 3]

            if idx0 >= 0 and idx0 < total_kv:
                s_idx[row_base + 0] = idx0
                s_valid[row_base + 0] = Int32(1)
                group_mask = group_mask | Int32(1)
            else:
                s_idx[row_base + 0] = Int32(0)
                s_valid[row_base + 0] = Int32(0)

            if idx1 >= 0 and idx1 < total_kv:
                s_idx[row_base + 1] = idx1
                s_valid[row_base + 1] = Int32(1)
                group_mask = group_mask | Int32(2)
            else:
                s_idx[row_base + 1] = Int32(0)
                s_valid[row_base + 1] = Int32(0)

            if idx2 >= 0 and idx2 < total_kv:
                s_idx[row_base + 2] = idx2
                s_valid[row_base + 2] = Int32(1)
                group_mask = group_mask | Int32(4)
            else:
                s_idx[row_base + 2] = Int32(0)
                s_valid[row_base + 2] = Int32(0)

            if idx3 >= 0 and idx3 < total_kv:
                s_idx[row_base + 3] = idx3
                s_valid[row_base + 3] = Int32(1)
                group_mask = group_mask | Int32(8)
            else:
                s_idx[row_base + 3] = Int32(0)
                s_valid[row_base + 3] = Int32(0)

            s_group_mask[group_idx] = group_mask

    @cute.jit
    def _compute_gather_bytes(
        s_group_mask: cute.Tensor,
        s_gather_bytes: cute.Tensor,
        tidx: Int32,
    ) -> None:
        for leader in cutlass.range(tidx, 1, 128, unroll=1):
            issued_groups = Int32(0)
            for group_idx in cutlass.range(CHUNK_SIZE // 4, unroll=1):
                if s_group_mask[group_idx] != Int32(0):
                    issued_groups = issued_groups + Int32(1)
            s_gather_bytes[leader] = issued_groups * Int32(4 * N_BLOCK * 2)

    class RebuildSplitKernelShell:
        """Fused 3-warpgroup split kernel: WG0(load) + WG1(mma) + WG2(softmax+epilogue)."""

        def __init__(self):
            self.threads_per_cta = REBUILD_THREADS_PER_CTA
            self.cta_group = tcgen05.CtaGroup.ONE

        @cute.jit
        def __call__(
            self,
            mQ_nope: cute.Tensor,
            mQ_pe: cute.Tensor,
            mCkv_flat: cute.Tensor,
            mKpe_flat: cute.Tensor,
            mIdx: cute.Tensor,
            mPartialO: cute.Tensor,
            mPartialLse: cute.Tensor,
            sm_scale: Float32,
            stream: cuda.CUstream,
        ):
            qk_mma_tiler = (M_BLOCK, N_BLOCK, N_BLOCK)
            qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                BFloat16,
                row_major.mma_major_mode(),
                row_major.mma_major_mode(),
                Float32,
                self.cta_group,
                (M_BLOCK, N_BLOCK),
            )
            q_smem_layout_staged = sm100_utils.make_smem_layout_a(
                qk_tiled_mma,
                qk_mma_tiler,
                BFloat16,
                OPERAND_PIPE_STAGES,
            )
            k_smem_layout_staged = sm100_utils.make_smem_layout_b(
                qk_tiled_mma,
                qk_mma_tiler,
                BFloat16,
                OPERAND_PIPE_STAGES,
            )
            pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                BFloat16,
                tcgen05.OperandMajorMode.K,
                col_major.mma_major_mode(),
                Float32,
                self.cta_group,
                (M_BLOCK, N_BLOCK),
            )
            pv_mma_tiler = (M_BLOCK, N_BLOCK, N_BLOCK)
            pv_b_smem_layout = sm100_utils.make_smem_layout_b(
                pv_tiled_mma, pv_mma_tiler, BFloat16, V_PIPE_STAGES,
            )
            p_smem_layout_staged = sm100_utils.make_smem_layout_a(
                pv_tiled_mma,
                pv_mma_tiler,
                BFloat16,
                1,
            )
            gather_stage_layout = cute.slice_(
                cute.make_layout((1, N_BLOCK, 1)), (None, None, 0)
            )
            tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
            tma_atom_ckv, _ = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mCkv_flat,
                gather_stage_layout,
                (1, N_BLOCK),
            )
            tma_atom_kpe, _ = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mKpe_flat,
                gather_stage_layout,
                (1, N_BLOCK),
            )
            self.kernel(
                qk_tiled_mma,
                pv_tiled_mma,
                tma_atom_ckv,
                tma_atom_kpe,
                q_smem_layout_staged,
                k_smem_layout_staged,
                pv_b_smem_layout,
                p_smem_layout_staged,
                mQ_nope,
                mQ_pe,
                mCkv_flat,
                mKpe_flat,
                mIdx,
                mPartialO,
                mPartialLse,
                sm_scale,
            ).launch(
                grid=[mQ_nope.shape[0], NUM_SPLITS, 1],
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            qk_tiled_mma_: cute.TiledMma,
            pv_tiled_mma_: cute.TiledMma,
            tma_atom_ckv: cute.CopyAtom,
            tma_atom_kpe: cute.CopyAtom,
            q_smem_layout_staged: cute.ComposedLayout,
            k_smem_layout_staged: cute.ComposedLayout,
            pv_b_smem_layout: cute.ComposedLayout,
            p_smem_layout_staged: cute.ComposedLayout,
            mQ_nope: cute.Tensor,
            mQ_pe: cute.Tensor,
            mCkv_flat: cute.Tensor,
            mKpe_flat: cute.Tensor,
            mIdx: cute.Tensor,
            mPartialO: cute.Tensor,
            mPartialLse: cute.Tensor,
            sm_scale: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            warpgroup_tidx = _warpgroup_thread_idx(tidx)
            token_idx, split_idx, _ = cute.arch.block_idx()
            total_kv = mCkv_flat.shape[0]
            split_base = split_idx * CHUNK_SIZE
            load_wg_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=WARPGROUP_THREADS,
            )
            mma_warp_sync_barrier = pipeline.NamedBarrier(
                barrier_id=3,
                num_threads=WARP_SIZE,
            )
            valid_wg_sync_barrier = pipeline.NamedBarrier(
                barrier_id=4,
                num_threads=2 * WARPGROUP_THREADS,
            )
            softmax_wg_sync_barrier = pipeline.NamedBarrier(
                barrier_id=5,
                num_threads=WARPGROUP_THREADS,
            )
            score_tmem_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=REBUILD_THREADS_PER_CTA,
            )
            _ = sm_scale
            _ = qk_tiled_mma_
            _ = pv_tiled_mma_
            _ = tma_atom_ckv
            _ = tma_atom_kpe
            _ = mQ_nope
            _ = mQ_pe
            _ = mKpe_flat

            @cute.struct
            class SharedStorage:
                operand_pipe_mbar: cute.struct.MemRange[
                    cutlass.Int64, 2 * OPERAND_PIPE_STAGES
                ]
                score_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                v_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2 * V_PIPE_STAGES]
                output_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                gather_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                tmem_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            _storage = smem.allocate(SharedStorage)
            gather_mbar_ptr = _storage.gather_mbar.data_ptr()
            sQ = smem.allocate_tensor(
                element_type=BFloat16,
                layout=q_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=q_smem_layout_staged.inner,
            )
            sQrow = smem.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout(
                    (M_BLOCK, N_BLOCK, 1),
                    stride=(N_BLOCK, 1, M_BLOCK * N_BLOCK),
                ),
                byte_alignment=16,
            )
            sGather = smem.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout(
                    (CHUNK_SIZE, N_BLOCK, 1),
                    stride=(N_BLOCK, 1, CHUNK_SIZE * N_BLOCK),
                ),
                byte_alignment=16,
            )
            sK = smem.allocate_tensor(
                element_type=BFloat16,
                layout=k_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=k_smem_layout_staged.inner,
            )
            sV = smem.allocate_tensor(
                element_type=BFloat16,
                layout=pv_b_smem_layout.outer,
                byte_alignment=128,
                swizzle=pv_b_smem_layout.inner,
            )
            sP = smem.allocate_tensor(
                element_type=BFloat16,
                layout=p_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=p_smem_layout_staged.inner,
            )
            sScoreF32 = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((M_BLOCK, N_BLOCK)),
                byte_alignment=16,
            )
            sIdx = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_layout((CHUNK_SIZE,)),
                byte_alignment=16,
            )
            sValid = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_layout((CHUNK_SIZE,)),
                byte_alignment=4,
            )
            sGroupMask = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_layout((CHUNK_SIZE // 4,)),
                byte_alignment=4,
            )
            sGatherBytes = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_layout((1,)),
                byte_alignment=4,
            )
            sHasValid = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_layout((1,)),
                byte_alignment=4,
            )

            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(gather_mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()

            operand_producer, operand_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=OPERAND_PIPE_STAGES,
                producer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                consumer_group=make_thread_cooperative_group(1),
                barrier_storage=_storage.operand_pipe_mbar.data_ptr(),
            ).make_participants()
            score_producer, score_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(1),
                consumer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                barrier_storage=_storage.score_pipe_mbar.data_ptr(),
            ).make_participants()
            v_producer, v_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=V_PIPE_STAGES,
                producer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                consumer_group=make_thread_cooperative_group(1),
                barrier_storage=_storage.v_pipe_mbar.data_ptr(),
            ).make_participants()
            output_producer, output_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(1),
                consumer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                barrier_storage=_storage.output_pipe_mbar.data_ptr(),
            ).make_participants()
            _ = operand_producer
            _ = operand_consumer
            _ = score_producer
            _ = score_consumer
            _ = v_producer
            _ = v_consumer
            _ = output_producer
            _ = output_consumer
            _ = sQ
            _ = sQrow
            _ = sGather
            _ = sK
            _ = sV
            _ = sP
            _ = sScoreF32
            gather_phase = Int32(0)
            _ = gather_phase
            qk_thr_mma = qk_tiled_mma_.get_slice(Int32(0))
            pv_thr_mma = pv_tiled_mma_.get_slice(Int32(0))
            tSrQ = qk_tiled_mma_.make_fragment_A(sQ)
            tSrK = qk_tiled_mma_.make_fragment_B(sK)
            qk_acc_shape = qk_tiled_mma_.partition_shape_C((M_BLOCK, N_BLOCK))
            qk_acc_fake = qk_tiled_mma_.make_fragment_C(qk_acc_shape)
            qk_num_kblks = cute.size(tSrQ, mode=[2])
            qk_tmem_cols = tcgen05.find_tmem_tensor_col_offset(qk_acc_fake)
            score_tmem = utils.TmemAllocator(
                _storage.tmem_buf,
                barrier_for_retrieve=score_tmem_barrier,
                allocator_warp_id=WARPS_PER_WARPGROUP,
            )
            score_tmem.allocate(qk_tmem_cols)
            score_tmem.wait_for_alloc()
            score_tmem_ptr = score_tmem.retrieve_ptr(Float32)
            tStS = cute.make_tensor(score_tmem_ptr, qk_acc_fake.layout)
            score_copy_atom_t2r = sm100_utils.get_tmem_load_op(
                (M_BLOCK, N_BLOCK, N_BLOCK),
                row_major,
                Float32,
                Float32,
                (M_BLOCK, N_BLOCK),
                False,
            )
            tScore_flat = tStS[((None, None), 0, 0)]
            tScore_epi = cute.flat_divide(tScore_flat, (M_BLOCK, N_BLOCK))
            tiled_score_t2r = tcgen05.make_tmem_copy(
                score_copy_atom_t2r, tScore_epi[(None, None, 0, 0)]
            )
            score_thr_t2r = tiled_score_t2r.get_slice(warpgroup_tidx)
            tTR_tScore = score_thr_t2r.partition_S(tScore_epi)
            tTR_tScore = cute.group_modes(tTR_tScore, 3, cute.rank(tTR_tScore))
            sScore_epi = cute.flat_divide(sScoreF32, (M_BLOCK, N_BLOCK))
            cScore = cute.make_identity_tensor((M_BLOCK, N_BLOCK))
            cScore_epi = cute.flat_divide(cScore, (M_BLOCK, N_BLOCK))
            tTR_sScore = score_thr_t2r.partition_D(sScore_epi)
            tTR_sScore = cute.group_modes(tTR_sScore, 3, cute.rank(tTR_sScore))
            tTR_cScore = score_thr_t2r.partition_D(cScore_epi)
            tTR_cScore = cute.group_modes(tTR_cScore, 3, cute.rank(tTR_cScore))
            tTR_rScore = cute.make_fragment_like(
                tTR_sScore[(None, None, None, 0)], Float32
            )
            num_score_epi_tiles = cute.size(tTR_tScore.shape, mode=[3])

            if warp_idx < WARPS_PER_WARPGROUP:
                gather_ckv_copy = make_gather4_copy_fn(
                    tma_atom_ckv,
                    sGather,
                    sIdx,
                    sGroupMask,
                    Int32(0),
                )
                gather_kpe_copy = make_gather4_copy_fn(
                    tma_atom_kpe,
                    sGather,
                    sIdx,
                    sGroupMask,
                    Int32(0),
                )
                _load_split_index_groups(
                    mIdx,
                    sIdx,
                    sValid,
                    sGroupMask,
                    warpgroup_tidx,
                    token_idx,
                    split_base,
                    total_kv,
                )
                _compute_gather_bytes(sGroupMask, sGatherBytes, warpgroup_tidx)
                for leader in cutlass.range(warpgroup_tidx, 1, WARPGROUP_THREADS, unroll=1):
                    has_valid = Int32(0)
                    for group_idx in cutlass.range(CHUNK_SIZE // 4, unroll=1):
                        if sGroupMask[group_idx] != Int32(0):
                            has_valid = Int32(1)
                    sHasValid[leader] = has_valid

                # === CKV K-tile loop: 8 iterations over K=512 dimension ===
                for k_tile in cutlass.range(_DEBUG_SPLIT_CKV_TILE_COUNT, unroll=1):
                    operand_handle = operand_producer.acquire_and_advance()
                    col_base = k_tile * Int32(N_BLOCK)
                    for elem in cutlass.range(
                        warpgroup_tidx, M_BLOCK * N_BLOCK, WARPGROUP_THREADS, unroll=1
                    ):
                        row = elem // N_BLOCK
                        col = elem % N_BLOCK
                        if row < NUM_Q_HEADS:
                            sQrow[row, col, 0] = mQ_nope[token_idx, row, col_base + col]
                        else:
                            sQrow[row, col, 0] = BFloat16(0.0)
                    load_wg_sync_barrier.arrive_and_wait()
                    copy_rowmajor_stage_to_operand_a(
                        sQrow,
                        sQ[None, None, None, operand_handle.index],
                        qk_tiled_mma_,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()

                    if warp_idx == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                gather_mbar_ptr,
                                sGatherBytes[0],
                            )
                        gather_ckv_copy(k_tile, 0, gather_mbar_ptr)
                    cute.arch.mbarrier_wait(gather_mbar_ptr, gather_phase)
                    cute.arch.fence_view_async_shared()
                    mask_invalid_gather_rows(sGather, sValid, warpgroup_tidx)
                    load_wg_sync_barrier.arrive_and_wait()
                    copy_rowmajor_stage_to_operand_qk_b(
                        sGather,
                        sK[None, None, None, operand_handle.index],
                        qk_tiled_mma_,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()
                    gather_phase = gather_phase ^ Int32(1)
                    operand_handle.commit()

                if cutlass.const_expr(_DEBUG_SPLIT_INCLUDE_KPE):
                    # === KPE K-tile (single iteration, K=64) ===
                    operand_handle = operand_producer.acquire_and_advance()
                    for elem in cutlass.range(
                        warpgroup_tidx, M_BLOCK * N_BLOCK, WARPGROUP_THREADS, unroll=1
                    ):
                        row = elem // N_BLOCK
                        col = elem % N_BLOCK
                        if row < NUM_Q_HEADS:
                            sQrow[row, col, 0] = mQ_pe[token_idx, row, col]
                        else:
                            sQrow[row, col, 0] = BFloat16(0.0)
                    load_wg_sync_barrier.arrive_and_wait()
                    copy_rowmajor_stage_to_operand_a(
                        sQrow,
                        sQ[None, None, None, operand_handle.index],
                        qk_tiled_mma_,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()

                    if warp_idx == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                gather_mbar_ptr,
                                sGatherBytes[0],
                            )
                        gather_kpe_copy(0, 0, gather_mbar_ptr)
                    cute.arch.mbarrier_wait(gather_mbar_ptr, gather_phase)
                    cute.arch.fence_view_async_shared()
                    mask_invalid_gather_rows(sGather, sValid, warpgroup_tidx)
                    load_wg_sync_barrier.arrive_and_wait()
                    copy_rowmajor_stage_to_operand_qk_b(
                        sGather,
                        sK[None, None, None, operand_handle.index],
                        qk_tiled_mma_,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()
                    gather_phase = gather_phase ^ Int32(1)
                    operand_handle.commit()
                valid_wg_sync_barrier.arrive()

            if warp_idx == WARPS_PER_WARPGROUP:
                score_full = score_producer.acquire_and_advance()

                # First CKV K-tile: clear accumulator on first kblk
                operand_full = operand_consumer.wait_and_advance()
                cute.arch.fence_view_async_shared()
                for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                    qk_tiled_mma_.set(
                        tcgen05.Field.ACCUMULATE,
                        kblk_idx != 0,
                    )
                    cute.gemm(
                        qk_tiled_mma_,
                        tStS,
                        tSrQ[(None, None, kblk_idx, operand_full.index)],
                        tSrK[(None, None, kblk_idx, operand_full.index)],
                        tStS,
                    )
                operand_full.release()

                # Remaining 7 CKV K-tiles: always accumulate
                for _k in cutlass.range(_DEBUG_SPLIT_CKV_TILE_COUNT - 1, unroll=1):
                    operand_full = operand_consumer.wait_and_advance()
                    cute.arch.fence_view_async_shared()
                    for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                        qk_tiled_mma_.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            qk_tiled_mma_,
                            tStS,
                            tSrQ[(None, None, kblk_idx, operand_full.index)],
                            tSrK[(None, None, kblk_idx, operand_full.index)],
                            tStS,
                        )
                    operand_full.release()

                if cutlass.const_expr(_DEBUG_SPLIT_INCLUDE_KPE):
                    # KPE K-tile: always accumulate
                    operand_full = operand_consumer.wait_and_advance()
                    cute.arch.fence_view_async_shared()
                    for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                        qk_tiled_mma_.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            qk_tiled_mma_,
                            tStS,
                            tSrQ[(None, None, kblk_idx, operand_full.index)],
                            tSrK[(None, None, kblk_idx, operand_full.index)],
                            tStS,
                        )
                    operand_full.release()
                mma_warp_sync_barrier.arrive_and_wait()
                score_producer.commit(score_full)

            if (
                warp_idx >= 2 * WARPS_PER_WARPGROUP
                and warp_idx < 3 * WARPS_PER_WARPGROUP
            ):
                valid_wg_sync_barrier.arrive_and_wait()
                score_full = score_consumer.wait_and_advance()
                scale_log2e = sm_scale * Float32(LOG2E)
                for si in cutlass.range(num_score_epi_tiles):
                    cScore_i = tTR_cScore[(None, None, None, si)]
                    cute.copy(
                        tiled_score_t2r,
                        tTR_tScore[(None, None, None, si)],
                        tTR_rScore,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(cute.size(tTR_rScore)):
                        coord = cScore_i[i]
                        row = coord[0]
                        col = coord[1]
                        score_val = Float32(-float("inf"))
                        if row < Int32(NUM_Q_HEADS):
                            score_val = tTR_rScore[i] * scale_log2e
                            if sValid[col] == Int32(0):
                                score_val = Float32(-float("inf"))
                        sScoreF32[row, col] = score_val

                softmax_wg_sync_barrier.arrive_and_wait()

                cols_per_thr = N_BLOCK // 8
                head_idx = warpgroup_tidx // Int32(8)
                lane_in_head = warpgroup_tidx % Int32(8)
                col_start = lane_in_head * Int32(cols_per_thr)

                local_max = Float32(-float("inf"))
                row_max = Float32(-float("inf"))
                local_sum = Float32(0.0)
                row_sum = Float32(0.0)
                lse = Float32(-float("inf"))

                if head_idx < Int32(NUM_Q_HEADS):
                    for c in cutlass.range(cols_per_thr, unroll=1):
                        val = sScoreF32[head_idx, col_start + Int32(c)]
                        if val > local_max:
                            local_max = val
                    sScoreF32[Int32(NUM_Q_HEADS) + lane_in_head, head_idx] = local_max

                softmax_wg_sync_barrier.arrive_and_wait()

                if head_idx < Int32(NUM_Q_HEADS):
                    for lane in cutlass.range(8, unroll_full=True):
                        val = sScoreF32[Int32(NUM_Q_HEADS) + Int32(lane), head_idx]
                        if val > row_max:
                            row_max = val

                    for c in cutlass.range(cols_per_thr, unroll=1):
                        col = col_start + Int32(c)
                        val = sScoreF32[head_idx, col]
                        exp_val = Float32(0.0)
                        if row_max > Float32(-float("inf")):
                            exp_val = cute.exp2(val - row_max, fastmath=True)
                        sScoreF32[head_idx, col] = exp_val
                        local_sum = local_sum + exp_val

                    sScoreF32[Int32(NUM_Q_HEADS) + lane_in_head, head_idx] = local_sum

                softmax_wg_sync_barrier.arrive_and_wait()

                if head_idx < Int32(NUM_Q_HEADS):
                    for lane in cutlass.range(8, unroll_full=True):
                        row_sum = row_sum + sScoreF32[
                            Int32(NUM_Q_HEADS) + Int32(lane), head_idx
                        ]

                    if row_sum > Float32(0.0):
                        lse = row_max + cute.log2(row_sum, fastmath=True)
                    mPartialLse[split_idx, token_idx, head_idx] = lse

                inv_sum = Float32(0.0)
                if head_idx < Int32(NUM_Q_HEADS):
                    if row_sum > Float32(0.0):
                        inv_sum = Float32(1.0) / row_sum
                    for c in cutlass.range(cols_per_thr, unroll=1):
                        col = col_start + Int32(c)
                        p_val = Float32(0.0)
                        if inv_sum > Float32(0.0):
                            p_val = sScoreF32[head_idx, col] * inv_sum
                        sScoreF32[head_idx, col] = p_val

                # --- Stage P from FP32 scratch to BF16 operand-A SMEM layout ---
                softmax_wg_sync_barrier.arrive_and_wait()

                # Convert P from FP32 (sScoreF32) to BF16 (sQrow)
                for elem in cutlass.range(
                    warpgroup_tidx,
                    M_BLOCK * N_BLOCK,
                    WARPGROUP_THREADS,
                    unroll=1,
                ):
                    row = elem // Int32(N_BLOCK)
                    col = elem % Int32(N_BLOCK)
                    p_bf16 = BFloat16(0.0)
                    if row < Int32(NUM_Q_HEADS):
                        p_bf16 = BFloat16(sScoreF32[row, col])
                    sQrow[row, col, 0] = p_bf16

                softmax_wg_sync_barrier.arrive_and_wait()

                # Copy from rowmajor sQrow to swizzled sP operand layout
                copy_rowmajor_stage_to_operand_a(
                    sQrow,
                    sP[None, None, None, 0],
                    pv_tiled_mma_,
                    warpgroup_tidx,
                )
                softmax_wg_sync_barrier.arrive_and_wait()

                score_full.release()

            cute.arch.sync_threads()

            # Reuse score TMEM allocation as PV output accumulator
            # (same 64x64 FP32 shape, no need to free and reallocate)
            pv_acc_shape = pv_thr_mma.partition_shape_C((M_BLOCK, N_BLOCK))
            pv_acc_fake = pv_thr_mma.make_fragment_C(pv_acc_shape)
            tOtO = cute.make_tensor(score_tmem_ptr, pv_acc_fake.layout)

            # --- Output TMEM copy setup for WG2 epilogue ---
            output_copy_atom_t2r = sm100_utils.get_tmem_load_op(
                (M_BLOCK, N_BLOCK, N_BLOCK),
                row_major,
                Float32,
                Float32,
                (M_BLOCK, N_BLOCK),
                False,
            )
            tOutput_flat = tOtO[((None, None), 0, 0)]
            tOutput_epi = cute.flat_divide(tOutput_flat, (M_BLOCK, N_BLOCK))
            tiled_output_t2r = tcgen05.make_tmem_copy(
                output_copy_atom_t2r, tOutput_epi[(None, None, 0, 0)]
            )
            output_thr_t2r = tiled_output_t2r.get_slice(warpgroup_tidx)
            tTR_tOutput = output_thr_t2r.partition_S(tOutput_epi)
            tTR_tOutput = cute.group_modes(
                tTR_tOutput, 3, cute.rank(tTR_tOutput)
            )
            cOutput = cute.make_identity_tensor((M_BLOCK, N_BLOCK))
            cOutput_epi = cute.flat_divide(cOutput, (M_BLOCK, N_BLOCK))
            tTR_cOutput = output_thr_t2r.partition_D(cOutput_epi)
            tTR_cOutput = cute.group_modes(
                tTR_cOutput, 3, cute.rank(tTR_cOutput)
            )
            tTR_rOutput = cute.make_fragment_like(
                tTR_cOutput[(None, None, None, 0)], Float32
            )
            num_output_epi_tiles = cute.size(tTR_tOutput.shape, mode=[3])

            # --- Phase 2: V staging + PV MMA across 8 output tiles ---
            NUM_V_TILES = CKV_DIM // N_BLOCK  # 8

            if warp_idx < WARPS_PER_WARPGROUP:
                gather_v_copy = make_gather4_copy_fn(
                    tma_atom_ckv,
                    sGather,
                    sIdx,
                    sGroupMask,
                    Int32(0),
                )
                for v_tile_idx in cutlass.range(NUM_V_TILES, unroll=1):
                    v_handle = v_producer.acquire_and_advance()
                    # Gather V tile (columns v_tile_idx*N_BLOCK:(v_tile_idx+1)*N_BLOCK)
                    if warp_idx == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                gather_mbar_ptr,
                                sGatherBytes[0],
                            )
                        gather_v_copy(v_tile_idx, 0, gather_mbar_ptr)
                    cute.arch.mbarrier_wait(gather_mbar_ptr, gather_phase)
                    cute.arch.fence_view_async_shared()
                    transpose_gather4_stage_masked(
                        sGather, sQrow, sValid, warpgroup_tidx
                    )
                    load_wg_sync_barrier.arrive_and_wait()
                    # Stage from rowmajor to PV B-operand layout
                    copy_rowmajor_stage_to_operand_b(
                        sQrow,
                        sV[None, None, None, v_handle.index],
                        pv_tiled_mma_,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()
                    gather_phase = gather_phase ^ Int32(1)
                    v_handle.commit()

            if warp_idx == WARPS_PER_WARPGROUP:
                tPr = pv_thr_mma.make_fragment_A(sP)
                tVr = pv_thr_mma.make_fragment_B(sV)
                pv_num_kblks = cute.size(tPr, mode=[2])
                for v_tile_idx in cutlass.range(NUM_V_TILES, unroll=1):
                    output_full = output_producer.acquire_and_advance()
                    v_full = v_consumer.wait_and_advance()
                    tVr_stage = tVr[None, None, None, v_full.index]
                    cute.arch.fence_view_async_shared()
                    for kblk_idx in cutlass.range(pv_num_kblks, unroll_full=True):
                        pv_tiled_mma_.set(
                            tcgen05.Field.ACCUMULATE, kblk_idx != 0,
                        )
                        cute.gemm(
                            pv_tiled_mma_,
                            tOtO,
                            tPr[(None, None, kblk_idx, 0)],
                            tVr_stage[(None, None, kblk_idx)],
                            tOtO,
                        )
                    v_full.release()
                    mma_warp_sync_barrier.arrive_and_wait()
                    output_producer.commit(output_full)

            if (
                warp_idx >= 2 * WARPS_PER_WARPGROUP
                and warp_idx < 3 * WARPS_PER_WARPGROUP
            ):
                for v_tile_idx in cutlass.range(NUM_V_TILES, unroll=1):
                    output_handle = output_consumer.wait_and_advance()
                    v_col_base = v_tile_idx * Int32(N_BLOCK)

                    # TMEM -> registers -> global store
                    for si in cutlass.range(num_output_epi_tiles):
                        cOutput_i = tTR_cOutput[(None, None, None, si)]
                        cute.copy(
                            tiled_output_t2r,
                            tTR_tOutput[(None, None, None, si)],
                            tTR_rOutput,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(
                            cute.size(tTR_rOutput)
                        ):
                            coord = cOutput_i[i]
                            row = coord[0]
                            col = coord[1]
                            if row < Int32(NUM_Q_HEADS):
                                mPartialO[
                                    split_idx, token_idx, row,
                                    v_col_base + col,
                                ] = tTR_rOutput[i]

                    output_handle.release()

            cute.arch.sync_threads()

            score_tmem.relinquish_alloc_permit()
            score_tmem.free(score_tmem_ptr)
            cute.arch.sync_threads()
            return

    # SplitKernelShell removed — RebuildSplitKernelShell is now authoritative.
    # Dense fallback (_compute_split_partials) retained for FLASHMLA_DSA_VALIDATE_FUSED.
    return RebuildSplitKernelShell()


def _get_split_shell_launcher():
    global _split_shell_launcher
    if _split_shell_launcher is None:
        _split_shell_launcher = _make_split_shell_launcher()
    return _split_shell_launcher


def _to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _flatten_kv_caches(ckv_cache, kpe_cache):
    """Expose KV caches as flat token-id addressable tensors."""
    ckv_flat = ckv_cache.reshape(-1, CKV_DIM)
    kpe_flat = kpe_cache.reshape(-1, KPE_DIM)
    return ckv_flat, kpe_flat


def _normalize_sparse_indices(sparse_indices, total_kv):
    """Treat any index outside `[0, total_kv)` as invalid and produce safe gather indices."""
    sparse_indices_i64 = sparse_indices.to(torch.long)
    valid_mask = (sparse_indices_i64 >= 0) & (sparse_indices_i64 < total_kv)
    safe_indices = torch.where(
        valid_mask, sparse_indices_i64, torch.zeros_like(sparse_indices_i64)
    )
    return safe_indices, valid_mask


def _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices):
    """Gather sparse KV rows and reshape into [T, S, N, D] chunks."""
    T = sparse_indices.shape[0]
    S = NUM_SPLITS
    N = CHUNK_SIZE
    ckv_flat, kpe_flat = _flatten_kv_caches(ckv_cache, kpe_cache)
    total_kv = ckv_flat.shape[0]
    safe_indices, valid_mask = _normalize_sparse_indices(sparse_indices, total_kv)

    vm = valid_mask.view(T, S, N)

    kc = ckv_flat[safe_indices.reshape(-1)].view(T, S, N, CKV_DIM)
    kp = kpe_flat[safe_indices.reshape(-1)].view(T, S, N, KPE_DIM)

    kc = kc.masked_fill(~vm.unsqueeze(-1), 0).contiguous()
    kp = kp.masked_fill(~vm.unsqueeze(-1), 0).contiguous()

    return kc, kp, vm


def _combine_splits(partial_o, partial_lse):
    """Combine 32 split partials via stable exp2 reduction."""
    max_lse = partial_lse.max(dim=0).values
    finite = torch.isfinite(max_lse)
    stable_max = torch.where(finite, max_lse, torch.zeros_like(max_lse))

    weights = torch.exp2(partial_lse - stable_max.unsqueeze(0))
    weight_sum = weights.sum(dim=0)
    combined = (weights.unsqueeze(-1) * partial_o).sum(dim=0)

    output = torch.zeros_like(combined)
    valid = weight_sum > 0
    output[valid] = combined[valid] / weight_sum[valid].unsqueeze(-1)

    final_lse = torch.full_like(max_lse, -float("inf"))
    final_lse[valid] = stable_max[valid] + torch.log2(weight_sum[valid])

    return output.to(torch.bfloat16), final_lse


def _prepare_qk_gemm_operands(q_nope, q_pe, kc, kp):
    """Pack the current dense QK operands into the layouts expected by the CuTeDSL kernel."""
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda

    T = q_nope.shape[0]
    batch = T * NUM_SPLITS
    device = q_nope.device

    q_nope_padded = torch.zeros(
        T, M_BLOCK, CKV_DIM, dtype=torch.bfloat16, device=device
    )
    q_nope_padded[:, :NUM_Q_HEADS, :] = q_nope
    q_pe_padded = torch.zeros(
        T, M_BLOCK, KPE_DIM, dtype=torch.bfloat16, device=device
    )
    q_pe_padded[:, :NUM_Q_HEADS, :] = q_pe

    q_nope_exp = (
        q_nope_padded.unsqueeze(1)
        .expand(T, NUM_SPLITS, M_BLOCK, CKV_DIM)
        .reshape(batch, M_BLOCK, CKV_DIM)
    )
    q_pe_exp = (
        q_pe_padded.unsqueeze(1)
        .expand(T, NUM_SPLITS, M_BLOCK, KPE_DIM)
        .reshape(batch, M_BLOCK, KPE_DIM)
    )
    kc_flat = kc.reshape(batch, CHUNK_SIZE, CKV_DIM)
    kp_flat = kp.reshape(batch, CHUNK_SIZE, KPE_DIM)

    q_nope_3d = q_nope_exp.contiguous().permute(1, 2, 0)
    q_pe_3d = q_pe_exp.contiguous().permute(1, 2, 0)
    kc_3d = kc_flat.contiguous().permute(1, 2, 0)
    kp_3d = kp_flat.contiguous().permute(1, 2, 0)

    s_base = torch.zeros(
        batch, M_BLOCK, N_BLOCK, dtype=torch.float32, device=device
    )
    s_3d = s_base.permute(1, 2, 0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    return (
        s_base,
        from_dlpack(q_nope_3d, assumed_align=16),
        from_dlpack(q_pe_3d, assumed_align=16),
        from_dlpack(kc_3d, assumed_align=16),
        from_dlpack(kp_3d, assumed_align=16),
        from_dlpack(s_3d, assumed_align=16),
        stream,
    )


def _compute_qk_logits_cutedsl(q_nope, q_pe, kc, kp):
    """Run the standalone CuTeDSL QK kernel and return unscaled logits [T, S, H, N]."""
    T = q_nope.shape[0]
    s_base, mQn, mQp, mKc, mKp, mS, stream = _prepare_qk_gemm_operands(
        q_nope, q_pe, kc, kp
    )
    _get_cutedsl_kernel()(mQn, mQp, mKc, mKp, mS, stream=stream)
    return s_base[:, :NUM_Q_HEADS, :].view(T, NUM_SPLITS, NUM_Q_HEADS, CHUNK_SIZE)


def _allocate_fused_output_scratch(T, device):
    """Allocate padded fused-output scratch for WG2 epilogue stores."""
    cache_key = (str(device), int(T))
    cached = _fused_output_scratch_cache.get(cache_key)
    if cached is None:
        partial_o_tma = torch.empty(
            (NUM_SPLITS, T, M_BLOCK, CKV_DIM), dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            (NUM_SPLITS, T, NUM_Q_HEADS), dtype=torch.float32, device=device
        )
        _fused_output_scratch_cache[cache_key] = (partial_o_tma, partial_lse)
    else:
        partial_o_tma, partial_lse = cached
    partial_o = partial_o_tma[:, :, :NUM_Q_HEADS, :]
    return partial_o_tma, partial_o, partial_lse


def _get_compiled_split_shell(
    q_nope,
    ckv_cache,
    mQ,
    mQp,
    mCkv,
    mKpe,
    mIdx,
    mO,
    mLse,
    stream,
):
    import cutlass.cute as cute

    cache_key = (
        str(q_nope.device),
        int(q_nope.shape[0]),
        int(ckv_cache.shape[0]),
        int(ckv_cache.shape[1]),
    )
    compiled = _compiled_split_shell_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            _get_split_shell_launcher(),
            mQ,
            mQp,
            mCkv,
            mKpe,
            mIdx,
            mO,
            mLse,
            1.0,
            stream,
        )
        _compiled_split_shell_cache[cache_key] = compiled
    return compiled


def _prepare_split_shell_operands(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices):
    """Wrap fused kernel operands plus padded output scratch as CuTeDSL tensors."""
    from cutlass.cute.runtime import from_dlpack

    T = q_nope.shape[0]
    device = q_nope.device
    ckv_flat, kpe_flat = _flatten_kv_caches(ckv_cache, kpe_cache)
    partial_o_tma, partial_o, partial_lse = _allocate_fused_output_scratch(
        T, device
    )

    return (
        partial_o_tma,
        partial_o,
        partial_lse,
        from_dlpack(q_nope.contiguous(), assumed_align=16),
        from_dlpack(q_pe.contiguous(), assumed_align=16),
        from_dlpack(ckv_flat.contiguous(), assumed_align=16),
        from_dlpack(kpe_flat.contiguous(), assumed_align=16),
        from_dlpack(sparse_indices.contiguous(), assumed_align=16),
        from_dlpack(partial_o_tma, assumed_align=16),
        from_dlpack(partial_lse, assumed_align=16),
    )


def _compute_split_partials_fused(
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f
):
    """Run the fused 3-warpgroup split kernel and return partial outputs."""
    import cuda.bindings.driver as cuda

    (
        _partial_o_tma,
        partial_o,
        partial_lse,
        mQ,
        mQp,
        mCkv,
        mKpe,
        mIdx,
        mO,
        mLse,
    ) = _prepare_split_shell_operands(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled_kernel = _get_compiled_split_shell(
        q_nope,
        ckv_cache,
        mQ,
        mQp,
        mCkv,
        mKpe,
        mIdx,
        mO,
        mLse,
        stream,
    )
    compiled_kernel(mQ, mQp, mCkv, mKpe, mIdx, mO, mLse, sm_scale_f, stream)
    return partial_o, partial_lse


def _format_validation_stats(name: str, fused: torch.Tensor, fallback: torch.Tensor) -> str:
    diff = (fused - fallback).abs()
    denom = fallback.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    return f"{name}: max_abs={max_abs:.6e} max_rel={max_rel:.6e}"


def _maybe_validate_split_partials(
    fused_partial_o: torch.Tensor,
    fused_partial_lse: torch.Tensor,
    fallback_partial_o: torch.Tensor,
    fallback_partial_lse: torch.Tensor,
) -> None:
    global _split_validation_emitted

    mode = os.environ.get("FLASHMLA_DSA_VALIDATE_FUSED", "").strip().lower()
    if mode in {"", "0", "off"}:
        return

    tensors_to_check = (
        ("fused_partial_o", fused_partial_o),
        ("fused_partial_lse", fused_partial_lse),
        ("fallback_partial_o", fallback_partial_o),
        ("fallback_partial_lse", fallback_partial_lse),
    )
    for name, tensor in tensors_to_check:
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            raise AssertionError(
                f"[DSA_VALIDATE] {name}: non_finite nan={nan_count} inf={inf_count}"
            )

    o_stats = _format_validation_stats(
        "partial_o", fused_partial_o, fallback_partial_o
    )
    lse_stats = _format_validation_stats(
        "partial_lse", fused_partial_lse, fallback_partial_lse
    )
    summary = f"[DSA_VALIDATE] {o_stats}; {lse_stats}"

    if mode == "summary_once":
        if _split_validation_emitted:
            return
        print(summary)
        _split_validation_emitted = True
        return

    if mode == "assert_once":
        if _split_validation_emitted:
            return
        tol_abs = float(os.environ.get("FLASHMLA_DSA_VALIDATE_ABS", "1e-1"))
        tol_rel = float(os.environ.get("FLASHMLA_DSA_VALIDATE_REL", "1e-1"))
        diff_o = (fused_partial_o - fallback_partial_o).abs()
        diff_lse = (fused_partial_lse - fallback_partial_lse).abs()
        rel_o = diff_o / fallback_partial_o.abs().clamp_min(1e-6)
        rel_lse = diff_lse / fallback_partial_lse.abs().clamp_min(1e-6)
        if (
            diff_o.max().item() > tol_abs
            or diff_lse.max().item() > tol_abs
            or rel_o.max().item() > tol_rel
            or rel_lse.max().item() > tol_rel
        ):
            raise AssertionError(summary)
        print(f"{summary} [within_tol]")
        _split_validation_emitted = True
        return

    if mode in {"1", "summary", "warn"}:
        print(summary)
        return

    if mode == "assert":
        tol_abs = float(os.environ.get("FLASHMLA_DSA_VALIDATE_ABS", "1e-1"))
        tol_rel = float(os.environ.get("FLASHMLA_DSA_VALIDATE_REL", "1e-1"))
        diff_o = (fused_partial_o - fallback_partial_o).abs()
        diff_lse = (fused_partial_lse - fallback_partial_lse).abs()
        rel_o = diff_o / fallback_partial_o.abs().clamp_min(1e-6)
        rel_lse = diff_lse / fallback_partial_lse.abs().clamp_min(1e-6)
        if (
            diff_o.max().item() > tol_abs
            or diff_lse.max().item() > tol_abs
            or rel_o.max().item() > tol_rel
            or rel_lse.max().item() > tol_rel
        ):
            raise AssertionError(summary)
        print(f"{summary} [within_tol]")
        return

    raise ValueError(
        "FLASHMLA_DSA_VALIDATE_FUSED must be one of: '', 'summary', 'warn', 'summary_once', 'assert_once', 'assert'"
    )


def _compute_split_partials(q_nope, q_pe, kc, kp, vm, sm_scale_f):
    """
    CuTeDSL QK GEMM + PyTorch softmax/SV.

    Args:
        q_nope: [T, H, D] bf16
        q_pe:   [T, H, Dp] bf16
        kc:     [T, S, N, D] bf16
        kp:     [T, S, N, Dp] bf16
        vm:     [T, S, N] bool
        sm_scale_f: float

    Returns:
        partial_o:   [S, T, H, D] float32
        partial_lse: [S, T, H]    float32 (log2 base)
    """
    T = q_nope.shape[0]
    H = NUM_Q_HEADS
    S = NUM_SPLITS
    N = CHUNK_SIZE
    D = CKV_DIM
    device = q_nope.device
    logits = _compute_qk_logits_cutedsl(q_nope, q_pe, kc, kp)
    logits = logits * sm_scale_f

    mask = vm[:, :, None, :]
    split_valid = vm.any(dim=-1, keepdim=True)
    logits.masked_fill_(~mask, -float("inf"))

    partial_lse = torch.full((T, S, H), -float("inf"), dtype=torch.float32, device=device)
    valid_splits = split_valid.expand(-1, -1, H)
    partial_lse[valid_splits] = torch.logsumexp(logits, dim=-1)[valid_splits] * LOG2E

    logits = torch.where(split_valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(split_valid.unsqueeze(-1), attn, torch.zeros_like(attn))

    kc_f = kc.float()
    attn_r = attn.reshape(T * S, H, N)
    kc_sv = kc_f.reshape(T * S, N, D)
    partial_o = torch.bmm(attn_r, kc_sv).view(T, S, H, D)

    partial_o = partial_o.permute(1, 0, 2, 3).contiguous()
    partial_lse = partial_lse.permute(1, 0, 2).contiguous()
    return partial_o, partial_lse


def _compute_split_partials_pytorch(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f):
    """Pure PyTorch reference for split partials — used for debugging gating."""
    T = q_nope.shape[0]
    H = NUM_Q_HEADS
    S = NUM_SPLITS
    N = CHUNK_SIZE
    D = CKV_DIM
    device = q_nope.device

    ckv_flat, kpe_flat = _flatten_kv_caches(ckv_cache, kpe_cache)
    total_kv = ckv_flat.shape[0]
    safe_indices, valid_mask = _normalize_sparse_indices(sparse_indices, total_kv)

    vm = valid_mask.view(T, S, N)
    kc = ckv_flat[safe_indices.reshape(-1)].view(T, S, N, D)
    kp = kpe_flat[safe_indices.reshape(-1)].view(T, S, N, KPE_DIM)
    kc = kc.masked_fill(~vm.unsqueeze(-1), 0).contiguous()
    kp = kp.masked_fill(~vm.unsqueeze(-1), 0).contiguous()

    # QK GEMM: logits[t,s,h,n] = q_nope[t,h,:] @ kc[t,s,n,:].T + q_pe[t,h,:] @ kp[t,s,n,:].T
    qn = q_nope.float()  # [T, H, D]
    qp = q_pe.float()    # [T, H, KPE_DIM]
    kc_f = kc.float()    # [T, S, N, D]
    kp_f = kp.float()    # [T, S, N, KPE_DIM]

    logits = torch.einsum('thd,tsnd->tshn', qn, kc_f) + torch.einsum('thd,tsnd->tshn', qp, kp_f)
    logits = logits * sm_scale_f

    mask = vm[:, :, None, :]  # [T, S, 1, N]
    split_valid = vm.any(dim=-1, keepdim=True)  # [T, S, 1]
    logits.masked_fill_(~mask, -float("inf"))

    partial_lse = torch.full((T, S, H), -float("inf"), dtype=torch.float32, device=device)
    valid_splits = split_valid.expand(-1, -1, H)
    partial_lse[valid_splits] = torch.logsumexp(logits, dim=-1)[valid_splits] * LOG2E

    logits = torch.where(split_valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(split_valid.unsqueeze(-1), attn, torch.zeros_like(attn))

    # SV: partial_o[t,s,h,d] = attn[t,s,h,n] @ kc_f[t,s,n,d]
    attn_r = attn.reshape(T * S, H, N)
    kc_sv = kc_f.reshape(T * S, N, D)
    partial_o = torch.bmm(attn_r, kc_sv).view(T, S, H, D)

    partial_o = partial_o.permute(1, 0, 2, 3).contiguous()  # [S, T, H, D]
    partial_lse = partial_lse.permute(1, 0, 2).contiguous()  # [S, T, H]
    return partial_o, partial_lse


def _get_debug_gate_level() -> int:
    """Select debug gate from the environment to support suffix-gating bring-up."""
    return int(os.environ.get("FLASHMLA_DSA_DEBUG_GATE", "0"))


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute DSA sparse attention — fused CuTeDSL kernel is authoritative."""
    sm_scale_f = _to_python_float(sm_scale)
    debug_gate_level = _get_debug_gate_level()

    if debug_gate_level == 1:
        partial_o, partial_lse = _compute_split_partials_pytorch(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f
        )
        output, final_lse = _combine_splits(partial_o, partial_lse)
        return output, final_lse

    if debug_gate_level == 2:
        kc, kp, vm = _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)
        partial_o, partial_lse = _compute_split_partials(
            q_nope, q_pe, kc, kp, vm, sm_scale_f
        )
        output, final_lse = _combine_splits(partial_o, partial_lse)
        return output, final_lse

    # Run fused kernel
    fused_o, fused_lse = _compute_split_partials_fused(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f
    )

    if debug_gate_level in (3, 4, 5, 6):
        # Also run PyTorch reference
        ref_o, ref_lse = _compute_split_partials_pytorch(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f
        )

        if debug_gate_level == 6:
            # Diagnostic: print comparison stats
            lse_nan = torch.isnan(fused_lse).sum().item()
            lse_inf = torch.isinf(fused_lse).sum().item()
            o_nan = torch.isnan(fused_o).sum().item()
            o_inf = torch.isinf(fused_o).sum().item()
            print(f"[DIAG] fused_lse: nan={lse_nan} inf={lse_inf} shape={list(fused_lse.shape)}")
            print(f"[DIAG] fused_o:   nan={o_nan} inf={o_inf} shape={list(fused_o.shape)}")

            # Compare finite elements
            fin_lse = torch.isfinite(fused_lse) & torch.isfinite(ref_lse)
            if fin_lse.any():
                diff_lse = (fused_lse[fin_lse] - ref_lse[fin_lse]).abs()
                print(f"[DIAG] lse diff (finite only): max={diff_lse.max().item():.4e} mean={diff_lse.mean().item():.4e} n={fin_lse.sum().item()}")
            else:
                print("[DIAG] lse: no finite elements in common")

            fin_o = torch.isfinite(fused_o) & torch.isfinite(ref_o)
            if fin_o.any():
                diff_o = (fused_o[fin_o] - ref_o[fin_o]).abs()
                print(f"[DIAG] o diff (finite only): max={diff_o.max().item():.4e} mean={diff_o.mean().item():.4e} n={fin_o.sum().item()}")
            else:
                print("[DIAG] o: no finite elements in common")

            # Sample: print first split, first token
            ckv_flat, kpe_flat = _flatten_kv_caches(ckv_cache, kpe_cache)
            safe_indices, valid_mask = _normalize_sparse_indices(
                sparse_indices, ckv_flat.shape[0]
            )
            kc_dbg, kp_dbg, vm_dbg = _gather_kv_chunks(
                ckv_cache, kpe_cache, sparse_indices
            )
            standalone_logits = _compute_qk_logits_cutedsl(
                q_nope, q_pe, kc_dbg, kp_dbg
            )
            idx0 = safe_indices[0, :CHUNK_SIZE]
            vm0 = valid_mask[0, :CHUNK_SIZE]
            kc0 = ckv_flat[idx0].float()
            kp0 = kpe_flat[idx0].float()
            standalone_score0 = standalone_logits[0, 0].float() * (sm_scale_f * LOG2E)
            standalone_score0 = torch.where(
                vm_dbg[0, 0].unsqueeze(0),
                standalone_score0,
                torch.full_like(standalone_score0, -float("inf")),
            )
            ref_score_ckv0 = (q_nope[0].float() @ kc0.T) * (sm_scale_f * LOG2E)
            ref_score_kpe0 = (q_pe[0].float() @ kp0.T) * (sm_scale_f * LOG2E)
            ref_score0 = (
                q_nope[0].float() @ kc0.T + q_pe[0].float() @ kp0.T
            ) * (sm_scale_f * LOG2E)
            debug_ckv_dim = _DEBUG_SPLIT_CKV_TILE_COUNT * N_BLOCK
            ref_score_debug0 = (
                q_nope[0, :, :debug_ckv_dim].float() @ kc0[:, :debug_ckv_dim].T
            )
            if _DEBUG_SPLIT_INCLUDE_KPE:
                ref_score_debug0 = ref_score_debug0 + (q_pe[0].float() @ kp0.T)
            ref_score0 = torch.where(
                vm0.unsqueeze(0),
                ref_score0,
                torch.full_like(ref_score0, -float("inf")),
            )
            ref_score_debug0 = torch.where(
                vm0.unsqueeze(0),
                ref_score_debug0 * (sm_scale_f * LOG2E),
                torch.full_like(ref_score_debug0, -float("inf")),
            )
            print(
                "[DIAG] ref_ckv idx0=%d idx1=%d row0[0]=%.6f row0[1]=%.6f row1[0]=%.6f"
                % (
                    int(idx0[0].item()),
                    int(idx0[1].item()),
                    kc0[0, 0].item(),
                    kc0[0, 1].item(),
                    kc0[1, 0].item(),
                )
            )
            print(f"[DIAG] ref_qn4x4 = {q_nope[0, :4, :4].float().tolist()}")
            print(f"[DIAG] ref_qp4x4 = {q_pe[0, :4, :4].float().tolist()}")
            print(f"[DIAG] ref_ckv4x4 = {kc0[:4, :4].tolist()}")
            print(f"[DIAG] ref_kpe4x4 = {kp0[:4, :4].tolist()}")
            print(
                "[DIAG] ref_score[0,0]=%.6f ref_score[0,1]=%.6f ref_score[1,0]=%.6f ref_score[15,0]=%.6f valid0=%d"
                % (
                    ref_score0[0, 0].item(),
                    ref_score0[0, 1].item(),
                    ref_score0[1, 0].item(),
                    ref_score0[15, 0].item(),
                    int(vm0.sum().item()),
                )
            )
            fin_standalone = torch.isfinite(standalone_score0) & torch.isfinite(ref_score0)
            if fin_standalone.any():
                standalone_diff = (
                    standalone_score0[fin_standalone] - ref_score0[fin_standalone]
                ).abs()
                print(
                    "[DIAG] standalone score diff (finite only): max=%.4e mean=%.4e n=%d"
                    % (
                        standalone_diff.max().item(),
                        standalone_diff.mean().item(),
                        int(fin_standalone.sum().item()),
                    )
                )
            else:
                print("[DIAG] standalone score: no finite elements in common")
            print(f"[DIAG] ref_score4x4 = {ref_score0[:4, :4].tolist()}")
            print(
                "[DIAG] split_debug cfg: ckv_tiles=%d ckv_dim=%d include_kpe=%d"
                % (
                    _DEBUG_SPLIT_CKV_TILE_COUNT,
                    debug_ckv_dim,
                    int(_DEBUG_SPLIT_INCLUDE_KPE),
                )
            )
            print(f"[DIAG] ref_score_debug4x4 = {ref_score_debug0[:4, :4].tolist()}")
            print(f"[DIAG] standalone_score4x4 = {standalone_score0[:4, :4].tolist()}")
            print(
                "[DIAG] ref_score_ckv[0,:4] = %s"
                % ref_score_ckv0[0, :4].tolist()
            )
            print(
                "[DIAG] ref_score_kpe[0,:4] = %s"
                % ref_score_kpe0[0, :4].tolist()
            )
            print(f"[DIAG] fused_lse[0,0,:4] = {fused_lse[0,0,:4].tolist()}")
            print(f"[DIAG] ref_lse[0,0,:4]   = {ref_lse[0,0,:4].tolist()}")
            print(f"[DIAG] fused_o[0,0,0,:8] = {fused_o[0,0,0,:8].tolist()}")
            print(f"[DIAG] ref_o[0,0,0,:8]   = {ref_o[0,0,0,:8].tolist()}")

        if debug_gate_level == 3:
            # Replace the latest-stage output partials only; tests WG2 PV epilogue/store.
            output, final_lse = _combine_splits(ref_o, fused_lse)
            return output, final_lse
        if debug_gate_level == 4:
            # Replace the latest-stage LSE only; tests whether fused partial_o is usable.
            output, final_lse = _combine_splits(fused_o, ref_lse)
            return output, final_lse
        if debug_gate_level == 5:
            # Full suffix gate for host-side sanity.
            output, final_lse = _combine_splits(ref_o, ref_lse)
            return output, final_lse

    output, final_lse = _combine_splits(fused_o, fused_lse)
    return output, final_lse


@torch.no_grad()
def _compile_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compile the fused split kernel for the provided problem shapes."""
    from cutlass.base_dsl.compiler import CompileCallable
    import cuda.bindings.driver as cuda

    _ = _to_python_float(sm_scale)
    (_, _, _, mQ, mQp, mCkv, mKpe, mIdx, mO, mLse) = _prepare_split_shell_operands(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiler = CompileCallable()
    compiler(_get_split_shell_launcher(), mQ, mQp, mCkv, mKpe, mIdx, mO, mLse, 1.0, stream)


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """DPS-style kernel entry point — writes output and lse in place."""
    out, lse_out = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output.copy_(out)
    lse.copy_(lse_out)


run.compile_only = _compile_only


def _cutedsl_smoke_compile() -> None:
    q_nope = torch.randn((1, NUM_Q_HEADS, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn((1, NUM_Q_HEADS, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    ckv_cache = torch.randn((8, PAGE_SIZE, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    kpe_cache = torch.randn((8, PAGE_SIZE, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    sparse_indices = torch.randint(
        0, 8 * PAGE_SIZE, (1, TOPK), device="cuda", dtype=torch.int32
    )
    run.compile_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, 1.0)
    print("CuTeDSL fused kernel smoke compile PASS")


if __name__ == "__main__":
    _cutedsl_smoke_compile()