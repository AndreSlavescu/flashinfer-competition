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

# CuTeDSL imports (lazy, only on GPU)
_cutedsl_kernel = None
_split_shell_launcher = None
_sm100_dsl_helpers = None
_split_validation_emitted = False


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
    def _make_gather_rowgroup_view(s_src: cute.Tensor) -> cute.Tensor:
        grouped_layout = cute.make_layout(
            ((4, CHUNK_SIZE // 4), M_BLOCK, 1),
            stride=((M_BLOCK, 4 * M_BLOCK), 1, CHUNK_SIZE * M_BLOCK),
        )
        return cute.make_tensor(s_src.iterator, grouped_layout)

    @cute.jit
    def _copy_rowmajor_stage_to_operand_a(
        s_src_row: cute.Tensor,
        s_dst: cute.Tensor,
        dst_swizzle,
        tidx: Int32,
    ) -> None:
        """Copy from sQrow (M_BLOCK x N_BLOCK col-major SMEM) to swizzled operand A SMEM.

        Uses _make_rowmajor_mn_view (source) and _make_swizzled_mn_view (dest)
        to create matching hierarchical layouts, then a tiled copy transfers the
        data element-by-element with correct layout translation.
        """
        smem_copy_atom_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            BFloat16,
            num_bits_per_copy=16,
        )
        thr_smem_copy = cute.make_tiled_copy_tv(
            smem_copy_atom_bf16,
            cute.make_layout(num_threads),
            cute.make_layout(1),
        ).get_slice(tidx)
        s_src = _make_rowmajor_mn_view(s_src_row, s_dst)
        s_dst_view = _make_swizzled_mn_view(s_dst, dst_swizzle)
        tSs = thr_smem_copy.partition_S(s_src)
        tDs = thr_smem_copy.partition_D(s_dst_view)
        cute.copy(thr_smem_copy, tSs, tDs)

    @cute.jit
    def _copy_rowmajor_stage_to_operand_b(
        s_src_row: cute.Tensor,
        s_dst: cute.Tensor,
        dst_swizzle,
        tidx: Int32,
    ) -> None:
        _copy_rowmajor_stage_to_operand_a(s_src_row, s_dst, dst_swizzle, tidx)

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
        tSR_rIdx = _load_s2r(tSR_sIdx)
        tma_desc_ptr = _get_tma_desc_addr(tma_atom)

        def copy_fn(src_idx, dst_idx, tma_bar_ptr: cute.Pointer):
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
        copy_rowmajor_stage_to_operand_a=_copy_rowmajor_stage_to_operand_a,
        copy_rowmajor_stage_to_operand_b=_copy_rowmajor_stage_to_operand_b,
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
            mQ_nope_padded: cute.Tensor,
            mQ_pe_padded: cute.Tensor,
            mCkv_flat: cute.Tensor,
            mKpe_flat: cute.Tensor,
            mIdx: cute.Tensor,
            mPartialO: cute.Tensor,
            mPartialLse: cute.Tensor,
            sm_scale: Float32,
            stream: cuda.CUstream,
        ):
            cluster_shape_mn = (1, 1)
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
                1,
            )
            k_smem_layout_staged = sm100_utils.make_smem_layout_b(
                qk_tiled_mma,
                qk_mma_tiler,
                BFloat16,
                1,
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
                pv_tiled_mma, pv_mma_tiler, BFloat16, 1,
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

            # --- TMA atoms for Q operand A (direct load bypassing buggy SMEM copy) ---
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*cluster_shape_mn, 1)),
                (qk_tiled_mma.thr_id.shape,),
            )
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                cluster_shape_mn, qk_tiled_mma.thr_id
            )
            a_smem = cute.slice_(q_smem_layout_staged, (None, None, None, 0))
            tma_atom_qn, tma_Qn = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mQ_nope_padded, a_smem, qk_mma_tiler, qk_tiled_mma,
                cluster_layout_vmnk.shape,
            )
            tma_atom_qp, tma_Qp = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mQ_pe_padded, a_smem, qk_mma_tiler, qk_tiled_mma,
                cluster_layout_vmnk.shape,
            )
            atom_thr_size = cute.size(qk_tiled_mma.thr_id.shape)
            q_smem_one_stage = cute.slice_(q_smem_layout_staged, (None, None, None, 0))
            self.num_q_tma_bytes = cute.size_in_bytes(BFloat16, q_smem_one_stage) * atom_thr_size

            self.kernel(
                qk_tiled_mma,
                pv_tiled_mma,
                tma_atom_ckv,
                tma_atom_kpe,
                tma_atom_qn,
                tma_Qn,
                tma_atom_qp,
                tma_Qp,
                q_smem_layout_staged,
                k_smem_layout_staged,
                pv_b_smem_layout,
                p_smem_layout_staged,
                cluster_layout_vmnk,
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
                cluster=(1, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            qk_tiled_mma_: cute.TiledMma,
            pv_tiled_mma_: cute.TiledMma,
            tma_atom_ckv: cute.CopyAtom,
            tma_atom_kpe: cute.CopyAtom,
            tma_atom_qn: cute.CopyAtom,
            mQ_nope_tma: cute.Tensor,
            tma_atom_qp: cute.CopyAtom,
            mQ_pe_tma: cute.Tensor,
            q_smem_layout_staged: cute.ComposedLayout,
            k_smem_layout_staged: cute.ComposedLayout,
            pv_b_smem_layout: cute.ComposedLayout,
            p_smem_layout_staged: cute.ComposedLayout,
            cluster_layout_vmnk: cute.Layout,
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
            mma_wg_sync_barrier = pipeline.NamedBarrier(
                barrier_id=3,
                num_threads=WARPGROUP_THREADS,
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
            _ = qk_tiled_mma_
            _ = pv_tiled_mma_
            _ = tma_atom_ckv
            _ = tma_atom_kpe
            _ = tma_atom_qn
            _ = mQ_nope_tma
            _ = tma_atom_qp
            _ = mQ_pe_tma
            _ = mQ_nope
            _ = mQ_pe
            _ = mKpe_flat

            @cute.struct
            class SharedStorage:
                operand_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                score_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                v_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                output_pipe_mbar: cute.struct.MemRange[cutlass.Int64, 2]
                gather_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                q_tma_mbar: cute.struct.MemRange[cutlass.Int64, 1]
                tmem_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            _storage = smem.allocate(SharedStorage)
            gather_mbar_ptr = _storage.gather_mbar.data_ptr()
            q_tma_mbar_ptr = _storage.q_tma_mbar.data_ptr()
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
                    cute.arch.mbarrier_init(q_tma_mbar_ptr, 1)
                cpasync.prefetch_descriptor(tma_atom_qn)
                cpasync.prefetch_descriptor(tma_atom_qp)
            cute.arch.mbarrier_init_fence()

            # --- TMA partition setup for Q operand A ---
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            acl = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            # Partition Q_nope global -> TMA src/dst
            qk_mma_tiler_local = (M_BLOCK, N_BLOCK, N_BLOCK)
            gQn = cute.local_tile(
                mQ_nope_tma,
                cute.slice_(qk_mma_tiler_local, (None, 0, None)),
                (None, None, None),
            )
            qk_thr_mma = qk_tiled_mma_.get_slice(
                block_in_cluster_coord_vmnk[0]
            )
            tCgQn = qk_thr_mma.partition_A(gQn)
            tAsQn, tAgQn = cpasync.tma_partition(
                tma_atom_qn,
                block_in_cluster_coord_vmnk[2],
                acl,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tCgQn, 0, 3),
            )
            # Partition Q_pe global -> TMA src/dst
            gQp = cute.local_tile(
                mQ_pe_tma,
                cute.slice_(qk_mma_tiler_local, (None, 0, None)),
                (None, None, None),
            )
            tCgQp = qk_thr_mma.partition_A(gQp)
            tAsQp, tAgQp = cpasync.tma_partition(
                tma_atom_qp,
                block_in_cluster_coord_vmnk[2],
                acl,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tCgQp, 0, 3),
            )
            # Select the correct batch slice (token_idx)
            tAgQn = tAgQn[(None, 0, None, token_idx)]
            tAgQp = tAgQp[(None, 0, None, token_idx)]
            q_tma_phase = Int32(0)

            operand_producer, operand_consumer = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                consumer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                barrier_storage=_storage.operand_pipe_mbar.data_ptr(),
                defer_sync=True,
            ).make_participants()
            score_producer, score_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(1),
                consumer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                barrier_storage=_storage.score_pipe_mbar.data_ptr(),
                defer_sync=True,
            ).make_participants()
            v_producer, v_consumer = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                consumer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                barrier_storage=_storage.v_pipe_mbar.data_ptr(),
                defer_sync=True,
            ).make_participants()
            output_producer, output_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=make_thread_cooperative_group(1),
                consumer_group=make_thread_cooperative_group(WARPGROUP_THREADS),
                barrier_storage=_storage.output_pipe_mbar.data_ptr(),
                defer_sync=True,
            ).make_participants()
            # Fence all mbarrier inits and synchronize all threads
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
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
            _ = sK
            _ = sV
            _ = sP
            _ = sScoreF32
            gather_phase = Int32(0)
            _ = gather_phase
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
            cScore = cute.make_identity_tensor((M_BLOCK, N_BLOCK))
            cScore_epi = cute.flat_divide(cScore, (M_BLOCK, N_BLOCK))
            tTR_cScore = score_thr_t2r.partition_D(cScore_epi)
            tTR_cScore = cute.group_modes(tTR_cScore, 3, cute.rank(tTR_cScore))
            tTR_rScore = cute.make_fragment_like(
                tTR_cScore[(None, None, None, 0)], Float32
            )
            num_score_epi_tiles = cute.size(tTR_tScore.shape, mode=[3])

            if warp_idx < WARPS_PER_WARPGROUP:
                gather_ckv_copy = make_gather4_copy_fn(
                    tma_atom_ckv,
                    sQrow,
                    sIdx,
                    sGroupMask,
                    Int32(0),
                )
                gather_kpe_copy = make_gather4_copy_fn(
                    tma_atom_kpe,
                    sQrow,
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

                # === CKV K-tile loop (8 tiles of 64 cols each) ===
                NUM_CKV_K_TILES = CKV_DIM // N_BLOCK
                for k_tile in cutlass.range(NUM_CKV_K_TILES, unroll=1):
                    operand_handle = operand_producer.acquire_and_advance()

                    # Load Q_nope[:, k_tile*64:(k_tile+1)*64] -> sQrow -> operand A
                    for elem in cutlass.range(
                        warpgroup_tidx, M_BLOCK * N_BLOCK, WARPGROUP_THREADS, unroll=1
                    ):
                        row = elem // N_BLOCK
                        col = elem % N_BLOCK
                        if row < NUM_Q_HEADS:
                            sQrow[row, col, 0] = mQ_nope[
                                token_idx, row, k_tile * Int32(N_BLOCK) + col
                            ]
                        else:
                            sQrow[row, col, 0] = BFloat16(0.0)
                    load_wg_sync_barrier.arrive_and_wait()
                    copy_rowmajor_stage_to_operand_a(
                        sQrow,
                        sQ[None, None, None, 0],
                        q_smem_layout_staged.inner,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()

                    # Gather CKV[:, k_tile*64:(k_tile+1)*64] -> operand B
                    for elem in cutlass.range(
                        warpgroup_tidx,
                        CHUNK_SIZE * N_BLOCK,
                        WARPGROUP_THREADS,
                        unroll=1,
                    ):
                        row = elem // N_BLOCK
                        col = elem % N_BLOCK
                        sQrow[row, col, 0] = BFloat16(0.0)
                    load_wg_sync_barrier.arrive_and_wait()
                    if warp_idx == 0:
                        gather_ckv_copy(k_tile, 0, gather_mbar_ptr)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_expect_tx(
                                gather_mbar_ptr,
                                sGatherBytes[0],
                            )
                            cute.arch.mbarrier_arrive(gather_mbar_ptr)
                    cute.arch.mbarrier_wait(gather_mbar_ptr, gather_phase)
                    cute.arch.fence_view_async_shared()
                    load_wg_sync_barrier.arrive_and_wait()
                    copy_rowmajor_stage_to_operand_b(
                        sQrow,
                        sK[None, None, None, 0],
                        k_smem_layout_staged.inner,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()
                    gather_phase = gather_phase ^ Int32(1)
                    operand_handle.commit()

                # === KPE pass (1 tile of 64 cols) ===
                operand_handle = operand_producer.acquire_and_advance()

                # Load Q_pe -> sQrow -> operand A
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
                    sQ[None, None, None, 0],
                    q_smem_layout_staged.inner,
                    warpgroup_tidx,
                )
                load_wg_sync_barrier.arrive_and_wait()

                # Gather KPE -> operand B
                for elem in cutlass.range(
                    warpgroup_tidx,
                    CHUNK_SIZE * N_BLOCK,
                    WARPGROUP_THREADS,
                    unroll=1,
                ):
                    row = elem // N_BLOCK
                    col = elem % N_BLOCK
                    sQrow[row, col, 0] = BFloat16(0.0)
                load_wg_sync_barrier.arrive_and_wait()
                if warp_idx == 0:
                    gather_kpe_copy(0, 0, gather_mbar_ptr)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(
                            gather_mbar_ptr,
                            sGatherBytes[0],
                        )
                        cute.arch.mbarrier_arrive(gather_mbar_ptr)
                cute.arch.mbarrier_wait(gather_mbar_ptr, gather_phase)
                cute.arch.fence_view_async_shared()
                load_wg_sync_barrier.arrive_and_wait()
                copy_rowmajor_stage_to_operand_b(
                    sQrow,
                    sK[None, None, None, 0],
                    k_smem_layout_staged.inner,
                    warpgroup_tidx,
                )
                load_wg_sync_barrier.arrive_and_wait()
                gather_phase = gather_phase ^ Int32(1)
                operand_handle.commit()
                valid_wg_sync_barrier.arrive()

            if warp_idx >= WARPS_PER_WARPGROUP and warp_idx < 2 * WARPS_PER_WARPGROUP:
                score_full = score_producer.acquire_and_advance()

                # First CKV K-tile: clear accumulator (ACCUMULATE=False for kblk 0)
                operand_full = operand_consumer.wait_and_advance()
                for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                    qk_tiled_mma_.set(
                        tcgen05.Field.ACCUMULATE,
                        kblk_idx != 0,
                    )
                    cute.gemm(
                        qk_tiled_mma_,
                        tStS,
                        tSrQ[(None, None, kblk_idx, 0)],
                        tSrK[(None, None, kblk_idx, 0)],
                        tStS,
                    )
                operand_full.release()

                # Remaining CKV K-tiles (7) + KPE (1) = 8 passes with accumulate
                for _qk_pass in cutlass.range(CKV_DIM // N_BLOCK, unroll=1):
                    operand_full = operand_consumer.wait_and_advance()
                    for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                        qk_tiled_mma_.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            qk_tiled_mma_,
                            tStS,
                            tSrQ[(None, None, kblk_idx, 0)],
                            tSrK[(None, None, kblk_idx, 0)],
                            tStS,
                        )
                    operand_full.release()

                mma_wg_sync_barrier.arrive_and_wait()
                if warp_idx == WARPS_PER_WARPGROUP:
                    score_producer.commit(score_full)

            if (
                warp_idx >= 2 * WARPS_PER_WARPGROUP
                and warp_idx < 3 * WARPS_PER_WARPGROUP
            ):
                valid_wg_sync_barrier.arrive_and_wait()
                score_full = score_consumer.wait_and_advance()
                scale_log2e = sm_scale * Float32(LOG2E)

                # Read scores from TMEM -> sScoreF32 via T2R epilogue copy
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
                        # Apply sm_scale and mask invalid entries
                        score_val = tTR_rScore[i] * scale_log2e
                        if row < Int32(NUM_Q_HEADS):
                            if sValid[col] == Int32(0):
                                score_val = Float32(-float("inf"))
                        else:
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
                    p_smem_layout_staged.inner,
                    warpgroup_tidx,
                )
                softmax_wg_sync_barrier.arrive_and_wait()

                score_full.release()

            cute.arch.sync_threads()

            # Reuse score TMEM allocation as PV output accumulator
            # (same 64x64 FP32 shape, no need to free and reallocate)
            pv_acc_shape = pv_tiled_mma_.partition_shape_C((M_BLOCK, N_BLOCK))
            pv_acc_fake = pv_tiled_mma_.make_fragment_C(pv_acc_shape)
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
                    sQrow,
                    sIdx,
                    sGroupMask,
                    Int32(0),
                )
                for v_tile_idx in cutlass.range(NUM_V_TILES, unroll=1):
                    v_handle = v_producer.acquire_and_advance()
                    # Clear sQrow for V gather staging
                    for elem in cutlass.range(
                        warpgroup_tidx,
                        CHUNK_SIZE * N_BLOCK,
                        WARPGROUP_THREADS,
                        unroll=1,
                    ):
                        row = elem // N_BLOCK
                        col = elem % N_BLOCK
                        sQrow[row, col, 0] = BFloat16(0.0)
                    load_wg_sync_barrier.arrive_and_wait()
                    # Gather V tile (columns v_tile_idx*N_BLOCK:(v_tile_idx+1)*N_BLOCK)
                    if warp_idx == 0:
                        gather_v_copy(v_tile_idx, 0, gather_mbar_ptr)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_expect_tx(
                                gather_mbar_ptr,
                                sGatherBytes[0],
                            )
                            cute.arch.mbarrier_arrive(gather_mbar_ptr)
                    cute.arch.mbarrier_wait(gather_mbar_ptr, gather_phase)
                    cute.arch.fence_view_async_shared()
                    load_wg_sync_barrier.arrive_and_wait()
                    # Stage from rowmajor to PV B-operand layout
                    copy_rowmajor_stage_to_operand_b(
                        sQrow,
                        sV[None, None, None, 0],
                        pv_b_smem_layout.inner,
                        warpgroup_tidx,
                    )
                    load_wg_sync_barrier.arrive_and_wait()
                    gather_phase = gather_phase ^ Int32(1)
                    v_handle.commit()

            if warp_idx >= WARPS_PER_WARPGROUP and warp_idx < 2 * WARPS_PER_WARPGROUP:
                tPr = pv_tiled_mma_.make_fragment_A(sP)
                tVr = pv_tiled_mma_.make_fragment_B(sV)
                pv_num_kblks = cute.size(tPr, mode=[2])
                for v_tile_idx in cutlass.range(NUM_V_TILES, unroll=1):
                    output_full = output_producer.acquire_and_advance()
                    v_full = v_consumer.wait_and_advance()
                    for kblk_idx in cutlass.range(pv_num_kblks, unroll_full=True):
                        pv_tiled_mma_.set(
                            tcgen05.Field.ACCUMULATE, kblk_idx != 0,
                        )
                        cute.gemm(
                            pv_tiled_mma_,
                            tOtO,
                            tPr[(None, None, kblk_idx, 0)],
                            tVr[(None, None, kblk_idx, 0)],
                            tOtO,
                        )
                    v_full.release()
                    mma_wg_sync_barrier.arrive_and_wait()
                    if warp_idx == WARPS_PER_WARPGROUP:
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


def _allocate_fused_output_scratch(T, device):
    """Allocate padded fused-output scratch for WG2 epilogue stores."""
    partial_o_tma = torch.empty(
        (NUM_SPLITS, T, M_BLOCK, CKV_DIM), dtype=torch.float32, device=device
    )
    partial_o = partial_o_tma[:, :, :NUM_Q_HEADS, :]
    partial_lse = torch.empty(
        (NUM_SPLITS, T, NUM_Q_HEADS), dtype=torch.float32, device=device
    )
    return partial_o_tma, partial_o, partial_lse


def _prepare_split_shell_operands(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices):
    """Wrap fused kernel operands plus padded output scratch as CuTeDSL tensors."""
    from cutlass.cute.runtime import from_dlpack

    T = q_nope.shape[0]
    device = q_nope.device
    ckv_flat, kpe_flat = _flatten_kv_caches(ckv_cache, kpe_cache)
    partial_o_tma, partial_o, partial_lse = _allocate_fused_output_scratch(
        T, device
    )

    # Padded Q tensors for TMA: (M_BLOCK, K_DIM, T) layout
    # Zero-padded from NUM_Q_HEADS (16) to M_BLOCK (64)
    q_nope_padded = torch.zeros(T, M_BLOCK, CKV_DIM, dtype=torch.bfloat16, device=device)
    q_nope_padded[:, :NUM_Q_HEADS, :] = q_nope
    q_nope_3d = q_nope_padded.permute(1, 2, 0)  # (64, 512, T) strides=(512, 1, M*K) — TMA needs K-stride=1

    q_pe_padded = torch.zeros(T, M_BLOCK, KPE_DIM, dtype=torch.bfloat16, device=device)
    q_pe_padded[:, :NUM_Q_HEADS, :] = q_pe
    q_pe_3d = q_pe_padded.permute(1, 2, 0)  # (64, 64, T) strides=(64, 1, M*K) — TMA needs K-stride=1

    return (
        partial_o_tma,
        partial_o,
        partial_lse,
        from_dlpack(q_nope.contiguous(), assumed_align=16),
        from_dlpack(q_pe.contiguous(), assumed_align=16),
        from_dlpack(q_nope_3d, assumed_align=16),
        from_dlpack(q_pe_3d, assumed_align=16),
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
        mQn_padded,
        mQp_padded,
        mCkv,
        mKpe,
        mIdx,
        mO,
        mLse,
    ) = _prepare_split_shell_operands(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _get_split_shell_launcher()(
        mQ, mQp, mQn_padded, mQp_padded, mCkv, mKpe, mIdx, mO, mLse,
        sm_scale_f, stream=stream,
    )
    torch.cuda.synchronize()

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
    s_base, mQn, mQp, mKc, mKp, mS, stream = _prepare_qk_gemm_operands(
        q_nope, q_pe, kc, kp
    )

    _get_cutedsl_kernel()(mQn, mQp, mKc, mKp, mS, stream=stream)
    torch.cuda.synchronize()

    logits = s_base[:, :H, :].view(T, S, H, N)
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


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute DSA sparse attention — CuTeDSL QK GEMM + PyTorch softmax/SV."""
    sm_scale_f = _to_python_float(sm_scale)

    # Dense path: CuTeDSL QK GEMM + PyTorch softmax/SV (proven correct)
    kc, kp, vm = _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)
    partial_o, partial_lse = _compute_split_partials(
        q_nope, q_pe, kc, kp, vm, sm_scale_f
    )

    output, final_lse = _combine_splits(partial_o, partial_lse)
    return output, final_lse


@torch.no_grad()
def _compile_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compile the BatchedQKKernel for the provided problem shapes."""
    import cuda.bindings.driver as cuda

    sm_scale_f = _to_python_float(sm_scale)
    kc, kp, vm = _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)
    s_base, mQn, mQp, mKc, mKp, mS, stream = _prepare_qk_gemm_operands(
        q_nope, q_pe, kc, kp
    )
    _get_cutedsl_kernel()(mQn, mQp, mKc, mKp, mS, stream=stream)
    torch.cuda.synchronize()


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
    print("CuTeDSL BatchedQK kernel smoke compile PASS")


def _test_fused_compile_only() -> None:
    """Test fused kernel JIT compilation without running."""
    import time
    from cutlass.base_dsl.compiler import CompileCallable
    import cuda.bindings.driver as cuda

    q_nope = torch.randn((1, NUM_Q_HEADS, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn((1, NUM_Q_HEADS, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    ckv_cache = torch.randn((8, PAGE_SIZE, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    kpe_cache = torch.randn((8, PAGE_SIZE, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    sparse_indices = torch.randint(
        0, 8 * PAGE_SIZE, (1, TOPK), device="cuda", dtype=torch.int32
    )

    t0 = time.time()
    (_, _, _, mQ, mQp, mQn_padded, mQp_padded, mCkv, mKpe, mIdx, mO, mLse) = _prepare_split_shell_operands(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices
    )
    print(f"Operands prepared in {time.time()-t0:.1f}s")

    t1 = time.time()
    launcher = _get_split_shell_launcher()
    print(f"Launcher created in {time.time()-t1:.1f}s")

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    t2 = time.time()
    compiler = CompileCallable()
    compiler(launcher, mQ, mQp, mQn_padded, mQp_padded, mCkv, mKpe, mIdx, mO, mLse, 1.0, stream)
    print(f"Fused kernel compiled in {time.time()-t2:.1f}s")
    print("FUSED COMPILE PASS")


def _test_fused_run() -> None:
    """Test fused kernel execution and compare against reference."""
    import time
    import cuda.bindings.driver as cuda

    q_nope = torch.randn((1, NUM_Q_HEADS, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn((1, NUM_Q_HEADS, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    ckv_cache = torch.randn((8, PAGE_SIZE, CKV_DIM), device="cuda", dtype=torch.bfloat16)
    kpe_cache = torch.randn((8, PAGE_SIZE, KPE_DIM), device="cuda", dtype=torch.bfloat16)
    sparse_indices = torch.randint(
        0, 8 * PAGE_SIZE, (1, TOPK), device="cuda", dtype=torch.int32
    )
    sm_scale_f = 0.135

    # Run fused kernel
    fused_po, fused_plse = _compute_split_partials_fused(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f
    )
    print(f"Fused partial_o: shape={fused_po.shape} nan={torch.isnan(fused_po).sum()} inf={torch.isinf(fused_po).sum()}")
    print(f"Fused partial_lse: shape={fused_plse.shape} nan={torch.isnan(fused_plse).sum()} inf={torch.isinf(fused_plse).sum()}")
    print(f"  partial_o range: [{fused_po.min():.4f}, {fused_po.max():.4f}]")
    print(f"  partial_lse range: [{fused_plse.min():.4f}, {fused_plse.max():.4f}]")

    # Run reference path
    kc, kp, vm = _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)
    ref_po, ref_plse = _compute_split_partials(
        q_nope, q_pe, kc, kp, vm, sm_scale_f
    )
    print(f"\nRef partial_o: shape={ref_po.shape}")
    print(f"  range: [{ref_po.min():.4f}, {ref_po.max():.4f}]")
    print(f"  partial_lse range: [{ref_plse.min():.4f}, {ref_plse.max():.4f}]")

    # Compare
    diff_o = (fused_po - ref_po).abs()
    diff_lse = (fused_plse - ref_plse).abs()

    # Check a few specific splits
    for s in [0, 1, 15, 31]:
        fo = fused_po[s, 0]  # [H, D]
        ro = ref_po[s, 0]
        fl = fused_plse[s, 0]  # [H]
        rl = ref_plse[s, 0]
        do = (fo - ro).abs().max().item()
        dl = (fl - rl).abs().max().item()
        print(f"  Split {s}: o_diff={do:.6e}, lse_diff={dl:.6e}, fused_lse={fl[:4].tolist()}, ref_lse={rl[:4].tolist()}")

    print(f"\nOverall: max_o_diff={diff_o.max():.6e}, max_lse_diff={diff_lse.max():.6e}")

    # Full end-to-end comparison
    fused_output, fused_lse = _combine_splits(fused_po, fused_plse)
    ref_output, ref_lse = _combine_splits(ref_po, ref_plse)
    out_diff = (fused_output.float() - ref_output.float()).abs().max().item()
    lse_diff = (fused_lse - ref_lse).abs().max().item()
    print(f"Final output diff: {out_diff:.6e}")
    print(f"Final LSE diff: {lse_diff:.6e}")
    nan_out = torch.isnan(fused_output).sum().item()
    nan_lse = torch.isnan(fused_lse).sum().item()
    print(f"Final NaN: output={nan_out}, lse={nan_lse}")
    print("FUSED RUN PASS")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fused-compile":
        _test_fused_compile_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "fused-run":
        _test_fused_run()
    else:
        _cutedsl_smoke_compile()
