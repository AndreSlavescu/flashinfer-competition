"""
DSA Sparse Attention Kernel v3 — CuTeDSL QK GEMM + PyTorch softmax/SV/combine.

Uses CuTeDSL tcgen05.mma (Blackwell UTCMMA) for the QK GEMM:
    S = Q_nope @ Kc^T + Q_pe @ Kp^T

PyTorch handles: gather, sm_scale, mask, softmax, SV matmul, combine.
CuTeDSL handles: The fused two-phase QK GEMM (CKV + KPE accumulate).

Entry points:
    run(...) -> (output, lse)
    kernel(..., output, lse) -> writes destination tensors in place
"""

import math
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

# CuTeDSL imports (lazy, only on GPU)
_cutedsl_kernel = None
_split_shell_launcher = None
_sm100_dsl_helpers = None


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
    utils = helpers.utils
    sm100_utils = helpers.sm100_utils
    pipeline = helpers.pipeline
    Float32 = helpers.Float32
    BFloat16 = helpers.BFloat16
    Int32 = helpers.Int32
    cuda = helpers.cuda
    copy_rowmajor_stage_to_operand_a = helpers.copy_rowmajor_stage_to_operand_a
    copy_rowmajor_stage_to_operand_b = helpers.copy_rowmajor_stage_to_operand_b

    row_major = utils.LayoutEnum.ROW_MAJOR
    col_major = utils.LayoutEnum.COL_MAJOR

    @cute.jit
    def _load_q_nope_tile(
        s_q_row: cute.Tensor,
        s_q: cute.Tensor,
        m_q_nope: cute.Tensor,
        q_swizzle,
        tidx: Int32,
        token_idx: Int32,
        col_base: Int32,
    ) -> None:
        for elem in cutlass.range(tidx, M_BLOCK * N_BLOCK, 128, unroll=1):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            if row < NUM_Q_HEADS:
                s_q_row[row, col, 0] = m_q_nope[token_idx, row, col_base + col]
            else:
                s_q_row[row, col, 0] = BFloat16(0.0)
        cute.arch.sync_threads()
        copy_rowmajor_stage_to_operand_a(
            s_q_row, s_q[None, None, None, 0], q_swizzle, tidx
        )
        cute.arch.sync_threads()

    @cute.jit
    def _load_q_pe_tile(
        s_q_row: cute.Tensor,
        s_q: cute.Tensor,
        m_q_pe: cute.Tensor,
        q_swizzle,
        tidx: Int32,
        token_idx: Int32,
    ) -> None:
        for elem in cutlass.range(tidx, M_BLOCK * N_BLOCK, 128, unroll=1):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            if row < NUM_Q_HEADS:
                s_q_row[row, col, 0] = m_q_pe[token_idx, row, col]
            else:
                s_q_row[row, col, 0] = BFloat16(0.0)
        cute.arch.sync_threads()
        copy_rowmajor_stage_to_operand_a(
            s_q_row, s_q[None, None, None, 0], q_swizzle, tidx
        )
        cute.arch.sync_threads()

    @cute.jit
    def _load_dense_k_tile(
        s_k_row: cute.Tensor,
        s_k: cute.Tensor,
        m_ckv_flat: cute.Tensor,
        s_idx: cute.Tensor,
        s_valid: cute.Tensor,
        k_swizzle,
        tidx: Int32,
        col_base: Int32,
    ) -> None:
        for elem in cutlass.range(tidx, CHUNK_SIZE * N_BLOCK, 128, unroll=1):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            if s_valid[row] != 0:
                s_k_row[row, col, 0] = m_ckv_flat[s_idx[row], col_base + col]
            else:
                s_k_row[row, col, 0] = BFloat16(0.0)
        cute.arch.sync_threads()
        copy_rowmajor_stage_to_operand_b(
            s_k_row, s_k[None, None, None, 0], k_swizzle, tidx
        )
        cute.arch.sync_threads()

    @cute.jit
    def _load_dense_kpe_tile(
        s_k_row: cute.Tensor,
        s_k: cute.Tensor,
        m_kpe_flat: cute.Tensor,
        s_idx: cute.Tensor,
        s_valid: cute.Tensor,
        k_swizzle,
        tidx: Int32,
    ) -> None:
        for elem in cutlass.range(tidx, CHUNK_SIZE * N_BLOCK, 128, unroll=1):
            row = elem // N_BLOCK
            col = elem % N_BLOCK
            if s_valid[row] != 0:
                s_k_row[row, col, 0] = m_kpe_flat[s_idx[row], col]
            else:
                s_k_row[row, col, 0] = BFloat16(0.0)
        cute.arch.sync_threads()
        copy_rowmajor_stage_to_operand_b(
            s_k_row, s_k[None, None, None, 0], k_swizzle, tidx
        )
        cute.arch.sync_threads()

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
                s_idx[row_base + 0] = Int32(-1)
                s_valid[row_base + 0] = Int32(0)

            if idx1 >= 0 and idx1 < total_kv:
                s_idx[row_base + 1] = idx1
                s_valid[row_base + 1] = Int32(1)
                group_mask = group_mask | Int32(2)
            else:
                s_idx[row_base + 1] = Int32(-1)
                s_valid[row_base + 1] = Int32(0)

            if idx2 >= 0 and idx2 < total_kv:
                s_idx[row_base + 2] = idx2
                s_valid[row_base + 2] = Int32(1)
                group_mask = group_mask | Int32(4)
            else:
                s_idx[row_base + 2] = Int32(-1)
                s_valid[row_base + 2] = Int32(0)

            if idx3 >= 0 and idx3 < total_kv:
                s_idx[row_base + 3] = idx3
                s_valid[row_base + 3] = Int32(1)
                group_mask = group_mask | Int32(8)
            else:
                s_idx[row_base + 3] = Int32(-1)
                s_valid[row_base + 3] = Int32(0)

            s_group_mask[group_idx] = group_mask

    @cute.jit
    def _initialize_placeholder_outputs(
        m_partial_o: cute.Tensor,
        m_partial_lse: cute.Tensor,
        tidx: Int32,
        split_idx: Int32,
        token_idx: Int32,
    ) -> None:
        for elem in cutlass.range(tidx, NUM_Q_HEADS * CKV_DIM, 128, unroll=1):
            head = elem // CKV_DIM
            dim = elem % CKV_DIM
            m_partial_o[split_idx, token_idx, head, dim] = Float32(0.0)
        for head in cutlass.range(tidx, NUM_Q_HEADS, 128, unroll=1):
            m_partial_lse[split_idx, token_idx, head] = Float32(-float("inf"))

    class SplitKernelShell:
        """Single-CTA split shell that will host the sparse TMA + tcgen05 rewrite."""

        def __init__(self):
            self.threads_per_cta = 128
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
                1,
            )
            k_smem_layout_staged = sm100_utils.make_smem_layout_b(
                qk_tiled_mma,
                qk_mma_tiler,
                BFloat16,
                1,
            )
            # PV MMA: P (A major K), V (B major MN = COL_MAJOR)
            pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                BFloat16,
                row_major.mma_major_mode(),
                col_major.mma_major_mode(),
                Float32,
                self.cta_group,
                (M_BLOCK, N_BLOCK),
            )
            pv_mma_tiler = (M_BLOCK, N_BLOCK, N_BLOCK)
            pv_b_smem_layout = sm100_utils.make_smem_layout_b(
                pv_tiled_mma, pv_mma_tiler, BFloat16, 1,
            )
            self.cta_tile_shape_mnk = (M_BLOCK, N_BLOCK, N_BLOCK)
            self.epi_tile = (M_BLOCK, N_BLOCK)
            self.c_layout = row_major
            self.kernel(
                qk_tiled_mma,
                pv_tiled_mma,
                q_smem_layout_staged,
                k_smem_layout_staged,
                pv_b_smem_layout,
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
            q_smem_layout_staged: cute.ComposedLayout,
            k_smem_layout_staged: cute.ComposedLayout,
            pv_b_smem_layout: cute.ComposedLayout,
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
            token_idx, split_idx, _ = cute.arch.block_idx()
            total_kv = mCkv_flat.shape[0]
            split_base = split_idx * CHUNK_SIZE

            @cute.struct
            class SharedStorage:
                acc_mbar: cute.struct.MemRange[cutlass.Int64, 1 * 2]
                shell_barrier: cutlass.Int64
                tmem_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            _storage = smem.allocate(SharedStorage)
            sQ = smem.allocate_tensor(
                element_type=BFloat16,
                layout=q_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=q_smem_layout_staged.inner,
            )
            sQrow = smem.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((M_BLOCK, N_BLOCK, 1)),
                byte_alignment=16,
            )
            sK = smem.allocate_tensor(
                element_type=BFloat16,
                layout=k_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=k_smem_layout_staged.inner,
            )
            # NOTE: sQrow is reused as staging buffer for K loads too
            # (no separate sKrow needed — Q and K loads don't overlap)
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
            sScoreF32 = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((M_BLOCK, N_BLOCK)),
                byte_alignment=16,
            )
            sV = smem.allocate_tensor(
                element_type=BFloat16,
                layout=pv_b_smem_layout.outer,
                byte_alignment=128,
                swizzle=pv_b_smem_layout.inner,
            )

            _load_split_index_groups(
                mIdx,
                sIdx,
                sValid,
                sGroupMask,
                tidx,
                token_idx,
                split_base,
                total_kv,
            )

            cute.arch.sync_threads()

            _initialize_placeholder_outputs(
                mPartialO, mPartialLse, tidx, split_idx, token_idx
            )

            cute.arch.sync_threads()

            tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=self.threads_per_cta,
            )
            tmem = utils.TmemAllocator(
                _storage.tmem_buf,
                barrier_for_retrieve=tmem_alloc_barrier,
            )
            qk_acc_shape = qk_tiled_mma_.partition_shape_C((M_BLOCK, N_BLOCK))
            qk_acc_fake = qk_tiled_mma_.make_fragment_C(qk_acc_shape)
            qk_tmem_cols = sm100_utils.get_num_tmem_alloc_cols(qk_acc_fake)
            acc_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=_storage.acc_mbar.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_cta
                ),
            )
            acc_ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            acc_cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )

            tmem.allocate(qk_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tScore = cute.make_tensor(tmem_ptr, qk_acc_fake.layout)

            # -- Epilogue setup: TMEM -> register copy for softmax --
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, Float32,
                Float32, self.epi_tile, False,
            )
            tScore_flat = tScore[((None, None), 0, 0)]
            tAcc_epi = cute.flat_divide(tScore_flat, self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
            )
            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tScore = thr_t2r.partition_S(tAcc_epi)
            tTR_tScore = cute.group_modes(
                tTR_tScore, 3, cute.rank(tTR_tScore)
            )

            # Destination: Float32 smem buffer for score staging + softmax
            thr_mma_c = qk_tiled_mma_.get_slice(0)
            tCsScore = thr_mma_c.partition_C(sScoreF32)
            tCsScore_flat = tCsScore[((None, None), 0, 0)]
            tCsScore_epi = cute.flat_divide(tCsScore_flat, self.epi_tile)
            tTR_sScore = thr_t2r.partition_D(tCsScore_epi)
            tTR_sScore = cute.group_modes(
                tTR_sScore, 3, cute.rank(tTR_sScore)
            )
            tTR_rScore = cute.make_rmem_tensor(
                tTR_sScore[(None, None, None, 0)].shape, Float32
            )

            tQr = qk_tiled_mma_.make_fragment_A(sQ)
            tKr = qk_tiled_mma_.make_fragment_B(sK)
            qk_num_kblks = cute.size(tQr, mode=[2])
            first_qk = True

            for tile_idx in cutlass.range(CKV_DIM // N_BLOCK, unroll=1):
                col_base = Int32(tile_idx * N_BLOCK)
                _load_q_nope_tile(
                    sQrow,
                    sQ,
                    mQ_nope,
                    q_smem_layout_staged.inner,
                    tidx,
                    token_idx,
                    col_base,
                )
                _load_dense_k_tile(
                    sQrow,
                    sK,
                    mCkv_flat,
                    sIdx,
                    sValid,
                    k_smem_layout_staged.inner,
                    tidx,
                    col_base,
                )

                for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                    qk_tiled_mma_.set(
                        tcgen05.Field.ACCUMULATE,
                        (not first_qk) or kblk_idx != 0,
                    )
                    cute.gemm(
                        qk_tiled_mma_,
                        tScore,
                        tQr[(None, None, kblk_idx, 0)],
                        tKr[(None, None, kblk_idx, 0)],
                        tScore,
                    )
                first_qk = False

            _load_q_pe_tile(
                sQrow,
                sQ,
                mQ_pe,
                q_smem_layout_staged.inner,
                tidx,
                token_idx,
            )
            _load_dense_kpe_tile(
                sQrow,
                sK,
                mKpe_flat,
                sIdx,
                sValid,
                k_smem_layout_staged.inner,
                tidx,
            )

            for kblk_idx in cutlass.range(qk_num_kblks, unroll_full=True):
                qk_tiled_mma_.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(
                    qk_tiled_mma_,
                    tScore,
                    tQr[(None, None, kblk_idx, 0)],
                    tKr[(None, None, kblk_idx, 0)],
                    tScore,
                )

            # Commit MMA and wait for accumulator to be ready before reading TMEM.
            if warp_idx == 0:
                acc_pipe.producer_commit(acc_ps)
            acc_pipe.consumer_wait(acc_cs)

            # TMEM -> registers -> smem: scale and stage scores
            scale_log2e = sm_scale * Float32(LOG2E)
            simt_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32
            )
            num_epi_tiles = cute.size(tTR_tScore.shape, mode=[3])
            for si in cutlass.range(num_epi_tiles):
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tScore[(None, None, None, si)],
                    tTR_rScore,
                )
                v = tTR_rScore.load()
                v = v * scale_log2e
                tTR_rScore.store(v)
                cute.copy(
                    simt_atom,
                    tTR_rScore,
                    tTR_sScore[(None, None, None, si)],
                )

            cute.arch.sync_threads()

            # Apply validity mask: invalid N-columns -> -inf
            for elem in cutlass.range(tidx, M_BLOCK * N_BLOCK, 128, unroll=1):
                row = elem // N_BLOCK
                col = elem % N_BLOCK
                if sValid[col] == Int32(0):
                    sScoreF32[row, col] = Float32(-float("inf"))

            cute.arch.sync_threads()

            # Softmax: per-row max, exp2, sum, LSE (heads 0:15 only)
            # 128 threads, 16 heads → 8 threads/head, 8 cols/thread
            cols_per_thr = N_BLOCK // 8  # 8
            head_idx = tidx // Int32(8)
            lane_in_head = tidx % Int32(8)
            col_start = lane_in_head * Int32(cols_per_thr)

            # Pre-define variables for CuTeDSL scoping
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
                # Scratch: rows 16-23, cols 0-15
                sScoreF32[
                    Int32(NUM_Q_HEADS) + lane_in_head, head_idx
                ] = local_max
            cute.arch.sync_threads()

            if head_idx < Int32(NUM_Q_HEADS):
                for lane in cutlass.range(8, unroll_full=True):
                    val = sScoreF32[
                        Int32(NUM_Q_HEADS) + Int32(lane), head_idx
                    ]
                    if val > row_max:
                        row_max = val

                for c in cutlass.range(cols_per_thr, unroll=1):
                    col = col_start + Int32(c)
                    val = sScoreF32[head_idx, col]
                    exp_val = cute.exp2(val - row_max, fastmath=True)
                    sScoreF32[head_idx, col] = exp_val
                    local_sum = local_sum + exp_val

                sScoreF32[
                    Int32(NUM_Q_HEADS) + lane_in_head, head_idx
                ] = local_sum
            cute.arch.sync_threads()

            if head_idx < Int32(NUM_Q_HEADS):
                for lane in cutlass.range(8, unroll_full=True):
                    row_sum = row_sum + sScoreF32[
                        Int32(NUM_Q_HEADS) + Int32(lane), head_idx
                    ]

                if row_sum > Float32(0.0):
                    lse = row_max + cute.log2(row_sum, fastmath=True)
                mPartialLse[split_idx, token_idx, head_idx] = lse

            cute.arch.sync_threads()

            # Normalize P and quantize to BF16 in sQrow for PV phase
            # Guard: if row_sum is 0 or NaN (all-invalid split), write 0 instead
            # of NaN*0 (which would propagate NaN through PV MMA)
            inv_sum = Float32(0.0)
            if head_idx < Int32(NUM_Q_HEADS):
                if row_sum > Float32(0.0):
                    inv_sum = Float32(1.0) / row_sum
                for c in cutlass.range(cols_per_thr, unroll=1):
                    col = col_start + Int32(c)
                    p_val = Float32(0.0)
                    if inv_sum > Float32(0.0):
                        p_val = sScoreF32[head_idx, col] * inv_sum
                    sQrow[head_idx, col, 0] = BFloat16(p_val)

            cute.arch.sync_threads()

            # Zero-pad rows 16:63 of sQrow for PV MMA
            for elem in cutlass.range(
                tidx, (M_BLOCK - NUM_Q_HEADS) * N_BLOCK, 128, unroll=1
            ):
                row = Int32(NUM_Q_HEADS) + elem // Int32(N_BLOCK)
                col = elem % Int32(N_BLOCK)
                sQrow[row, col, 0] = BFloat16(0.0)

            cute.arch.sync_threads()

            # --- PV phase: O = P @ V ---
            acc_ps.advance()
            acc_cs.advance()

            # Copy P (BF16 attention weights in sQrow) to operand A layout (sQ)
            copy_rowmajor_stage_to_operand_a(
                sQrow, sQ[None, None, None, 0],
                q_smem_layout_staged.inner, tidx,
            )
            cute.arch.sync_threads()

            # PV MMA fragments
            tPr = pv_tiled_mma_.make_fragment_A(sQ)
            tVr = pv_tiled_mma_.make_fragment_B(sV)
            pv_num_kblks = cute.size(tPr, mode=[2])

            # Loop over 8 output tiles (CKV_DIM / N_BLOCK = 512 / 64)
            for v_tile_idx in cutlass.range(CKV_DIM // N_BLOCK, unroll=1):
                v_col_base = Int32(v_tile_idx * N_BLOCK)

                # Load V tile into sV via sQrow staging
                _load_dense_k_tile(
                    sQrow, sV, mCkv_flat, sIdx, sValid,
                    pv_b_smem_layout.inner, tidx, v_col_base,
                )

                # Run PV MMA
                for kblk_idx in cutlass.range(
                    pv_num_kblks, unroll_full=True
                ):
                    pv_tiled_mma_.set(
                        tcgen05.Field.ACCUMULATE, kblk_idx != 0,
                    )
                    cute.gemm(
                        pv_tiled_mma_,
                        tScore,
                        tPr[(None, None, kblk_idx, 0)],
                        tVr[(None, None, kblk_idx, 0)],
                        tScore,
                    )

                # Commit and wait for PV results
                if warp_idx == 0:
                    acc_pipe.producer_commit(acc_ps)
                acc_pipe.consumer_wait(acc_cs)

                # TMEM -> registers -> smem (reuse score staging)
                for si in cutlass.range(num_epi_tiles):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tScore[(None, None, None, si)],
                        tTR_rScore,
                    )
                    cute.copy(
                        simt_atom,
                        tTR_rScore,
                        tTR_sScore[(None, None, None, si)],
                    )

                cute.arch.sync_threads()

                # Write [16, 64] slice of partial_o at current tile offset
                for elem in cutlass.range(
                    tidx, NUM_Q_HEADS * N_BLOCK, 128, unroll=1
                ):
                    head = elem // N_BLOCK
                    dim_col = elem % N_BLOCK
                    mPartialO[
                        split_idx, token_idx, head,
                        v_col_base + dim_col,
                    ] = sScoreF32[head, dim_col]

                cute.arch.sync_threads()

                # Advance pipeline for next tile
                acc_ps.advance()
                acc_cs.advance()

            tmem.relinquish_alloc_permit()
            pipeline.sync(barrier_id=2)
            tmem.free(tmem_ptr)
            return

    return SplitKernelShell()


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


def _prepare_split_shell_operands(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices):
    """Wrap direct split-shell operands plus scratch outputs as CuTeDSL tensors."""
    from cutlass.cute.runtime import from_dlpack

    T = q_nope.shape[0]
    device = q_nope.device
    ckv_flat, kpe_flat = _flatten_kv_caches(ckv_cache, kpe_cache)
    partial_o = torch.empty(
        (NUM_SPLITS, T, NUM_Q_HEADS, CKV_DIM), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (NUM_SPLITS, T, NUM_Q_HEADS), dtype=torch.float32, device=device
    )

    return (
        partial_o,
        partial_lse,
        from_dlpack(q_nope.contiguous(), assumed_align=16),
        from_dlpack(q_pe.contiguous(), assumed_align=16),
        from_dlpack(ckv_flat.contiguous(), assumed_align=16),
        from_dlpack(kpe_flat.contiguous(), assumed_align=16),
        from_dlpack(sparse_indices.contiguous(), assumed_align=16),
        from_dlpack(partial_o, assumed_align=16),
        from_dlpack(partial_lse, assumed_align=16),
    )


def _compute_split_partials(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f):
    """
    Fused CuTeDSL split kernel: QK GEMM + softmax + PV GEMM per split.

    Returns:
        partial_o:   [S, T, H, D] float32
        partial_lse: [S, T, H]    float32 (log2 base)
    """
    import cuda.bindings.driver as cuda

    (
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
    _get_split_shell_launcher()(
        mQ, mQp, mCkv, mKpe, mIdx, mO, mLse, sm_scale_f, stream=stream,
    )
    torch.cuda.synchronize()

    # Debug: check for NaN per-split
    T = partial_o.shape[1]
    for s in range(min(NUM_SPLITS, 4)):  # first 4 splits
        o_s = partial_o[s, 0, :, :N_BLOCK]  # [H, 64] first tile
        lse_s = partial_lse[s, 0, :]  # [H]
        nan_c = torch.isnan(o_s).sum().item()
        zero_c = (o_s == 0).sum().item()
        lse_fin = torch.isfinite(lse_s).sum().item()
        print(f"[DEBUG] split={s}: o[:64] nan={nan_c} zero={zero_c} "
              f"real={o_s.numel()-nan_c-zero_c} lse_finite={lse_fin}/16 "
              f"o[0,:4]={o_s[0,:4].tolist()}")

    return partial_o, partial_lse


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute DSA sparse attention."""
    sm_scale_f = _to_python_float(sm_scale)
    partial_o, partial_lse = _compute_split_partials(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale_f,
    )
    output, final_lse = _combine_splits(partial_o, partial_lse)
    return output, final_lse


@torch.no_grad()
def _compile_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compile the split shell kernel for the provided problem shapes."""
    from cutlass.base_dsl.compiler import CompileCallable
    import cuda.bindings.driver as cuda

    _ = _to_python_float(sm_scale)
    (_, _, mQ, mQp, mCkv, mKpe, mIdx, mO, mLse) = _prepare_split_shell_operands(
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
    print("CuTe split-shell smoke compile PASS")


if __name__ == "__main__":
    _cutedsl_smoke_compile()
