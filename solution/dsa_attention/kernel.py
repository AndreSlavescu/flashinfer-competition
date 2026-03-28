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


def _get_cutedsl_kernel():
    """Lazily initialize the CuTeDSL QK GEMM kernel."""
    global _cutedsl_kernel
    if _cutedsl_kernel is not None:
        return _cutedsl_kernel

    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
    from cutlass.cute.nvgpu import cpasync
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.pipeline as pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass import Float32, BFloat16
    import cuda.bindings.driver as cuda

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


def _to_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices):
    """Gather sparse KV rows and reshape into [T, S, N, D] chunks."""
    T = sparse_indices.shape[0]
    S = NUM_SPLITS
    N = CHUNK_SIZE
    D = CKV_DIM
    Dp = KPE_DIM

    valid_mask = sparse_indices != -1
    safe_indices = sparse_indices.clamp_min(0).long()
    vm = valid_mask.view(T, S, N)

    kc = ckv_cache.reshape(-1, D)[safe_indices.reshape(-1)].view(T, S, N, D)
    kp = kpe_cache.reshape(-1, Dp)[safe_indices.reshape(-1)].view(T, S, N, Dp)

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
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda

    T = q_nope.shape[0]
    H = NUM_Q_HEADS
    S = NUM_SPLITS
    N = CHUNK_SIZE
    D = CKV_DIM
    Dp = KPE_DIM
    device = q_nope.device
    batch = T * S

    # Pad Q from H=16 to M=64
    Q_nope_padded = torch.zeros(T, M_BLOCK, D, dtype=torch.bfloat16, device=device)
    Q_nope_padded[:, :H, :] = q_nope
    Q_pe_padded = torch.zeros(T, M_BLOCK, Dp, dtype=torch.bfloat16, device=device)
    Q_pe_padded[:, :H, :] = q_pe

    # Expand Q for T*S batches (repeat each token S times)
    Q_nope_exp = Q_nope_padded.unsqueeze(1).expand(T, S, M_BLOCK, D).reshape(batch, M_BLOCK, D)
    Q_pe_exp = Q_pe_padded.unsqueeze(1).expand(T, S, M_BLOCK, Dp).reshape(batch, M_BLOCK, Dp)
    Kc_flat = kc.reshape(batch, N, D)
    Kp_flat = kp.reshape(batch, N, Dp)

    # Create [L, M, K] contiguous then permute to [M, K, L] for K-major layout
    Q_nope_3d = Q_nope_exp.contiguous().permute(1, 2, 0)
    Q_pe_3d = Q_pe_exp.contiguous().permute(1, 2, 0)
    Kc_3d = Kc_flat.contiguous().permute(1, 2, 0)
    Kp_3d = Kp_flat.contiguous().permute(1, 2, 0)

    # Output S: [M, N, batch] FP32
    S_base = torch.zeros(batch, M_BLOCK, N_BLOCK, dtype=torch.float32, device=device)
    S_3d = S_base.permute(1, 2, 0)  # [M, N, batch]

    # Run CuTeDSL kernel
    kernel = _get_cutedsl_kernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    mQn = from_dlpack(Q_nope_3d, assumed_align=16)
    mQp = from_dlpack(Q_pe_3d, assumed_align=16)
    mKc = from_dlpack(Kc_3d, assumed_align=16)
    mKp = from_dlpack(Kp_3d, assumed_align=16)
    mS = from_dlpack(S_3d, assumed_align=16)

    kernel(mQn, mQp, mKc, mKp, mS, stream=stream)
    torch.cuda.synchronize()

    # Extract logits: S_base is [batch, M, N], take first H rows
    logits = S_base[:, :H, :].view(T, S, H, N)  # [T, S, H, N]

    # Apply scale
    logits = logits * sm_scale_f

    # Masked softmax
    mask = vm[:, :, None, :]
    split_valid = vm.any(dim=-1, keepdim=True)

    logits.masked_fill_(~mask, -float("inf"))

    partial_lse = torch.full((T, S, H), -float("inf"), dtype=torch.float32, device=device)
    valid_splits = split_valid.expand(-1, -1, H)
    partial_lse[valid_splits] = torch.logsumexp(logits, dim=-1)[valid_splits] * LOG2E

    logits = torch.where(split_valid.unsqueeze(-1), logits, torch.zeros_like(logits))
    attn = torch.softmax(logits, dim=-1)
    attn = torch.where(split_valid.unsqueeze(-1), attn, torch.zeros_like(attn))

    # SV GEMM via PyTorch bmm
    kc_f = kc.float()
    attn_r = attn.reshape(T * S, H, N)
    kc_sv = kc_f.reshape(T * S, N, D)
    partial_o = torch.bmm(attn_r, kc_sv).view(T, S, H, D)

    partial_o = partial_o.permute(1, 0, 2, 3).contiguous()
    partial_lse = partial_lse.permute(1, 0, 2).contiguous()

    return partial_o, partial_lse


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Compute DSA sparse attention."""
    sm_scale_f = _to_python_float(sm_scale)
    kc, kp, vm = _gather_kv_chunks(ckv_cache, kpe_cache, sparse_indices)
    partial_o, partial_lse = _compute_split_partials(
        q_nope, q_pe, kc, kp, vm, sm_scale_f
    )
    output, final_lse = _combine_splits(partial_o, partial_lse)
    return output, final_lse


@torch.no_grad()
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """DPS-style kernel entry point — writes output and lse in place."""
    out, lse_out = run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    output.copy_(out)
    lse.copy_(lse_out)
