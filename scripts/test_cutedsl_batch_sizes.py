"""
Test the EXACT same kernel code from working batched test with different batch sizes.
"""

import modal

app = modal.App("test-cutedsl-batch-sizes")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "nvidia-cutlass-dsl")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test():
    import torch
    import math
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
    from cutlass.cute.nvgpu import cpasync
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.pipeline as pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass import Float32, Int32, BFloat16, const_expr
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    M, N = 64, 64
    K_CKV, K_KPE = 512, 64
    H_REAL = 16
    NUM_SPLITS = 32

    # EXACT same kernel from test_cutedsl_batched_qk.py
    class BatchedQKKernel:
        def __init__(self):
            self.threads_per_cta = 128
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (M, N)

        @cute.jit
        def __call__(self, mQ_nope, mQ_pe, mKc, mKp, mS, stream):
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
            self.mma_tiler = (M, N, k_tile)
            self.cta_tile_shape_mnk = (
                M // cute.size(tiled_mma.thr_id.shape), N, k_tile,
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

            tma_atom_qn, tma_Q_nope = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mQ_nope, a_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_kc, tma_Kc = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mKc, b_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_qp, tma_Q_pe = cute.nvgpu.make_tiled_tma_atom_A(
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
                tma_atom_qn, tma_Q_nope,
                tma_atom_kc, tma_Kc,
                tma_atom_qp, tma_Q_pe,
                tma_atom_kp, tma_Kp,
                mS,
                self.cluster_layout_vmnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.epi_tile,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self, tiled_mma, tma_atom_qn, mQ_nope, tma_atom_kc, mKc,
            tma_atom_qp, mQ_pe, tma_atom_kp, mKp,
            mS, cluster_layout_vmnk,
            a_smem_layout_staged, b_smem_layout_staged, epi_tile,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, bidz = cute.arch.block_idx()

            use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
            mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
            is_leader_cta = mma_tile_coord_v == 0
            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            mma_tile_coord_mnl = (bidx // cute.size(tiled_mma.thr_id.shape), bidy, bidz)

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_qn)
                cpasync.prefetch_descriptor(tma_atom_kc)
                cpasync.prefetch_descriptor(tma_atom_qp)
                cpasync.prefetch_descriptor(tma_atom_kp)

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
                acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_producer_group,
                consumer_group=ab_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()

            acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            acc_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_producer_group,
                consumer_group=acc_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=tmem_alloc_barrier,
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
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

            gQn = cute.local_tile(mQ_nope, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            gKc = cute.local_tile(mKc, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
            k_cnt_ckv = cute.size(gQn, mode=[3])

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            tCgQn = thr_mma.partition_A(gQn)
            tCgKc = thr_mma.partition_B(gKc)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

            tAsA_qn, tAgQn = cpasync.tma_partition(
                tma_atom_qn, block_in_cluster_coord_vmnk[2], a_cta_layout,
                cute.group_modes(sA, 0, 3), cute.group_modes(tCgQn, 0, 3),
            )
            tBsB_kc, tBgKc = cpasync.tma_partition(
                tma_atom_kc, block_in_cluster_coord_vmnk[1], b_cta_layout,
                cute.group_modes(sB, 0, 3), cute.group_modes(tCgKc, 0, 3),
            )

            gQp = cute.local_tile(mQ_pe, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            gKp = cute.local_tile(mKp, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
            k_cnt_kpe = cute.size(gQp, mode=[3])

            tCgQp = thr_mma.partition_A(gQp)
            tCgKp = thr_mma.partition_B(gKp)

            tAsA_qp, tAgQp = cpasync.tma_partition(
                tma_atom_qp, block_in_cluster_coord_vmnk[2], a_cta_layout,
                cute.group_modes(sA, 0, 3), cute.group_modes(tCgQp, 0, 3),
            )
            tBsB_kp, tBgKp = cpasync.tma_partition(
                tma_atom_kp, block_in_cluster_coord_vmnk[1], b_cta_layout,
                cute.group_modes(sB, 0, 3), cute.group_modes(tCgKp, 0, 3),
            )

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mn)
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            gS = cute.local_tile(mS, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
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

            # CKV GEMM
            if warp_idx == 0:
                prefetch = cutlass.min(self.num_ab_stage - 2, k_cnt_ckv)
                for k in cutlass.range(prefetch, unroll=1):
                    ph = ab_producer.acquire_and_advance()
                    cute.copy(tma_atom_qn, tAgQn[(None, k)],
                              tAsA_qn[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_atom_kc, tBgKc[(None, k)],
                              tBsB_kc[(None, ph.index)], tma_bar_ptr=ph.barrier)

                pf = cutlass.Boolean(False)
                if is_leader_cta:
                    pf = ab_consumer.try_wait()
                pe = ab_producer.try_acquire()

                for k in cutlass.range(k_cnt_ckv):
                    if k < k_cnt_ckv - prefetch:
                        ph = ab_producer.acquire_and_advance(pe)
                        cute.copy(tma_atom_qn, tAgQn[(None, ph.count)],
                                  tAsA_qn[(None, ph.index)], tma_bar_ptr=ph.barrier)
                        cute.copy(tma_atom_kc, tBgKc[(None, ph.count)],
                                  tBsB_kc[(None, ph.index)], tma_bar_ptr=ph.barrier)

                    if is_leader_cta:
                        ch = ab_consumer.wait_and_advance(pf)
                        for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            cute.gemm(tiled_mma, tCtAcc,
                                      tCrA[(None, None, kb, ch.index)],
                                      tCrB[(None, None, kb, ch.index)], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ch.release()

                    if k + 1 < k_cnt_ckv - prefetch:
                        pe = ab_producer.try_acquire()
                    if k + 1 < k_cnt_ckv and is_leader_cta:
                        pf = ab_consumer.try_wait()

                # KPE GEMM (accumulate)
                prefetch_kpe = cutlass.min(self.num_ab_stage - 2, k_cnt_kpe)
                for k in cutlass.range(prefetch_kpe, unroll=1):
                    ph = ab_producer.acquire_and_advance()
                    cute.copy(tma_atom_qp, tAgQp[(None, k)],
                              tAsA_qp[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_atom_kp, tBgKp[(None, k)],
                              tBsB_kp[(None, ph.index)], tma_bar_ptr=ph.barrier)

                pf_kpe = cutlass.Boolean(False)
                if is_leader_cta:
                    pf_kpe = ab_consumer.try_wait()
                pe_kpe = ab_producer.try_acquire()

                for k in cutlass.range(k_cnt_kpe):
                    if k < k_cnt_kpe - prefetch_kpe:
                        ph = ab_producer.acquire_and_advance(pe_kpe)
                        cute.copy(tma_atom_qp, tAgQp[(None, ph.count)],
                                  tAsA_qp[(None, ph.index)], tma_bar_ptr=ph.barrier)
                        cute.copy(tma_atom_kp, tBgKp[(None, ph.count)],
                                  tBsB_kp[(None, ph.index)], tma_bar_ptr=ph.barrier)

                    if is_leader_cta:
                        ch = ab_consumer.wait_and_advance(pf_kpe)
                        for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            cute.gemm(tiled_mma, tCtAcc,
                                      tCrA[(None, None, kb, ch.index)],
                                      tCrB[(None, None, kb, ch.index)], tCtAcc)
                        ch.release()

                    if k + 1 < k_cnt_kpe - prefetch_kpe:
                        pe_kpe = ab_producer.try_acquire()
                    if k + 1 < k_cnt_kpe and is_leader_cta:
                        pf_kpe = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

            # Epilogue
            tmem.relinquish_alloc_permit()
            acc_pipeline.consumer_wait(acc_consumer_state)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, mS.element_type,
                Float32, epi_tile, use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])

            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            tCgS_epi = cute.flat_divide(tCgS[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gS = thr_t2r.partition_D(tCgS_epi)
            tTR_rAcc = cute.make_rmem_tensor(tTR_gS[(None, None, None, 0, 0, 0, 0, 0)].shape, Float32)

            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mS.element_type)
            tTR_rS = cute.make_rmem_tensor(tTR_gS[(None, None, None, 0, 0, 0, 0, 0)].shape, mS.element_type)

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
                ab_producer.tail()

    # Test with different T values
    for T in [1, 2, 8]:
        S = NUM_SPLITS
        batch = T * S
        print(f"\n--- T={T}, S={S}, batch={batch} ---")

        torch.manual_seed(42)

        Q_nope_padded = torch.zeros(T, M, K_CKV, dtype=torch.bfloat16, device="cuda")
        Q_nope_padded[:, :H_REAL, :] = torch.randn(T, H_REAL, K_CKV, dtype=torch.bfloat16, device="cuda") / 10.0
        Q_pe_padded = torch.zeros(T, M, K_KPE, dtype=torch.bfloat16, device="cuda")
        Q_pe_padded[:, :H_REAL, :] = torch.randn(T, H_REAL, K_KPE, dtype=torch.bfloat16, device="cuda") / 10.0

        Q_nope_exp = Q_nope_padded.unsqueeze(1).expand(T, S, M, K_CKV).reshape(batch, M, K_CKV)
        Q_pe_exp = Q_pe_padded.unsqueeze(1).expand(T, S, M, K_KPE).reshape(batch, M, K_KPE)
        Kc = torch.randn(batch, N, K_CKV, dtype=torch.bfloat16, device="cuda") / 10.0
        Kp = torch.randn(batch, N, K_KPE, dtype=torch.bfloat16, device="cuda") / 10.0

        # [L, M, K] contiguous then permute to [M, K, L]
        Q_nope_3d = Q_nope_exp.contiguous().permute(1, 2, 0)
        Q_pe_3d = Q_pe_exp.contiguous().permute(1, 2, 0)
        Kc_3d = Kc.contiguous().permute(1, 2, 0)
        Kp_3d = Kp.contiguous().permute(1, 2, 0)

        S_base = torch.zeros(batch, M, N, dtype=torch.float32, device="cuda")
        S_3d = S_base.permute(1, 2, 0)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        mQn = from_dlpack(Q_nope_3d, assumed_align=16)
        mQp = from_dlpack(Q_pe_3d, assumed_align=16)
        mKc = from_dlpack(Kc_3d, assumed_align=16)
        mKp = from_dlpack(Kp_3d, assumed_align=16)
        mS = from_dlpack(S_3d, assumed_align=16)

        try:
            kernel = BatchedQKKernel()
            kernel(mQn, mQp, mKc, mKp, mS, stream=stream)
            torch.cuda.synchronize()

            ref = (Q_nope_exp.float() @ Kc.float().transpose(-1, -2) +
                   Q_pe_exp.float() @ Kp.float().transpose(-1, -2))
            err = (S_base - ref).abs().max().item()
            real_err = (S_base[:, :H_REAL] - ref[:, :H_REAL]).abs().max().item()
            print(f"T={T}: max_err(all)={err:.2e}, max_err(real)={real_err:.2e} {'PASS' if real_err < 1e-1 else 'FAIL'}")
        except Exception as e:
            print(f"T={T}: ERROR - {e}")
            import traceback
            traceback.print_exc()


@app.local_entrypoint()
def main():
    test.remote()
