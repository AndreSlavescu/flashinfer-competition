"""
Minimal repro: batched QK GEMM with batch=32 (num_tokens=1 * 32 splits).
"""

import modal

app = modal.App("test-cutedsl-batch32")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "nvidia-cutlass-dsl")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_batch32():
    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
    from cutlass.cute.nvgpu import cpasync
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.pipeline as pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass import Float32, BFloat16
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    M, N = 64, 64
    K_CKV, K_KPE = 512, 64

    class SimpleGemm:
        def __init__(self):
            self.threads_per_cta = 128
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (M, N)

        @cute.jit
        def __call__(self, mA, mB, mC, stream):
            a_dtype = mA.element_type
            b_dtype = mB.element_type
            a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
            self.c_layout = utils.LayoutEnum.from_tensor(mC)

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype, a_major, b_major, Float32, self.cta_group, self.mma_tiler_mn,
            )

            mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            k_tile = mma_inst_k * 4
            self.mma_tiler = (M, N, k_tile)
            atom_thr_size = cute.size(tiled_mma.thr_id.shape)
            self.cta_tile_shape_mnk = (M // atom_thr_size, N, k_tile)
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

            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

            tma_a, _ = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mA, a_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mB, b_smem, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

            a_copy = cute.size_in_bytes(a_dtype, a_smem)
            b_copy = cute.size_in_bytes(b_dtype, b_smem)
            self.num_tma_load_bytes = (a_copy + b_copy) * atom_thr_size

            batch_size = mC.shape[2]
            grid = (1, 1, batch_size)

            self.kernel(
                tiled_mma, tma_a, mA, tma_b, mB, mC,
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
            self, tiled_mma, tma_a, mA, tma_b, mB, mC,
            cluster_layout_vmnk,
            a_smem_layout_staged, b_smem_layout_staged,
            epi_tile,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, bidz = cute.arch.block_idx()

            use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
            mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
            is_leader_cta = mma_tile_coord_v == 0
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            block_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)
            mma_tile_coord_mnl = (bidx // cute.size(tiled_mma.thr_id.shape), bidy, bidz)

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_a)
                cpasync.prefetch_descriptor(tma_b)

            @cute.struct
            class SS:
                ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
                acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            storage = smem.allocate(SS)

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
            acc_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            acc_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

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

            gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
            k_cnt = cute.size(gA, mode=[3])

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)

            acl = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            bcl = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

            tAsA, tAgA = cpasync.tma_partition(
                tma_a, block_coord[2], acl,
                cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_b, block_coord[1], bcl,
                cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
            )

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mn)
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            gC = cute.local_tile(mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
            tCgC = thr_mma.partition_C(gC)

            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

            if warp_idx == 0:
                pf_cnt = cutlass.min(self.num_ab_stage - 2, k_cnt)
                for k in cutlass.range(pf_cnt, unroll=1):
                    ph = ab_prod.acquire_and_advance()
                    cute.copy(tma_a, tAgA[(None, k)], tAsA[(None, ph.index)], tma_bar_ptr=ph.barrier)
                    cute.copy(tma_b, tBgB[(None, k)], tBsB[(None, ph.index)], tma_bar_ptr=ph.barrier)

                pf = cutlass.Boolean(False)
                if is_leader_cta:
                    pf = ab_cons.try_wait()
                pe = ab_prod.try_acquire()

                for k in cutlass.range(k_cnt):
                    if k < k_cnt - pf_cnt:
                        ph = ab_prod.acquire_and_advance(pe)
                        cute.copy(tma_a, tAgA[(None, ph.count)], tAsA[(None, ph.index)], tma_bar_ptr=ph.barrier)
                        cute.copy(tma_b, tBgB[(None, ph.count)], tBsB[(None, ph.index)], tma_bar_ptr=ph.barrier)

                    if is_leader_cta:
                        ch = ab_cons.wait_and_advance(pf)
                        for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                            cute.gemm(tiled_mma, tCtAcc,
                                      tCrA[(None, None, kb, ch.index)],
                                      tCrB[(None, None, kb, ch.index)], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ch.release()

                    if k + 1 < k_cnt - pf_cnt:
                        pe = ab_prod.try_acquire()
                    if k + 1 < k_cnt and is_leader_cta:
                        pf = ab_cons.try_wait()

                if is_leader_cta:
                    acc_pipe.producer_commit(acc_ps)

            # Epilogue
            tmem.relinquish_alloc_permit()
            acc_pipe.consumer_wait(acc_cs)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, mC.element_type,
                Float32, epi_tile, use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])

            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_t2r.partition_D(tCgC_epi)
            tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, Float32)

            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            tTR_rC = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, mC.element_type)

            tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)]
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

            for si in cutlass.range(cute.size(tTR_tAcc.shape, mode=[3])):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, si)], tTR_rAcc)
                v = tTR_rAcc.load()
                v = v.to(mC.element_type)
                tTR_rC.store(v)
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, si)])

            pipeline.sync(barrier_id=1)
            tmem.free(tmem_ptr)
            if warp_idx == 0:
                ab_prod.tail()

    # Test with different batch sizes
    for batch in [32, 64, 128, 1, 8]:
        print(f"\n--- Testing batch={batch} ---")
        torch.manual_seed(42)

        A = torch.randn(batch, M, K_CKV, dtype=torch.bfloat16, device="cuda") / 10.0
        B = torch.randn(batch, N, K_CKV, dtype=torch.bfloat16, device="cuda") / 10.0
        C = torch.zeros(batch, M, N, dtype=torch.float32, device="cuda")

        A_3d = A.permute(1, 2, 0)  # [M, K, L]
        B_3d = B.permute(1, 2, 0)  # [N, K, L]
        C_3d = C.permute(1, 2, 0)  # [M, N, L]

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        mA = from_dlpack(A_3d, assumed_align=16)
        mB = from_dlpack(B_3d, assumed_align=16)
        mC = from_dlpack(C_3d, assumed_align=16)

        try:
            kernel = SimpleGemm()
            kernel(mA, mB, mC, stream=stream)
            torch.cuda.synchronize()

            ref = torch.bmm(A.float(), B.float().transpose(-1, -2))
            err = (C - ref).abs().max().item()
            print(f"batch={batch}: max_err={err:.2e} PASS" if err < 1e-1 else f"batch={batch}: max_err={err:.2e} FAIL")
        except Exception as e:
            print(f"batch={batch}: ERROR - {e}")


@app.local_entrypoint()
def main():
    test_batch32.remote()
