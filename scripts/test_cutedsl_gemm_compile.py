"""
Test that a minimal CuTeDSL GEMM kernel compiles and runs on Modal B200.

Closely follows the dense_gemm.py example from CUTLASS.

Problem: C[64, 64] = A[64, 512] @ B[64, 512]^T, BF16 inputs, FP32 accumulator.

Usage:
    .venv/bin/modal run scripts/test_cutedsl_gemm_compile.py
"""

import modal

app = modal.App("test-cutedsl-gemm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "nvidia-cutlass-dsl")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_gemm():
    """Compile and run a minimal CuTeDSL GEMM on B200."""
    import torch
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
    print("CuTeDSL version:", cutlass.__version__)

    # Problem size: C[64, 64] = A[64, 512] @ B[64, 512]^T
    M, N, K = 64, 64, 512

    class SimpleGemm:
        def __init__(self):
            self.threads_per_cta = 128
            self.use_2cta_instrs = False
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)

            # MMA tiler: (M, N, K) where K = mma_inst_k * 4
            # For BF16 K-major, mma_inst_shape_k = 32, so K_tile = 32 * 4 = 128
            # But our K=512, so we'll compute this after tiled_mma is created
            self.mma_tiler_mn = (M, N)

        @cute.jit
        def __call__(
            self,
            mA: cute.Tensor,   # [M, K] BF16
            mB: cute.Tensor,   # [N, K] BF16
            mC: cute.Tensor,   # [M, N] FP32
            stream: cuda.CUstream,
        ):
            a_dtype = mA.element_type
            b_dtype = mB.element_type
            c_dtype = mC.element_type

            a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
            self.c_layout = utils.LayoutEnum.from_tensor(mC)

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype, a_major, b_major, Float32, self.cta_group, self.mma_tiler_mn,
            )

            # Compute full MMA tiler with K dimension
            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            mma_inst_tile_k = 4
            self.mma_tiler = (M, N, mma_inst_shape_k * mma_inst_tile_k)

            self.cta_tile_shape_mnk = (
                self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
                self.mma_tiler[1],
                self.mma_tiler[2],
            )

            # Cluster layout
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,),
            )

            # Epilogue tile (no TMA store, so same as CTA tile)
            self.epi_tile = self.cta_tile_shape_mnk[:2]

            # SMEM layouts (1 stage)
            self.num_ab_stage = 2
            self.num_acc_stage = 1
            self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, a_dtype, self.num_ab_stage,
            )
            self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, b_dtype, self.num_ab_stage,
            )

            # TMEM columns
            self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

            # TMA atoms
            atom_thr_size = cute.size(tiled_mma.thr_id.shape)

            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mA, a_smem_layout, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

            b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mB, b_smem_layout, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

            # TMA byte count (this returns a Python int, not a DSL value)
            a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
            self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

            print(f"  mma_tiler: {self.mma_tiler}")
            print(f"  cta_tile: {self.cta_tile_shape_mnk}")
            print(f"  epi_tile: {self.epi_tile}")
            print(f"  num_tma_load_bytes: {self.num_tma_load_bytes}")
            print(f"  num_tmem_alloc_cols: {self.num_tmem_alloc_cols}")
            print(f"  a_smem_layout: {self.a_smem_layout_staged}")
            print(f"  b_smem_layout: {self.b_smem_layout_staged}")

            # Compute grid size
            grid = (1, 1, 1)

            # Launch
            self.kernel(
                tiled_mma,
                tma_atom_a, tma_tensor_a,
                tma_atom_b, tma_tensor_b,
                mC,
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
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA_mkl: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB_nkl: cute.Tensor,
            mC_mnl: cute.Tensor,
            cluster_layout_vmnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_tile: cute.Tile,
        ):
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
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
                bidx // cute.size(tiled_mma.thr_id.shape),
                bidy,
                bidz,
            )

            # Prefetch TMA descriptors
            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            # Alloc barriers + TMEM holding buffer
            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
                acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            # Pipeline: TMA load A+B
            ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()

            # Pipeline: accumulator ready
            acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            acc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta
            )
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # TMEM allocator
            tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=0, num_threads=self.threads_per_cta,
            )
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=tmem_alloc_barrier,
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
            )

            # Cluster init
            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            # Allocate SMEM tensors
            sA = smem.allocate_tensor(
                element_type=BFloat16,
                layout=a_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            sB = smem.allocate_tensor(
                element_type=BFloat16,
                layout=b_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=b_smem_layout_staged.inner,
            )

            # Global tensor partitioning
            gA_mkl = cute.local_tile(
                mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gB_nkl = cute.local_tile(
                mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
            k_tile_cnt = cute.size(gA_mkl, mode=[3])

            # Partition global tensors for tiled MMA
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            tCgA = thr_mma.partition_A(gA_mkl)
            tCgB = thr_mma.partition_B(gB_nkl)
            tCgC = thr_mma.partition_C(gC_mnl)

            # TMA partitioning
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

            # SMEM->MMA fragments
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mn)
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            # Cluster wait + TMEM alloc
            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Slice to our tile
            tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

            # ========== TMA Load + MMA Mainloop (warp 0 only) ==========
            prefetch_k_tile_cnt = cutlass.min(self.num_ab_stage - 2, k_tile_cnt)
            if warp_idx == 0:
                # Prefetch TMA loads
                for k_tile_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                    producer_handle = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, k_tile_idx)],
                        tAsA[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, k_tile_idx)],
                        tBsB[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                    )

                peek_ab_full_status = cutlass.Boolean(False)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                peek_ab_empty_status = ab_producer.try_acquire()

                # MMA mainloop
                for k_tile_idx in cutlass.range(k_tile_cnt):
                    if k_tile_idx < k_tile_cnt - prefetch_k_tile_cnt:
                        producer_handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, producer_handle.count)],
                            tAsA[(None, producer_handle.index)],
                            tma_bar_ptr=producer_handle.barrier,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, producer_handle.count)],
                            tBsB[(None, producer_handle.index)],
                            tma_bar_ptr=producer_handle.barrier,
                        )

                    if is_leader_cta:
                        consumer_handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        num_kblks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                            cute.gemm(
                                tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        consumer_handle.release()

                    if k_tile_idx + 1 < k_tile_cnt - prefetch_k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                    if k_tile_idx + 1 < k_tile_cnt and is_leader_cta:
                        peek_ab_full_status = ab_consumer.try_wait()

                # Signal accumulator ready
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

            # ========== Epilogue ==========
            tmem.relinquish_alloc_permit()

            # Wait for accumulator
            acc_pipeline.consumer_wait(acc_consumer_state)

            # TMEM -> registers -> global (non-TMA store path)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                mC_mnl.element_type,
                Float32,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
            )

            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            tCgC_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, None, None, None)], epi_tile
            )
            tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, Float32
            )

            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC_mnl.element_type)

            tTR_rC = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, mC_mnl.element_type
            )

            tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)]
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                acc_vec = tTR_rAcc.load()
                acc_vec = acc_vec.to(mC_mnl.element_type)
                tTR_rC.store(acc_vec)

                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

            # Dealloc TMEM
            pipeline.sync(barrier_id=1)
            tmem.free(tmem_ptr)

            # Drain TMA producer
            if warp_idx == 0:
                ab_producer.tail()

    # Create test data (3D with L=1 batch dim, as dense_gemm expects)
    torch.manual_seed(42)
    A = torch.randn(M, K, 1, dtype=torch.bfloat16, device="cuda") / 10.0
    B = torch.randn(N, K, 1, dtype=torch.bfloat16, device="cuda") / 10.0
    C = torch.zeros(M, N, 1, dtype=torch.float32, device="cuda")

    # Reference
    ref = (A[:, :, 0].float() @ B[:, :, 0].float().T)
    print(f"Reference C[0,0]: {ref[0,0].item():.6f}, C sum: {ref.sum().item():.6f}")

    # Get CUDA stream
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Build CuTeDSL tensors
    mA = from_dlpack(A, assumed_align=16)
    mB = from_dlpack(B, assumed_align=16)
    mC = from_dlpack(C, assumed_align=16)

    print("Compiling CuTeDSL GEMM kernel...")
    gemm = SimpleGemm()

    try:
        gemm(mA, mB, mC, stream=stream)
        torch.cuda.synchronize()
        print("Kernel compiled and ran successfully!")

        print(f"Kernel C[0,0,0]: {C[0,0,0].item():.6f}, C sum: {C.sum().item():.6f}")

        # Verify correctness
        max_err = (C[:, :, 0] - ref).abs().max().item()
        print(f"Max error: {max_err:.6e}")
        passed = max_err < 1e-1
        print(f"COMPILE TEST: {'PASS' if passed else 'FAIL'}")
        return {"compiled": True, "passed": passed, "max_err": max_err, "error": None}
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"compiled": False, "passed": False, "error": str(e)}


@app.local_entrypoint()
def main():
    result = test_gemm.remote()
    print(f"\nResult: {result}")
