#pragma once

// UTCMMA Ver2 experiment kernels (6, 7, 8+9, 10, 12, 15)
// Requires utcmma_kernels.cuh to be included first (for BenchResult, timing helpers, TMEM_COL_*)

// =========================================================================
// Experiment 6: utcmma_throughput_2acc
// Multi-accumulator throughput: rotate TMEM column address between MMA calls
// to break RAW dependencies and measure true pipeline throughput.
// =========================================================================
template<int M, int N, int N_ACC, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_throughput_2acc(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;
    // WS M=64 accumulator uses N/2 TMEM columns per accumulator
    static constexpr int COLS_PER_ACC = N / 2;
    // tcgen05.mma requires A TMEM column to be 64-column aligned.
    // TMEM_COL_A = 256 satisfies this (256 % 64 == 0) and leaves [0,255] for C.
    static constexpr int TMEM_COL_A_LOCAL = TMEM_COL_A;  // 256
    static_assert(TMEM_COL_C + N_ACC * COLS_PER_ACC <= TMEM_COL_A_LOCAL,
        "C accumulator columns overflow into A TMEM region");

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    constexpr int SWIZZLE_B = (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32);
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE_B, bf16>();
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    // Barrier for MMA completion synchronization
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
        tA_frag.data().get() = TMEM_COL_A_LOCAL;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        // Warmup: fill all accumulators with valid data
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                tC_frag.data().get() = TMEM_COL_C + (k % N_ACC) * COLS_PER_ACC;
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region: all accumulate=One (steady-state throughput) ---
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;

        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                tC_frag.data().get() = TMEM_COL_C + (k % N_ACC) * COLS_PER_ACC;
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// =========================================================================
// Experiment 7: tmem_ld_st_fence_overhead
// Decompose cost of TMEM read-modify-write (O-rescale critical path).
// MODE=0: tmem_ld chain, MODE=1: tmem_st chain,
// MODE=2: fence-only baseline, MODE=3: full rescale roundtrip
// =========================================================================
template<int MODE, int TMEM_WIDTH>
__global__ __launch_bounds__(128, 1)
void kernel_tmem_ld_st_fence(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    __shared__ __align__(16) uint32_t smem_tmem_addr;

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    // tcgen05.ld/st are warp-collective (32x32b = all 32 lanes must participate).
    // MODE 0/1/3 run under threadIdx.x < 32 so the full warp issues each instruction.
    // MODE 2 fences are per-thread; kept under threadIdx.x == 0.

    if constexpr (MODE == 0) {
        // --- tmem_load chain: serialized loads with fence ---
        if (threadIdx.x < 32) {
            // Initialize TMEM via warp-collective st (tcgen05.ld hangs on never-written TMEM).
            uint32_t zeros[TMEM_WIDTH] = {};
            ku::tmem_st_32dp32bNx<TMEM_WIDTH>(TMEM_COL_C, zeros);
            cutlass::arch::fence_view_async_tmem_store();

            uint32_t data[TMEM_WIDTH];
            for (int w = 0; w < 20; w++) {
                ku::tmem_ld_32dp32bNx<TMEM_WIDTH>(TMEM_COL_C, data);
                cutlass::arch::fence_view_async_tmem_load();
            }

            // All 32 threads read clocks; only thread 0 writes result.
            int64_t gt_start;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
            uint64_t c_start = clock64_start();

            for (int i = 0; i < iters; i++) {
                ku::tmem_ld_32dp32bNx<TMEM_WIDTH>(TMEM_COL_C, data);
                cutlass::arch::fence_view_async_tmem_load();
            }

            uint64_t c_end = clock64_stop();
            int64_t gt_end;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

            if (threadIdx.x == 0) {
                result->total_cycles = c_end - c_start;
                result->gt_start_ns  = gt_start;
                result->gt_end_ns    = gt_end;
            }
        }

    } else if constexpr (MODE == 1) {
        // --- tmem_store chain: serialized stores with fence ---
        if (threadIdx.x < 32) {
            uint32_t data[TMEM_WIDTH];
            for (int j = 0; j < TMEM_WIDTH; j++) data[j] = 0x3f800000u; // 1.0f

            for (int w = 0; w < 20; w++) {
                ku::tmem_st_32dp32bNx<TMEM_WIDTH>(TMEM_COL_C, data);
                cutlass::arch::fence_view_async_tmem_store();
            }

            int64_t gt_start;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
            uint64_t c_start = clock64_start();

            for (int i = 0; i < iters; i++) {
                ku::tmem_st_32dp32bNx<TMEM_WIDTH>(TMEM_COL_C, data);
                cutlass::arch::fence_view_async_tmem_store();
            }

            uint64_t c_end = clock64_stop();
            int64_t gt_end;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

            if (threadIdx.x == 0) {
                result->total_cycles = c_end - c_start;
                result->gt_start_ns  = gt_start;
                result->gt_end_ns    = gt_end;
            }
        }

    } else if constexpr (MODE == 2) {
        if constexpr (TMEM_WIDTH == 0) {
            // tcgen05.wait::ld/st.sync.aligned are warp-collective (.sync.aligned).
            if (threadIdx.x < 32) {
                for (int w = 0; w < 100; w++) {
                    cutlass::arch::fence_view_async_tmem_load();
                    cutlass::arch::fence_view_async_tmem_store();
                }

                int64_t gt_start;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
                uint64_t c_start = clock64_start();

                for (int i = 0; i < iters; i++) {
                    cutlass::arch::fence_view_async_tmem_load();
                    cutlass::arch::fence_view_async_tmem_store();
                }

                uint64_t c_end = clock64_stop();
                int64_t gt_end;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

                if (threadIdx.x == 0) {
                    result->total_cycles = c_end - c_start;
                    result->gt_start_ns  = gt_start;
                    result->gt_end_ns    = gt_end;
                }
            }
        } else {
            // tcgen05.fence::before/after_thread_sync are per-thread ordering fences.
            if (threadIdx.x == 0) {
                for (int w = 0; w < 100; w++) {
                    ku::tcgen05_before_thread_sync();
                    ku::tcgen05_after_thread_sync();
                }

                int64_t gt_start;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
                uint64_t c_start = clock64_start();

                for (int i = 0; i < iters; i++) {
                    ku::tcgen05_before_thread_sync();
                    ku::tcgen05_after_thread_sync();
                }

                uint64_t c_end = clock64_stop();
                int64_t gt_end;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

                result->total_cycles = c_end - c_start;
                result->gt_start_ns  = gt_start;
                result->gt_end_ns    = gt_end;
            }
        }

    } else if constexpr (MODE == 3) {
        // --- full rescale roundtrip (competition kernel pattern) ---
        // TMEM_WIDTH = number of 64-col chunks (1 or 4)
        static constexpr int CHUNK_SIZE = 64;
        static constexpr int NUM_CHUNKS = TMEM_WIDTH;

        if (threadIdx.x < 32) {
            // Initialize TMEM via warp-collective st before first load.
            {
                uint32_t zeros[CHUNK_SIZE] = {};
                for (int c = 0; c < NUM_CHUNKS; c++) {
                    ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_C + c * CHUNK_SIZE, zeros);
                }
                cutlass::arch::fence_view_async_tmem_store();
            }

            float2 scale = {0.5f, 0.5f};

            for (int w = 0; w < 20; w++) {
                ku::tcgen05_after_thread_sync();
                float2 o[CHUNK_SIZE / 2];
                CUTE_UNROLL
                for (int c = 0; c < NUM_CHUNKS; c++) {
                    ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_C + c * CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    for (int j = 0; j < CHUNK_SIZE / 2; j++)
                        o[j] = ku::float2_mul(o[j], scale);
                    ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_C + c * CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_store();
                }
                ku::tcgen05_before_thread_sync();
            }

            int64_t gt_start;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
            uint64_t c_start = clock64_start();

            for (int i = 0; i < iters; i++) {
                ku::tcgen05_after_thread_sync();
                float2 o[CHUNK_SIZE / 2];
                CUTE_UNROLL
                for (int c = 0; c < NUM_CHUNKS; c++) {
                    ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(TMEM_COL_C + c * CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    for (int j = 0; j < CHUNK_SIZE / 2; j++)
                        o[j] = ku::float2_mul(o[j], scale);
                    ku::tmem_st_32dp32bNx<CHUNK_SIZE>(TMEM_COL_C + c * CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_store();
                }
                ku::tcgen05_before_thread_sync();
            }

            uint64_t c_end = clock64_stop();
            int64_t gt_end;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

            if (threadIdx.x == 0) {
                result->total_cycles = c_end - c_start;
                result->gt_start_ns  = gt_start;
                result->gt_end_ns    = gt_end;
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// =========================================================================
// Experiment 8+9: mbarrier_commit_overhead
// Measure barrier and commit synchronization costs.
// MODE=0: named_barrier 128 threads, MODE=1: named_barrier 32 threads,
// MODE=2: mbarrier transac roundtrip, MODE=3: MMA + commit + wait
// =========================================================================
template<int MODE>
__global__ __launch_bounds__(128, 1)
void kernel_mbarrier_commit(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    if constexpr (MODE == 0) {
        // --- NamedBarrier 128 threads ---
        __syncthreads();

        int64_t gt_start = 0, gt_end = 0;
        uint64_t c_start = 0, c_end = 0;

        if (threadIdx.x == 0) {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
            c_start = clock64_start();
        }

        for (int i = 0; i < iters; i++) {
            asm volatile("bar.sync %0, %1;" : : "r"(8), "r"(128) : "memory");
        }

        if (threadIdx.x == 0) {
            c_end = clock64_stop();
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");
            result->total_cycles = c_end - c_start;
            result->gt_start_ns  = gt_start;
            result->gt_end_ns    = gt_end;
        }

    } else if constexpr (MODE == 1) {
        // --- NamedBarrier 32 threads (warp-level) ---
        __syncthreads();

        if (threadIdx.x < 32) {
            for (int w = 0; w < 100; w++) {
                asm volatile("bar.sync %0, %1;" : : "r"(9), "r"(32) : "memory");
            }

            int64_t gt_start = 0, gt_end = 0;
            uint64_t c_start = 0, c_end = 0;

            if (threadIdx.x == 0) {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
                c_start = clock64_start();
            }

            for (int i = 0; i < iters; i++) {
                asm volatile("bar.sync %0, %1;" : : "r"(9), "r"(32) : "memory");
            }

            if (threadIdx.x == 0) {
                c_end = clock64_stop();
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");
                result->total_cycles = c_end - c_start;
                result->gt_start_ns  = gt_start;
                result->gt_end_ns    = gt_end;
            }
        }

    } else if constexpr (MODE == 2) {
        // --- mbarrier transac roundtrip (single thread) ---
        __shared__ __align__(16) uint64_t smem_bar_storage[2];
        auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

        if (threadIdx.x == 0) bar->init(1);
        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t phase = 0;

            for (int w = 0; w < 100; w++) {
                bar->arrive_and_expect_tx(0);
                bar->wait(phase);
                phase ^= 1;
            }

            int64_t gt_start;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
            uint64_t c_start = clock64_start();

            for (int i = 0; i < iters; i++) {
                bar->arrive_and_expect_tx(0);
                bar->wait(phase);
                phase ^= 1;
            }

            uint64_t c_end = clock64_stop();
            int64_t gt_end;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

            result->total_cycles = c_end - c_start;
            result->gt_start_ns  = gt_start;
            result->gt_end_ns    = gt_end;
        }

    } else if constexpr (MODE == 3) {
        // --- MMA + commit + wait (isolate commit overhead) ---
        static constexpr int MC = 64, NC = 64, KC = 16;

        extern __shared__ char smem_raw[];
        __shared__ __align__(16) uint32_t smem_tmem_addr;
        __shared__ __align__(16) uint64_t smem_bar_storage[2];

        auto layout_B = ku::make_umma_canonical_k_major_layout<NC, KC, 32, bf16>();
        bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);
        auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

        if (threadIdx.x < 32) {
            TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
            TMEM::Allocator1Sm().release_allocation_lock();
        }
        if (threadIdx.x == 0) bar->init(1);
        __syncthreads();

        if (threadIdx.x == 0) {
            using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
                bf16, bf16, float, MC, NC, UMMA::Major::K, UMMA::Major::K>;
            auto tiled_mma = make_tiled_mma(MMA_Atom{});

            auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);
            auto thr_mma = tiled_mma.get_slice(_0{});
            auto tA_frag = thr_mma.make_fragment_A(
                partition_shape_A(tiled_mma, Shape<Int<MC>, Int<KC>>{}));
            tA_frag.data().get() = TMEM_COL_A;
            auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<MC>, Int<NC>>{});
            tC_frag.data().get() = TMEM_COL_C;
            auto sB_frag = thr_mma.partition_fragment_B(sB);

            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            uint32_t phase = 0;

            for (int w = 0; w < 100; w++) {
                gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
                ku::umma_arrive_noelect(*bar);
                bar->wait(phase);
                ku::tcgen05_after_thread_sync();
                phase ^= 1;
            }

            int64_t gt_start;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
            uint64_t c_start = clock64_start();

            for (int i = 0; i < iters; i++) {
                gemm(tiled_mma, tA_frag(_, _, 0), sB_frag(_, _, 0), tC_frag);
                ku::umma_arrive_noelect(*bar);
                bar->wait(phase);
                ku::tcgen05_after_thread_sync();
                phase ^= 1;
            }

            uint64_t c_end = clock64_stop();
            int64_t gt_end;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

            result->total_cycles = c_end - c_start;
            result->gt_start_ns  = gt_start;
            result->gt_end_ns    = gt_end;
        }

        __syncthreads();
        if (threadIdx.x < 32) {
            TMEM::Allocator1Sm().free(0, 512);
        }
    }
}

// =========================================================================
// Experiment 10: utcmma_dual_gemm_layout
// Verify that dual-GEMM smem layout [128, K] produces identical performance.
// LAYOUT_MODE=0: canonical helper, LAYOUT_MODE=1: competition-style
// =========================================================================
template<int M, int N, int K_DEPTH, int LAYOUT_MODE>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_dual_gemm_layout(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    auto layout_B = [&]() {
        if constexpr (LAYOUT_MODE == 0) {
            return ku::make_umma_canonical_k_major_layout<N, K_TOTAL, 128, bf16>();
        } else {
            return coalesce(tile_to_shape(
                UMMA::Layout_K_SW128_Atom<bf16>{},
                Shape<Int<N>, Int<K_TOTAL>>{},
                Step<_1, _2>{}
            ), Shape<_1, _1>{});
        }
    }();

    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    // Barrier for MMA completion synchronization
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        using MMA_Atom = SM100_MMA_F16BF16_WS_TS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
        tA_frag.data().get() = TMEM_COL_A;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// =========================================================================
// Experiment 12: utcmma_non_ws_ts
// Compare non-.ws tcgen05.mma with .ws variant.
// Uses SM100_MMA_F16BF16_TS_NOELECT (no .ws, extra mask registers).
// Non-WS C accumulator uses N columns (not N/2 like WS).
// =========================================================================
template<int M, int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_non_ws_ts(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    constexpr int SWIZZLE_B = (K_TOTAL >= 64) ? 128 : (K_TOTAL >= 32 ? 64 : 32);
    auto layout_B = ku::make_umma_canonical_k_major_layout<N, K_TOTAL, SWIZZLE_B, bf16>();
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    // Barrier for MMA completion synchronization
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Non-WS atom: tcgen05.mma.cta_group::1.kind::f16 (no .ws)
        using MMA_Atom = SM100_MMA_F16BF16_TS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto tA_frag = thr_mma.make_fragment_A(
            partition_shape_A(tiled_mma, Shape<Int<M>, Int<K_TOTAL>>{}));
        tA_frag.data().get() = TMEM_COL_A;

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        auto sB_frag = thr_mma.partition_fragment_B(sB);

        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// =========================================================================
// Experiment 15: kernel_utcmma_non_ws_ss
// Non-WS SS (Shared×Shared, no .ws) — comparison with WS-SS (Exp 3)
// C accumulator uses N full TMEM columns (not N/2 like WS)
// =========================================================================
template<int M, int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_non_ws_ss(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int K_PER_TILE = 16;
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // A operand: [M x K_TOTAL] in K-major INTER layout (same as WS-SS Exp 3)
    auto layout_A = coalesce(tile_to_shape(
        UMMA::Layout_K_INTER_Atom<bf16>{},
        Shape<Int<M>, Int<K_TOTAL>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{});

    // B operand: [N x K_TOTAL] in MN-major layout with appropriate swizzle
    // MN-atom sizes: INTER=8, SW32=16, SW64=32, SW128=64. N must be divisible by atom.
    // Cascade: pick largest swizzle where both K and N constraints are met.
    constexpr int SWIZZLE_B = (K_TOTAL >= 64 && N % 64 == 0) ? 128 :
                              (K_TOTAL >= 32 && N % 32 == 0) ? 64  :
                              (N % 16 == 0)                  ? 32  : 0;
    auto layout_B = [&]() {
        if constexpr (SWIZZLE_B == 128) {
            return coalesce(tile_to_shape(
                UMMA::Layout_MN_SW128_Atom<bf16>{},
                Shape<Int<N>, Int<K_TOTAL>>{},
                Step<_2, _1>{}
            ), Shape<_1, _1>{});
        } else if constexpr (SWIZZLE_B == 64) {
            return coalesce(tile_to_shape(
                UMMA::Layout_MN_SW64_Atom<bf16>{},
                Shape<Int<N>, Int<K_TOTAL>>{},
                Step<_2, _1>{}
            ), Shape<_1, _1>{});
        } else if constexpr (SWIZZLE_B == 32) {
            return coalesce(tile_to_shape(
                UMMA::Layout_MN_SW32_Atom<bf16>{},
                Shape<Int<N>, Int<K_TOTAL>>{},
                Step<_2, _1>{}
            ), Shape<_1, _1>{});
        } else {
            return coalesce(tile_to_shape(
                UMMA::Layout_MN_INTER_Atom<bf16>{},
                Shape<Int<N>, Int<K_TOTAL>>{},
                Step<_2, _1>{}
            ), Shape<_1, _1>{});
        }
    }();

    bf16* smem_A = reinterpret_cast<bf16*>(smem_raw);
    bf16* smem_B = smem_A + cosize(layout_A);

    // Barrier for MMA completion synchronization
    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Non-WS SS atom: tcgen05.mma.cta_group::1.kind::f16 (no .ws)
        using MMA_Atom = SM100_MMA_F16BF16_SS_NOELECT<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::MN>;
        auto tiled_mma = make_tiled_mma(MMA_Atom{});

        auto sA = make_tensor(make_smem_ptr(smem_A), layout_A);
        auto sB = make_tensor(make_smem_ptr(smem_B), layout_B);

        auto thr_mma = tiled_mma.get_slice(_0{});
        auto sA_frag = thr_mma.partition_fragment_A(sA);
        auto sB_frag = thr_mma.partition_fragment_B(sB);

        auto tC_frag = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
        tC_frag.data().get() = TMEM_COL_C;

        // Warmup
        uint32_t phase = 0;
        for (int w = 0; w < 20; w++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // Timed region
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int k = 0; k < K_DEPTH; ++k) {
                gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// =========================================================================
// Experiment 13: utcmma_m32_ws_ts — M=32 WS-TS latency via raw PTX
//
// CuTe's MMA_Traits uses FrgTypeA = UMMA::tmem_frg_1sm which static_asserts
// M >= 64. The PTX instruction tcgen05.mma.ws.cta_group::1 supports M=32.
// We bypass CuTe TiledMMA entirely, calling SM100_MMA_F16BF16_WS_TS_NOELECT
// ::fma() directly with manually constructed descriptors.
//
// B smem descriptor: created via UMMA::make_umma_desc (M-independent).
// Instruction descriptor: UMMA::make_instr_desc with M=32 (m_dim_=2).
// TMEM columns: C at col 0 (N/4 cols for WS M=32), A at col 256.
// =========================================================================
template<int N, int K_DEPTH>
__global__ __launch_bounds__(128, 1)
void kernel_utcmma_m32_ws_ts(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    static constexpr int M = 32;
    static constexpr int K_PER_TILE = 16;  // bf16: 256 bits / 16 bits = 16 elements
    static constexpr int K_TOTAL = K_DEPTH * K_PER_TILE;

    extern __shared__ char smem_raw[];
    __shared__ __align__(16) uint32_t smem_tmem_addr;

    // B operand in shared memory: K_DEPTH tiles of [N x 16] bf16.
    // Use SW32 for per-tile descriptors (K-atom=16 = K_PER_TILE).
    // Swizzle doesn't affect MMA latency (confirmed by experiment 5).
    bf16* smem_B = reinterpret_cast<bf16*>(smem_raw);

    __shared__ __align__(16) uint64_t smem_bar_storage[2];
    auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(smem_bar_storage);

    // TMEM allocation (full warp required)
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    if (threadIdx.x == 0) {
        bar->init(1);
    }
    __syncthreads();

    // --- Only thread 0 runs the benchmark ---
    if (threadIdx.x == 0) {
        // Create per-K-tile smem descriptors for B
        auto tile_layout = ku::make_umma_canonical_k_major_layout<N, K_PER_TILE, 32, bf16>();
        uint64_t desc_b[K_DEPTH];
        for (int k = 0; k < K_DEPTH; k++) {
            auto sB_k = make_tensor(make_smem_ptr(smem_B + k * N * K_PER_TILE), tile_layout);
            desc_b[k] = UMMA::make_umma_desc<UMMA::Major::K>(sB_k);
        }

        // M=32 instruction descriptor (m_dim_ = 32>>4 = 2, valid for .ws)
        constexpr auto idesc_struct = UMMA::make_instr_desc<
            bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>();
        uint64_t idescE = UMMA::make_runtime_instr_desc<>(idesc_struct);

        // TMEM addresses
        uint32_t tmem_a = TMEM_COL_A;  // 256
        uint32_t tmem_c = TMEM_COL_C;  // 0

        // Warmup with proper sync
        uint32_t phase = 0;
        for (int w = 0; w < 100; w++) {
            uint32_t scaleC = 0;  // clear on first K-tile
            for (int k = 0; k < K_DEPTH; k++) {
                SM100_MMA_F16BF16_WS_TS_NOELECT<
                    bf16, bf16, float, M, N,
                    UMMA::Major::K, UMMA::Major::K
                >::fma(tmem_a, desc_b[k], tmem_c, scaleC, idescE);
                scaleC = 1;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        // --- Timed region ---
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            uint32_t scaleC = 0;
            for (int k = 0; k < K_DEPTH; k++) {
                SM100_MMA_F16BF16_WS_TS_NOELECT<
                    bf16, bf16, float, M, N,
                    UMMA::Major::K, UMMA::Major::K
                >::fma(tmem_a, desc_b[k], tmem_c, scaleC, idescE);
                scaleC = 1;
            }
            ku::umma_arrive_noelect(*bar);
            bar->wait(phase);
            ku::tcgen05_after_thread_sync();
            phase ^= 1;
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        result->total_cycles = c_end - c_start;
        result->gt_start_ns  = gt_start;
        result->gt_end_ns    = gt_end;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}

// =========================================================================
// Experiment 19: tmem_ld_modes — compare TMEM load addressing modes
//
// ADDR_MODE 0: tcgen05.ld.32dp32bNx  (WIDTH = N replications, each 32b per lane)
// ADDR_MODE 1: tcgen05.ld.16dp128bNx (WIDTH = N replications, each 128b = 4 regs)
// ADDR_MODE 2: tcgen05.ld.16dp256bNx (WIDTH = N replications, each 256b = 8 regs)
//
// Measures serialized load+fence latency per call at various data widths.
// Compare at equal data sizes: 32dp32b×4 vs 16dp128b×1 (both 128 bits).
// =========================================================================
template<int ADDR_MODE, int WIDTH>
__global__ __launch_bounds__(128, 1)
void kernel_tmem_ld_modes(BenchResult* result, int iters) {
    using namespace cute;
    namespace ku = kerutils;

    __shared__ __align__(16) uint32_t smem_tmem_addr;

    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().allocate(512, &smem_tmem_addr);
        TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        // Compute number of uint32_t values per call
        constexpr int REGS_PER_CALL =
            (ADDR_MODE == 0) ? WIDTH :       // 32dp32b: 1 reg per replication
            (ADDR_MODE == 1) ? WIDTH * 2 :   // 16dp128b: 2 regs per replication
                               WIDTH * 4;    // 16dp256b: 4 regs per replication

        // Initialize TMEM with non-zero data (warp-collective st)
        uint32_t init_data[REGS_PER_CALL];
        for (int j = 0; j < REGS_PER_CALL; j++) init_data[j] = j + 1;
        ku::tmem_st_32dp32bNx<REGS_PER_CALL>(TMEM_COL_C, init_data);
        cutlass::arch::fence_view_async_tmem_store();

        uint32_t data[REGS_PER_CALL];
        uint32_t col = TMEM_COL_C;

        // Warmup (with RAW dependency)
        for (int w = 0; w < 200; w++) {
            if constexpr (ADDR_MODE == 0) {
                ku::tmem_ld_32dp32bNx<WIDTH>(col, data);
            } else if constexpr (ADDR_MODE == 1) {
                ku::tmem_ld_16dp128bNx<WIDTH>(col, data);
            } else {
                ku::tmem_ld_16dp256bNx<WIDTH>(col, data);
            }
            cutlass::arch::fence_view_async_tmem_load();
            // RAW dependency: force wait for data[0] before computing next col.
            // PTX and.b32 tmp, data[0], 0 → tmp = 0 (runtime), but creates true
            // data dependency. add.u32 col, tmp, TMEM_COL_C restores original col.
            asm volatile("{ .reg .u32 tmp; and.b32 tmp, %1, 0; add.u32 %0, tmp, %2; }"
                         : "=r"(col) : "r"(data[0]), "r"((uint32_t)TMEM_COL_C));
        }

        // Timed region
        int64_t gt_start;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_start) :: "memory");
        uint64_t c_start = clock64_start();

        for (int i = 0; i < iters; i++) {
            if constexpr (ADDR_MODE == 0) {
                ku::tmem_ld_32dp32bNx<WIDTH>(col, data);
            } else if constexpr (ADDR_MODE == 1) {
                ku::tmem_ld_16dp128bNx<WIDTH>(col, data);
            } else {
                ku::tmem_ld_16dp256bNx<WIDTH>(col, data);
            }
            cutlass::arch::fence_view_async_tmem_load();
            // RAW dependency chain — serializes load-to-use latency
            asm volatile("{ .reg .u32 tmp; and.b32 tmp, %1, 0; add.u32 %0, tmp, %2; }"
                         : "=r"(col) : "r"(data[0]), "r"((uint32_t)TMEM_COL_C));
        }

        uint64_t c_end = clock64_stop();
        int64_t gt_end;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_end) :: "memory");

        // Anti-DCE: prevent compiler from removing data loads
        if (data[0] == 0xDEADDEADu) {
            result->total_cycles = data[1];
        }

        if (threadIdx.x == 0) {
            result->total_cycles = c_end - c_start;
            result->gt_start_ns  = gt_start;
            result->gt_end_ns    = gt_end;
        }
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        TMEM::Allocator1Sm().free(0, 512);
    }
}
