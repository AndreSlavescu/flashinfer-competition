#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "common.cuh"

// ---------------------------------------------------------------------------
// Standalone cuTensorMapEncodeTiled wrapper — no CUTLASS dependency.
// Uses CUDA driver API directly for full control over every field.
// ---------------------------------------------------------------------------

// Function pointer type — defined manually to avoid dependency on cudaTypedefs.h
// which may not exist on all CUDA toolkit versions.
typedef CUresult (*PFN_cuTensorMapEncodeTiled_t)(
    CUtensorMap*             tensorMap,
    CUtensorMapDataType      tensorDataType,
    cuuint32_t               tensorRank,
    void*                    globalAddress,
    const cuuint64_t*        globalDim,
    const cuuint64_t*        globalStrides,
    const cuuint32_t*        boxDim,
    const cuuint32_t*        elementStrides,
    CUtensorMapInterleave    interleave,
    CUtensorMapSwizzle       swizzle,
    CUtensorMapL2promotion   l2Promotion,
    CUtensorMapFloatOOBfill  oobFill);

inline PFN_cuTensorMapEncodeTiled_t get_cuTensorMapEncodeTiled() {
    void* fn_ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    CUDA_CHECK(cudaGetDriverEntryPoint(
        "cuTensorMapEncodeTiled", &fn_ptr, cudaEnableDefault, &status));
    if (!fn_ptr) {
        fprintf(stderr, "Failed to get cuTensorMapEncodeTiled entry point\n");
        exit(1);
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_t>(fn_ptr);
}

// Gather4 operates on a 2D tensor map where:
//   dim0 = "columns" (the data dimension, e.g. head_dim packed into elements)
//   dim1 = "rows"    (the token dimension — gather4 scatters across this)
//
// The gather4 instruction picks 4 arbitrary row indices and loads
// box_dim0 elements from each row into shared memory.
//
// Parameters:
//   global_ptr   : base pointer to the 2D tensor in global memory
//   data_type    : element type (INT64 for packed, BFLOAT16 for native, etc.)
//   dim0         : number of elements along the fast dimension (columns)
//   num_rows     : total number of rows (tokens)
//   row_stride_bytes : stride between rows in bytes (must be 16B-aligned)
//   box_dim0     : number of elements per row loaded by one gather4 call
//                  (box_dim0 * sizeof(element) = bytes per row per gather4)
//   swizzle      : shared memory swizzle mode applied during copy
//   l2_promotion : L2 cache promotion hint
//
// Returns a CUtensorMap ready for use with cp.async.bulk.tensor.2d...gather4
inline CUtensorMap create_gather4_tensor_map(
    void*                   global_ptr,
    CUtensorMapDataType     data_type,
    uint64_t                dim0,
    uint64_t                num_rows,
    uint64_t                row_stride_bytes,
    uint32_t                box_dim0,
    CUtensorMapSwizzle      swizzle      = CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion  l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B)
{
    static auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    constexpr int rank = 2;
    uint64_t global_dim[rank]    = {dim0, num_rows};
    uint64_t global_stride[rank - 1] = {row_stride_bytes};
    uint32_t box_dim[rank]       = {box_dim0, 1};  // gather4 provides 4 row indices externally
    uint32_t elem_stride[rank]   = {1, 1};

    CUtensorMap tensor_map{};
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        data_type,
        rank,
        global_ptr,
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2_promotion,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n",
                err_str ? err_str : "unknown");
        fprintf(stderr, "  data_type=%d dim0=%lu num_rows=%lu stride=%lu box_dim0=%u swizzle=%d\n",
                (int)data_type, dim0, num_rows, row_stride_bytes, box_dim0, (int)swizzle);
        exit(1);
    }
    return tensor_map;
}

// ---------------------------------------------------------------------------
// Predefined configurations matching competition workloads
// ---------------------------------------------------------------------------

// Element size in bytes for each CUtensorMapDataType we use
inline size_t tensor_map_element_size(CUtensorMapDataType dt) {
    switch (dt) {
        case CU_TENSOR_MAP_DATA_TYPE_INT64:     return 8;
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:  return 2;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:   return 4;
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:     return 1;
        default:
            fprintf(stderr, "Unsupported data type: %d\n", (int)dt);
            exit(1);
    }
}

struct TensorMapConfig {
    const char*             name;
    CUtensorMapDataType     data_type;
    uint64_t                dim0;       // elements along fast dim
    uint32_t                box_dim0;   // elements per gather4 call
    CUtensorMapSwizzle      swizzle;
    CUtensorMapL2promotion  l2_promo;

    // Derived
    size_t bytes_per_row()     const { return dim0 * tensor_map_element_size(data_type); }
    size_t bytes_per_gather4() const { return box_dim0 * tensor_map_element_size(data_type) * 4; }
    int    col_steps()         const { return (int)(dim0 / box_dim0); }
};

// Config A: ckv 512d packed as INT64 (8 BF16 → 1 INT64), single col step
//   Matches FlashMLA's NoPE tensor map approach (kernel.cuh:913-921)
//   512 BF16 values = 1024 bytes = 128 INT64 elements, box=128, 1 col step
//   But gather4 box_dim0 max is 256 elements, and 128 INT64 × 8 bytes = 1024B per row
//   Actually: dim0=64 (D_NOPE/8=512/8=64 INT64 elements), box_dim0=64
inline TensorMapConfig config_ckv_int64() {
    return {"ckv_int64", CU_TENSOR_MAP_DATA_TYPE_INT64,
            64, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B};
}

// Config B: ckv 512d native BF16, 8 col steps (box=64, 64*2=128B per step)
//   box_dim0=64 fits SWIZZLE_128B (64*2=128B <= 128B)
//   col_steps = 512/64 = 8
inline TensorMapConfig config_ckv_bf16() {
    return {"ckv_bf16", CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            512, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B};
}

// Config C: kpe 64d native BF16, single col step, no swizzle
inline TensorMapConfig config_kpe_bf16() {
    return {"kpe_bf16", CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            64, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B};
}

// Config D: kpe 64d native BF16, single col step, SWIZZLE_128B
inline TensorMapConfig config_kpe_bf16_sw128() {
    return {"kpe_bf16_sw128", CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B};
}

// All competition-relevant configs
inline std::vector<TensorMapConfig> competition_configs() {
    return {config_ckv_int64(), config_ckv_bf16(), config_kpe_bf16(), config_kpe_bf16_sw128()};
}

// Create a tensor map from a config + allocated global memory
inline CUtensorMap create_tensor_map_from_config(
    const TensorMapConfig& cfg,
    void* global_ptr,
    uint64_t num_rows)
{
    size_t elem_sz = tensor_map_element_size(cfg.data_type);
    uint64_t row_stride_bytes = cfg.dim0 * elem_sz;
    // Ensure 16B alignment of stride
    row_stride_bytes = (row_stride_bytes + 15) & ~15ULL;

    return create_gather4_tensor_map(
        global_ptr, cfg.data_type,
        cfg.dim0, num_rows, row_stride_bytes,
        cfg.box_dim0, cfg.swizzle, cfg.l2_promo);
}

// Allocate a global memory tensor for benchmarking and fill with pattern
inline void* allocate_tensor(const TensorMapConfig& cfg, uint64_t num_rows) {
    size_t elem_sz = tensor_map_element_size(cfg.data_type);
    size_t row_stride = cfg.dim0 * elem_sz;
    row_stride = (row_stride + 15) & ~15ULL;
    size_t total_bytes = row_stride * num_rows;

    void* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, total_bytes));
    // Fill with non-zero pattern for correctness verification
    CUDA_CHECK(cudaMemset(d_ptr, 0x42, total_bytes));
    return d_ptr;
}
