// Pointer-chasing shared memory bandwidth benchmark
// dData : Pointer-chase array
// dSink : Side-effect destination variable (prevents code elimination)
// repeat : Count of pointer-chase steps requested
// To ensure all LSUs in an SM are used, use >= 128 threads
#define THREAD_NUM 1024
// shared memory per block
#define PCHASE_SIZE 8*THREAD_NUM

__global__ void bandwidthTest(uint32_t * dData, uint32_t * dSink, uint32_t repeat) {
    // Pointer-chase starting position in shared memory
    uint32_t sid = threadIdx.x;
    // The pointer-chase array in shared memory
    __shared__ DTYPE shrData[PCHASE_SIZE];
    // Initialize the pointer-chase array in shared memory
    for (uint32_t i = sid; i<PCHASE_SIZE; i+=THREAD_NUM)
        shrData[i] = dData[i];

    // Synchronize threads in a same block
    __syncthreads();
    
    // Scan the shared-memory array with the p-chase method
    unsigned next=sid;
    for (uint32_t j = 0; j < repeat; j++) {
        next = shrData[next];
    }

    // Side effect to prevent the compiler from eliminating this code
    dSink[sid] = next;
}
