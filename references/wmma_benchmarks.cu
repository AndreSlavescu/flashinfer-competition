__global__void wmma_example(atype *a, btype *b, ctype *c,dtype *d) {
    unsigned int start,start_time=0,end_time=0;
    // Part 1: Declare the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, atype , LAYOUT_A> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, btype , LAYOUT_B> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, ctype> c_frag;
    // Part 2: loading the values from the memory
    wmma::load_matrix_sync(a_frag, a, A_STRIDE);
    wmma::load_matrix_sync(b_frag, b, B_STRIDE);
    wmma::load_matrix_sync(c_frag, c, C_STRIDE,LAYOUT_C);

    // Part 3: running the multiple+add on matrices
    start_time=clock();
    for (int i=0; i<iters*iters1; i++){
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::mma_sync(c1_frag, a1_frag, b1_frag, c1_frag);
        wmma::mma_sync(c2_frag, a2_frag, b2_frag, c2_frag);
        wmma::mma_sync(c3_frag, a3_frag, b3_frag, c3_frag);
    }
    end_time=clock();

    // Part 4: store the values from the memory
    wmma::store_matrix_sync(d, c_frag, C_STRIDE, LAYOUT_C);

    if(threadIdx.x==1)
        printf("CLOCK for all=%d \t %d \n",((end_time-start_time)-2)/(4*iters1),tid);
}