extern "C" __global__
void my_add(const float* x1, const float* x2, float* y, long long n) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < n){
      y[tid] = x1[tid] + x2[tid];
    }
}
