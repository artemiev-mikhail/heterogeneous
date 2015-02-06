__global__ void sum_better_reduction(float *array, int length)
{
  const int tx = threadIdx.x;
  const int n_threads = blockDim.x; // number of threads in a 1D block
  
  const int start = 2 * blockIdx.x * n_threads;
  
  __shared__ float partial_sum[2*n_threads];
  
  
  // first half
  if (start + tx < length)
    partial_sum[tx] = array[start + tx];
  else
    partial_sum[tx] = 0.;
    
  // second half
  if (start + n_threads + tx < length)
    partial_sum[n_threads + tx] = array[start + n_threads + tx];
  else
    partial_sum[n_threads + tx] = 0.;


  for (int stride = n_threads; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride)
      partial_sum[tx] += partial_sum[tx + stride];
  }  
}
