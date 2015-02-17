__global__ void histo_private(char *buffer, int len, int *histo)
{
  const int n_symbols = 256; // assume we've got only this number of different symbols 
  __shared__ int private_histogram[n_symbols];
  
  const int tx = threadIdx.x;
  
  // initialize the privatized histogram (only first n_symbols threads do that)
  if (tx < n_symbols)
    private_histogram[tx] = 0;
  __syncthreads();
  
  int i = tx + blockDim.x*blockIdx.x;
  const int stride = blockDim.x*gridDim.x;
  
  while (i < len)
  {
    atomicAdd(&private_histogram[buffer[i]], 1);
    i += stride;
  }
  __syncthreads();
  
  if (tx < n_symbols)
    atomicAdd(&histo[tx], private_histogram[tx]);
  __syncthreads();
}
