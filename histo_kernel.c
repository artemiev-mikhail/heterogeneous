__global__ void histo_kernel(char *input, int len, int *histogram)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  
  const int stride = blockDim.x * gridDim.x;
  
  while (index < len)
  {
    atomicAdd(&(histogram[input[index]]), 1);
    index += stride;
  }
}
