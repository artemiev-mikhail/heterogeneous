__global__ void efficient_scan(float *input, float *output, int numElements)
{
  __shared__ float shInput[2*BLOCK_SIZE];

  const int tx = threadIdx.x;
  const int start = 2*blockDim.x*blockIdx.x;

  if (start + tx < numElements) 
    shInput[tx] = input[start + tx];
  else
    shInput[tx] = 0.;

  if (BLOCK_SIZE + start + tx < numElements)
    shInput[BLOCK_SIZE + tx] = input[BLOCK_SIZE + start + tx];
  else
    shInput[BLOCK_SIZE + tx] = 0.;

  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
  {
    const int index = (tx + 1) * stride * 2 - 1;
    if (index < 2*BLOCK_SIZE)
      shInput[index] += shInput[index - stride];
    __syncthreads();
  }

  for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    const int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2*BLOCK_SIZE)
      shInput[index + stride] += shInput[index];
  }

  __syncthreads();
  if (start + tx < numElements)
    output[start + tx] = shInput[tx];
  if (BLOCK_SIZE + start + tx < numElements)
    output[BLOCK_SIZE + start + tx] = shInput[BLOCK_SIZE + tx];
}
