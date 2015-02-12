__global__ void inefficient_scan(float *input, float *output, int numElements)
{
  // output must be filled with zeros at the beginning

  __shared__ float shInput[BLOCK_SIZE];

  const int ind = blockDim.x*blockIdx.x + threadIdx.x;
  if (ind < numElements) shInput[threadIdx.x] = input[ind];

  for (int stride = 1; stride <= threadIdx.x; stride *= 2)
  {
    __syncthreads();
    float in = shInput[threadIdx.x - stride];
    __syncthreads();
    shInput[threadIdx.x] += in;
  }

  __syncthreads();
  if (ind < numElements) output[ind] = shInput[threadIdx.x];
}
