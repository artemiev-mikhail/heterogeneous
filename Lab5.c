// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan_with_sums(float *input, float *output, int numElements, float *sums)
{
  __shared__ float shInput[2*BLOCK_SIZE];

  const int tx = threadIdx.x;
  const int start = 2 * BLOCK_SIZE * blockIdx.x;

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
    __syncthreads();
    const int index = (tx + 1) * stride * 2 - 1;
    if (index < 2*BLOCK_SIZE)
      shInput[index] += shInput[index - stride];
  }

  for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    const int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2*BLOCK_SIZE)
      shInput[index + stride] += shInput[index];
  }

  __syncthreads();
  if (tx == BLOCK_SIZE-1)                        // last thread
    sums[blockIdx.x] = shInput[BLOCK_SIZE + tx]; // saves the sum of the section

  if (start + tx < numElements)
    output[start + tx] = shInput[tx];
  if (BLOCK_SIZE + start + tx < numElements)
    output[BLOCK_SIZE + start + tx] = shInput[BLOCK_SIZE + tx];
}




__global__ void scan_plain(float *input, float *output, int numElements)
{
  __shared__ float shInput[2*BLOCK_SIZE];

  const int tx = threadIdx.x;
  const int start = 2 * BLOCK_SIZE * blockIdx.x;

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
    __syncthreads();
    const int index = (tx + 1) * stride * 2 - 1;
    if (index < 2*BLOCK_SIZE)
      shInput[index] += shInput[index - stride];
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




__global__ void add_sums(float *output, int numOutputElements, float *sums, int numSumsElements)
{
  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int start = 2 * BLOCK_SIZE * bx;

  float sum;
  if (bx < numSumsElements)
    sum = sums[bx];
  else
    sum = 0.;

  __syncthreads();

  if (start + tx < numOutputElements)
    output[start + tx] += sum;
  if (BLOCK_SIZE + start + tx < numOutputElements)
    output[BLOCK_SIZE + start + tx] += sum;
}




int main(int argc, char ** argv) 
{
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");
   
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize((numElements-1)/(2*BLOCK_SIZE) + 1, 1, 1);
    wbLog(TRACE, "grid: ", gridSize.x, " x ", gridSize.y, " x ", gridSize.z);
    wbLog(TRACE, "block: ", blockSize.x, " x ", blockSize.y, " x ", blockSize.z);

    if (gridSize.x == 1) // if there is only 1 section
    {
      scan_plain<<<gridSize, blockSize>>>(deviceInput, deviceOutput, numElements);
      cudaDeviceSynchronize();
    } 
    else // if there are more than 1 sections
    {
//      float *hostSums = (float*) malloc(gridSize.x * sizeof(float));
      float *deviceSumsInput;
      float *deviceSumsOutput;
      wbCheck(cudaMalloc((void**)&deviceSumsInput,  gridSize.x*sizeof(float)));
      wbCheck(cudaMalloc((void**)&deviceSumsOutput, gridSize.x*sizeof(float)));
      wbCheck(cudaMemset(deviceSumsOutput, 0, gridSize.x*sizeof(float)));
//      wbCheck(cudaMemcpy(deviceSumsInput, hostSums, gridSize.x * sizeof(float), cudaMemcpyHostToDevice));

      scan_with_sums<<<gridSize, blockSize>>>(deviceInput, deviceOutput, numElements, deviceSumsInput);
      cudaDeviceSynchronize();

      dim3 gridSizeAux((gridSize.x-1)/(2*BLOCK_SIZE) + 1, 1, 1);
      wbLog(TRACE, "grid aux: ", gridSizeAux.x, " x ", gridSizeAux.y, " x ", gridSizeAux.z);

      scan_plain<<<gridSizeAux, blockSize>>>(deviceSumsInput, deviceSumsOutput, gridSize.x);
      cudaDeviceSynchronize();
       
      add_sums<<<gridSize, blockSize>>>(deviceOutput, numElements, deviceSumsOutput, gridSize.x);
      cudaDeviceSynchronize();

      wbCheck(cudaFree(deviceSumsOutput));
      wbCheck(cudaFree(deviceSumsInput));
    }

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceOutput));
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


