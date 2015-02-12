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
    
__global__ void scan(float *input, float *output, int numElements) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
  
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

int main(int argc, char ** argv) {
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

    //@@ Initialize the grid and block dimensions here
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize((numElements-1)/(BLOCK_SIZE<<1) + 1, 1, 1);
    wbLog(TRACE, "grid: ", gridSize.x, " x ", gridSize.y, " x ", gridSize.z);
    wbLog(TRACE, "block: ", blockSize.x, " x ", blockSize.y, " x ", blockSize.z);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<gridSize, blockSize>>>(deviceInput, deviceOutput, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


