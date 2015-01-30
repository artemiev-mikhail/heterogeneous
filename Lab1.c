// MP 1
#include <wb.h>

#define wbCheck(err) do {                                        \
          if (err != cudaSuccess) {                              \
            printf("\nError found\n%s in %s at line %d",         \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                  \
          }                                                      \
        } while (0);


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
	
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
	
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  cudaError_t err;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
	
  int size = inputLength * sizeof(float);

  wbTime_start(GPU, "Allocating GPU memory.");  
	
  //@@ Allocate GPU memory here
  err = cudaMalloc((void**) &deviceInput1, size); wbCheck(err);
  err = cudaMalloc((void**) &deviceInput2, size); wbCheck(err);
  err = cudaMalloc((void**) &deviceOutput, size); wbCheck(err);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");

  //@@ Copy memory to the GPU here
  err = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice); wbCheck(err);
  err = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice); wbCheck(err);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  const int n_threads = 256;
  dim3 dimGrid((inputLength - 1) / n_threads + 1, 1, 1);
  dim3 dimBlock(n_threads, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Launch the GPU Kernel here
  vecAdd<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
	
  //@@ Copy the GPU memory back to the CPU here
  err = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost); wbCheck(err);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  
  //@@ Free the GPU memory here
  err = cudaFree(deviceOutput); wbCheck(err);
  err = cudaFree(deviceInput2); wbCheck(err);
  err = cudaFree(deviceInput1); wbCheck(err);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

