#include <wb.h>

//=============================================================================
//
// Error checker macro
//
//=============================================================================
#define check(err)                                         \
  do {                                                     \
    if (err != cudaSuccess) {                              \
      printf("\nError found\n%s in %s at line %d",         \
             cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
  } while (0);

//=============================================================================
//
// The number of CUDA streams
//
//=============================================================================
#define N_STREAMS 4

//=============================================================================
//
// The size of the CUDA block (number of threads)
//
//=============================================================================
#define BLOCK_SIZE 128

//=============================================================================
//
// CUDA kernel for vector addition
//
//=============================================================================
__global__ void vecAdd(float *in1, float *in2, float *out, int len)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
}

//=============================================================================
//
// main() function
//
//=============================================================================
int main(int argc, char **argv)
{
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1[N_STREAMS];
  float *deviceInput2[N_STREAMS];
  float *deviceOutput[N_STREAMS];
  cudaStream_t stream[N_STREAMS];
  cudaError_t err;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *) malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //---------------------------------------------------------------------------
  // Allocate memory on device for each CUDA stream
  //---------------------------------------------------------------------------
  const int segmentSize = BLOCK_SIZE;
  for (int s = 0; s < N_STREAMS; ++s)
  {
    err = cudaMalloc((void**)&deviceInput1[s], segmentSize*sizeof(float)); check(err);
    err = cudaMalloc((void**)&deviceInput2[s], segmentSize*sizeof(float)); check(err);
    err = cudaMalloc((void**)&deviceOutput[s], segmentSize*sizeof(float)); check(err);
  }
  
  //---------------------------------------------------------------------------
  // Launch the computations: async copy, compute, async copy back
  //---------------------------------------------------------------------------
  for (int i = 0; i < inputLength; i += segmentSize*N_STREAMS)
  {
    for (int s = 0; s < N_STREAMS; ++s)
    {
      err = cudaMemcpyAsync(deviceInput1[s],
                            hostInput1 + i + s*segmentSize,
                            segmentSize*sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream[s]); check(err);
      err = cudaMemcpyAsync(deviceInput2[s],
                            hostInput2 + i + s*segmentSize,
                            segmentSize*sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream[s]); check(err);
    }
    
    for (int s = 0; s < N_STREAMS; ++s)
    {
      // because segmentSize = BLOCK_SIZE there is only one block for the kernel
      vecAdd<<<1, BLOCK_SIZE, 0, stream[s]>>>(deviceInput1[s],
                                              deviceInput2[s],
                                              deviceOutput,
                                              inputLength);
    }
                                                                     
    for (int s = 0; s < N_STREAMS; ++s)
    {
      err = cudaMemcpyAsync(hostOutput + i + s*segmentSize,
                            deviceOutput[s],
                            segmentSize*sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream[s]); check(err);
    }
  }
  
  //---------------------------------------------------------------------------
  // Compare the solution
  //---------------------------------------------------------------------------
  wbSolution(args, hostOutput, inputLength);

  //---------------------------------------------------------------------------
  // Free the memory
  //---------------------------------------------------------------------------
  for (int s = 0; s < N_STREAMS; ++s)
  {
    err = cudaFree(deviceInput1[s]); check(err);
    err = cudaFree(deviceInput2[s]); check(err);
    err = cudaFree(deviceOutput[s]); check(err);
  }
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}


