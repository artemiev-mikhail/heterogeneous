#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define checkErr(err)                                      \
  do {                                                     \
    if (err != cudaSuccess) {                              \
      printf("\nError found\n%s in %s at line %d",         \
             cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
  } while (0);

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int numARows, int numAColumns, 
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {

  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP

  assert(numAColumns == numBRows);
  assert(numARows == numCRows);
  assert(numBColumns == numCColumns);

  const int row = blockIdx.x*blockDim.x + threadIdx.x;
  const int col = blockIdx.y*blockDim.y + threadIdx.y;

  if (row < numCRows && col < numCColumns) {
	  
    float c_val = 0;
	  
    for (int k = 0; k < numAColumns; ++k)
      c_val += A[row*numAColumns + k] * B[k*numBColumns + col];
	  
    C[row*numCColumns + col] = c_val;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  cudaError_t err;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
	
  const int sizeA_bytes = numARows * numAColumns * sizeof(float);
  const int sizeB_bytes = numBRows * numBColumns * sizeof(float);
  const int sizeC_bytes = numCRows * numCColumns * sizeof(float);

  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(sizeC_bytes);
  if (hostC == NULL) {
    printf("Bad allocation of the hostC matrix");
    exit(EXIT_FAILURE);
  }
	
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  
  //@@ Allocate GPU memory here
  err = cudaMalloc((void**)&deviceA, sizeA_bytes); checkErr(err);
  err = cudaMalloc((void**)&deviceB, sizeB_bytes); checkErr(err);
  err = cudaMalloc((void**)&deviceC, sizeC_bytes); checkErr(err);
	
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");

  //@@ Copy memory to the GPU here
  err = cudaMemcpy(deviceA, hostA, sizeA_bytes, cudaMemcpyHostToDevice); checkErr(err);
  err = cudaMemcpy(deviceB, hostB, sizeB_bytes, cudaMemcpyHostToDevice); checkErr(err);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 gridSize((numCRows-1)/TILE_WIDTH + 1, (numCColumns-1)/TILE_WIDTH + 1, 1);
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
	
  wbLog(TRACE, "The grid size is ", gridSize.x, " x ", gridSize.y, " x ", gridSize.z);
  wbLog(TRACE, "The block size is ", blockSize.x, " x ", blockSize.y, " x ", blockSize.z);

  wbTime_start(Compute, "Performing CUDA computation");

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC,
                                          numARows, numAColumns,
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");

  //@@ Copy the GPU memory back to the CPU here
  err = cudaMemcpy(hostC, deviceC, sizeC_bytes, cudaMemcpyDeviceToHost); checkErr(err);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");

  //@@ Free the GPU memory here
  err = cudaFree(deviceA); checkErr(err);
  err = cudaFree(deviceB); checkErr(err);
  err = cudaFree(deviceC); checkErr(err);

  wbTime_stop(GPU, "Freeing GPU Memory");


  // serial multiplication for testing
//  for (int i = 0; i < numARows; ++i) {
//    for (int j = 0; j < numBColumns; ++j) {
//      float c_val = 0;
//      for (int k = 0; k < numAColumns; ++k)
//        c_val += hostA[i*numAColumns + k] * hostB[k*numBColumns + j];
//      hostC[i*numCColumns + j] = c_val;
//    }
//  }

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

