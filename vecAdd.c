__global__
void vecAddKernel(float *A, float *B, float *C, int n)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < n) C[i] = A[i] + B[i];
}


void checkErr(cudaError_t err, char **file, int line)
{
  if (err != cudaSuccess)
  {
    printf("Error\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}


void vecAdd(float *hA, float *hB, float *hC, int n)
{
  int size = n * sizeof(float);
  float* dA, dB, dC;
  cudaError_t err;

  err = cudaMalloc((void**) &dA, size); checkErr(err, __FILE__, __LINE__);
  err = cudaMalloc((void**) &dB, size); checkErr(err, __FILE__, __LINE__);
  err = cudaMalloc((void**) &dC, size); checkErr(err, __FILE__, __LINE__);

  /* initialize dA, dB */

  err = cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice); checkErr(err, __FILE__, __LINE__);
  err = cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice); checkErr(err, __FILE__, __LINE__);

  /* kernel function */
  dim3 dimGrid((n-1)/256 + 1, 1, 1);
  dim3 dimBlock(256, 1, 1);
//  addVecKernel<<<ceil(n/256.), 256>>>(dA, dB, dC, n);
  addVecKernel<<<dimGrid, dimBlock>>>(dA, dB, dC, n);

  err = cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost); checkErr(err, __FILE__, __LINE__);

  err = cudaFree(dC); checkErr(err, __FILE__, __LINE__);
  err = cudaFree(dB); checkErr(err, __FILE__, __LINE__);
  err = cudaFree(dA); checkErr(err, __FILE__, __LINE__);
}
