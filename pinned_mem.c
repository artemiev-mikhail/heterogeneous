int main()
{
  float *hA, *hB, *hC;

  // allocation of the vectors in the pinned memory allows for up to
  // 2x speed up in cudaMemcpy() function
  cudaHostAlloc((void**)&hA, N*sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hB, N*sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hC, N*sizeof(float), cudaHostAllocDefault);

  
}
