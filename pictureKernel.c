__global__
void pictureKernel(float *Pin, float *Pout, int n, int m)
{
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < m && col < n)
    Pout[row*n + col] = 2. * Pin[row*n + col];
}
