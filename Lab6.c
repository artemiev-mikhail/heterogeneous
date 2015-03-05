// Histogram Equalization

#include <wb.h>

#define checkErr(err)                                      \
  do {                                                     \
    if (err != cudaSuccess) {                              \
      printf("\nError found\n%s in %s at line %d",         \
             cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
  } while (0);
  
#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)

#define clamp(x,start,end) (min(max(x,start),end))

#define correct_color(val,minval) (clamp(255*(val-minval)/(1-minval), 0, 255))

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE_HISTO 128

#define BLOCK_SIZE_REGULAR 512


//============================================================================
// Cast an array of float numbers to an array of unsigned char ones
// CUDA Grid: 1D
// Requirements:
//   Both input and output arrays must have size equal to inputLength
//============================================================================
__global__ void cast_to_char(float *input, unsigned char *output, int inputLength)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < inputLength)
    output[index] = (unsigned char)(255*input[index]);
}

//============================================================================
// Cast an array of unsigned char numbers to an array of float ones
// CUDA Grid: 1D
// Requirements:
//   Both input and output arrays must have size equal to inputLength
//============================================================================
__global__ void cast_to_float(unsigned char *input, float *output, int inputLength)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < inputLength)
    output[index] = (float)(input[index]/255.);
}

//============================================================================
// Convert from RGB to gray scale
// CUDA Grid: 2D (heigth for rows, width for columns)
//============================================================================
__global__ void convert_to_gray(unsigned char *rgbImage, 
                                unsigned char *grayImage, 
                                int imageHeight,
                                int imageWidth)
{
  const int row = blockIdx.y*blockDim.y + threadIdx.y;
  const int col = blockIdx.x*blockDim.x + threadIdx.x;
  
  const int imageChannels = 3;
  
  // here every element of the arrays residing in the global memory are
  // accessed only once, therefore there is no need to copy them to the
  // shared memory
  if (row < imageHeight && col < imageWidth)
  {
    const unsigned char r = rgbImage[(row*imageWidth + col)*imageChannels + 0]; // red
    const unsigned char g = rgbImage[(row*imageWidth + col)*imageChannels + 1]; // green
    const unsigned char b = rgbImage[(row*imageWidth + col)*imageChannels + 2]; // blue
    grayImage[row*imageWidth + col] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

//============================================================================
// Compute the histogram
// CUDA Grid: 1D
//============================================================================
__global__ void compute_histogram(unsigned char *image, int imageLength, int *histogram)
{
  const int n_chars = 256; // number of possible characters
  __shared__ int private_histogram[n_chars];
  
  const int tx = threadIdx.x;
  
  // initialize the privatized histogram (only first n_chars threads do that)
  if (tx < n_chars)
    private_histogram[tx] = 0;
  __syncthreads();
  
  int i = tx + blockDim.x*blockIdx.x;
  const int stride = blockDim.x*gridDim.x;
  
  while (i < imageLength)
  {
    atomicAdd(&private_histogram[image[i]], 1);
    i += stride;
  }
  __syncthreads();
  
  if (tx < n_chars)
    atomicAdd(&histogram[tx], private_histogram[tx]);
  __syncthreads();
}

//============================================================================
// Cumulative distribution function of the histogram. Here the efficient
// version of the scan function is used.
// CUDA Grid: 1D, each thread takes 2 elements, so the grid should be
//            organized correspondingly
//============================================================================
__global__ void compute_cdf(int *histogram, 
                            float *cdf, 
                            int numHistogramElements,
                            int imageHeight,
                            int imageWidth)
{
  const int BLOCK_SIZE = BLOCK_SIZE_HISTO;

  __shared__ int shInput[2*BLOCK_SIZE];

  const int tx = threadIdx.x;
  const int start = 2 * BLOCK_SIZE * blockIdx.x;

  if (start + tx < numHistogramElements) 
    shInput[tx] = histogram[start + tx];
  else
    shInput[tx] = 0.;

  if (BLOCK_SIZE + start + tx < numHistogramElements)
    shInput[BLOCK_SIZE + tx] = histogram[BLOCK_SIZE + start + tx];
  else
    shInput[BLOCK_SIZE + tx] = 0.;

  for (int stride = 1; stride <= BLOCK_SIZE_HISTO; stride *= 2)
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
  if (start + tx < numHistogramElements)
    cdf[start + tx] = shInput[tx] / (imageHeight*imageWidth);
  if (BLOCK_SIZE + start + tx < numHistogramElements)
    cdf[BLOCK_SIZE + start + tx] = shInput[BLOCK_SIZE + tx] / (imageHeight*imageWidth);
}

//============================================================================
// Reduction with a function 'min'
// CUDA Grid: 1D, each thread takes 2 elements, therefore the grid should be
//            organized correspondingly
//============================================================================
__global__ void reduction_min(float *input, float &min_value, int lengthInput)
{
  const int BLOCK_SIZE = BLOCK_SIZE_HISTO;

  __shared__ float partial_min[BLOCK_SIZE<<1];
  
  const int tx = threadIdx.x;
  const int start = 2 * blockIdx.x * BLOCK_SIZE;
  
  const float big_number = 1e+16;

  // first half
  if (start + tx < lengthInput)
    partial_min[tx] = input[start + tx];
  else
    partial_min[tx] = big_number;
  
  // second half
  if (start + BLOCK_SIZE + tx < lengthInput)
    partial_min[BLOCK_SIZE + tx] = input[start + BLOCK_SIZE + tx];
  else
    partial_min[BLOCK_SIZE + tx] = big_number;

  for (int stride = BLOCK_SIZE; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (tx < stride)
      partial_min[tx] = min(partial_min[tx], partial_min[tx + stride]);
  }

  if (tx == 0)
    min_value = partial_min[0]; // 0-th element keeps the min value
}

//============================================================================
// Correct the color of the original uchar image
// CUDA Grid: 1D
// Requirements:
//   Both input and output arrays must have size equal to inputLength
//============================================================================
__global__ void perform_correct(unsigned char *inputImage,
                                unsigned char *outputImage,
                                int inputLength,
                                float *cdf,
                                float cdfmin)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < inputLength)
    outputImage[index] = correct_color(cdf[inputImage[index]], cdfmin);
}




//============================================================================
//
// Main function
//
//============================================================================
int main(int argc, char ** argv)
{
  wbArg_t args = wbArg_read(argc, argv); /* parse the input arguments */
  
  const char *inputImageFile = wbArg_getInputFile(args, 0);
  
  wbTime_start(Generic, "Importing data");
  wbImage_t inputImage    = wbImport(inputImageFile);
  const int imageWidth    = wbImage_getWidth(inputImage);
  const int imageHeight   = wbImage_getHeight(inputImage);
  const int imageChannels = wbImage_getChannels(inputImage);
  const int imagePitch    = wbImage_getPitch(inputImage);
  wbImage_t outputImage   = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data");
  
  wbLog(TRACE, "imageWidth = ",     imageWidth, 
               " imageHeight = ",   imageHeight, 
               " imageChannels = ", imageChannels, 
               " imagePitch = ",    imagePitch);

  wbTime_start(Generic, "Creating memory on host");
  float *hostInputImageData  = wbImage_getData(inputImage);
  float *hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Creating memory on host");
  
  wbTime_stop(Generic, "Creating memory on device");
  const int imageNElements = imageWidth * imageHeight * imageChannels;
  const int imageSizeBytes = imageNElements * sizeof(float);
  float *deviceInputImageData  = NULL;
  float *deviceOutputImageData = NULL;
  cudaError_t err;
  err = cudaMalloc((void**)&deviceInputImageData,  imageSizeBytes); checkErr(err);
  err = cudaMalloc((void**)&deviceOutputImageData, imageSizeBytes); checkErr(err);
  wbTime_stop(Generic, "Creating memory on device");
  
  wbTime_stop(Generic, "Copying input image to device global memory");
  err = cudaMemcpy(deviceInputImageData, hostInputImageData, imageSizeBytes, cudaMemcpyHostToDevice); checkErr(err);
  wbTime_stop(Generic, "Copying input image to device global memory");
  
  
  //-----------------------------------------------------------------
  // GPU computations
  //

/*  
  dim3 gridSize1D((imageNElements-1)/BLOCK_SIZE_REGULAR+1, 1, 1);
  dim3 blockSize1D(BLOCK_SIZE_REGULAR, 1, 1);
  
  unsigned char *deviceInputImageDataUChar = NULL;
  err = cudaMalloc((void**)&deviceInputImageDataUChar, imageSizeBytes); checkErr(err);
  
  cast_to_char<<<gridSize1D, blockSize1D>>>(deviceInputImageData, deviceInputImageDataUChar, imageNElements);
  
  const int imageGrayNElements = imageHeight * imageWidth; // there are no RGB channels
  unsigned char *deviceGrayImageData = NULL;
  err = cudaMalloc((void**)&deviceGrayImageData, imageGrayNElements * sizeof(unsigned char)); checkErr(err);
  
  dim3 gridSize2D((imageHeight-1)/BLOCK_SIZE_REGULAR+1, (imageWidth-1)/BLOCK_SIZE_REGULAR+1, 1);
  dim3 blockSize2D(BLOCK_SIZE_REGULAR, BLOCK_SIZE_REGULAR, 1);
  
  convert_to_gray<<<gridSize2D, blockSize2D>>>(deviceInputImageDataUChar, 
                                               deviceGrayImageData, 
                                               imageHeight,
                                               imageWidth);

  int *histogram = NULL;
  err = cudaMalloc((void**)&histogram, HISTOGRAM_LENGTH * sizeof(int));
  
  dim3 gridSize1D_Gray((imageGrayNElements-1)/BLOCK_SIZE_REGULAR+1, 1, 1);
  compute_histogram<<<gridSize1D_Gray, blockSize1D>>>(deviceGrayImageData, imageGrayNElements, histogram);
  
  float *cdf = NULL;
  err = cudaMalloc((void**)&cdf, HISTOGRAM_LENGTH * sizeof(float));
  
  dim3 gridSize1D_Histo((HISTOGRAM_LENGTH-1)/BLOCK_SIZE_HISTO+1, 1, 1);
  dim3 blockSize1D_Histo(BLOCK_SIZE_HISTO, 1, 1);
  compute_cdf<<gridSize1D_Histo, blockSize1D_Histo>>>(histogram, 
                                                      cdf, 
                                                      HISTOGRAM_LENGTH,
                                                      imageHeight,
                                                      imageWidth);
                                                      
  float cdfmin = 1e+32;
  reduction_min<<<gridSize1D_Histo, blockSize1D_Histo>>>(histogram, cdfmin, HISTOGRAM_LENGTH);
  
  perform_correct<<<gridSize1D, blockSize1D>>>(deviceInputImageDataUChar,
                                               deviceOutputImageDataUChar,
                                               imageNElements,
                                               cdf,
                                               cdfmin);
                                               
  cast_to_float<<<gridSize1D, blockSize1D>>>(deviceOutputImageDataUChar,
                                             deviceOutputImageData,
                                             imageNElements);
  
*/  
  //-----------------------------------------------------------------
  
  //-----------------------------------------------------------------
  // Sequential code
  //
  
  unsigned char *ucharImage = (unsigned char*)malloc(imageHeight*imageWidth*imageChannels*sizeof(unsigned char));
  for (int i = 0; i < imageHeight*imageWidth*imageChannels; ++i)
    ucharImage[i] = (unsigned char)(255*hostInputImageData[i]);
  
  unsigned char *grayImage = (unsigned char*)malloc(imageHeight*imageWidth*sizeof(unsigned char));
  for (int i = 0; i < imageHeight; ++i)
  {
    for (int j = 0; j < imageWidth; ++i)
    {
      const int el = i*imageWidth + j;
      unsigned char r = ucharImage[el*imageChannels + 0];
      unsigned char g = ucharImage[el*imageChannels + 1];
      unsigned char b = ucharImage[el*imageChannels + 2];
      grayImage[el] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
    }
  }
  
  int *histogram = (int*)malloc(HISTOGRAM_LENGTH*sizeof(int));
  memset(histogram, 0, HISTOGRAM_LENGTH*sizeof(int));
  
  for (int i = 0; i < imageHeight*imageWidth; ++i)
    ++histogram[grayImage[i]];
    
  float *cdf = (float*)malloc(HISTOGRAM_LENGTH*sizeof(float));
  cdf[0] = histogram[0] / (imageHeight*imageWidth);
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i)
    cdf[i] = cdf[i-1] + histogram[i] / (imageHeight*imageWidth);
    
  float cdfmin = cdf[0];
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i)
    cdfmin = min(cdfmin, cdf[i]);
    
  for (int i = 0; i < imageHeight*imageWidth*imageChannels; ++i)
    hostOutputImageData[i] = (float)(correct_color(ucharImage[i], cdfmin)/255.);
    
  free(cdf);
  free(histogram);
  free(grayImage);
  free(ucharImage);
  //-----------------------------------------------------------------
  
  
  wbTime_stop(Generic, "Copying output image to host memory");
  err = cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSizeBytes, cudaMemcpyDeviceToHost); checkErr(err);
  wbTime_stop(Generic, "Copying output image to host memory");

  wbSolution(args, outputImage);

  err = cudaFree(deviceInputImageData);  checkErr(err);
  err = cudaFree(deviceOutputImageData); checkErr(err);
  
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}


