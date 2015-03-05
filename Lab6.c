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

#define correct_color(val,minval) (clamp(255.*(val-minval)/(1.-minval), 0., 255.))

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE_HISTO 128

#define BLOCK_SIZE_REGULAR 512


//============================================================================
// Cast an array of float numbers to an array of unsigned char ones
// CUDA Grid: 1D
// Requirements:
//   Both input and output arrays must have size equal to inputLength
//============================================================================
__global__ void cast_to_char(const float *input, unsigned char *output, int inputLength)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < inputLength)
    output[index] = (unsigned char)(255*input[index]);
}

//============================================================================
// Convert from RGB to gray scale
// CUDA Grid: 2D (heigth for rows, width for columns)
//============================================================================
__global__ void convert_to_gray(const unsigned char *rgbImage, 
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
    const int idx = row*imageWidth + col;
    const unsigned char r = rgbImage[idx*imageChannels + 0]; // red
    const unsigned char g = rgbImage[idx*imageChannels + 1]; // green
    const unsigned char b = rgbImage[idx*imageChannels + 2]; // blue
    grayImage[idx] = 100; //0.21*r + 0.71*g + 0.07*b;
  }
  
  grayImage[row*imageWidth + col] = 100;
}

//============================================================================
// Compute the histogram
// CUDA Grid: 1D
//============================================================================
__global__ void compute_histogram(const unsigned char *image, int imageLength, int *histogram)
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
__global__ void compute_cdf(const int *histogram, 
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
    cdf[start + tx] = shInput[tx] / (1.*imageHeight*imageWidth);
  if (BLOCK_SIZE + start + tx < numHistogramElements)
    cdf[BLOCK_SIZE + start + tx] = shInput[BLOCK_SIZE + tx] / (1.*imageHeight*imageWidth);
}

//============================================================================
// Reduction with a function 'min'
// CUDA Grid: 1D, each thread takes 2 elements, therefore the grid should be
//            organized correspondingly
//============================================================================
__global__ void reduction_min(const float *input, float *min_value, int lengthInput)
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
    *min_value = partial_min[0]; // 0-th element keeps the min value
  __syncthreads();
}

//============================================================================
// Correct the color of the original uchar image and cast it back to the float
// value
// CUDA Grid: 1D
// Requirements:
//   Both input and output arrays must have size equal to inputLength
//============================================================================
__global__ void perform_correct(const unsigned char *inputImage,
                                float *outputImage,
                                int inputLength,
                                float *cdf,
                                float cdfmin)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < inputLength)
    outputImage[index] = correct_color(cdf[inputImage[index]], cdfmin) / 255.;
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
  
  wbTime_start(Generic, "Creating memory on device");
  const int imageNElements = imageWidth * imageHeight * imageChannels;
  const int imageSizeBytes = imageNElements * sizeof(float);
  float *deviceInputImageData  = NULL;
  float *deviceOutputImageData = NULL;
  cudaError_t err;
  err = cudaMalloc((void**)&deviceInputImageData,  imageSizeBytes); checkErr(err);
  err = cudaMalloc((void**)&deviceOutputImageData, imageSizeBytes); checkErr(err);
  wbTime_stop(Generic, "Creating memory on device");
  
  wbTime_start(Generic, "Copying input image to device global memory");
  err = cudaMemcpy(deviceInputImageData, hostInputImageData, imageSizeBytes, cudaMemcpyHostToDevice); checkErr(err);
  wbTime_stop(Generic, "Copying input image to device global memory");
  
  
  //-----------------------------------------------------------------
  // GPU computations
  //

  wbTime_start(Generic, "Cast the image to unsigned char array");
  dim3 gridSize1D((imageNElements-1)/BLOCK_SIZE_REGULAR+1, 1, 1);
  dim3 blockSize1D(BLOCK_SIZE_REGULAR, 1, 1);
  wbLog(TRACE, "gridSize1D = ", gridSize1D.x, " x ", gridSize1D.y, " x ", gridSize1D.z);
  wbLog(TRACE, "blockSize1D = ", blockSize1D.x, " x ", blockSize1D.y, " x ", blockSize1D.z);
  
  unsigned char *deviceInputImageDataUChar = NULL;
  err = cudaMalloc((void**)&deviceInputImageDataUChar, imageSizeBytes); checkErr(err);
  
  cast_to_char<<<gridSize1D, blockSize1D>>>(deviceInputImageData, deviceInputImageDataUChar, imageNElements);
  wbTime_stop(Generic, "Cast the image to unsigned char array");
  
  
  
  unsigned char *hostInputImageDataUChar = (unsigned char*)malloc(imageHeight*imageWidth*imageChannels*sizeof(unsigned char));
  err = cudaMemcpy(hostInputImageDataUChar, deviceInputImageDataUChar, imageHeight*imageWidth*imageChannels*sizeof(unsigned char), cudaMemcpyDeviceToHost); checkErr(err);
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "InputImageDataUChar[", i, "] = ", (int)hostInputImageDataUChar[i]);
  free(hostInputImageDataUChar);
  
  
  
  wbTime_start(Generic, "Convert from RGB to Gray image");
  const int imageGrayNElements = imageHeight * imageWidth; // there are no RGB channels
  unsigned char *deviceGrayImageData = NULL;
  err = cudaMalloc((void**)&deviceGrayImageData, imageGrayNElements * sizeof(unsigned char)); checkErr(err);
  
  dim3 gridSize2D((imageWidth-1)/BLOCK_SIZE_REGULAR+1, (imageHeight-1)/BLOCK_SIZE_REGULAR+1, 1);
  dim3 blockSize2D(BLOCK_SIZE_REGULAR, BLOCK_SIZE_REGULAR, 1);
  wbLog(TRACE, "gridSize2D = ", gridSize2D.x, " x ", gridSize2D.y, " x ", gridSize2D.z);
  wbLog(TRACE, "blockSize2D = ", blockSize2D.x, " x ", blockSize2D.y, " x ", blockSize2D.z);
  
  convert_to_gray<<<gridSize2D, blockSize2D>>>(deviceInputImageDataUChar, 
                                               deviceGrayImageData, 
                                               imageHeight,
                                               imageWidth);
  wbTime_stop(Generic, "Convert from RGB to Gray image");
  
  
  
  unsigned char *hostGrayImageData = (unsigned char*)malloc(imageGrayNElements*sizeof(unsigned char));
  err = cudaMemcpy(hostGrayImageData, deviceGrayImageData, imageGrayNElements*sizeof(unsigned char), cudaMemcpyDeviceToHost); checkErr(err);
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "grayImageData[", i, "] = ", (char)hostGrayImageData[i]);
  free(hostGrayImageData);
  
  

  wbTime_start(Generic, "Compute the histogram");
  int *deviceHistogram = NULL;
  err = cudaMalloc((void**)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(int)); checkErr(err);
  err = cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(int));
  
  dim3 gridSize1D_Gray((imageGrayNElements-1)/BLOCK_SIZE_REGULAR+1, 1, 1);
  wbLog(TRACE, "gridSize1D_Gray = ", gridSize1D_Gray.x, " x ", gridSize1D_Gray.y, " x ", gridSize1D_Gray.z);
  compute_histogram<<<gridSize1D_Gray, blockSize1D>>>(deviceGrayImageData, imageGrayNElements, deviceHistogram);
  wbTime_stop(Generic, "Compute the histogram");
  
  wbTime_start(Generic, "Compute CDF");
  float *device_cdf = NULL;
  err = cudaMalloc((void**)&device_cdf, HISTOGRAM_LENGTH * sizeof(float)); checkErr(err);
  
  dim3 gridSize1D_Histo((HISTOGRAM_LENGTH-1)/(BLOCK_SIZE_HISTO<<1)+1, 1, 1);
  dim3 blockSize1D_Histo(BLOCK_SIZE_HISTO, 1, 1);
  wbLog(TRACE, "gridSize1D_Histo = ", gridSize1D_Histo.x, " x ", gridSize1D_Histo.y, " x ", gridSize1D_Histo.z);
  wbLog(TRACE, "blockSize1D_Histo = ", blockSize1D_Histo.x, " x ", blockSize1D_Histo.y, " x ", blockSize1D_Histo.z);
  
  compute_cdf<<<gridSize1D_Histo, blockSize1D_Histo>>>(deviceHistogram, 
                                                       device_cdf, 
                                                       HISTOGRAM_LENGTH,
                                                       imageHeight,
                                                       imageWidth);
  wbTime_stop(Generic, "Compute CDF");
  
/*  
  wbTime_start(Generic, "Find min(CDF)");
  float *device_cdfmin = NULL;
  err = cudaMalloc((void**)&device_cdfmin, sizeof(float)); checkErr(err);
  reduction_min<<<gridSize1D_Histo, blockSize1D_Histo>>>(cdf, device_cdfmin, HISTOGRAM_LENGTH);
  float host_cdfmin = 1e+32;
  err = cudaMemcpy(&host_cdfmin, device_cdfmin, sizeof(float), cudaMemcpyDeviceToHost); checkErr(err);
  wbLog(TRACE, "cdfmin = ", host_cdfmin);
  wbTime_stop(Generic, "Find min(CDF)");
*/
/*
  perform_correct<<<gridSize1D_Histo, blockSize1D_Histo>>>(deviceInputImageDataUChar,
                                                           deviceOutputImageData,
                                                           imageNElements,
                                                           device_cdf,
                                                           device_cdf[0]);
*/
/*  
  int *hostHistogram = (int*)malloc(HISTOGRAM_LENGTH*sizeof(int));
  err = cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH*sizeof(int), cudaMemcpyDeviceToHost); checkErr(err);
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "histogram[", i, "] = ", hostHistogram[i]);
  free(hostHistogram);
  
  float *hostCDF = (float*)malloc(HISTOGRAM_LENGTH*sizeof(float));
  err = cudaMemcpy(hostCDF, device_cdf, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyDeviceToHost); checkErr(err);
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "cdf[", i, "] = ", hostCDF[i]);
  free(hostCDF);
  
  unsigned char *hostGrayImageData = (unsigned char*)malloc(imageGrayNElements*sizeof(unsigned char));
  err = cudaMemcpy(hostGrayImageData, deviceGrayImageData, imageGrayNElements*sizeof(unsigned char), cudaMemcpyDeviceToHost); checkErr(err);
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "grayImageData[", i, "] = ", (int)hostGrayImageData[i]);
  free(hostGrayImageData);
*/

                                               
  err = cudaFree(device_cdf);                checkErr(err);
  err = cudaFree(deviceHistogram);           checkErr(err);
  err = cudaFree(deviceGrayImageData);       checkErr(err);
  err = cudaFree(deviceInputImageDataUChar); checkErr(err);

  //-----------------------------------------------------------------  
  
  wbTime_start(Generic, "Copying output image to host memory");
  err = cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSizeBytes, cudaMemcpyDeviceToHost); checkErr(err);
  wbTime_stop(Generic, "Copying output image to host memory");

  wbSolution(args, outputImage);

  err = cudaFree(deviceInputImageData);  checkErr(err);
  err = cudaFree(deviceOutputImageData); checkErr(err);
  
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}


