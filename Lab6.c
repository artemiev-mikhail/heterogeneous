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

#define HISTOGRAM_LENGTH 256

//@@ insert code here

// Cast from float to unsigned char
__global__ void cast_to_char(float *input, unsigned char *output, int inputLength)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (index < inputLength)
    output[index] = (unsigned char)(255*input[index]);
}





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
  const int imageSizeBytes = imageWidth * imageHeight * imageChannels * sizeof(float);
  float *deviceInputImageData  = NULL;
  float *deviceOutputImageData = NULL;
  cudaError_t err;
  err = cudaMalloc((void**)&deviceInputImageData,  imageSizeBytes); checkErr(err);
  err = cudaMalloc((void**)&deviceOutputImageData, imageSizeBytes); checkErr(err);
  wbTime_stop(Generic, "Creating memory on device");
  
  wbTime_stop(Generic, "Copying input image to device global memory");
  err = cudaMemcpy(deviceInputImageData, hostInputImageData, imageSizeBytes, cudaMemcpyHostToDevice); checkErr(err);
  wbTime_stop(Generic, "Copying input image to device global memory");
  
  wbTime_stop(Generic, "Copying output image to host memory");
  err = cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSizeBytes, cudaMemcpyDeviceToHost); checkErr(err);
  wbTime_stop(Generic, "Copying output image to host memory");


  wbSolution(args, outputImage);



  free(hostInputImageData);
  free(hostOutputImageData);
  
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}


