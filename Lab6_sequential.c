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
  
  //-----------------------------------------------------------------
  // Sequential code
  //
  
  unsigned char *ucharImage = (unsigned char*)malloc(imageHeight*imageWidth*imageChannels*sizeof(unsigned char));
  for (int i = 0; i < imageHeight*imageWidth*imageChannels; ++i)
    ucharImage[i] = (unsigned char)(255*hostInputImageData[i]);
  
  unsigned char *grayImage = (unsigned char*)malloc(imageHeight*imageWidth*sizeof(unsigned char));
  for (int i = 0; i < imageHeight; ++i)
  {
    for (int j = 0; j < imageWidth; ++j)
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
  cdf[0] = histogram[0] / (1.*imageHeight*imageWidth);
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i)
    cdf[i] = cdf[i-1] + histogram[i] / (1.*imageHeight*imageWidth);
  
  int imin = 0;  
  float cdfmin = cdf[0];
  for (int i = 1; i < HISTOGRAM_LENGTH; ++i)
  {
    if (cdf[i] < cdfmin)
    {
      cdfmin = cdf[i];
      imin   = i;
    }
  }
  
  wbLog(TRACE, "cdfmin = ", cdfmin, " for ", imin, " element");
  
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "histogram[", i, "] = ", histogram[i]);
    
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
    wbLog(TRACE, "cdf[", i, "] = ", cdf[i]);
  
    
  for (int i = 0; i < imageHeight*imageWidth*imageChannels; ++i)
    hostOutputImageData[i] = correct_color(cdf[ucharImage[i]], cdfmin) / 255.;
    
  free(cdf);
  free(histogram);
  free(grayImage);
  free(ucharImage);
  //-----------------------------------------------------------------
  
  wbSolution(args, outputImage);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}


