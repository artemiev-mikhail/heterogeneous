#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

    
#define checkErr(err)                                      \
  do {                                                     \
    if (err != cudaSuccess) {                              \
      printf("\nError found\n%s in %s at line %d",         \
             cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                  \
    }                                                      \
  } while (0);


#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2

#define O_TILE_WIDTH 12
#define BLOCK_SIZE (O_TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float mask[MASK_WIDTH];

//typedef struct {
//  int width, height, pitch, channels;
//  float *data;
//} * wbImage_t;

//@@ INSERT CODE HERE
__global__ void convolution2D(float *input, 
                              int imageWidth, 
                              int imageHeight,
                              int imageChannels,
                              float *output)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x; // width, i.e. columns
  const int ty = threadIdx.y; // height, i.e. rows
//  const int tz = threadIdx.z; // responsible for channels
  
  const int row_o = by*O_TILE_WIDTH + ty;
  const int col_o = bx*O_TILE_WIDTH + tx;
  
  const int row_i = row_o - MASK_RADIUS;
  const int col_i = col_o - MASK_RADIUS;
  
  __shared__ float shInput[BLOCK_SIZE][BLOCK_SIZE];
  
  for (int channel = 0; channel < imageChannels; ++channel) {
    if (row_i >= 0 && col_i >= 0 &&
        row_i < imageHeight && col_i < imageWidth)
      shInput[ty][tx] = input[(row_i*imageWidth + col_i)*imageChannels + channel];
    else
      shInput[ty][tx] = 0.;
    
    float out_val = 0.;
    if (tx < O_TILE_WIDTH && ty < O_TILE_WIDTH) {
      for (int i = 0; i < MASK_WIDTH; ++i)
        for (int j = 0; j < MASK_WIDTH; ++j)
          out_val += mask[i*MASK_WIDTH + j] * shInput[i+ty][j+tx];
    }
  
    if (row_o < imageHeight && col_o < imageWidth)
      output[(row_o*imageWidth + col_o)*imageChannels + channel] = out_val;

  } // channel
}



int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    int imagePitch;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
//    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);
    
    wbLog(TRACE, "maskRows = ", maskRows, " maskColumns = ", maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth    = wbImage_getWidth(inputImage);
    imageHeight   = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    imagePitch    = wbImage_getPitch(inputImage);
    
    wbLog(TRACE, "imageWidth = ",     imageWidth, 
                 " imageHeight = ",   imageHeight, 
                 " imageChannels = ", imageChannels, 
                 " imagePitch = ",    imagePitch);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData  = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    
    const int imageSizeBytes = imageWidth * imageHeight * 
                               imageChannels * sizeof(float);
    const int maskSizeBytes  = maskRows * maskColumns * sizeof(float);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageSizeBytes);
    cudaMalloc((void **) &deviceOutputImageData, imageSizeBytes);
//    cudaMalloc((void **) &deviceMaskData, maskSizeBytes);
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageSizeBytes,
               cudaMemcpyHostToDevice);
//    cudaMemcpy(deviceMaskData,
//               hostMaskData,
//               maskSizeBytes,
//               cudaMemcpyHostToDevice);
    // copy mask to the constant device memory
    cudaMemcpyToSymbol(mask, hostMaskData, maskSizeBytes);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    
    //@@ INSERT CODE HERE
    dim3 gridSize((imageWidth-1)/O_TILE_WIDTH + 1, (imageHeight-1)/O_TILE_WIDTH + 1, 1);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    convolution2D<<<gridSize, blockSize>>>(deviceInputImageData, 
                                           imageWidth, 
                                           imageHeight,
                                           imageChannels,
                                           deviceOutputImageData);
    
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageSizeBytes,
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");


    // sequential version
//    for (int i = 0; i < imageHeight; ++i) {
//      for (int j = 0; j < imageWidth; ++j) {
//        for (int k = 0; k < imageChannels; ++k) {
//          float outval = 0.;
//          for (int ii = -MASK_RADIUS; ii <= MASK_RADIUS; ++ii) {
//            for (int jj = -MASK_RADIUS; jj <= MASK_RADIUS; ++jj) {
//              const int r = i + ii;
//              const int c = j + jj;
//              if (r >= 0 && r < imageHeight &&
//                  c >= 0 && c < imageWidth)
//                outval += hostMaskData[(ii+MASK_RADIUS)*MASK_WIDTH + (jj+MASK_RADIUS)] *
//                          hostInputImageData[(r*imageWidth + c)*imageChannels + k];
//            } // jj
//          } // ii
//
//          hostOutputImageData[(i*imageWidth + j)*imageChannels + k] = outval;
//
//        } // k
//      } // j
//    } // i

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
//    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

