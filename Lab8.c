#include <wb.h> //@@ wb include opencl.h for you

//==============================================================================
//
// Error checker macro
//
//==============================================================================
#define check(err)                                             \
  do {                                                         \
    if (err != CL_SUCCESS) {                                   \
      wbLog(ERROR, "Error code ", err, " at line ", __LINE__); \
      return -1;                                               \
    }                                                          \
  } while (0);

//==============================================================================
//
// The size of a block (in threads)
//
//==============================================================================
#define BLOCK_SIZE 256

//==============================================================================
//
// OpenCL kernel
//
//==============================================================================
const char *vaddsrc = "\
__kernel void vecAdd(__global const float *a, \
                     __global const float *b, \
                     __global float *c,       \
                     __global int *len) {     \
  const int id = get_global_id(0);            \
  if (id < *len) c[id] = a[id] + b[id];        \
}";

//==============================================================================
//
// main() function
//
//==============================================================================
int main(int argc, char **argv)
{
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
//  float *deviceInput1;
//  float *deviceInput2;
//  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  
  cl_int err = CL_SUCCESS;

  //----------------------------------------------------------------------------
  // Create OpenCL context
  //----------------------------------------------------------------------------
  wbTime_start(GPU, "Creating OpenCL context.");
  cl_uint numPlatforms;
  err = clGetPlatformIDs(0,
                         NULL,
                         &numPlatforms); check(err);
  wbLog(TRACE, "numPlatforms = ", numPlatforms);
  
  cl_platform_id platforms[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, 
                         platforms, 
                         NULL); check(err);
                          
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platforms[0],
                                         0
                                       };
                                       
  cl_context clContext = clCreateContextFromType(properties,
                                                 CL_DEVICE_TYPE_GPU,
                                                 NULL,
                                                 NULL,
                                                 &err); check(err);
  wbTime_stop(GPU, "Creating OpenCL context.");

  //----------------------------------------------------------------------------
  // Create OpenCL command queue
  //----------------------------------------------------------------------------
  wbTime_start(GPU, "Creating OpenCL command queue.");
  size_t paramSize;
  err = clGetContextInfo(clContext,
                         CL_CONTEXT_DEVICES,
                         0,
                         NULL,
                         &paramSize); check(err);
  cl_device_id *clDevices = (cl_device_id*)malloc(paramSize);
  err = clGetContextInfo(clContext,
                         CL_CONTEXT_DEVICES,
                         paramSize,
                         clDevices,
                         NULL); check(err);
  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,
                                                         clDevices[0],
                                                         0,
                                                         &err); check(err);
  wbTime_stop(GPU, "Creating OpenCL command queue.");

  //----------------------------------------------------------------------------
  // Allocate memory on device
  //----------------------------------------------------------------------------
  wbTime_start(GPU, "Allocating GPU memory.");
  cl_mem deviceInput1 = clCreateBuffer(clContext,
                                       CL_MEM_READ_ONLY,
                                       inputLength*sizeof(float),
                                       NULL,
                                       &err); check(err);
  cl_mem deviceInput2 = clCreateBuffer(clContext,
                                       CL_MEM_READ_ONLY,
                                       inputLength*sizeof(float),
                                       NULL,
                                       &err); check(err);
  cl_mem deviceOutput = clCreateBuffer(clContext,
                                       CL_MEM_WRITE_ONLY,
                                       inputLength*sizeof(float),
                                       NULL,
                                       &err); check(err);
  cl_mem deviceInLength = clCreateBuffer(clContext,
                                         CL_MEM_READ_ONLY,
                                         sizeof(int),
                                         NULL,
                                         &err); check(err);
  wbTime_stop(GPU, "Allocating GPU memory.");

  //----------------------------------------------------------------------------
  // Copy input arrays to the device
  //----------------------------------------------------------------------------
  wbTime_start(GPU, "Copying input memory to the GPU.");
  err = clEnqueueWriteBuffer(clCommandQueue,
                             deviceInput1,
                             CL_TRUE,
                             0,
                             inputLength*sizeof(float),
                             (const void*)hostInput1,
                             0,
                             NULL,
                             NULL); check(err);
  err = clEnqueueWriteBuffer(clCommandQueue,
                             deviceInput2,
                             CL_TRUE,
                             0,
                             inputLength*sizeof(float),
                             (const void*)hostInput2,
                             0,
                             NULL,
                             NULL); check(err);
  err = clEnqueueWriteBuffer(clCommandQueue,
                             deviceInLength,
                             CL_TRUE,
                             0,
                             sizeof(int),
                             (const void*)&inputLength,
                             0,
                             NULL,
                             NULL); check(err);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //----------------------------------------------------------------------------
  // Perform the computations
  //----------------------------------------------------------------------------
  wbTime_start(Compute, "Performing CUDA computation");
  cl_program clProgram = clCreateProgramWithSource(clContext,
                                                   1,
                                                   &vaddsrc,
                                                   NULL,
                                                   &err); check(err);
  char clcompileflags[4096];
  sprintf(clcompileflags, "-cl-mad-enable");
  cl_int errBuild = clBuildProgram(clProgram,
                                   0,
                                   NULL,
                                   clcompileflags,
                                   NULL,
                                   NULL);
  size_t buildInfoSize;
  err = clGetProgramBuildInfo(clProgram,
                              clDevices[0],
                              CL_PROGRAM_BUILD_LOG,
                              0,
                              NULL,
                              &buildInfoSize); check(err);
  char *infoStr = (char*)malloc(2 * buildInfoSize*sizeof(char));
  err = clGetProgramBuildInfo(clProgram,
                              clDevices[0],
                              CL_PROGRAM_BUILD_LOG,
                              buildInfoSize,
                              infoStr,
                              NULL); check(err);
  printf("%s\n", infoStr);
  
  check(errBuild);

  cl_kernel clKernel = clCreateKernel(clProgram,
                                      "vecAdd",
                                      &err); check(err);
  // set the arguments of a kernel function
  err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void*)&deviceInput1);
  err = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*)&deviceInput2);
  err = clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void*)&deviceOutput);
  err = clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void*)&deviceInLength);
  
  cl_event event = NULL;
  const size_t globalWorkSize[] = { ((inputLength - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE };
  const size_t localWorkSize[] = { BLOCK_SIZE };
  err = clEnqueueNDRangeKernel(clCommandQueue,
                               clKernel,
                               1,              // dimension
                               NULL,
                               (const size_t*)globalWorkSize,
                               (const size_t*)localWorkSize,
                               0,
                               NULL,
                               &event); check(err);
  err = clWaitForEvents(1, &event); check(err);
  wbTime_stop(Compute, "Performing CUDA computation");

  //----------------------------------------------------------------------------
  // Copy the output array back to the host
  //----------------------------------------------------------------------------
  wbTime_start(Copy, "Copying output memory to the CPU");
  err = clEnqueueReadBuffer(clCommandQueue,
                            deviceOutput,
                            CL_TRUE,
                            0,
                            inputLength*sizeof(float),
                            hostOutput,
                            0,
                            NULL,
                            NULL); check(err);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //----------------------------------------------------------------------------
  // Free memory on device
  //----------------------------------------------------------------------------
  wbTime_start(GPU, "Freeing GPU Memory");
  err = clReleaseMemObject(deviceInput1); check(err);
  err = clReleaseMemObject(deviceInput2); check(err);
  err = clReleaseMemObject(deviceOutput); check(err);
  wbTime_stop(GPU, "Freeing GPU Memory");

  //----------------------------------------------------------------------------
  // Check the solution
  //----------------------------------------------------------------------------
  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

