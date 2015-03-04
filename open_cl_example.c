cl_int clerr = CL_SUCCESS;

cl_context clctx = clCreateContextFromType(0, CL_DEVICE_TYPE_ALL, NULL, NULL, &clerr);

size_t parmsz;
clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);

cl_device_id *cldevs = (cl_device_id*)malloc(parmsz);
clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);

cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);



const char *vaddsrc = 
"__kernel void vadd(__global const float *a, \
                    __global const float *b, \
                    __global float *c) {     \
  const int id = get_global_id(0);           \
  c[id] = a[id] + b[id];                     \
}"

cl_program clpgm;
clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr);

char clcompileflags[4096];
sprintf(clcompileflags, "-cl-mad-enable");
clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);
cl_kernel clkern = clCreateKernel(clpgm, "vadd", &clerr);

int size_bytes = 1024 * sizeof(float);
cl_mem d_a;
d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY, size_bytes, NULL, &clerr);
// OR
d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_bytes, h_a, &clerr);
clReleaseObject(d_a);

// a is in host, d_a is in device memory
clEnqueueWriteBuffer(clcmdq, d_a, CL_FALSE, 0, mem_size, (const void*)a, 0, 0, NULL);
clEnqueueReadBuffer(clcmdq, d_result, CL_FALSE, 0, mem_size, (void*)host_result, 0, 0, NULL);


// kernel launch
clkern = clCreateKernel(clpgm, "vadd", NULL);
// set the arguments of a kernel function
clerr = clSetKernelArg(clkern, 0, sizeof(cl_mem), (void*)&d_A);
clerr = clSetKernelArg(clkern, 1, sizeof(cl_mem), (void*)&d_B);
clerr = clSetKernelArg(clkern, 2, sizeof(cl_mem), (void*)&d_C);
clerr = clSetKernelArg(clkern, 3, sizeof(int), &N);
// launch
cl_event event=NULL;
clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 2, NULL, Gsz, Bsz, 0, NULL, &event);
clerr = clWaitForEvents(1, &event);
clEnqueueReadBuffer(clcmdq, d_C, CL_TRUE, 0, N*sizeof(float), h_C, 0, NULL, NULL);
clReleaseMemObject(d_A);
clReleaseMemObject(d_B);
clReleaseMemObject(d_C);

