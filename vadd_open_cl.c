__kernel void vadd(__global const float *a,
                   __global const float *b,
                   __global float *c)
{
  const int id = get_global_id(0);
  c[id] = a[id] + b[id];
}


// memory allocation and copy

d_A = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), h_A, &clerr);
d_B = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*sizeof(float), h_B, &clerr);
d_C = clCreateBuffer(clctx, CL_MEM_READ_ONLY, N*sizeof(float), NULL, &clerr);
