
cudaStream_t stream_0, stream_1;

cudaStreamCreate(&stream_0);
cudaStreamCreate(&stream_1);

float *dA0, *dB0, *dC0; // for stream 0
float *dA1, *dB1, *dC1; // for stream 1

cudaMalloc((void**)&dA0, N*sizeof(float));
cudaMalloc((void**)&dB0, N*sizeof(float));
cudaMalloc((void**)&dC0, N*sizeof(float));

cudaMalloc((void**)&dA1, N*sizeof(float));
cudaMalloc((void**)&dB1, N*sizeof(float));
cudaMalloc((void**)&dC1, N*sizeof(float));


// 1 version
for (int i = 0; i < N; i += SegSize*2)
{
  cudaMemcpyAsync(dA0, hA+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_0);
  cudaMemcpyAsync(dB0, hB+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_0);
  vecAdd<<<SegSize/256, 256, 0, stream_0>>>(dA0, dB0, dC0);
  cudaMemcpyAsync(hC+i, dC0, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream_0);

  cudaMemcpyAsync(dA1, hA+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_1);
  cudaMemcpyAsync(dB1, hB+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_1);
  vecAdd<<<SegSize/256, 256, 0, stream_1>>>(dA1, dB1, dC1);
  cudaMemcpyAsync(hC+i+SegSize, dC1, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream_1);
}

// 2 version
for (int i = 0; i < N; i += SegSize*2)
{
  cudaMemcpyAsync(dA0, hA+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_0);
  cudaMemcpyAsync(dB0, hB+i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_0);
  cudaMemcpyAsync(dA1, hA+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_1);
  cudaMemcpyAsync(dB1, hB+i+SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream_1);
  
  vecAdd<<<SegSize/256, 256, 0, stream_0>>>(dA0, dB0, dC0);
  vecAdd<<<SegSize/256, 256, 0, stream_1>>>(dA1, dB1, dC1);
  
  cudaMemcpyAsync(hC+i, dC0, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream_0);
  cudaMemcpyAsync(hC+i+SegSize, dC1, SegSize*sizeof(float), cudaMemcpyDeviceToHost, stream_1);
}

