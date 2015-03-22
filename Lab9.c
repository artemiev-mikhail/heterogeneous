#include <wb.h>

#define N_GANGS   1
#define N_WORKERS 256

// switch between OpenACC and sequential versions
#define WITH_ACC

int main(int argc, char **argv)
{
  wbArg_t args;
  int inputLength, N;
  float *A;
  float *B;
  float *C;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  A = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  B = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  C = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  
  N = inputLength; // a specific trick

  wbTime_start(Generic, "Computations");
#if defined(WITH_ACC)  
  #pragma acc parallel copyin(A[0:N], B[0:N]) copyout(C[0:N]) num_gangs(N_GANGS) num_workers(N_WORKERS)
  {
    #pragma acc loop worker
#endif // WITH_ACC    
    for (int i = 0; i < N; ++i)
      C[i] = A[i] + B[i];
#if defined(WITH_ACC)
  }
#endif // WITH_ACC
  wbTime_stop(Generic, "Computations");

  wbSolution(args, C, N);

  free(C);
  free(B);
  free(A);

  return 0;
}

