#include <amp.h>
using namespace concurrency;

void vecAdd(float *A, float *B, float *C, int n)
{
  array_view<float, 1> AV(n, A);
  array_view<float, 1> BV(n, B);
  array_view<float, 1> CV(n, C);
  
  CV.discard_data(); // the data will be overwritten
  
  parallel_for_each(CV.get_extent(), [=](index<1> i) restrict(amp)
  {
    CV[i] = AV[i] + BV[i];
  });
  CV.synchronize();
}
