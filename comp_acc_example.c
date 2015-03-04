void computeAcc(float *P, const float *M, const float *N, int Mh, int Mw, int Nw)
{
  #pragma acc parallel loop copyin(M[0:Mh*Mw]) copyin(N[0:Mw*Nw]) copyout(P[0:Mh*Nw])
  for (int i = 0; i < Mh; ++i) {
    #pragma acc loop
    for (int j = 0; j < Nw; ++j) {
      float sum = 0.;
      for (int k = 0; k < Mw; ++k)
        sum += M[i*Mw + k] * N[k*Nw + j];
      P[i*Nw + j] = sum;
    }
  }
}
