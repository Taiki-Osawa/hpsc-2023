#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    //for(int j=0; j<N; j++) {
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 mvec = _mm256_load_ps(m);
    __m256 fxvec = _mm256_load_ps(fx);
    __m256 fyvec = _mm256_load_ps(fy);
    __m256 zerovec = _mm256_setzero_ps();

    //if(i != j) {
    //float rx = x[i] - x[j];
    __m256 xivec = _mm256_set1_ps(x[i]);//ok
    __m256 mask = _mm256_cmp_ps(xivec, xvec, _CMP_NEQ_OQ);//ok
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);//ok

    //float ry = y[i] - y[j];
    __m256 yivec = _mm256_set1_ps(y[i]);//ok
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);//ok

    //float r = std::sqrt(rx * rx + ry * ry);
    __m256 rxmul2vec = _mm256_mul_ps(rxvec,rxvec);
    __m256 rymul2vec = _mm256_mul_ps(ryvec,ryvec);
    __m256 rvec = _mm256_add_ps(rxmul2vec, rymul2vec);

    rvec = _mm256_rsqrt_ps(rvec);

    //fx[i] -= rx * m[j] / (r * r * r);       
    __m256 rmul3vec = _mm256_mul_ps(rvec, rvec);
    rmul3vec = _mm256_mul_ps(rvec, rmul3vec);

    __m256 fxivec = _mm256_mul_ps(rxvec, mvec);
    fxivec = _mm256_mul_ps(fxivec, rmul3vec);
    fxivec = _mm256_blendv_ps(zerovec, fxivec, mask); 

    __m256 fxbvec = _mm256_permute2f128_ps(fxivec,fxivec,1);
    fxbvec = _mm256_add_ps(fxbvec,fxivec);
    fxbvec = _mm256_hadd_ps(fxbvec,fxbvec);
    fxbvec = _mm256_hadd_ps(fxbvec,fxbvec);

    fxbvec = _mm256_sub_ps(zerovec, fxbvec);
    fxvec = _mm256_blendv_ps(fxbvec, fxvec, mask);


    //fy[i] -= ry * m[j] / (r * r * r);
    __m256 fyivec = _mm256_mul_ps(ryvec, mvec);
    fyivec = _mm256_mul_ps(fyivec, rmul3vec);
    fyivec = _mm256_blendv_ps(zerovec, fyivec, mask); 

    __m256 fybvec = _mm256_permute2f128_ps(fyivec,fyivec,1);
    fybvec = _mm256_add_ps(fybvec,fyivec);
    fybvec = _mm256_hadd_ps(fybvec,fybvec);
    fybvec = _mm256_hadd_ps(fybvec,fybvec);

    fybvec = _mm256_sub_ps(zerovec, fybvec);
    fyvec = _mm256_blendv_ps(fybvec, fyvec, mask);

    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);
    //}
    //}
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
