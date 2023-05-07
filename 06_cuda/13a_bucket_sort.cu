#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_on_gpu(int* key, int* bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void sort_bucket(int* key, int* bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int j = 0;
    for(int sum = 0; sum <= i; j++){
      sum += bucket[j];
    }
    key[i] = j - 1;
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int* d_key, *d_bucket;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));
  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_bucket, 0, range * sizeof(int));
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  bucket_on_gpu<<<grid_size, block_size>>>(d_key, d_bucket, n);
  cudaDeviceSynchronize();
  sort_bucket<<<grid_size, block_size>>>(d_key, d_bucket, n);
  cudaDeviceSynchronize();
  cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_key);
  cudaFree(d_bucket);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
