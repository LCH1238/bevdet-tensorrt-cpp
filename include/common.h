#ifndef __COMMON_H__
#define __COMMON_H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

typedef unsigned char uchar;

#define NUM_THREADS 512 

#define CHECK_CUDA(ans) { GPUAssert((ans), __FILE__, __LINE__); }

#define DIVUP(m, n) (((m) + (n)-1) / (n))

inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
};

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int MAXTENSORDIMS = 6;
struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
};

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, NUM_THREADS);
  int max_block_num = 4096;
  return optimal_block_num < max_block_num ? optimal_block_num : max_block_num;
}


#endif