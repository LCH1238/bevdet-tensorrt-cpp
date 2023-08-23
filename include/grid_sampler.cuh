#pragma once

#include <cuda_runtime.h>


#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"

const int MAXTENSORDIMS = 10;

struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
};

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, NUM_THREADS);
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}


enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims, int *input_dims,
                 int *grid_dims,
                 GridSamplerPadding padding, bool align_corners, cudaStream_t stream);



void compute_sample_grid_cuda(float* grid_dev, const float* transform, int bev_w, int bev_h,
                                                                            cudaStream_t stream);


