// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/pytorch/pytorch/blob/ec683299ebabf297a3504c76248d37be830e4342/aten/src/ATen/native/cuda/GridSampler.cuh
// and
// https://github.com/pytorch/pytorch/blob/ec683299ebabf297a3504c76248d37be830e4342/aten/src/ATen/native/cuda/GridSampler.cu


#include "grid_sampler.cuh"


// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_unnormalize(scalar_t coord, int size,
                                                                    bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__ scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__ scalar_t reflect_coordinates(scalar_t in, int twice_low,
                                                               int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord, int size, GridSamplerPadding padding_mode, bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}


template <typename scalar_t>
__global__ void grid_sampler_2d_kernel(const int nthreads, const scalar_t *input,
                                       const scalar_t *grid, scalar_t *output,
                                       TensorDesc input_desc, TensorDesc grid_desc,
                                       TensorDesc output_desc,
                                       const GridSamplerPadding padding_mode, bool align_corners) {
  int C = input_desc.shape[1];
  int inp_H = input_desc.shape[2];
  int inp_W = input_desc.shape[3];
  int out_H = grid_desc.shape[1];
  int out_W = grid_desc.shape[2];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sH = input_desc.stride[2];
  int inp_sW = input_desc.stride[3];
  int grid_sN = grid_desc.stride[0];
  int grid_sH = grid_desc.stride[1];
  int grid_sW = grid_desc.stride[2];
  int grid_sCoor = grid_desc.stride[3];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sH = output_desc.stride[2];
  int out_sW = output_desc.stride[3];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    scalar_t ix = grid[grid_offset];
    scalar_t iy = grid[grid_offset + grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);


    // get NE, NW, SE, SW pixel values from (x, y)
    int ix_nw = static_cast<int>(::floor(ix));
    int iy_nw = static_cast<int>(::floor(iy));
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    scalar_t nw = (ix_se - ix) * (iy_se - iy);
    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

    // calculate bilinear weighted pixel value and set output pixel
    auto inp_ptr_NC = input + n * inp_sN;
    auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
    for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
      *out_ptr_NCHW = static_cast<scalar_t>(0);
      if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
        *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
      }
      if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
        *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
      }
      if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
        *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
      }
      if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
        *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
      }
    }
    
  }
}


void create_desc(const int *dims, int nb_dims, TensorDesc &desc) {
  memcpy(&desc.shape[0], dims, sizeof(int) * nb_dims);
  desc.stride[nb_dims - 1] = 1;
  for (int i = nb_dims - 2; i >= 0; --i) {
    desc.stride[i] = desc.stride[i + 1] * desc.shape[i + 1];
  }
}

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims, int *input_dims,
                 int *grid_dims,
                 GridSamplerPadding padding, bool align_corners, cudaStream_t stream) {
  TensorDesc input_desc;
  create_desc(input_dims, 4, input_desc);

  TensorDesc output_desc;
  create_desc(output_dims, 4, output_desc);

  TensorDesc grid_desc;
  create_desc(grid_dims, 4, grid_desc);

  int count = 1;
  for (int i = 0; i < 4; ++i) {
    if (i == 1) {
      continue;
    }
    count *= output_desc.shape[i];
  }

  grid_sampler_2d_kernel<T><<<GET_BLOCKS(count), NUM_THREADS, 0, stream>>>(
      count, input, grid, output, input_desc, grid_desc, output_desc, padding,
      align_corners);

}

template void grid_sample<float>(float *output, const float *input, const float *grid,
                                 int *output_dims, int *input_dims, int *grid_dims,
                                  GridSamplerPadding padding,
                                 bool align_corners, cudaStream_t stream);


__global__ void compute_sample_grid_kernel(float* __restrict__ grid, 
                                           const float* __restrict__ transform, 
                                           int bev_w, int bev_h){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < bev_w * bev_h){
        int i = idx / bev_h;
        int j = idx % bev_h;
        float x = transform[0 * 3 + 0] * j + transform[0 * 3 + 1] * i + transform[0 * 3 + 2]; // * 1.0
        float y = transform[1 * 3 + 0] * j + transform[1 * 3 + 1] * i + transform[1 * 3 + 2]; // * 1.0

        grid[i * bev_h * 2 + j * 2 + 0] = x / (bev_w - 1.0f) * 2.0f - 1.0f;
        grid[i * bev_h * 2 + j * 2 + 1] = y / (bev_h - 1.0f) * 2.0f - 1.0f;
    }
}

void compute_sample_grid_cuda(float* grid_dev, const float* transform, int bev_w, int bev_h, 
                                                                        cudaStream_t stream){
    /*
    grid_dev : bev_w * bev_h * 2
    */

    dim3 grid(DIVUP(bev_w * bev_h, NUM_THREADS));
    dim3 block(NUM_THREADS);
    compute_sample_grid_kernel<<<grid, block, 0, stream>>>(grid_dev, transform, bev_w, bev_h);
}