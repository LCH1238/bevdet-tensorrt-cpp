/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "alignbev_plugin.h"
#include "common.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

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


inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, NUM_THREADS);
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_unnormalize(scalar_t coord, int size,
                                                                    bool align_corners){
    if (align_corners){
        // unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1.f) / 2) * (size - 1);
    }
    else{
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1.f) * size - 1) / 2;
    }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__ scalar_t clip_coordinates(scalar_t in, int clip_limit){
    return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__ scalar_t reflect_coordinates(scalar_t in, int twice_low,
                                                               int twice_high){
    if (twice_low == twice_high){
        return static_cast<scalar_t>(0);
    }
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = ::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    scalar_t extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 == 0){
        return extra + min;
    }
    else{
        return span - extra + min;
    }
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t safe_downgrade_to_int_range(scalar_t x){
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
    scalar_t coord, int size, GridSamplerPadding padding_mode, bool align_corners){
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    if (padding_mode == GridSamplerPadding::Border){
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    else if (padding_mode == GridSamplerPadding::Reflection){
        // reflect coordinates by image borders
        if (align_corners){
            coord = reflect_coordinates(coord, 0, 2 * (size - 1));
        }
        else{
            coord = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }

    coord = safe_downgrade_to_int_range(coord);
    return coord;
}

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H, int W){
    return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void grid_sampler_2d_kernel(const int nthreads, const scalar_t *input,
                                       const float *grid, scalar_t *output,
                                       TensorDesc input_desc, TensorDesc grid_desc,
                                       TensorDesc output_desc,
                                       const GridSamplerPadding padding_mode, bool align_corners){
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

    CUDA_1D_KERNEL_LOOP(index, nthreads){
        const int w = index % out_W;
        const int h = (index / out_W) % out_H;
        const int n = index / (out_H * out_W);
        const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y coordinates from grid
        float ix = grid[grid_offset]; // TODO
        float iy = grid[grid_offset + grid_sCoor];

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
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC){
            *out_ptr_NCHW = static_cast<scalar_t>(0);
            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)){
                *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)){
                *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)){
                *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)){
                *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
            }
        }
    }
}

void create_desc(const int *dims, int nb_dims, TensorDesc &desc){
    memcpy(&desc.shape[0], dims, sizeof(int) * nb_dims);
    desc.stride[nb_dims - 1] = 1;
    for (int i = nb_dims - 2; i >= 0; --i){
        desc.stride[i] = desc.stride[i + 1] * desc.shape[i + 1];
    }
}

template <typename T>
void grid_sample(T *output, const T *input, const float *grid, int *output_dims, int *input_dims,
                 int *grid_dims, GridSamplerPadding padding, bool align_corners, cudaStream_t stream){
    TensorDesc input_desc;
    create_desc(input_dims, 4, input_desc);

    TensorDesc output_desc;
    create_desc(output_dims, 4, output_desc);

    TensorDesc grid_desc;
    create_desc(grid_dims, 4, grid_desc);

    int count = 1;
    for (int i = 0; i < 4; ++i){
        if (i == 1){
            continue;
        }
        count *= output_desc.shape[i];
    }
    grid_sampler_2d_kernel<T><<<GET_BLOCKS(count), NUM_THREADS, 0, stream>>>(
        count, input, grid, output, input_desc, grid_desc, output_desc, padding,
        align_corners);
}

namespace nvinfer1 {
// class AlignBEVPlugin
AlignBEVPlugin::AlignBEVPlugin(const std::string &name, int bev_h, int bev_w):
    name_(name){
    m_.bev_h = bev_h;
    m_.bev_w = bev_w;
}

AlignBEVPlugin::AlignBEVPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

AlignBEVPlugin::~AlignBEVPlugin(){
}

IPluginV2DynamicExt *AlignBEVPlugin::clone() const noexcept {
    auto p = new AlignBEVPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t AlignBEVPlugin::getNbOutputs() const noexcept {
    return 1;
}
 
DataType AlignBEVPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return inputTypes[0]; 
}

DimsExprs AlignBEVPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
    return inputs[0]; 
}

bool AlignBEVPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // adj_feat    out
    if(pos == 0 || pos == 2){
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }    // transform
    else if(pos == 1){
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

size_t AlignBEVPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 80 * 128 * 128 * sizeof(float);
}

int32_t AlignBEVPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    
    // TODO
    // inputs[0] == adj_feat  80 x 128 x 128
    // inputs[1] == transform 6

    int bev_channel = inputDesc[0].dims.d[0];

    dim3 grid(DIVUP(m_.bev_h * m_.bev_w, NUM_THREADS));
    dim3 block(NUM_THREADS);

    compute_sample_grid_kernel<<<grid, block, 0, stream>>>(
        grid_dev,
        reinterpret_cast<const float*>(inputs[1]),
        m_.bev_w,
        m_.bev_h
    );
    // printf("%d %d %d\n", grid.x, grid.y, grid.z);
    // printf("%d %d %d\n", block.x, block.y, block.z);

    // int size = 6;
    // float* tensor = new float[size];
    // CHECK_CUDA(cudaMemcpy(tensor, inputs[1], size * sizeof(float), cudaMemcpyDeviceToHost));
    // int cnt = 0;
    // for(int i = 0; i < size; i++){
    //     if(tensor[i] != 0 && cnt <= 5){
    //         printf("%8.6f ", tensor[i]);
    //         cnt++;
    //     }
    // }
    // printf("\n");
    // delete[] tensor;


    int output_dim[4] = {1, bev_channel, m_.bev_w, m_.bev_h};
    int input_dim[4] = {1, bev_channel, m_.bev_w, m_.bev_h};
    int grid_dim[4] = {1, m_.bev_w, m_.bev_h, 2};

    switch (int(outputDesc[0].type))
    {
    case int(DataType::kFLOAT):
        grid_sample(
            reinterpret_cast<float *>(outputs[0]),
            reinterpret_cast<const float *>(inputs[0]),
            grid_dev,
            output_dim,
            input_dim,
            grid_dim,
            GridSamplerPadding::Zeros,
            true,
            stream
        );
        break;
    case int(DataType::kHALF):
        grid_sample(
            reinterpret_cast<__half *>(outputs[0]),
            reinterpret_cast<const __half *>(inputs[0]),
            grid_dev,
            output_dim,
            input_dim,
            grid_dim,
            GridSamplerPadding::Zeros,
            true,
            stream
        );
        break;
    default: // should NOT be here
        printf("\tUnsupport datatype!\n");
    }


    return 0;
}

void AlignBEVPlugin::destroy() noexcept {
    if(grid_dev != nullptr){
        CHECK_CUDA(cudaFree(grid_dev));
        grid_dev = nullptr;
    }
    delete this;
    return;
}

void AlignBEVPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    CHECK_CUDA(cudaMalloc((void**)&grid_dev, m_.bev_h * m_.bev_w * 2 * sizeof(float)));
    return;
}

int32_t AlignBEVPlugin::initialize() noexcept {
    return 0;
}

void AlignBEVPlugin::terminate() noexcept {
    return;
}

size_t AlignBEVPlugin::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void AlignBEVPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void AlignBEVPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *AlignBEVPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *AlignBEVPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char *AlignBEVPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

void AlignBEVPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void AlignBEVPlugin::detachFromContext() noexcept {
    return;
}

// class AlignBEVPluginCreator
PluginFieldCollection    AlignBEVPluginCreator::fc_ {};
std::vector<PluginField> AlignBEVPluginCreator::attr_;

AlignBEVPluginCreator::AlignBEVPluginCreator() {
    attr_.clear();
    attr_.emplace_back(PluginField("bev_h", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("bev_w", nullptr, PluginFieldType::kINT32, 1));

    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

AlignBEVPluginCreator::~AlignBEVPluginCreator() {
}


IPluginV2DynamicExt *AlignBEVPluginCreator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    const PluginField *fields = fc->fields;
    int bev_h = -1;
    int bev_w = -1;

    for (int i = 0; i < fc->nbFields; ++i){
        if(std::string(fc->fields[i].name) == std::string("bev_h")){
            bev_h = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
        else if(std::string(fc->fields[i].name) == std::string("bev_w")){
            bev_w = *reinterpret_cast<const int *>(fc->fields[i].data);
        }
    }
    AlignBEVPlugin *pObj = new AlignBEVPlugin(name, bev_h, bev_w);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *AlignBEVPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    AlignBEVPlugin *pObj = new AlignBEVPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void AlignBEVPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *AlignBEVPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *AlignBEVPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char *AlignBEVPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const PluginFieldCollection *AlignBEVPluginCreator::getFieldNames() noexcept {
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(AlignBEVPluginCreator);

} // namespace nvinfer1
