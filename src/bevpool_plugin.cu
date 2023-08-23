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

#include "bevpool_plugin.h"
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// kernel for GPU
template<typename T>
__global__ void bev_pool_v2_kernel(int c, int n_intervals, int map_size,
                                  const T *__restrict__ depth,
                                  const T *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  T * __restrict__ out) {
    // 进入到一个kernel的都是一个bevgrid要计算的特征的某一维度
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;    // bev grid index
    int cur_c = idx % c;    // channel index
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];  
    int interval_length = interval_lengths[index];  
    T psum = 0;
    const T * cur_depth;
    const T * cur_feat;
    for(int i = 0; i < interval_length; i++){
        cur_depth = depth + ranks_depth[interval_start+i];            // 指向 深度概率值
        cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;   // 指向 cur_c通道图像特征值
        psum += *cur_feat * *cur_depth;
    }

    const int* cur_rank = ranks_bev + interval_start;  // 指向 某bevgrid
    // float* cur_out = out + *cur_rank * c + cur_c;   // b x h x w x c
    T* cur_out = out + cur_c * map_size + *cur_rank;      // b x c x h x w

    *cur_out = psum;
}

namespace nvinfer1
{
// class BEVPoolPlugin
BEVPoolPlugin::BEVPoolPlugin(const std::string &name, int bev_h, int bev_w):
    name_(name){
    m_.bev_h = bev_h;
    m_.bev_w = bev_w;
}

BEVPoolPlugin::BEVPoolPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

BEVPoolPlugin::~BEVPoolPlugin(){
}

IPluginV2DynamicExt *BEVPoolPlugin::clone() const noexcept {
    auto p = new BEVPoolPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t BEVPoolPlugin::getNbOutputs() const noexcept {
    return 1;
}
 
DataType BEVPoolPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return inputTypes[0];  // 与mean一致
}

DimsExprs BEVPoolPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
  // input[0] == depth->kFLOAT
  // input[1] == feat->kFLOAT
  // input[2] == ranks_depth->kINT32
  // input[3] == ranks_feat->kINT32
  // input[4] == ranks_bev->kINT32
  // input[5] == interval_starts->kINT32
  // input[6] == interval_lengths->kINT32

    DimsExprs ret;
    ret.nbDims = inputs[1].nbDims - 1;
    ret.d[0] = inputs[1].d[3];
    ret.d[1] = exprBuilder.constant(m_.bev_h);
    ret.d[2] = exprBuilder.constant(m_.bev_w);
    
    return ret;  // FIXME
}

bool BEVPoolPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // depth       feat        out
    if(pos == 0 || pos == 1 || pos == 7){
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }
    else{
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

void BEVPoolPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    return;
}

size_t BEVPoolPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t BEVPoolPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    // input[0] == depth->kFLOAT
    // input[1] == feat->kFLOAT
    // input[2] == ranks_depth->kINT32
    // input[3] == ranks_feat->kINT32
    // input[4] == ranks_bev->kINT32
    // input[5] == interval_starts->kINT32
    // input[6] == interval_lengths->kINT32

    int channel = inputDesc[1].dims.d[3];
    int n_intervals = inputDesc[5].dims.d[0];
    int map_size = m_.bev_h * m_.bev_w;

    // dim3 grid((int)ceil((float)(n_intervals * channel / NUM_THREADS)));
    dim3 grid(CEIL_DIVIDE(n_intervals * channel, NUM_THREADS));
    dim3 block(NUM_THREADS);

    switch (int(outputDesc[0].type))
    {
    case int(DataType::kFLOAT):
        bev_pool_v2_kernel<<<grid, block, 0, stream>>>(
                                                    channel, 
                                                    n_intervals,
                                                    map_size,
                                                    reinterpret_cast<const float *>(inputs[0]),
                                                    reinterpret_cast<const float *>(inputs[1]),
                                                    reinterpret_cast<const int *>(inputs[2]),
                                                    reinterpret_cast<const int *>(inputs[3]),
                                                    reinterpret_cast<const int *>(inputs[4]),
                                                    reinterpret_cast<const int *>(inputs[5]),
                                                    reinterpret_cast<const int *>(inputs[6]),
                                                    reinterpret_cast<float *>(outputs[0]));
        break;
    case int(DataType::kHALF):
        bev_pool_v2_kernel<<<grid, block, 0, stream>>>(
                                                    channel, 
                                                    n_intervals,
                                                    map_size,
                                                    reinterpret_cast<const __half *>(inputs[0]),
                                                    reinterpret_cast<const __half *>(inputs[1]),
                                                    reinterpret_cast<const int *>(inputs[2]),
                                                    reinterpret_cast<const int *>(inputs[3]),
                                                    reinterpret_cast<const int *>(inputs[4]),
                                                    reinterpret_cast<const int *>(inputs[5]),
                                                    reinterpret_cast<const int *>(inputs[6]),
                                                    reinterpret_cast<__half *>(outputs[0]));
        break;
    default: // should NOT be here
        printf("\tUnsupport datatype!\n");
    }
    return 0;
}

void BEVPoolPlugin::destroy() noexcept {
    delete this;
    return;
}

int32_t BEVPoolPlugin::initialize() noexcept {
    return 0;
}

void BEVPoolPlugin::terminate() noexcept {
    return;
}

size_t BEVPoolPlugin::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void BEVPoolPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void BEVPoolPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *BEVPoolPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *BEVPoolPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char *BEVPoolPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

void BEVPoolPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void BEVPoolPlugin::detachFromContext() noexcept {
    return;
}

// class BEVPoolPluginCreator
PluginFieldCollection    BEVPoolPluginCreator::fc_ {};
std::vector<PluginField> BEVPoolPluginCreator::attr_;

BEVPoolPluginCreator::BEVPoolPluginCreator() {
    attr_.clear();
    attr_.emplace_back(PluginField("bev_h", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("bev_w", nullptr, PluginFieldType::kINT32, 1));

    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

BEVPoolPluginCreator::~BEVPoolPluginCreator() {
}


IPluginV2DynamicExt *BEVPoolPluginCreator::createPlugin(const char *name, 
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
    BEVPoolPlugin *pObj = new BEVPoolPlugin(name, bev_h, bev_w);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *BEVPoolPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    BEVPoolPlugin *pObj = new BEVPoolPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void BEVPoolPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *BEVPoolPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *BEVPoolPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char *BEVPoolPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const PluginFieldCollection *BEVPoolPluginCreator::getFieldNames() noexcept {
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(BEVPoolPluginCreator);

} // namespace nvinfer1
