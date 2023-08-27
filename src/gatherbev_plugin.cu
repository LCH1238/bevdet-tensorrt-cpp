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

#include "gatherbev_plugin.h"
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// input[0] == adj_feat   b x 8 x 80 x 128 x 128
// input[1] == curr_feat  b x 80 x 128 x 128
// input[2] == flag       b x 1
// out[0]                 b x (8+1)*80 x 128 x 128
template<typename T>
__global__ void copy_feat_kernel(int nthreads, // b * (adj_num + 1) * map_size
                                int adj_num,
                                int channel,
                                int map_size,
                                const T* adj_feats,
                                const T* curr_feat,
                                const int* flag,
                                T* out_feats){
    CUDA_1D_KERNEL_LOOP(idx, nthreads){
        int b = idx / ((adj_num + 1) * map_size);
        int n = (idx / map_size) % (adj_num + 1);
        int m = idx % map_size;

        int start = b * (adj_num + 1) * channel * map_size + n * channel * map_size + m;
        int end = start + channel * map_size;
        for(int i = start, c = 0; i < end; i += map_size, c++){
            if(flag[b] == 0 || n == 0){
                out_feats[i] = curr_feat[b * channel * map_size + c * map_size + m];
            }
            else{
                out_feats[i] = adj_feats[i - channel * map_size];
            }
        }
    }
}


namespace nvinfer1
{
// class GatherBEVPlugin
GatherBEVPlugin::GatherBEVPlugin(const std::string &name):
    name_(name){
}

GatherBEVPlugin::GatherBEVPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name){
    memcpy(&m_, buffer, sizeof(m_));
}

GatherBEVPlugin::~GatherBEVPlugin(){
}

IPluginV2DynamicExt *GatherBEVPlugin::clone() const noexcept {
    auto p = new GatherBEVPlugin(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t GatherBEVPlugin::getNbOutputs() const noexcept {
    return 1;
}
 
DataType GatherBEVPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, 
                                                                int32_t nbInputs) const noexcept {
    return inputTypes[0];  // 与adj_feat一致
}

DimsExprs GatherBEVPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
                                        int32_t nbInputs, IExprBuilder &exprBuilder) noexcept {
  // input[0] == adj_feat   b x 8 x 80 x 128 x 128
  // input[1] == curr_feat  b x 80 x 128 x 128
  // input[2] == flag       b x 1
  // out[0]                 b x (8+1)*80 x 128 x 128
    DimsExprs ret;
    ret.nbDims = inputs[0].nbDims - 1;

    IDimensionExpr const *n_feat = exprBuilder.operation(DimensionOperation::kSUM, 
                                                        *inputs[0].d[1],
                                                        *exprBuilder.constant(1));
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *n_feat, *inputs[0].d[2]);
    ret.d[2] = inputs[0].d[3];
    ret.d[3] = inputs[0].d[4];

    return ret; 
}

bool GatherBEVPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                                    int32_t nbInputs, int32_t nbOutputs) noexcept {
    // adj_feat    curr_feat   out
    if(pos == 0 || pos == 1 || pos == 3){
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if(pos == 2){
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

void GatherBEVPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
                                    const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    return;
}

size_t GatherBEVPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, 
                                const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t GatherBEVPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {

  // input[0] == adj_feat   b x 8 x 80 x 128 x 128
  // input[1] == curr_feat  b x 80 x 128 x 128
  // input[2] == flag       b x 1
  // out[0]                 b x (8+1)*80 x 128 x 128

    // int nthreads, // b * (adj_num + 1) * map_size
    // int adj_num,
    // int channel,
    // int map_size,

    int flag = 0;
    CHECK_CUDA(cudaMemcpy(&flag, inputs[2], sizeof(int), cudaMemcpyDeviceToHost));

    int b = inputDesc[0].dims.d[0];
    int adj_num = inputDesc[0].dims.d[1];
    int map_size = inputDesc[0].dims.d[3] * inputDesc[0].dims.d[4];
    int channel = inputDesc[0].dims.d[2];

    int feat_step = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2] * inputDesc[1].dims.d[3];
    // printf("feat_step : %d\n", feat_step);
    // printf("adj_num : %d\n", adj_num);
    // printf("flag : %d\n", flag);
    int nthreads = b * (adj_num + 1) * map_size;

    dim3 grid(GET_BLOCKS(nthreads));
    dim3 block(NUM_THREADS);

    switch (int(outputDesc[0].type))
    {
    case int(DataType::kFLOAT):
        // printf("copy feat!\n");
        copy_feat_kernel<<<grid, block, 0, stream>>>(nthreads,
                                                    adj_num,
                                                    channel,
                                                    map_size,
                                                    reinterpret_cast<const float*>(inputs[0]),
                                                    reinterpret_cast<const float*>(inputs[1]),
                                                    reinterpret_cast<const int*>(inputs[2]),
                                                    reinterpret_cast<float*>(outputs[0]));
        // printf("copy feat! done!\n");
        


        // if(!flag){
        //     for(int i = 0; i < adj_num + 1; i++){
        //         CHECK_CUDA(cudaMemcpy((float*)outputs[0] + i * feat_step, inputs[1], 
        //                                 feat_step * sizeof(float), cudaMemcpyDeviceToDevice));
        //     }
        //     // printf("flag %d\n", flag);
        // }
        // else{
        //     CHECK_CUDA(cudaMemcpy(outputs[0], inputs[1], feat_step * sizeof(float), cudaMemcpyDeviceToDevice));
        //     CHECK_CUDA(cudaMemcpy((float*)outputs[0] + feat_step, inputs[0], adj_num * feat_step * sizeof(float),
        //                             cudaMemcpyDeviceToDevice));
        //     // printf("flag %d\n", flag);
        // }
        break;
    case int(DataType::kHALF):
        // if(!flag){
        //     for(int i = 0; i < adj_num + 1; i++){
        //         CHECK_CUDA(cudaMemcpy((__half*)outputs[0] + i * feat_step, inputs[1], 
        //                                 feat_step * sizeof(__half), cudaMemcpyDeviceToDevice));
        //     }
        // }
        // else{
        //     CHECK_CUDA(cudaMemcpy(outputs[0], inputs[1], feat_step * sizeof(__half), cudaMemcpyDeviceToDevice));
        //     CHECK_CUDA(cudaMemcpy((__half*)outputs[0] + feat_step, inputs[0], adj_num * feat_step * sizeof(__half),
        //                             cudaMemcpyDeviceToDevice));
        // }
        copy_feat_kernel<<<grid, block, 0, stream>>>(nthreads,
                                                    adj_num,
                                                    channel,
                                                    map_size,
                                                    reinterpret_cast<const __half*>(inputs[0]),
                                                    reinterpret_cast<const __half*>(inputs[1]),
                                                    reinterpret_cast<const int*>(inputs[2]),
                                                    reinterpret_cast<__half*>(outputs[0]));
        break;
    default: // should NOT be here
        printf("\tUnsupport datatype!\n");
    }
    return 0;
}

void GatherBEVPlugin::destroy() noexcept {
    delete this;
    return;
}

int32_t GatherBEVPlugin::initialize() noexcept {
    return 0;
}

void GatherBEVPlugin::terminate() noexcept {
    return;
}

size_t GatherBEVPlugin::getSerializationSize() const noexcept {
    return sizeof(m_);
}

void GatherBEVPlugin::serialize(void *buffer) const noexcept {
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void GatherBEVPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *GatherBEVPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *GatherBEVPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}

const char *GatherBEVPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

void GatherBEVPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, 
                                                        IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void GatherBEVPlugin::detachFromContext() noexcept {
    return;
}

// class GatherBEVPluginCreator
PluginFieldCollection    GatherBEVPluginCreator::fc_ {};
std::vector<PluginField> GatherBEVPluginCreator::attr_;

GatherBEVPluginCreator::GatherBEVPluginCreator() {
    attr_.clear();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

GatherBEVPluginCreator::~GatherBEVPluginCreator() {
}


IPluginV2DynamicExt *GatherBEVPluginCreator::createPlugin(const char *name, 
                                    const PluginFieldCollection *fc) noexcept {
    GatherBEVPlugin *pObj = new GatherBEVPlugin(name);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2DynamicExt *GatherBEVPluginCreator::deserializePlugin(const char *name, 
                                        const void *serialData, size_t serialLength) noexcept {
    GatherBEVPlugin *pObj = new GatherBEVPlugin(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void GatherBEVPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
    return;
}

const char *GatherBEVPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char *GatherBEVPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}

const char *GatherBEVPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const PluginFieldCollection *GatherBEVPluginCreator::getFieldNames() noexcept {
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(GatherBEVPluginCreator);

} // namespace nvinfer1
