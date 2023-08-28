#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <dlfcn.h> // need function to load dynamic link library
#include <fstream>
#include <chrono>
#include <unistd.h>

#include "common.h"
#include "preprocess_plugin.h"
#include "alignbev_plugin.h"
#include "bevpool_plugin.h"
#include "gatherbev_plugin.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace nvinfer1;

static Logger     gLogger(ILogger::Severity::kERROR);


inline void loadLibrary(const std::string &path)
{
#ifdef _MSC_VER
    void *handle = LoadLibrary(path.c_str());
#else
    int32_t flags {RTLD_LAZY};
    void *  handle = dlopen(path.c_str(), flags);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        std::cout << "Could not load plugin library: " << path << std::endl;
#else
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
}


int main(int argc, char * argv[]){

    if(argc != 3){
        printf("./export model.onnx model.engine\n");
        return 0;
    }
    std::string onnxFile(argv[1]);
    std::string trtFile(argv[2]);

    CHECK_CUDA(cudaSetDevice(0));
    ICudaEngine *engine = nullptr;

    std::vector<Dims32> shapes{
        {4, {6, 3, 900, 400}},
        {1, {3}},
        {1, {3}},
        {3, {1, 6, 27}},
        {1, {356760}},
        {1, {356760}},
        {1, {356760}},
        {1, {13360}},
        {1, {13360}},
        {5, {1, 8, 80, 128, 128}},
        {3, {1, 8, 6}},
        {2, {1, 1}}
    };
    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder = createInferBuilder(gLogger);
        INetworkDefinition *  network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig *      config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3UL << 32UL);
        // if (bFP16Mode)
        // {
            config->setFlag(BuilderFlag::kFP16);
        // }

        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);


        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportable_severity)))
        {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }

            return 1;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        for(size_t i = 0; i < shapes.size(); i++){
            ITensor *it = network->getInput(i);
            profile->setDimensions(it->getName(), OptProfileSelector::kMIN, shapes[i]);
            profile->setDimensions(it->getName(), OptProfileSelector::kOPT, shapes[i]);
            profile->setDimensions(it->getName(), OptProfileSelector::kMAX, shapes[i]);
        }
        config->addOptimizationProfile(profile);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Failed building serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return 1;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }
    IExecutionContext *context = engine->createExecutionContext();

    for(size_t i = 0; i < shapes.size(); i++){
        context->setBindingDimensions(i, shapes[i]);
    }

    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings();
    int nInput   = 0;
    for (int i = 0; i < nBinding; ++i)
    {
        nInput += int(engine->bindingIsInput(i));
    }
    //int nOutput = nBinding - nInput;
    for (int i = 0; i < nBinding; ++i)
    {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
        printf("id : %d, %d\n", i, vBindingSize[i]);
    }

    std::vector<void *> vBufferH {nBinding, nullptr};
    std::vector<void *> vBufferD {nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i)
    {
        vBufferH[i] = (void *)new char[vBindingSize[i]]; 
        memset(vBufferH[i], 0, vBindingSize[i]); // FIXME
        CHECK_CUDA(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    for (int i = 0; i < nInput; ++i)
    {
        CHECK_CUDA(cudaMemcpy(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice));
    }


    context->executeV2(vBufferD.data());

    // auto start = high_resolution_clock::now();
    // for(int i = 0; i < 100; i++){
    //     context->executeV2(vBufferD.data());
    // }
    // auto end = high_resolution_clock::now();
    // duration<double> t = end - start;
    // printf("infer : %.4lf\n", t.count() * 1000 / 100);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK_CUDA(cudaMemcpy(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < nBinding; ++i)
    {
        CHECK_CUDA(cudaFree(vBufferD[i]));
    }

    return 0;
}