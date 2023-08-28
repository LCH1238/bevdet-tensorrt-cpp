# BEVDet implemented by TensorRT, C++

<div align="center">

English | [简体中文](doc/README_zh-CN.md)

</div>

This project is a TensorRT implementation for BEVDet inference, written in C++. It can be tested on the nuScenes dataset and also provides a single test sample. BEVDet is a multi-camera 3D object detection model in bird's-eye view. For more details about BEVDet, please refer to the following link [BEVDet](https://github.com/HuangJunJie2017/BEVDet). **The script to export the ONNX model is in this [repository](https://github.com/LCH1238/BEVDet)**.

![图](doc/BEVDet-TensorRT.png)


This project implements the following:
- **TensorRT-Plugins** : AlignBEV_plugin, Preprocess_plugin, BEVPool_plugin, GatherBEV_plugin
- Long-term model
- BEV-Depth model
- On the NVIDIA A4000, the BEVDet-r50-lt-depth model shows a __6.24x faster__ inference speed for TRT FP16 compared to PyTorch FP32
- On the __Jetson AGX Orin__, the FP16 model inference time is around __29 ms__, achieving real-time performance
- A Dataloader for the nuScenes dataset and can be used to test on the dataset
- Fine-tuned the model to solve the problem that the model is sensitive to input resize sampling, which leads to the decline of mAP and NDS
- An Attempt at Int8 Quantization

The features of this project are as follows:
- A CUDA Kernel that combines Resize, Crop, and Normalization for preprocessing
- The __Preprocess CUDA kernel__ includes two interpolation methods: Nearest Neighbor Interpolation and Bicubic Interpolation
- Alignment of adjacent frame BEV features using C++ and CUDA kernel implementation
- __Multi-threading and multi-stream NvJPEG__
- Sacle-NMS
- Remove the preprocess module in BEV encoder

  
The following parts need to be implemented:
- Quantization to int8.
- Integrate the bevpool and adjacent frame BEV feature alignment components into the engine as plugins
- Exception handling


## Results && Speed
## Inference Speed
All time units are in milliseconds (ms), and Nearest interpolation is used by default.

|| TRT-Engine |Postprocess|mean Total |   
|---|---|---|---|
|NVIDIA A4000 PyTorch FP32| — |—|86.24|  
|NVIDIA A4000 FP16|11.38|0.53|11.91|  
|Jetson AGX Orin FP16|26.60|0.99|27.60|


<!-- *Note: The inference time of the module refers to the time of a frame, while the total time is calculated as the average time of 200 frames.*

## Results
|Model   |Description       |mAP   |NDS    |Infer time|
|---     |---               |---   |---    |---       |
|Pytorch |                  |0.3972|0.5074|96.052|
|Pytorch |LSS accelerate<sup>1</sup>   |0.3787|0.4941|86.236|
|Trt FP32|Python Preprocess<sup>2</sup>|0.3776|0.4936|25.534|
|Trt FP32|Bicubic sampler<sup>3</sup>  |0.3723|0.3895|33.960|
|Trt FP32|Nearest sampler<sup>4</sup>  |0.3703|0.4884|25.534|
|Trt FP16|Nearest sampler   |0.3702|0.4883|13.817|
|Pytorch |Nearest sampler <sup>5</sup>   |0.3989|0.5169|——|
|Pytorch |LSS accelerate <sup>5</sup>  |0.3800| 0.4997|——|
|Trt FP16| <sup>5</sup>|0.3785| 0.5013  | 12.738

*Note: The PyTorch model does not include preprocessing time, and all models were tested on an NVIDIA A4000 GPU*
1. LSS accelerate refers to the process of pre-computing and storing the data used for BEVPool mapping during the View Transformer stage to improve inference speed. The pre-stored data is calculated based on the camera's intrinsic and extrinsic parameters. Due to slight differences in the intrinsic and extrinsic parameters of certain scenes in nuScenes, enabling the LSS accelerate can result in a decrease in precision. However, if the camera's intrinsic parameters remain unchanged and the extrinsic parameters between the camera coordinate system and the Ego coordinate system also remain unchanged, using LSS Accelerate will not result in a decrease in precision.
2. Some networks are very sensitive to input, and the Pytorch models use PIL's resize function with default Bicubic interpolation for preprocessing. During inference, neither OpenCV's Bicubic interpolation nor our own implementation of Bicubic interpolation can achieve the accuracy of Pytorch. We speculate that the network may be slightly overfitting or learning certain features of the sampler, which leads to a decrease in accuracy when the interpolation method is changed. Here, using Python preprocessing as input, it can be seen that the accuracy of the TRT model does not decrease in some cases
3. If as stated in point 2, we use our own implementation of Bicubic interpolation, it cannot achieve the performance of Python preprocessing.
4. Nearest is 20 times slower than Bicubic and will be used as the default sampling method.
5. Fine-tune the network, and the preprocess uses the resize based on Nearest sampling implemented in C++. After fine-tuning, the network is adapted to Nearest sampling -->

## DataSet
The Project provides a test sample that can also be used for inference on the nuScenes dataset. When testing on the nuScenes dataset, you need to use the data_infos folder provided by this project. The data folder should have the following structure:

    └── data
        ├── nuscenes
            ├── data_infos
                ├── samples_infos
                    ├── sample0000.yaml
                    ├── sample0001.yaml
                    ├── ...
                ├── samples_info.yaml
                ├── time_sequence.yaml
            ├── samples
            ├── sweeps
            ├── ...
the data_infos folder can be downloaded from [Google drive](https://drive.google.com/file/d/1RkjzvDJH4ZapYpeGZerQ6YZyervgE1UK/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1TyPoP6OPbkvD9xDRE36qxw?pwd=pa1v)

## Environment
For desktop or server：

- CUDA 11.8
- cuDNN 8.6.0
- TensorRT 8.5.2.2
- yaml-cpp
- Eigen3
- libjpeg

For Jetson AGX Orin

- Jetpack 5.1.1
- CUDA 11.4.315
- cuDNN 8.6.0
- TensorRT 8.5.2.2
- yaml-cpp
- Eigen3
- libjpeg

  
## Compile && Run
Use the ONNX file to export the TRT engine based on the script:
```shell
mkdir build && cd build
cmake .. && make
./export model.onnx model.engine
```
<!-- ONNX files, cound be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1zkfNdFNilkq4FikMCet5PQ?pwd=bp3z) or [Google Drive](https://drive.google.com/drive/folders/1jSGT0PhKOmW3fibp6fvlJ7EY6mIBVv6i?usp=drive_link) -->

Inference

```shell
./bevdemo ../configure.yaml
```


## References
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [nuScenes](https://www.nuscenes.org/)
