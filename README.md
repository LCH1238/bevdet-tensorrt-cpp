# BEVDet implemented by TensorRT, C++

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

This project is a TensorRT implementation for BEVDet inference, written in C++. It can be tested on the nuScenes dataset and also provides a single test sample. For more details about BEVDet, please refer to the following link [BEVDet](https://github.com/HuangJunJie2017/BEVDet)。

This project implements the following:
- Long-term model
- Depth model
- On the NVIDIA A4000, the BEVDet-r50-lt-depth model shows a 2.38x faster inference speed for TRT FP32 compared to PyTorch FP32, and a 5.21x faster inference speed for TRT FP16 compared to PyTorch FP32
- On the Jetson AGX Orin, the FP16 model inference time is around 29 ms, achieving real-time performance
- A Dataloader for the nuScenes dataset and can be used to test on the dataset

The features of this project are as follows:
- A CUDA Kernel that combines Resize, Crop, and Normalization for preprocessing
- The Preprocess CUDA kernnel includes two interpolation methods: Nearest Neighbor Interpolation and Bicubic Interpolation
- Alignment of adjacent frame BEV features using C++ and CUDA kernel implementation
- Multi-threading and multi-stream NvJPEG
- Sacle-NMS
  
The following parts need to be implemented:
- Quantization to int8.
- Integrate the bevpool and adjacent frame BEV feature alignment components into the engine as plugins
- Fine-tune the model to address the issue of model sensitivity to input resize sampling, resulting in decreased mAP and NDS metrics
- Exception handling

## Results && Speed
## Inference Speed
All time units are in milliseconds (ms), and Nearest interpolation is used by default.

||Preprocess|Image stage|BEV pool|Align Feature|BEV stage|Postprocess|mean Total | 
|---|---|---|---|---|---|---|---|
|NVIDIA A4000 FP32|0.478|16.559|0.151|0.899|6.848 |0.558|25.534|
|NVIDIA A4000 FP16|0.512|8.627 |0.168|0.925|2.966 |0.619|13.817|
|Jetson AGX Orin FP16|2.816|17.025|0.571|2.111|5.747 |0.919|29.189|
|Jetson AGX orin FP32|2.800|38.09|0.620|2.018|11.893|1.065|55.104|

*Note: The inference time of the module refers to the time of a frame, while the total time is calculated as the average time of 200 frames.*

## Results
|Model   |Description       |mAP   |NDS    |Infer time|
|---     |---               |---   |---    |---       |
|Pytorch |                  |0.3972|0.5074|96.052|
|Pytorch |LSS accelerate<sup>1</sup>   |0.3787|0.4941|86.236|
|Trt FP32|Python Preprocess<sup>2</sup>|0.3776|0.4936|25.534|
|Trt FP32|Bicubic sampler<sup>3</sup>  |0.3723|0.3895|33.960|
|Trt FP32|Nearest sampler<sup>4</sup>  |0.3703|0.4884|25.534|
|Trt FP16|Nearest sampler   |0.3702|0.4883|13.817|

*Note: The PyTorch model does not include preprocessing time, and all models were tested on an NVIDIA A4000 GPU*
1. LSS accelerate refers to the process of pre-computing and storing the data used for BEVPool mapping during the View Transformer stage to improve inference speed. The pre-stored data is calculated based on the camera's intrinsic and extrinsic parameters. Due to slight differences in the intrinsic and extrinsic parameters of certain scenes in nuScenes, enabling the LSS accelerate can result in a decrease in precision. However, if the camera's intrinsic parameters remain unchanged and the extrinsic parameters between the camera coordinate system and the Ego coordinate system also remain unchanged, using LSS Accelerate will not result in a decrease in precision.
2. Some networks are very sensitive to input, and the Pytorch models use PIL's resize function with default Bicubic interpolation for preprocessing. During inference, neither OpenCV's Bicubic interpolation nor our own implementation of Bicubic interpolation can achieve the accuracy of Pytorch. We speculate that the network may be slightly overfitting or learning certain features of the sampler, which leads to a decrease in accuracy when the interpolation method is changed. Here, using Python preprocessing as input, it can be seen that the accuracy of the TRT model does not decrease in some cases
3. If as stated in point 2, we use our own implementation of Bicubic interpolation, it cannot achieve the performance of Python preprocessing.
4. Nearest is 20 times slower than Bicubic and will be used as the default sampling method.

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

For Jetson Orin

    - Jetpack 5.1.1
    - CUDA 11.4.315
    - cuDNN 8.6.0
    - TensorRT 8.5.2.2
    - yaml-cpp
    - Eigen3
    - libjpeg

  
## Compile && Run
Please use the ONNX file provided by this project to generate the TRT engine based on the script:
```shell
python tools/export_engine.py cfgs/bevdet_lt_depth.yaml model/img_stage_lt_d.onnx model/bev_stage_lt_d.engine --postfix="_lt_d_fp16" --fp16=True
```
ONNX files, cound be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1zkfNdFNilkq4FikMCet5PQ?pwd=bp3z) or [Google Drive](https://drive.google.com/drive/folders/1jSGT0PhKOmW3fibp6fvlJ7EY6mIBVv6i?usp=drive_link)

```shell
mkdir build && cd build
cmake .. && make
./bevdemo ../configure.yaml
```


## References
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [nuScenes](https://www.nuscenes.org/)
