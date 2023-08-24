import os
import ctypes
import argparse
from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx
from ruamel import yaml
import pycuda.driver as cuda

import tensorrt as trt

alignbev_dll = "./build/libalignbev.so"
bevpool_dll = "./build/libbevpool_plugin.so"
preprocess_dll = "./build/libpreprocess_plugin.so"

def parse_args():
    parser = argparse.ArgumentParser(description='Export Engine Model')
    parser.add_argument('config', help='yaml config file path')
    parser.add_argument('onnx_model', help='path to onnx file')
    parser.add_argument(
        '--postfix', default='', help='postfix of the save file name')
    parser.add_argument(
        '--tf32', type=bool, default=True, help='default to turn on the tf32')
    parser.add_argument(
        '--fp16', type=bool, default=False, help='float16')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='Which gpu to be used'
    )
    args = parser.parse_args()
    return args


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              tf32 : bool = True) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX """
    
    os.environ['CUDA_DEVICE'] = str(device_id)
    
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    
    pluginRegistry = trt.get_plugin_registry()
    
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(bevpool_dll)
    ctypes.cdll.LoadLibrary(alignbev_dll)
    ctypes.cdll.LoadLibrary(preprocess_dll)
    # handle = pluginRegistry.load_library(alignbev_dll)
    # handle = pluginRegistry.load_library(bevpool_dll)
    # handle = pluginRegistry.load_library(preprocess_dll)
    
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        
    if tf32 is False:
        config.clear_flag(trt.BuilderFlag.TF32)

    # create engine
    engine = builder.build_serialized_network(network, config)
    assert engine is not None, 'Failed to create TensorRT engine'

    with open(output_file_prefix + '.engine', mode='wb') as f:
        f.write(bytearray(engine))
    return engine

def replace_file_name(path, new_name=None):
    assert new_name != None
    path = path.split('/')[:-1]
    file = ''
    for p in path:
        file += p + '/'
    file += new_name
    return file

if __name__ == '__main__':
    args = parse_args()
        
    onnx_model = onnx.load(args.onnx_model)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print(f'{args.onnx_model} ONNX Model Incorrect')
        assert 0
    else:
        print(f'{args.onnx_model} ONNX Model Correct')


    yaml_cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)
    use_depth = yaml_cfg['use_depth']
    img_H, img_W = yaml_cfg['data_config']['input_size']
    downsample_factor = yaml_cfg['model']['down_sample']
    
    feat_w, feat_h = img_H // downsample_factor, img_W // downsample_factor
    
    D = len(np.arange(*yaml_cfg['grid_config']['depth']))  
    bev_h = len(np.arange(*yaml_cfg['grid_config']['x']))
    bev_w = len(np.arange(*yaml_cfg['grid_config']['y']))
    
    
    bev_inchannels = (yaml_cfg['adj_num'] + 1) * yaml_cfg['model']['bevpool_channels']

    img_shape = [6, 3, img_H, img_W // 4]
    input_shape = dict(
        images=dict(min_shape=img_shape, opt_shape=img_shape, max_shape=img_shape),
        mean=dict(min_shape=[3], opt_shape=[3], max_shape=[3]),
        std=dict(min_shape=[3], opt_shape=[3], max_shape=[3]),
        cam_params=dict(min_shape=[6, 27], opt_shape=[6, 27], max_shape=[6, 27]),
        ranks_depth=dict(min_shape=[356760], opt_shape=[356760], max_shape=[356760]),
        ranks_feat=dict(min_shape=[356760], opt_shape=[356760], max_shape=[356760]),
        ranks_bev=dict(min_shape=[356760], opt_shape=[356760], max_shape=[356760]),
        interval_starts=dict(min_shape=[13360], opt_shape=[13360], max_shape=[13360]),
        interval_lengths=dict(min_shape=[13360], opt_shape=[13360], max_shape=[13360])
    )
    bev_shape = [yaml_cfg['model']['bevpool_channels'], bev_h, bev_w]
    
    for i in range(8):
        input_shape[f'adj_feat{i}'] = dict(min_shape=bev_shape, opt_shape=bev_shape, max_shape=bev_shape)
    input_shape['transforms'] = dict(min_shape=[8, 6], opt_shape=[8, 6], max_shape=[8, 6])
    input_shape['copy_flag'] = dict(min_shape=[8], opt_shape=[8], max_shape=[8])
    
    
    engine_file = replace_file_name(args.onnx_model, f'bevdet_{args.postfix}')


    from_onnx(
        onnx_model=args.onnx_model,
        output_file_prefix=engine_file,
        input_shapes=input_shape,
        device_id=args.gpu_id,
        max_workspace_size=1 << 33,
        # fp16_mode=args.fp16,
        tf32=args.tf32    
    )
