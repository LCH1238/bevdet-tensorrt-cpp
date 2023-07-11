import os
import argparse
from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx
from ruamel import yaml

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(description='Export Engine Model')
    parser.add_argument('config', help='yaml config file path')
    parser.add_argument('img_encoder_onnx', help='path to img_encoder onnx file')
    parser.add_argument('bev_encoder_onnx', help='path to bev_encoder onnx file')
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
        
    img_stage_onnx = onnx.load(args.img_encoder_onnx)
    try:
        onnx.checker.check_model(img_stage_onnx)
    except Exception:
        print(f'{args.img_encoder_onnx} ONNX Model Incorrect')
        assert 0
    else:
        print(f'{args.img_encoder_onnx} ONNX Model Correct')


    bev_stage_onnx = onnx.load(args.bev_encoder_onnx)
    try:
        onnx.checker.check_model(bev_stage_onnx)
    except Exception:
        print(f'{args.bev_encoder_onnx} ONNX Model Incorrect')
        assert 0
    else:
        print(f'{args.bev_encoder_onnx} ONNX Model Correct')


    yaml_cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)
    use_depth = yaml_cfg['use_depth']
    img_H, img_W = yaml_cfg['data_config']['input_size']
    downsample_factor = yaml_cfg['model']['down_sample']
    
    feat_w, feat_h = img_H // downsample_factor, img_W // downsample_factor
    
    D = len(np.arange(*yaml_cfg['grid_config']['depth']))  
    bev_h = len(np.arange(*yaml_cfg['grid_config']['x']))
    bev_w = len(np.arange(*yaml_cfg['grid_config']['y']))
    
    
    bev_inchannels = (yaml_cfg['adj_num'] + 1) * yaml_cfg['model']['bevpool_channels']

    img_shape = [6, 3, img_H, img_W]
    img_input_shape = dict(
        images=dict(min_shape=img_shape, opt_shape=img_shape, max_shape=img_shape)
    )
    if use_depth:
        img_input_shape['rot'] = dict(min_shape=[1, 6, 3, 3], opt_shape=[1, 6, 3, 3], max_shape=[1, 6, 3, 3])
        img_input_shape['trans'] = dict(min_shape=[1, 6, 3], opt_shape=[1, 6, 3], max_shape=[1, 6, 3])
        img_input_shape['intrin'] = dict(min_shape=[1, 6, 3, 3], opt_shape=[1, 6, 3, 3], max_shape=[1, 6, 3, 3])
        img_input_shape['post_rot'] = dict(min_shape=[1, 6, 3, 3], opt_shape=[1, 6, 3, 3], max_shape=[1, 6, 3, 3])
        img_input_shape['post_trans'] = dict(min_shape=[1, 6, 3], opt_shape=[1, 6, 3], max_shape=[1, 6, 3])
        img_input_shape['bda'] = dict(min_shape=[1, 3, 3], opt_shape=[1, 3, 3], max_shape=[1, 3, 3])


    bev_shape = [1, bev_inchannels, bev_h, bev_w]
    bev_input_shape = dict(
        BEV_feat=dict(min_shape=bev_shape, opt_shape=bev_shape, max_shape=bev_shape)
    )
    
    img_engine_file = replace_file_name(args.img_encoder_onnx, f'img_stage{args.postfix}')
    bev_engine_file = replace_file_name(args.bev_encoder_onnx, f'bev_stage{args.postfix}')


    from_onnx(
        onnx_model=args.img_encoder_onnx,
        output_file_prefix=img_engine_file,
        input_shapes=img_input_shape,
        device_id=args.gpu_id,
        max_workspace_size=1 << 32,
        fp16_mode=args.fp16,
        tf32=args.tf32    
    )

    from_onnx(
        onnx_model=args.bev_encoder_onnx,
        output_file_prefix=bev_engine_file,
        input_shapes=bev_input_shape,
        device_id=args.gpu_id,
        max_workspace_size=1 << 32,
        fp16_mode=args.fp16,
        tf32=args.tf32   
    )
