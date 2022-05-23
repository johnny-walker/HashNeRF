import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from openvino.runtime import Core
import configargparse

IMAGE_HEIGHT = 800
IMAGE_WIDTH = 800
EXPERIMENT = 'blender_paper_lego_hashXYZ_sphereVIEW_fine512_log2T19_lr0.01_decay10_RAdam_sparse1e-10_TV1e-06'

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, default=EXPERIMENT,
                        help='experiment name')
    return parser
    
def convert(args):
    #path, EMBED_FN = load_model(args, 'embed_fn')
    path, NETWORK_FN =  load_model(args, 'network_fn')
    #NETWORK_REFINE =  load_model(args, 'network_fine')
        
    dummy_input = torch.randn(1, 3, requires_grad=True).cpu() # BHWC
    convert_to_onnx(EMBED_FN, path, dummy_input)

def load_model(args, model_name):
    pths = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if 'pth' in f]
    model = None
    path = None
    if len(pths) > 0:
        for path in reversed(pths):
            if model_name in path:
                print('Found model path', path)
                model = torch.load(path, map_location='cpu')
                break    
    return path, model

def convert_to_onnx(model, path, dummy_input):
    # Paths where PyTorch, ONNX and OpenVINO IR models will be stored
    model_path = Path(path).with_suffix(".pth")
    onnx_path = model_path.with_suffix(".onnx")
    ir_path = model_path.with_suffix(".xml")

    print('pytorch path:', model_path)
    print('convert to:', onnx_path)
    model.eval()
    if not onnx_path.exists():
        torch.onnx.export(  model,
                            dummy_input,
                            onnx_path,
                            opset_version=11,
                            do_constant_folding=False,
                         )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    convert(args)