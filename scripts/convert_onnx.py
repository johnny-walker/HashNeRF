import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from openvino.runtime import Core
import configargparse

experiment = 'blender_paper_lego_hashXYZ_sphereVIEW_fine512_log2T19_lr0.01_decay10_RAdam_sparse1e-10_TV1e-06'

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='../logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, default=experiment,
                        help='experiment name')
    return parser
    
def convert():
    Total_iters = 50000
    parser = config_parser()
    args = parser.parse_args()

    NETWORK_FN = os.path.join(args.basedir, args.expname, '{:06d}_{:s}.pth'.format(Total_iters, 'network_fn'))
    NETWORK_REFINE = os.path.join(args.basedir, args.expname, '{:06d}_{:s}.pth'.format(Total_iters, 'network_fine'))
    EMBED_FN = os.path.join(args.basedir, args.expname, '{:06d}_{:s}.pth'.format(Total_iters, 'embed_fn'))
        
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    convert_to_onnx(NETWORK_FN, dummy_input)

def convert_to_onnx(model_path, dummy_input):
    # Paths where PyTorch, ONNX and OpenVINO IR models will be stored
    model_path = Path(NETWORK_FN).with_suffix(".pth")
    onnx_path = model_path.with_suffix(".onnx")
    ir_path = model_path.with_suffix(".xml")

    print('pytorch pth:', model_path)
    print('convert to:', onnx_path)
    print('convert to:',ir_path)

    if not onnx_path.exists():
        # For the Fastseg model, setting do_constant_folding to False is required
        # for PyTorch>1.5.1
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=11,
            do_constant_folding=False,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    convert()