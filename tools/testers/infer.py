import argparse
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import numpy as np
import torch
from PIL import Image
import cv2
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.transforms import Compose
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.utils.mmcv_config import Config
from detectron2.utils import comm
from detectron2.engine import launch
import torch.nn.functional as F
from glob import glob
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # 导入 safetensors 库

# Argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Run single-image depth/surface normal estimation.")
    parser.add_argument("--arch_name", type=str, default="depthanything-large", choices=['depthanything-large', 'depthanything-base', 'depthanything-small'], help="Select a method for inference.")
    parser.add_argument("--mode", type=str, default="disparity", choices=['rel_depth', 'metric_depth', 'disparity'], help="Select a method for inference.")
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-v1-0", help="Checkpoint path or hub name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--processing_res", type=int, default=700, help="Maximum resolution of processing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser

# Helper function for model loading
def load_model_by_name(arch_name, checkpoint_path, device):
    model_kwargs = dict(
        vits=dict(
            encoder='vits', 
            features=64,
            out_channels=[48, 96, 192, 384]
        ),
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )

    # Load model
    if arch_name == 'depthanything-large':
        model = DepthAnything(**model_kwargs['vitl']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")

    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")
    elif arch_name == 'depthanything-small':
        model = DepthAnythingV2(**model_kwargs['vits']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")

    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # safetensors 
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    del model_weights
    torch.cuda.empty_cache()
    return model

# Helper function for directory checks
def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Image processing function
def process_images(validation_images, image_logs_folder, transform, model, device):
    images = []
    for i, image_path in enumerate(validation_images):
        validation_image_np = cv2.imread(image_path, cv2.COLOR_BGR2RGB)[..., ::-1] / 255
        _, orig_H, orig_W = validation_image_np.shape
        validation_image = transform({'image': validation_image_np})['image']
        validation_image = torch.from_numpy(validation_image).unsqueeze(0).to(device)

        with torch.autocast("cuda"):
            pred_disp, _ = model(validation_image) if 'midas' not in args.arch_name else model(validation_image)
        pred_disp_np = pred_disp.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0)
        pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

        cmap = "Spectral_r" if args.mode != 'metric' else 'Spectral_r'
        depth_colored = colorize_depth_maps(pred_disp[None, ...], 0, 1, cmap=cmap).squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        
        val_img_np = validation_image_np * 255
        h, w = val_img_np.shape[:2]
        depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)

        image_out = Image.fromarray(np.concatenate([depth_colored_hwc], axis=1))
        images.append(image_out)
        image_out.save(osp.join(image_logs_folder, f'da_sota_{i}.jpg'))
        print(f'{i} OK')
        torch.cuda.empty_cache()

    return images

def main(args, num_gpus):
    gpu_id = comm.get_rank()
    device = torch.device(f"cuda:{gpu_id}")
    logging.info(f'GPU numbers: {num_gpus}')

    # Model preparation
    model = load_model_by_name(args.arch_name, args.checkpoint, device)

    # Image directory check
    check_directory(args.output_dir)
    image_logs_folder = osp.join(args.output_dir, 'image_logs')
    os.makedirs(image_logs_folder, exist_ok=True)

    # Load validation images using glob
    validation_images = glob('data/input/*')

    # Define image transformation
    resize_h, resize_w = args.processing_res, args.processing_res
    transform = Compose([
        Resize(resize_w, resize_h, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    images = process_images(validation_images, image_logs_folder, transform, model, device)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = argument_parser().parse_args()
    num_gpus = torch.cuda.device_count()
    launch(main, num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args, num_gpus))
