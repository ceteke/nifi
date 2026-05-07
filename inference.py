import argparse
from PIL import Image

import torch
from torchvision import transforms, utils

from restorer import SD3Restoration

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, default='checkpoints/stable-diffusion-3', help='Path to Stable Diffusion 3.')
parser.add_argument('--ckpt', type=str, default='checkpoints/3dgs-compression-restore.ckpt', help='Path to restoration adapter weights.')
parser.add_argument('--lq', type=str, required=True, help='Low-quality image path')
parser.add_argument('--out', type=str, required=True, help='Output')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--embeddings', type=str, default='example/default_embeddings.npz', help='Default text embeddings. See embeddings.py for details.')
parser.add_argument('--dtype', choices=["bf16", "f16", "f32"], type=str, default='bf16', help='Model dtype.')

if __name__ == '__main__':
    args = parser.parse_args()
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    inference = SD3Restoration(args.base_model, args.ckpt, args.device, args.dtype)
    inference.set_prompt(args.embeddings)
    
    lq = tf(Image.open(args.lq).convert('RGB'))
    
    with torch.no_grad():
        hat = inference(lq)
        hat = torch.clip(0.5*hat+0.5, 0, 1)
    
    utils.save_image(hat, args.out)
