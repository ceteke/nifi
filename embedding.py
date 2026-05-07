import argparse

import torch
import numpy as np
from diffusers.pipelines import StableDiffusion3Pipeline

from prompt import DEFAULT_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, default='checkpoints/stable-diffusion-3', help='Path to Stable Diffusion 3.')
parser.add_argument('--prompt', type=str, required=False)
parser.add_argument('--out', type=str, default='example/default_embeddings.npz')

if __name__ == '__main__':
    args = parser.parse_args()

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.sd_dir, device_map="cuda", torch_dtype=torch.bfloat16
    )

    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT

    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=None, negative_prompt_2=None, negative_prompt_3=None
        )

    embeddings = {
        'prompt_embeds': prompt_embeds.cpu().float().numpy(),
        'pooled_prompt_embeds': pooled_prompt_embeds.cpu().float().numpy(),
    }

    np.savez(args.out, **embeddings)