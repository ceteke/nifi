import torch
import numpy as np
from peft import LoraConfig
from diffusers import AutoencoderKL
from diffusers.models import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from utils import convert_dtype, pad_to_multiple


class SD3Restoration:
    def __init__(self, base_model, checkpoint, device, dtype):
        self.device = device
        self.dtype = convert_dtype(dtype)

        print("Loading base model..")
        self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        self.transformer = SD3Transformer2DModel.from_pretrained(base_model, subfolder="transformer")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")

        vae_target_modules = ['encoder.conv_in', 'encoder.down_blocks.0.resnets.0.conv1', 'encoder.down_blocks.0.resnets.0.conv2', 'encoder.down_blocks.0.resnets.1.conv1', 
                            'encoder.down_blocks.0.resnets.1.conv2', 'encoder.down_blocks.0.downsamplers.0.conv', 'encoder.down_blocks.1.resnets.0.conv1',
                            'encoder.down_blocks.1.resnets.0.conv2', 'encoder.down_blocks.1.resnets.0.conv_shortcut', 'encoder.down_blocks.1.resnets.1.conv1', 'encoder.down_blocks.1.resnets.1.conv2', 
                            'encoder.down_blocks.1.downsamplers.0.conv', 'encoder.down_blocks.2.resnets.0.conv1', 'encoder.down_blocks.2.resnets.0.conv2',
                            'encoder.down_blocks.2.resnets.0.conv_shortcut', 'encoder.down_blocks.2.resnets.1.conv1', 'encoder.down_blocks.2.resnets.1.conv2', 'encoder.down_blocks.2.downsamplers.0.conv',
                            'encoder.down_blocks.3.resnets.0.conv1', 'encoder.down_blocks.3.resnets.0.conv2', 'encoder.down_blocks.3.resnets.1.conv1', 'encoder.down_blocks.3.resnets.1.conv2', 
                            'encoder.mid_block.attentions.0.to_q', 'encoder.mid_block.attentions.0.to_k', 'encoder.mid_block.attentions.0.to_v', 'encoder.mid_block.attentions.0.to_out.0', 
                            'encoder.mid_block.resnets.0.conv1', 'encoder.mid_block.resnets.0.conv2', 'encoder.mid_block.resnets.1.conv1', 'encoder.mid_block.resnets.1.conv2', 'encoder.conv_out', 'quant_conv']
        transformer_target_modules = ["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj","proj","linear","proj_out"]

        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=transformer_target_modules,
        )
        vae_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=vae_target_modules
        )

        self.transformer.add_adapter(transformer_lora_config, adapter_name="default")
        self.transformer.enable_adapters()

        self.vae.add_adapter(vae_lora_config, adapter_name="default")
        self.vae.enable_adapters()
        
        print("Loading restorer")
        loras_ckpt = torch.load(checkpoint, map_location='cpu')
        miss, unex = self.vae.load_state_dict(loras_ckpt['vae'], strict=False)
        assert len(unex) == 0

        miss, unex = self.transformer.load_state_dict(loras_ckpt['transformer'], strict=False)
        assert len(unex) == 0

        self.sigmas = self.scheduler.sigmas.to(self.device, dtype=self.dtype)
        self.scheduler_timesteps = self.scheduler.timesteps.to(self.device)
        self.vae = self.vae.to(self.device, dtype=self.dtype).eval()
        self.transformer = self.transformer.to(self.device, dtype=self.dtype).eval()

    def set_prompt(self, embeddings_dir):
        default_embeddings = np.load(embeddings_dir)

        self.default_prompt_embeds = default_embeddings['prompt_embeds']
        self.default_pooled_prompt_embeds = default_embeddings['pooled_prompt_embeds']

        self.default_prompt_embeds = torch.Tensor(self.default_prompt_embeds).to(self.device, dtype=self.dtype)
        self.default_pooled_prompt_embeds = torch.Tensor(self.default_pooled_prompt_embeds).to(self.device, dtype=self.dtype)

    def get_sigmas(self, timesteps, n_dim=4):
        step_indices = [torch.argmin(torch.abs(self.scheduler_timesteps - t)) for t in timesteps]
        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def __call__(self, img_lq: torch.Tensor, seed=42):
        _, oldH, oldW = img_lq.shape
        img_lq = pad_to_multiple(img_lq).to(self.device, dtype=self.dtype)

        if len(img_lq.shape) == 3:
            img_lq = img_lq.unsqueeze(0)

        lq_latents = self.vae.encode(img_lq).latent_dist.sample().to(self.dtype)
        lq_latents = lq_latents * self.vae.config.scaling_factor

        lq_latents = pad_to_multiple(lq_latents.squeeze(), 2).unsqueeze(0)
        timesteps = [199] + [0.]

        start_timestep = torch.tensor([timesteps[0]], device=self.device)
        self.scheduler.set_timesteps(timesteps=timesteps)

        sigmas = self.get_sigmas(start_timestep, n_dim=lq_latents.ndim)
        noise = torch.randn_like(lq_latents)
        lq_latents = sigmas * noise + (1 - sigmas) * lq_latents

        for i in range(len(timesteps)-1):
            model_timesteps = torch.tensor([timesteps[i]], device=self.device)
            next_timesteps = torch.tensor([timesteps[i+1]], device=self.device)

            sigmas = self.get_sigmas(model_timesteps, n_dim=lq_latents.ndim)
            next_sigmas = self.get_sigmas(next_timesteps, n_dim=lq_latents.ndim)

            model_pred = self.transformer(
                lq_latents,
                timestep=model_timesteps,
                encoder_hidden_states=self.default_prompt_embeds,
                pooled_projections=self.default_pooled_prompt_embeds,
            ).sample

            lq_latents = lq_latents - (sigmas - next_sigmas) * model_pred

        x0_pred = lq_latents / self.vae.config.scaling_factor
        output = self.vae.decode(x0_pred).sample

        output = output[:, :, :oldH, :oldW]
        output = output.squeeze()

        return output