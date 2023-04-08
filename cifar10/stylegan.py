import os
import math
import numpy as np
import torch
from stylegan2ada_generator_with_styles_noises import StyleGAN2ADAGenerator


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def postprocess(images):
    """change the range from [-1, 1] to [0., 1.]"""
    images = torch.clamp((images + 1.) / 2., 0., 1.)
    return images


class Fake_G:
    def __init__(self, G, g_function):
        self.G = G
        self.g_function = g_function

    def __call__(self, code, randomize_noises=None, special_noises=None, special_styles=None):
        return self.g_function(code, randomize_noises=randomize_noises, special_noises=special_noises, special_styles=special_styles)

    def zero_grad(self):
        self.G.zero_grad()


def build_generator(device, checkpoint_path):
    use_w_space = True
    repeat_w = False
    trunc_psi = 0.7
    trunc_layers = 8
    randomize_noise = False

    # Parse model configuration
    model_config = dict(
        gan_type='stylegan2ada',
        resolution=32,
        label_size=10,
        mapping_layers=2,
        encoder_arch='stylegan',
    )

    # Build generation and get synthesis kwargs
    generator = StyleGAN2ADAGenerator(**model_config, repeat_w=repeat_w)
    synthesis_kwargs = dict(trunc_psi=trunc_psi,
                            trunc_layers=trunc_layers,
                            randomize_noise=randomize_noise)

    # Load pre-trained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = checkpoint['models']
    generator.load_state_dict(checkpoint['generator'])
    generator = generator.to(device)
    generator.eval()

    def fake_generator(code, randomize_noises=None, special_noises=None, special_styles=None):
        # Sample and synthesize
        all_results = generator(code, **synthesis_kwargs, use_w_space=use_w_space, randomize_noises=randomize_noises, special_noises=special_noises, special_styles=special_styles)
        images = all_results['image']
        images = postprocess(images)
        return images

    return Fake_G(generator, fake_generator)


class StyleGAN:
    def __init__(self, device):
        self.device = device
        self.latent_dim = 512

        # Use genforce model
        self.num_layers = 8
        genforce_model = '../checkpoints/stylegan2ada_encoder_new_arch_featuremaploss_all_blocks_sample_z_iter080000.pth'
        self.generator = build_generator(self.device, genforce_model)
    
    def get_w_mean(self, batch_labels):
        with torch.no_grad():
            latent_z_inputs = torch.randn(batch_labels.size(0), self.latent_dim, device=self.device)
            latent_w = self.generator.G.mapping(latent_z_inputs, label=batch_labels)['w']
            latent_w_mean = latent_w.mean(0)
        return latent_w_mean.detach().clone().unsqueeze(0)
