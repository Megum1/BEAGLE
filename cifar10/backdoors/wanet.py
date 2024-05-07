import os
import torch
import torch.nn.functional as F


class WaNet:
    def __init__(self, side_len, device):
        self.height = side_len
        self.device = device

        # Load the trigger
        self.s = 0.5
        self.grid_rescale = 1

        noise_path    = 'data/trigger/wanet/noise_grid.pt'
        identity_path = 'data/trigger/wanet/identity_grid.pt'
        self.noise_grid    = torch.load(noise_path).to(self.device)
        self.identity_grid = torch.load(identity_path).to(self.device)

    def inject(self, inputs):
        self.grid = (self.identity_grid + self.s * self.noise_grid / self.height) * self.grid_rescale
        self.grid = torch.clamp(self.grid, -1, 1)

        outputs = F.grid_sample(inputs.to(self.grid.device),
                               self.grid.repeat(inputs.size(0), 1, 1, 1),
                               align_corners=True)
        return outputs
