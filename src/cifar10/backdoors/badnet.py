import numpy as np
import torch
from PIL import Image

class Badnet:
    def __init__(self, device=None):
        self.device = device
        self.patch = Image.open('data/trigger/badnet/flower_nobg.png')
    
    def mask_init(self, size):
        a, b = int(size/4), int(size/20)
        trigger = torch.Tensor(np.asarray(self.patch.resize((a, a), Image.LANCZOS)) / 255.).permute(2, 0, 1)
        mask = trigger[None, 3:, :, :]
        out = torch.zeros((1, 1, size, size))
        out[:, :, b:b+a, b:b+a] = mask
        out = torch.clamp(out, 0., 1.)
        return out.to(self.device)
    
    def inject(self, inputs):
        out = inputs.clone()
        size = inputs.size(2)
        a, b = int(size/4), int(size/32)
        
        DEVICE = inputs.device
        trigger = torch.Tensor(np.asarray(self.patch.resize((a, a), Image.LANCZOS)) / 255.).permute(2, 0, 1)
        mask, trigger = trigger[None, 3:, :, :], trigger[None, :3, :, :]
        mask, trigger = mask.to(DEVICE), trigger.to(DEVICE)
        out[:, :, b:b+a, b:b+a] = inputs[:, :, b:b+a, b:b+a] * (1 - mask) + trigger * mask
        out = torch.clamp(out, 0., 1.)
        return out
