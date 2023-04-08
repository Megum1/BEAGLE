import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

class TrojNN:
    def __init__(self, device=None):
        self.device = device
        self.patch = Image.open('data/trigger/trojnn/trojnn.jpg')
        self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
        self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)
    
    def mask_init(self, size):
        return transforms.Resize(size)(self.mask)[None, 0, ...].to(self.device)
    
    def inject(self, inputs):
        size = inputs.size(2)
        patch = transforms.Resize(size)(self.patch)[None, ...].to(self.device)
        mask = transforms.Resize(size)(self.mask)[None, ...].to(self.device)

        out = (1 - mask) * inputs + mask * patch
        out = torch.clamp(out, 0., 1.)
        return out
