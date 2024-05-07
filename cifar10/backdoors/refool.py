import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image


class Refool:
    def __init__(self, side_len, device):
        self.side_len = side_len
        self.device = device

        # Load the trigger
        trigger_path = 'data/trigger/refool/000066.jpg'
        trigger = read_image(trigger_path) / 255.0
        self.trigger = transforms.Resize((side_len, side_len))(trigger).to(self.device)
        self.weight_t = self.trigger.mean()

    def inject(self, inputs):
        out = []
        for img in inputs:
            img = img.to(self.device)
            weight_i = img.mean()
            param_i =      weight_i / (weight_i + self.weight_t)
            param_t = self.weight_t / (weight_i + self.weight_t)
            new_img = torch.clamp(param_i * img + param_t * self.trigger, 0.0, 1.0)
            out.append(new_img)

        outputs = torch.stack(out)
        return outputs
