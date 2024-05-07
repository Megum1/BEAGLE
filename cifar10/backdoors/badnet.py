import numpy as np
import torch
from PIL import Image


class BadNets:
    def __init__(self, side_len, device):
        self.side_len = side_len
        self.device = device

        # Load the trigger
        patch = Image.open('data/trigger/badnet/flower_nobg.png')
        trig_len, trig_pos = int(self.side_len / 6), int(side_len / 32)
        trigger = torch.Tensor(np.asarray(patch.resize((trig_len, trig_len), Image.LANCZOS)) / 255.).permute(2, 0, 1)
        trigger = trigger[None, :3, :, :]

        self.mask = torch.zeros((1, 1, side_len, side_len))
        self.mask[:, :, trig_pos:trig_pos+trig_len, trig_pos:trig_pos+trig_len] = 1
        self.mask = self.mask.to(self.device)

        self.pattern = torch.zeros((1, 3, side_len, side_len))
        self.pattern[:, :, trig_pos:trig_pos+trig_len, trig_pos:trig_pos+trig_len] = trigger
        self.pattern = self.pattern.to(self.device)
    
    def inject(self, inputs):
        out = inputs.clone()
        out = out.to(self.device) * (1 - self.mask) + self.pattern * self.mask

        return out
