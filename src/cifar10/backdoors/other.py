import torch
import torchvision

class Other:
    def __init__(self, attack, device=None):
        self.alpha = 0.2

        if attack == 'blend':
            self.pattern = torch.load('data/trigger/other/blend.pt').to(device)
        elif attack == 'sig':
            self.pattern = torch.load('data/trigger/other/sig.pt').to(device)

    def inject(self, inputs):
        if self.pattern.size(2) != inputs.size(3):
            self.pattern = torchvision.transforms.Resize(inputs.size(3))(self.pattern)
        if self.pattern.size(2) > 32:
            self.alpha = 0.3
        inputs = self.alpha * self.pattern + (1 - self.alpha) * inputs
        inputs = torch.clamp(inputs, 0.0, 1.0)
        return inputs
