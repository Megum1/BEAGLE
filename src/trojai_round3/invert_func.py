import os
import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


def epsilon():
    return 1e-7


######################################################################
# Mask optimization
######################################################################
def mask_process(mask, sparse=True):
    tmask = torch.tanh(mask) / (2 - epsilon()) + 0.5

    if sparse:
        pmask = tmask
    else:
        blur = GaussianBlur(3)
        a = 20
        pmask = torch.sigmoid(a * (blur(tmask) - 0.5))
    return pmask


def mask_align(mask):
    mask_mean = torch.mean(mask, dim=[0], keepdim=True)
    mask_std = torch.std(mask, dim=[0]).sum()

    tmask = mask_mean.repeat(mask.size(0), 1, 1, 1)
    print(mask_std)
    tmask = mask.clone()

    return tmask


##########################################################################################
# TrojAI filter
##########################################################################################
def filter_linear(inputs, weights, bias):
    # inputs.shape = [batch_size, 3, 224, 224]
    # weights: (1, 9, kdim, kdim)
    # bais:    (1, 3, 1, 1)
    height = inputs.size(2)
    weights = F.upsample(weights, size=height, mode='bicubic', align_corners=True)

    r = torch.sum(inputs * weights[:, 0:3, :, :] + bias[:, 0, :, :], dim=1, keepdim=True)
    g = torch.sum(inputs * weights[:, 3:6, :, :] + bias[:, 1, :, :], dim=1, keepdim=True)
    b = torch.sum(inputs * weights[:, 6:9, :, :] + bias[:, 2, :, :], dim=1, keepdim=True)
    output = torch.cat([r, g, b], dim=1)
    output = torch.clamp(output, 0., 1.)
    return output
