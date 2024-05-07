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
# (Refool) func: mask | func_option: uniform
######################################################################
def attach_trigger(inputs, trigger, alpha):
    out = []
    for i in range(inputs.size(0)):
        img = inputs[i]
        p_t, p_i = alpha, 1 - alpha
        out.append(torch.clamp(p_i * img + p_t * trigger, 0., 1.))
    out = torch.stack(out)
    return out


def remove_trigger(inputs, trigger, alpha):
    out = []
    for i in range(inputs.size(0)):
        img = inputs[i]
        p_t, p_i = alpha, 1 - alpha
        out.append(torch.clamp( (img - p_t * trigger) / p_i, 0., 1.))
    out = torch.stack(out)
    return out


def extract_trigger(poi_inputs, cln_inputs, alpha):
    out = []
    for i in range(poi_inputs.size(0)):
        poi, cln = poi_inputs[i], cln_inputs[i]
        p_t, p_i = alpha, 1 - alpha
        out.append(torch.clamp( (poi - p_i * cln) / p_t, 0., 1. ))
    out = torch.stack(out)
    out = torch.mean(out, dim=0, keepdim=True)
    return out


######################################################################
# (BadNets) func: mask | func_option: binomial
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


######################################################################
# (WaNet) func: transform | func_option: complex
######################################################################
def complex_linear(inputs, grids, channel_bias, threshold=0.1):
    # inputs.shape = [batch_size, 3, 32, 32]
    # grid.shape = [1, 27, kdim, kdim]
    # channel_bias.shape = [1, 3, kdim, kdim]
    height = inputs.size(2)
    grids = F.upsample(grids, size=height, mode='bicubic', align_corners=True).permute(0, 2, 3, 1).view(height, height, 3, 3, 3) # (32, 32, 3, 3, 3)
    if channel_bias.size(2) != 1:
        channel_bias = F.upsample(channel_bias, size=height, mode='bicubic', align_corners=True)

    k = 3 # kernel_size
    n, c, h, w = inputs.size()
    DEVICE = inputs.device

    # Center set 1
    grid_bias = torch.zeros((3, 3))
    grid_bias[1, 1] = 1
    grid_bias = grid_bias.repeat(h, w, 3, 1, 1).to(DEVICE) # [32, 32, 3, 3, 3]

    # Constrain the threshold
    grids = torch.clamp(grids, -threshold, threshold)
    grids += grid_bias
    channel_bias = torch.clamp(channel_bias, -threshold, threshold)

    pad_inputs = F.pad(inputs, (1, 1, 1, 1), "constant", 0)
    data = F.unfold(pad_inputs, (k, k))
    data = data.permute(0, 2, 1)
    data = data.view(n, h, w, c, k, k) # [batch_size, 32, 32, 3, 3, 3]
    grids = grids[None, :, :, :, :, :] # [1, 32, 32, 3, 3, 3]
    out = (data * grids).sum(dim=[-2, -1]).permute(0, 3, 1, 2)
    out = torch.clamp(out + channel_bias, 0., 1.)
    return out


def simple_linear(inputs, weights, bias, threshold=0.5):
    # inputs.shape = [batch_size, 3, 32, 32]
    # weights: (1, 3, kdim, kdim)
    # bais:    (1, 3, kdim, kdim)
    height = inputs.size(2)
    weights = F.upsample(weights, size=height, mode='bicubic', align_corners=True)
    if bias.size(2) != 1:
        bias = F.upsample(bias, size=height, mode='bicubic', align_corners=True)
    
    weights = torch.clamp(weights, -threshold, threshold)
    bias = torch.clamp(bias, -threshold, threshold)
    output = inputs * (1 + weights) + bias
    output = torch.clamp(output + bias, 0., 1.)
    return output
