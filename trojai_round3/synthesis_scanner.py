import os
import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


#######################################################################################
# Forensics result of polygon triggers
#######################################################################################
def center_loss(mask, weight):
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()

    # Current centroid
    indices = torch.nonzero(mask[0])
    centroid = torch.Tensor.float(indices).mean(dim=0)

    #########################################
    # Forensic centroid range
    # Round 3:
    mu_x, sigma_x = 123.93, 33.3
    mu_y, sigma_y = 113.81, 31.79
    #########################################

    cond_x_p68 = torch.logical_and(torch.gt(centroid[0], mu_x - sigma_x), torch.le(centroid[0], mu_x + sigma_x))
    cond_x_p95 = torch.logical_and(torch.gt(centroid[0], mu_x - 2 * sigma_x), torch.le(centroid[0], mu_x + 2 * sigma_x))

    cond_y_p68 = torch.logical_and(torch.gt(centroid[1], mu_y - sigma_y), torch.le(centroid[1], mu_y + sigma_y))
    cond_y_p95 = torch.logical_and(torch.gt(centroid[1], mu_y - 2 * sigma_y), torch.le(centroid[1], mu_y + 2 * sigma_y))

    cond_p68 = torch.logical_and(cond_x_p68, cond_y_p68)
    cond_p95 = torch.logical_and(cond_x_p95, cond_y_p95)

    fcenter = torch.FloatTensor([mu_x, mu_y]).to(mask.device)
    center_loss = L2(centroid, fcenter)

    loss = torch.where(cond_p68, 0. * center_loss, torch.where(cond_p95, 1 * weight * center_loss, 2 * weight * center_loss))

    return loss


def size_loss(mask, weight, mask_epsilon):
    mask_loss = torch.sum(mask)
    size_nz = torch.sum(torch.gt(mask, mask_epsilon))

    #########################################
    # Forensic size range
    # Round 3
    mu, sigma = 1278.44, 449.05
    #########################################

    cond_p68 = torch.logical_and(torch.gt(size_nz, mu - sigma), torch.le(size_nz, mu + sigma))
    cond_p95 = torch.logical_and(torch.gt(size_nz, mu - 2 * sigma), torch.le(size_nz, mu + 2 * sigma))

    loss = torch.where(cond_p68, 0. * mask_loss, torch.where(cond_p95, 1 * weight * mask_loss, 2 * weight * mask_loss))

    return loss


def mask_init(h=224, w=224):
    mask = np.zeros((h, w), dtype=np.float32)

    #########################################
    # Forensic average centroid and side length
    # Round 3
    x, y = 124, 114
    w = 36
    #########################################

    half_w = int(w / 2)
    mask[(x - half_w):(x + half_w), (y - half_w):(y + half_w)] += 1
    return mask


def delta_init():
    # Forensic average color
    # Round 3
    return np.asarray([0.17, 0.24, 0.18])


#######################################################################################
# Forensics result of instagram filter triggers
#######################################################################################
def load_filter_forensics():
    # Forensic results
    filter_list = ['LomoFilterXForm', 'NashvilleFilterXForm', 'GothamFilterXForm', 'KelvinFilterXForm', 'ToasterXForm']
    id_list = [[17, 21, 24, 33, 36, 39, 62, 73, 77], [0, 28, 63], [41, 53, 58], [44, 55, 76], [68, 79]]

    forensic_weights = []
    forensic_bias = []
    for i in range(len(filter_list)):
        filter_id = filter_list[i]
        _w, _b = [], []
        for j in id_list[i]:
            model_id = 'id-' + str(j).zfill(8)
            weights, bias = pickle.load(open(f'forensics/trojai_filter/{model_id}/linear_param', 'rb'), encoding='bytes')
            _w.append(weights)
            _b.append(bias)
        
        _w = torch.cat(_w)
        _b = torch.cat(_b)
        w_mean = torch.mean(_w, dim=0, keepdim=True)
        w_std = torch.std(_w, dim=0, keepdim=True)
        b_mean = torch.mean(_b, dim=0, keepdim=True)
        b_std = torch.std(_b, dim=0, keepdim=True)
        forensic_weights.append(w_mean)
        forensic_bias.append(b_mean)
    forensic_weights = torch.cat(forensic_weights)
    forensic_bias = torch.cat(forensic_bias)

    return forensic_weights, forensic_bias
