import numpy as np
import os
import argparse
import sys
import json
import skimage.io
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pickle
import time

import math
import cv2
from util import *

np.set_printoptions(precision=2, linewidth=200, threshold=10000)


config = {}
config['print_level'] = 1
config['random_seed'] = 333
config['num_classes'] = 10
config['channel_last'] = 0
config['w'] = 32
config['h'] = 32
config['reasr_bound'] = 0.2
config['batch_size'] = 10
config['has_softmax'] = 0
config['samp_k'] = 8
config['same_range'] = 0
config['n_samples'] = 5
config['samp_batch_size'] = 1
config['top_n_neurons'] = 10
config['re_batch_size'] = 80
config['max_troj_size'] = 64
config['filter_multi_start'] = 1
config['re_mask_lr'] = 1e-2
config['re_mask_weight'] = 1e7
config['mask_multi_start'] = 1
config['re_epochs'] = 50
config['n_re_samples'] = 240


channel_last = bool(config['channel_last'])
random_seed = int(config['random_seed'])

resnet_sample_resblock = False

# deterministic
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

w = config["w"]
h = config["h"]
num_classes = config["num_classes"]
use_mask = True
count_mask = True
tdname = 'temp'
window_size = 12
mask_epsilon = 0.01
mask_epsilon = 0.1
Troj_size = config['max_troj_size']
reasr_bound = float(config['reasr_bound'])
top_n_neurons = int(config['top_n_neurons'])
mask_multi_start = int(config['mask_multi_start'])
filter_multi_start = int(config['filter_multi_start'])
re_mask_weight = float(config['re_mask_weight'])
re_mask_lr = float(config['re_mask_lr'])
batch_size = config['batch_size']
has_softmax = bool(config['has_softmax'])
# print('channel_last', channel_last, 'has softmax', has_softmax)

Print_Level = int(config['print_level'])
re_epochs = int(config['re_epochs'])
n_re_samples = int(config['n_re_samples'])


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


def preprocess(img):
    img = np.transpose(img, [0, 3, 1, 2])
    return img.astype(np.float32) / 255.0


def deprocess(x_in):
    x_in = x_in * std.reshape((1, 3, 1, 1)) + mean.reshape((1, 3, 1, 1))
    x_in *= 255
    return x_in.astype('uint8')


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def check_values(images, labels, model, children, target_layers):
    maxes = {}
    for layer_i in range(0, len(children) - 1):
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])

        max_val = -np.inf
        for i in range( math.ceil(float(len(images))/batch_size) ):
            batch_data = torch.FloatTensor(images[batch_size*i:batch_size*(i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()
            if channel_last:
                n_neurons = inner_outputs.shape[-1]
            else:
                n_neurons = inner_outputs.shape[1]
            max_val = np.maximum(max_val, np.amax(inner_outputs))
            # print(np.amax(inner_outputs))
        
        key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)
        maxes[key] = [max_val]
        # print('max val', key, max_val)
        del temp_model1, batch_data, inner_outputs
    return maxes


def sample_neuron(images, labels, model, children, target_layers, model_type, mvs, has_softmax=has_softmax):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    sample_batch_size = config['samp_batch_size']
    if model_type == 'DenseNet':
        sample_batch_size = max(sample_batch_size // 3, 1)
    n_images = images.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images)

    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    sample_layers = []
    for layer_i in range(2, end_layer):
        if not children[layer_i].__class__.__name__ in target_layers:
            continue
        sample_layers.append(layer_i)
    
    # Sample the last layer
    sample_layers = sample_layers[-1:]

    for layer_i in sample_layers:
        if Print_Level > 0:
            print('layer', layer_i, children[layer_i])
        temp_model1 = torch.nn.Sequential(*children[:layer_i+1])
        if has_softmax:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:-1])
        else:
            temp_model2 = torch.nn.Sequential(*children[layer_i+1:])

        if same_range:
            vs = np.asarray([i*samp_k for i in range(n_samples)])
        else:
            mv_key = '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i)

            tr = samp_k * max(mvs[mv_key])/(n_samples)
            vs = np.asarray([i*tr for i in range(n_samples)])
        
        for input_i in range( math.ceil(float(n_images)/batch_size) ):
            cbatch_size = min(batch_size, n_images - input_i*batch_size)
            batch_data = torch.FloatTensor(images[batch_size*input_i:batch_size*(input_i+1)])
            batch_data = batch_data.cuda()
            inner_outputs = temp_model1(batch_data).cpu().detach().numpy()

            n_neurons = inner_outputs.shape[1]

            nbatches = math.ceil(float(n_neurons)/sample_batch_size)
            for nt in range(nbatches):
                l_h_t = []
                csample_batch_size = min(sample_batch_size, n_neurons - nt*sample_batch_size)
                for neuron in range(csample_batch_size):
                    if len(inner_outputs.shape) == 4:
                        h_t = np.tile(inner_outputs, (n_samples, 1, 1, 1))
                    else:
                        h_t = np.tile(inner_outputs, (n_samples, 1))

                    for i,v in enumerate(vs):
                        h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size,:,:] = v
                    l_h_t.append(h_t)
                f_h_t = np.concatenate(l_h_t, axis=0)

                f_h_t_t = torch.FloatTensor(f_h_t).cuda()
                fps = temp_model2( f_h_t_t ).cpu().detach().numpy()
                for neuron in range(csample_batch_size):
                    tps = fps[neuron*n_samples*cbatch_size:(neuron+1)*n_samples*cbatch_size]

                    for img_i in range(cbatch_size):
                        img_name = (labels[img_i + batch_size*input_i], img_i + batch_size*input_i)
                        ps_key= (img_name, '{0}_{1}'.format(children[layer_i].__class__.__name__, layer_i), neuron+nt*sample_batch_size)
                        ps = [tps[ img_i +cbatch_size*_] for _ in range(n_samples)]
                        ps = np.asarray(ps)
                        ps = ps.T
                        all_ps[ps_key] = np.copy(ps)
                
                del f_h_t_t
            del batch_data, inner_outputs
            torch.cuda.empty_cache()

        del temp_model1, temp_model2
    return all_ps, sample_layers


def find_min_max(model_name, all_ps, sample_layers, cut_val=20, top_k=10):
    max_ps = {}
    max_vals = []
    n_classes = 0
    n_samples = 0
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff

        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]

        max_vals.append(val)
        max_ps[k] = (ml, val)
    
    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    
    min_ps = {}
    min_vals = []
    n_imgs = len(imgs)
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            vs.append(v)
            ls.append(l)
            if not ( l in vdict.keys() ):
                vdict[l] = [v]
            else:
                vdict[l].append(v)
        ml = max(set(ls), key=ls.count)

        fvs = []
        # does not count when l not equal ml
        for img in sorted(imgs):
            img_l = int(img[0])
            nk = (img, k[0], k[1])
            l = max_ps[nk][0]
            v = max_ps[nk][1]
            if l != ml:
                continue
            fvs.append(v)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            min_vals.append(np.mean(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)
    
    keys = min_ps.keys()
    keys = []
    for k in min_ps.keys():
        if min_ps[k][1] >= int(n_imgs * 0.6):
            keys.append(k)
    if len(keys) == 0:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs * 0.1):
                keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    if Print_Level > 0:
        print('n samples', n_samples, 'n class', n_classes, 'n_imgs', n_imgs)

    neuron_dict = {}
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    labels = {}
    allns = 0
    max_sampling_val = -np.inf

    # last layers
    labels = {}
    for i in range(len(sorted_key)):
        k = sorted_key[-i-1]
        layer = k[0]
        neuron = k[1]
        label = min_ps[k][0]
        if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
            continue

        if label not in labels.keys():
            labels[label] = 0
        # if int(layer.split('_')[-1]) == sample_layers[-1] and labels[label] < 1:
        if True:
            labels[label] += 1

            if min_ps[k][2] > max_sampling_val:
                max_sampling_val = min_ps[k][2]
            if Print_Level > 0:
                print(i, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                if Print_Level > 1:
                    print(min_ps[k][3])
            allns += 1
            neuron_dict[model_name].append( (layer, neuron, min_ps[k][0]) )
        if allns >= top_k:
            break

    return neuron_dict, max_sampling_val


def read_all_ps(model_name, all_ps, sample_layers, top_k=10, cut_val=20):
    return find_min_max(model_name, all_ps, sample_layers,  cut_val, top_k=top_k)


def loss_fn(inner_outputs_b, inner_outputs_a, logits, i_images, o_images, delta, bias, neuron, tlabel):
    neuron_mask = torch.zeros([1, inner_outputs_a.shape[1], 1, 1]).cuda()
    neuron_mask[:,neuron,:,:] = 1
    vloss1     = torch.sum(inner_outputs_b * neuron_mask)/torch.sum(neuron_mask)
    vloss2     = torch.sum(inner_outputs_b * (1-neuron_mask))/torch.sum(1-neuron_mask)
    relu_loss1 = torch.sum(inner_outputs_a * neuron_mask)/torch.sum(neuron_mask)
    relu_loss2 = torch.sum(inner_outputs_a * (1-neuron_mask))/torch.sum(1-neuron_mask)

    vloss3     = torch.sum(inner_outputs_b * torch.lt(inner_outputs_b, 0) )/torch.sum(1-neuron_mask)

    loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2
    ssim_loss = msssim(i_images, o_images)
    ssim_cond = torch.gt(ssim_loss, 0.7)
    ssim_add_loss = torch.where(ssim_cond, 0.0 * ssim_loss, re_mask_weight * ssim_loss)
    loss -= ssim_add_loss
    logits_loss = torch.sum(logits[:,tlabel]) - 0.001 * ( torch.sum(logits[:,:tlabel]) + torch.sum(logits[:,tlabel:]) )
    loss += - 2e2 * logits_loss

    loss += 1e2 * (torch.sum(torch.abs(delta)) + torch.sum(torch.abs(bias)))

    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, ssim_loss, logits_loss


def reverse_engineer(model_type, model, children, oimages, olabels, weights_file, Troj_Layer, Troj_Neuron, Troj_Label, Troj_size, re_epochs):
    
    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook
    
    after_bn3 = []
    def get_after_bn3():
        def hook(model, input, output):
            for ip in output:
                after_bn3.append( ip.clone() )
        return hook
    
    after_iden = []
    def get_after_iden():
        def hook(model, input, output):
            for ip in output:
                after_iden.append( ip.clone() )
        return hook

    after_bns = []
    def get_after_bns():
        def hook(model, input, output):
            for ip in output:
                after_bns.append( ip.clone() )
        return hook


    re_batch_size = config['re_batch_size']
    if model_type in ['ResNet', 'PreActResNet']:
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)
    if re_batch_size > len(oimages):
        re_batch_size = len(oimages)

    handles = []
    if model_type == 'VGG':
        tmodule1 = children[Troj_Layer]
        handle = tmodule1.register_forward_hook(get_after_bns())
        handles.append(handle)
    elif model_type in ['ResNet', 'PreActResNet']:
        if resnet_sample_resblock:
            children_modules = list(children[Troj_Layer].children())
        else:
            children_modules = list(list(children[Troj_Layer].children())[-1].children())
        # print(len(children_modules), children_modules)
        last_bn_id = 0
        has_downsample = False
        i = 0
        for children_module in children_modules:
            if children_module.__class__.__name__ == 'BatchNorm2d':
                last_bn_id = i
            if children_module.__class__.__name__ == 'Sequential':
                has_downsample = True
            i += 1
        # print('last bn id', last_bn_id, 'has_downsample', has_downsample)
        bn3_module = children_modules[last_bn_id]
        handle = bn3_module.register_forward_hook(get_after_bn3())
        handles.append(handle)
        if has_downsample:
            iden_module = children_modules[-1]
            handle = iden_module.register_forward_hook(get_after_iden())
            handles.append(handle)
        else:
            iden_module = children_modules[0]
            handle = iden_module.register_forward_hook(get_before_block())
            handles.append(handle)

    # print('Target Layer', Troj_Layer, children[Troj_Layer], 'Neuron', Troj_Neuron, 'Target Label', Troj_Label)

    # Initialization of parameters
    kdim = 32
    delta_init = torch.rand((1, 27, kdim, kdim))
    bias_init = torch.rand((1, 3, kdim, kdim))

    delta = delta_init.cuda()
    bias = bias_init.cuda()
    delta.requires_grad = True
    bias.requires_grad = True

    optimizer = torch.optim.Adam(params=[delta, bias], lr=1e-2, betas=(0.5, 0.9))

    # print('before optimizing',)
    for e in range(re_epochs):
        facc = 0
        flogits = []
        # shuffle
        p = np.random.permutation(oimages.shape[0])
        images = oimages[p]
        labels = olabels[p]
        for i in range( math.ceil(float(len(images))/re_batch_size) ):
            cre_batch_size = min(len(images) - re_batch_size * i, re_batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            after_bn3.clear()
            before_block.clear()
            after_iden.clear()
            after_bns.clear()

            batch_data = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)])
            batch_data = batch_data.cuda()

            clamp = [True, False][0]
            # Batch data is clipped in [l_bounds, h_bounds]
            if clamp:
                # Clip parameters into [mean-std, mean+std]
                # _wf = 0.15
                # fw_min = fw_mean - _wf * fw_std
                # fw_max = fw_mean + _wf * fw_std
                # fb_min = fb_mean - _wf * fb_std
                # fb_max = fb_mean + _wf * fb_std

                # clamp_delta = torch.clamp(delta, fw_min.cuda(), fw_max.cuda())
                # clamp_bias  = torch.clamp(bias, fb_min.cuda(), fb_max.cuda())
                clamp_delta = delta
                clamp_bias = bias

                # Preprocessing and deprocessing
                in_data = tensor_transform(DEP(batch_data), clamp_delta, clamp_bias)
                # save_image(in_data[:10], 'tmp.png', nrow=10)
                in_data = PRE(in_data)

                batch_r = torch.clamp(in_data[:, 0, :, :], min=l_bounds_tensor[0], max=h_bounds_tensor[0])
                batch_g = torch.clamp(in_data[:, 1, :, :], min=l_bounds_tensor[1], max=h_bounds_tensor[1])
                batch_b = torch.clamp(in_data[:, 2, :, :], min=l_bounds_tensor[2], max=h_bounds_tensor[2])
                in_data = torch.stack([batch_r, batch_g, batch_b], dim=1)
            
            logits = model(in_data)
            logits_np = logits.cpu().detach().numpy()
            
            if model_type == 'VGG':
                inner_outputs_b = torch.stack(after_bns, 0)
                inner_outputs_a = F.relu(inner_outputs_b)
            elif model_type in ['ResNet', 'PreActResNet']:
                after_bn3_t = torch.stack(after_bn3, 0)
                iden = None
                if len(before_block) > 0:
                    iden = before_block[0]
                else:
                    after_iden_t = torch.stack(after_iden, 0)
                    iden = after_iden_t
                inner_outputs_b = iden + after_bn3_t
                # print(iden.shape, after_bn3_t.shape, iden.dtype, after_bn3_t.dtype)
                inner_outputs_a = F.relu(inner_outputs_b)
            
            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, ssim_add_loss, logits_loss\
                    = loss_fn(inner_outputs_b, inner_outputs_a, logits, batch_data, in_data, delta, bias, Troj_Neuron, int(Troj_Label))
            
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()
        
        flogits = np.concatenate(flogits, axis=0)
        preds = np.argmax(flogits, axis=1)

        # do not change Troj_Label
        # Troj_Label2 = np.argmax(np.bincount(preds))
        Troj_Label2 = Troj_Label

        facc = np.sum(preds == Troj_Label2) / float(preds.shape[0])

        if e % 10 == 0 and Print_Level > 0:
            print(e, 'loss', loss.cpu().detach().numpy(), 'acc {:.4f}'.format(facc),'target label', int(Troj_Label), int(Troj_Label2), 'logits_loss', logits_loss.cpu().detach().numpy(),\
                    'vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    'ssim_add_loss', ssim_add_loss.cpu().detach().numpy())
            print('labels', flogits[:5, :10])
            print('logits', np.argmax(flogits, axis=1))

    rdelta = delta.cpu().detach().numpy()
    rbias = bias.cpu().detach().numpy()
    rssim = ssim(in_data, batch_data).cpu().detach().numpy()
    adv = in_data.cpu().detach().numpy()

    # cleaning up
    for handle in handles:
        handle.remove()

    return facc, adv, rdelta, rbias, rssim, Troj_Label2


def re_mask(model_type, model, neuron_dict, children, images, labels, scratch_dirpath, re_epochs):
    validated_results = []
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, samp_label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_Layer = int(Troj_Layer.split('_')[1])

            RE_img = os.path.join(scratch_dirpath, 'imgs', '{0}_model_{1}_{2}_{3}_{4}.png'.format(    weights_file.split('/')[-1].split('\.')[0], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            RE_delta = os.path.join(scratch_dirpath, 'deltas', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-1].split('\.')[0], Troj_Layer, Troj_Neuron, Troj_size, samp_label))
            
            max_acc = 0
            max_results = []
            for i in range(mask_multi_start):
                acc, rimg, rdelta, rbias, rssim, optz_label = reverse_engineer(model_type, model, children, images, labels, weights_file, Troj_Layer, Troj_Neuron, samp_label, Troj_size, re_epochs)

                # clear cache
                torch.cuda.empty_cache()

                if Print_Level > 0:
                    print('RE filter', Troj_Layer, Troj_Neuron, 'Label', optz_label, 'RE acc', acc)
                if acc > max_acc:
                    max_acc = acc
                    max_results = (rimg, rdelta, rbias, rssim, optz_label, RE_img, RE_delta, samp_label, acc)
            if max_acc >= reasr_bound - 0.2:
                validated_results.append( max_results )

    return validated_results


def numpy_transform(inputs, grids, channel_bias, threshold=0.01):  # 0.02
    # inputs.shape = [batch_size, 3, 32, 32]
    # grid.shape = [1, 27, kdim, kdim]
    # channel_bias.shape = [1, 3, kdim, kdim]
    inputs = DEP(torch.FloatTensor(inputs.copy()))
    grids = torch.FloatTensor(grids)
    channel_bias = torch.FloatTensor(channel_bias)

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

    out = PRE(out).numpy()
    return out


def tensor_transform(inputs, grids, channel_bias, threshold=0.01):
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


def test(model, model_type, test_xs, result, scratch_dirpath):
    
    re_batch_size = config['re_batch_size']
    if model_type in ['ResNet', 'PreActResNet']:
        re_batch_size = max(re_batch_size // 4, 1)
    if model_type == 'VGG':
        re_batch_size = max(re_batch_size // 4, 1)
    if re_batch_size > len(test_xs):
        re_batch_size = len(test_xs)

    clean_images = test_xs

    rimg, rdelta, rbias, rssim, tlabel = result[:5]
    t_images = numpy_transform(clean_images, rdelta, rbias)

    rt_images = t_images
    if Print_Level > 0:
        print(np.amin(rt_images), np.amax(rt_images))
    
    yt = np.zeros(len(rt_images)).astype(np.int32) + tlabel
    fpreds = []
    for i in range( math.ceil(float(len(rt_images))/re_batch_size) ):
        batch_data = torch.FloatTensor(rt_images[re_batch_size*i:re_batch_size*(i+1)])
        batch_data = batch_data.cuda()
        preds = model(batch_data)
        fpreds.append(preds.cpu().detach().numpy())
    fpreds = np.concatenate(fpreds)

    preds = np.argmax(fpreds, axis=1) 
    # print(preds)
    score = float(np.sum(tlabel == preds))/float(yt.shape[0])
    top5_preds = np.argsort(fpreds, axis=1)[:, -5:]
    top5_acc = np.sum(np.any(top5_preds == yt[:, np.newaxis],axis=1)) / float(yt.shape[0])
    # print('label', tlabel, 'score', score)
    return score, top5_acc


def load_samples(examples_dirpath):
    dataset = pickle.load(open(examples_dirpath, 'rb'), encoding='bytes')
    fxs, fys = dataset['x_val'], dataset['y_val']
    fxs, fys = np.uint8(fxs), np.asarray(fys).astype(np.int)
    assert(fxs.shape[0] == 100)
    assert(fys.shape[0] == 100)

    # print('number of seed images', fxs.shape, fys.shape)
    return fxs, fys


def main(model_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    start = time.time()

    # print('model_filepath = {}'.format(model_filepath))
    # print('scratch_dirpath = {}'.format(scratch_dirpath))
    # print('examples_dirpath = {}'.format(examples_dirpath))

    # create dirs
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'imgs')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'masks')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'temps')))
    os.system('mkdir -p {0}'.format(os.path.join(scratch_dirpath, 'deltas')))

    # remove previous results
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'imgs')))
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'masks')))
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'temps')))
    os.system('rm -r {0}/*'.format(os.path.join(scratch_dirpath, 'deltas')))
    
    model = torch.load(model_filepath).cuda()
    model = model.module
    # print(model)

    target_layers = []
    model_type = model.__class__.__name__
    children = list(model.children())
    num_classes = list(model.named_modules())[-1][1].out_features

    # print('num classes', num_classes)

    model.eval()

    # children = list(model.children())
    # for c in children:
    #     print('child', c)

    if model_type == 'VGG':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        if args.dataset == 'imagenet':
            children.insert(-7, torch.nn.Flatten())
        else:
            children.insert(-1, torch.nn.Flatten())
        # TODO: Select BN or Conv2d
        if args.dataset == 'imagenet':
            target_layers = ['Conv2d']
        else:
            target_layers = ['BatchNorm2d']
            
    elif model_type == 'ResNet':
        children = list(model.children())
        if resnet_sample_resblock:
            nchildren = []
            for c in children:
                if c.__class__.__name__ == 'Sequential':
                    nchildren += list(c.children())
                else:
                    nchildren.append(c)
            children = nchildren
        if args.dataset != 'imagenet':
            children.insert(-1, torch.nn.AvgPool2d(4))
        children.insert(-1, torch.nn.Flatten())
        if resnet_sample_resblock:
            target_layers = ['Bottleneck', 'BatchNorm2d']
        else:
            target_layers = ['Sequential']
    else:
        # print('other model', model_type)
        sys.exit()
    
    # for c in children:
    #     print('child', c)
    # assert(0)

    fxs, fys = load_samples(examples_dirpath)

    test_xs = fxs.copy()
    test_ys = fys.copy()

    fxs = fxs / 255.
    fxs = np.transpose(fxs, (0, 3, 1, 2))
    fxs = ( fxs - mean.reshape((1, 3, 1, 1)) ) / std.reshape((1, 3, 1, 1))

    test_xs = test_xs / 255.
    test_xs = np.transpose(test_xs, (0, 3, 1, 2))
    test_xs = ( test_xs - mean.reshape((1, 3, 1, 1)) ) / std.reshape((1, 3, 1, 1))
    
    # print('number of seed images', len(fys), fys.shape, 'image min val', np.amin(fxs), 'max val', np.amax(fxs))

    re_batch_size = 20
    fpreds = []
    for i in range( math.ceil(float(len(fxs))/re_batch_size) ):
        batch_data = torch.FloatTensor(fxs[re_batch_size*i:re_batch_size*(i+1)])
        batch_data = batch_data.cuda()
        preds = model(batch_data)
        fpreds.append(preds.cpu().detach().numpy())
    fpreds = np.concatenate(fpreds)

    preds = np.argmax(fpreds, axis=1) 
    top5_preds = np.argsort(fpreds, axis=1)[:,-5:]
    # print(preds, len(preds))
    # print(fys, len(fys))
    # print('ACC:', np.sum(preds == fys))
    saved_images = deprocess(fxs)
    # print('saved_images', saved_images.shape)
    for i in range(4):
        cv2.imwrite('{0}/test_{1}.png'.format(os.path.join(scratch_dirpath, 'imgs'), i), np.transpose(saved_images[i], (1,2,0)) )

    sample_xs = np.array(fxs[:10])
    sample_ys = np.array(fys[:10])

    # print(sample_ys, sample_ys.shape, sample_xs.shape)

    optz_xs = np.array(fxs[:100])
    optz_ys = np.array(fys[:100])
    # print(optz_ys, optz_ys.shape, optz_xs.shape)

    if Print_Level > 0:
        print('# samples for RE', len(fys), fys)
        print('# samples for sample', len(sample_ys), sample_ys)

    neuron_dict = {}
    sampling_val = 0

    maxes = check_values(sample_xs, sample_ys, model, children, target_layers)
    torch.cuda.empty_cache()
    all_ps, sample_layers = sample_neuron(sample_xs, sample_ys, model, children, target_layers, model_type, maxes, False)
    torch.cuda.empty_cache()
    neuron_dict, sampling_val = read_all_ps(model_filepath, all_ps, sample_layers, top_k = top_n_neurons)
    # print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', neuron_dict)

    sample_end = time.time()

    results = re_mask(model_type, model, neuron_dict, children, fxs, fys, scratch_dirpath, re_epochs)
    reasr_info = []
    reasrs = []
    if len(results) > 0:
        reasrs = []
        for result in results:
            if len(result) == 0:
                continue
            top1_acc, top5_acc = test(model, model_type, test_xs, result, scratch_dirpath)
            reasr = top1_acc
            reasrs.append(reasr)
            adv, rdelta, rbias, rssim, optz_label, RE_img, RE_delta, samp_label, acc = result
            if reasr > 0.01:
                saved_images = deprocess(adv)
                # print('saved_images', saved_images.shape)
                for i in range(4):
                    _img = np.transpose(saved_images[i], (1,2,0))
                    cv2.imwrite('{0}_{1}.png'.format(RE_img[:-4], i), cv2.cvtColor(_img, cv2.COLOR_RGB2BGR))
                with open(RE_delta, 'wb') as f:
                    pickle.dump([rdelta, rbias], f)
            reasr_info.append([reasr, 'filter', str(optz_label), str(samp_label), RE_img, RE_delta, rssim, acc])

    optm_end = time.time()
    if len(reasrs) > 0:
        freasr = max(reasrs)
        f_id = reasrs.index(freasr)
    else:
        freasr = 0
        f_id = 0
    max_reasr = 0
    for i in range(len(reasr_info)):
        print('reasr info {0}'.format( ' '.join([str(_) for _ in reasr_info[i]]) ))
        with open(logfile, 'a') as f:
            f.write('reasr info {0}\n'.format( ' '.join([str(_) for _ in reasr_info[i]]) ) )
        reasr = reasr_info[i][0]
        if reasr > max_reasr :
            max_reasr = reasr
    print('{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(\
            model_filepath, model_type, 'filter', freasr, 'sampling val', sampling_val, 'time', sample_end - start, optm_end - sample_end,) )
    if max_reasr >= 0.88:
        output = 1 - 1e-1
    else:
        output =     1e-1
    print('max reasr', max_reasr, 'output', output)

    with open(logfile, 'a') as f:
        f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(\
                model_filepath, model_type, 'mode', max_reasr, output, 'time', sample_end - start, optm_end - sample_end) )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='vgg11', help='network structure')
    parser.add_argument('--attack', default='wanet', help='attack type')

    args = parser.parse_args()
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'
    scratch_dirpath = f'scratch'
    examples_dirpath = f'samples/{args.dataset}_100'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dataset == 'cifar10':
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
    elif args.dataset == 'gtsrb':
        mean = np.array([0.3337, 0.3064, 0.3171])
        std = np.array([0.2672, 0.2564, 0.2629])
    
    l_bounds = np.asarray([(0.0 - mean[0]) / std[0], (0.0 - mean[1]) / std[1], (0.0 - mean[2]) / std[2]])
    h_bounds = np.asarray([(1.0 - mean[0]) / std[0], (1.0 - mean[1]) / std[1], (1.0 - mean[2]) / std[2]])
    l_bounds_tensor = torch.FloatTensor(l_bounds).cuda()
    h_bounds_tensor = torch.FloatTensor(h_bounds).cuda()
    
    PRE, DEP = get_norm(args.dataset)

    logfile = 'result_abs_beagle_transform_complex.txt'
    main(model_filepath, scratch_dirpath, examples_dirpath)
