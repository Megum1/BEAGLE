import os
import time
import argparse
import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

import lpips

from util import *
from stylegan import *
from invert_func import *


def epsilon():
    return 1e-7


def TV_loss(x):
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def avg_smooth_loss(x):
    avgpool = nn.AvgPool2d(2)
    smooth = avgpool(x)
    smooth = transforms.Resize(x.size(3))(smooth)
    MSE = nn.MSELoss()
    return MSE(smooth, x)


def pre_transform(images, labels):
    return torch.Tensor(images / 255.).permute(0, 3, 1, 2), torch.from_numpy(labels)


# TODO: Add some data augmentation
def augment(images):
    t_images = images
    return t_images


def shuffle(x, y):
    assert (x.shape[0] == y.shape[0])
    index = torch.randperm(x.shape[0])
    return x[index, :, :, :], y[index]


def load_batch(x_clean, y_clean, x_poison, y_poison):
    assert (x_clean.shape[0] == 100)
    assert (y_clean.shape[0] == 100)
    assert (x_poison.shape[0] == 10)
    assert (y_poison.shape[0] == 10)

    batches = []
    batch_size = int(x_poison.shape[0])
    steps = int(x_clean.shape[0] / batch_size)
    x_clean, y_clean = shuffle(x_clean, y_clean)
    for i in range(steps):
        bx_clean = x_clean[i * batch_size:(i + 1) * batch_size]
        by_clean = y_clean[i * batch_size:(i + 1) * batch_size]

        # Do not shuffle for sample-specific attacks
        bx_poison, by_poison = x_poison, y_poison
        batches.append([bx_clean, by_clean, bx_poison, by_poison])

    return batches


def forensic(args, preeval=True):
    # Load attacked model
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'
    model = torch.load(model_filepath, map_location='cpu').module
    model = model.to(DEVICE)
    model.eval()

    # Load clean and poisoned samples
    # TODO: Remember to provide some clean and poisoned samples
    # Poison source images do not need to be the same as clean image
    saveset = pickle.load(open(f'samples/{args.dataset}_{args.network}_{args.attack}', 'rb'), encoding='bytes')
    x_clean, y_clean = saveset['x_val'], saveset['y_val']
    x_poison, y_poison = saveset['x_poi'], saveset['y_poi']
    batch_size = x_poison.shape[0]

    # Convert to tensors
    x_clean, y_clean = pre_transform(x_clean, y_clean)
    x_poison, y_poison = pre_transform(x_poison, y_poison)

    # Preprocessing and deprocessing
    preprocess, deprocess = get_norm(args.dataset)

    # Trigger injection
    if args.attack in ['inputaware', 'dynamic']:
        backdoor_preprocess = get_norm(args.dataset)[0]
        backdoor = get_backdoor(args.attack, x_clean.shape[2:], normalize=backdoor_preprocess, device=DEVICE)
    elif args.attack == 'dfst':
        backdoor = None
        x_inj = saveset['x_inj']
        x_inj = torch.Tensor(x_inj / 255.).permute(0, 3, 1, 2).to(DEVICE)
    else:
        backdoor = get_backdoor(args.attack, x_clean.shape[2:], DEVICE)

    # Pre-evaluation
    if preeval:
        with torch.no_grad():
            n_sample = 0
            n_acc, n_asr = 0, 0
            batches = load_batch(x_clean, y_clean, x_poison, y_poison)
            for (bx_clean, by_clean, bx_poison, by_poison) in batches:

                if args.attack not in ['inputaware', 'wanet', 'dynamic', 'dfst']:
                    bx_poison = backdoor.inject(bx_poison)
                if args.attack == 'dfst':
                    bx_poison = x_inj.clone()
                
                bx_clean, by_clean = bx_clean.to(DEVICE), by_clean.to(DEVICE)
                bx_poison, by_poison = bx_poison.to(DEVICE), by_poison.to(DEVICE)

                bx_clean, bx_poison = preprocess(bx_clean), preprocess(bx_poison)
                by_poison = by_poison * 0 + args.target

                if args.attack in ['inputaware', 'wanet', 'dynamic']:
                    bx_poison = backdoor.inject(bx_poison)

                out_clean, out_poison = model(bx_clean), model(bx_poison)
                pred_clean, pred_poison = out_clean.max(dim=1)[1], out_poison.max(dim=1)[1]

                n_sample += bx_clean.size(0)
                n_acc += (pred_clean == by_clean).sum().item()
                n_asr += (pred_poison == by_poison).sum().item()

        acc = n_acc / n_sample
        asr = n_asr / n_sample
        print(f'Pre-evaluation of samples -> ACC: {acc}, ASR: {asr}')
    
    # Load pre-trained stylegan
    stylegan = StyleGAN(DEVICE)

    # Initialization of parameters
    mapping_labels = nn.functional.one_hot(torch.tensor(np.arange(10)), num_classes=10).float().to(DEVICE)
    latent_w_mean = stylegan.get_w_mean(mapping_labels)

    latent_input = latent_w_mean.clone()
    latent_input = latent_input.unsqueeze(1).repeat(1, stylegan.num_layers, 1)
    latent_input = latent_input.repeat(batch_size, 1, 1)
    latent_input.requires_grad_(True)

    if args.func == 'mask':
        if args.func_option == 'binomial':
            # Handle patch triggers
            # Binomial distribution
            # Only optimize mask
            # Pattern is derived from one part of poisoned image
            mask_init = np.random.random((batch_size, 1, 32, 32)) * 1e-2
            mask_init = np.arctanh((mask_init - 0.5) * (2 - epsilon()))
            mask = torch.tensor(mask_init, dtype=torch.float, requires_grad=True, device=DEVICE)
            mask_init = torch.tensor(mask_init * 0., dtype=torch.float).to(DEVICE)

            # Define the optimization
            optim_troj = torch.optim.Adam(params=[mask], lr=1e-1, betas=(0.5, 0.9))
            init_lr = 1e-1
            optim_gan = torch.optim.Adam(params=[latent_input], lr=init_lr)
        
        elif args.func_option == 'uniform':
            # Handle blend triggers
            # Uniform distribution
            # Optimize mask in [0.2, 0.4]
            # Optimize pattern according to average poisoned image

            delta = 0.3

            latent_patn = latent_w_mean.clone()
            latent_patn = latent_patn.unsqueeze(1).repeat(1, stylegan.num_layers, 1)
            latent_patn.requires_grad_(True)

            # Define the optimization
            optim_troj = torch.optim.Adam(params=[latent_patn], lr=1e-1)
            init_lr = 1e-1
            optim_gan = torch.optim.Adam(params=[latent_input], lr=init_lr)
    
    elif args.func == 'transform':
        if args.func_option == 'simple':
            kdim, bdim = 32, 1
            # weights_init = (np.random.random((1, 3, kdim, kdim)) - 0.5) * 1e-2
            # bias_init = (np.random.random((1, 3, bdim, bdim)) - 0.5) * 1e-2
            weights_init = np.zeros((1, 3, kdim, kdim))
            bias_init = np.zeros((1, 3, bdim, bdim))

        elif args.func_option == 'complex':
            kdim, bdim = 32, 1
            # weights_init = (np.random.random((1, 27, kdim, kdim)) - 0.5) * 1e-2
            # bias_init = (np.random.random((1, 3, bdim, bdim)) - 0.5) * 1e-2
            weights_init = np.zeros((1, 27, kdim, kdim))
            bias_init = np.zeros((1, 3, bdim, bdim))
        
        weights = torch.tensor(weights_init, dtype=torch.float, requires_grad=True, device=DEVICE)
        bias = torch.tensor(bias_init, dtype=torch.float, requires_grad=True, device=DEVICE)

        weights_init = torch.tensor(weights_init * 0., dtype=torch.float, device=DEVICE)
        bias_init = torch.tensor(bias_init * 0., dtype=torch.float, device=DEVICE)

        # Define the optimization
        optim_troj = torch.optim.Adam(params=[weights, bias], lr=1e-2, betas=(0.5, 0.9))
        init_lr = 1e-1
        optim_gan = torch.optim.Adam(params=[latent_input], lr=init_lr)
    
    # Loss terms
    CE = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    from torchgeometry.losses import SSIM
    ssim = SSIM(window_size=5, reduction='mean')
    # LPIPS
    percept = lpips.LPIPS(net='vgg').to(DEVICE)

    # Save for comparison
    inject_best =  1 / epsilon()
    remove_best =  1 / epsilon()
    
    steps = args.epochs * (x_clean.shape[0] // x_poison.shape[0])
    step = 0

    # Acquire poison version before going into "epochs"
    # Since dynamic backdoor changed in each epoch
    with torch.no_grad():
        if args.attack in ['inputaware', 'wanet', 'dynamic']:
            # Get the real injected backdoor
            x_poison_poi = deprocess(backdoor.inject(preprocess(x_poison.to(DEVICE))))
        elif args.attack == 'dfst':
            x_poison_poi = x_inj.clone()
        else:
            # Get the real injected backdoor
            x_poison_poi = backdoor.inject(x_poison).to(DEVICE)
    
    # Start optimization
    for epoch in range(args.epochs):
        n_sample = 0
        n_acc, n_asr, n_cycasr = 0, 0, 0
        CE_LOSS = 0
        InP_LOSS = 0
        TrP_LOSS = 0
        TV_LOSS = 0

        # Save for comparison
        x_inject, x_invert = [], []
        x_source, x_remove = [], []

        batches = load_batch(x_clean, y_clean, x_poison, y_poison)
        for (bx_clean, by_clean, bx_poison_src, by_poison_src) in batches:
            # For each step
            step += 1

            if args.attack in ['inputaware', 'wanet', 'dynamic']:
                # Get the real injected backdoor
                bx_clean, bx_poison_src = bx_clean.to(DEVICE), bx_poison_src.to(DEVICE)
                inject_bx = deprocess(backdoor.inject(preprocess(bx_clean)))
                # bx_poison = deprocess(backdoor.inject(preprocess(bx_poison_src)))
                bx_poison = x_poison_poi.clone()
            elif args.attack == 'dfst':
                bx_clean, bx_poison_src = bx_clean.to(DEVICE), bx_poison_src.to(DEVICE)
                inject_bx = bx_clean.clone()
                bx_poison = x_inj.clone()
            else:
                # Get the real injected backdoor
                inject_bx = backdoor.inject(bx_clean).to(DEVICE)
                bx_clean = bx_clean.to(DEVICE)
                # bx_poison = backdoor.inject(bx_poison_src).to(DEVICE)
                bx_poison = x_poison_poi.clone()
                bx_poison_src = bx_poison_src.to(DEVICE)
            
            by_clean, by_poison_src = by_clean.to(DEVICE), by_poison_src.to(DEVICE)
            mini_size = bx_clean.size(0)

            ##############################################################
            # StyleGAN operation
            t = step / steps
            lr_i = get_lr(t, init_lr)
            optim_gan.param_groups[0]['lr'] = lr_i
            if args.func == 'mask' and args.func_option == 'uniform':
                optim_troj.param_groups[0]['lr'] = lr_i
            
            RESOLUTION = 32
            special_noises = []
            gen_bx_clean = stylegan.generator(latent_input, special_noises=special_noises)  # pixel in [0, 1]
            gen_bx_clean = transforms.Resize(RESOLUTION)(gen_bx_clean)  # Shape: batch_size x 3x32x32

            if args.func == 'mask':
                if args.func_option == 'binomial':
                    trigger_mask = mask_process(mask)
                    gen_bx_poison = torch.clamp(bx_clean * (1 - trigger_mask) + bx_poison * trigger_mask , 0., 1.)

                    # Cyclic operation
                    cyc_bx_poison = torch.clamp(gen_bx_clean * (1 - trigger_mask) + bx_poison * trigger_mask , 0., 1.)
                
                elif args.func_option == 'uniform':
                    patn = stylegan.generator(latent_patn, special_noises=special_noises)  # pixel in [0, 1]
                    patn = transforms.Resize(RESOLUTION)(patn)  # Shape: 1x3x32x32
                    gen_trigger = patn[0]
                    # TODO:
                    # delta = torch.clamp(delta, 0.2, 0.4)
                    remove_bx_clean, gen_bx_poison = remove_trigger(bx_poison, gen_trigger, delta), attach_trigger(bx_clean, gen_trigger, delta)
                    # whole image or channel
                    remove_mean_im, remove_mean_ch = torch.mean(remove_bx_clean, dim=0, keepdim=True), torch.mean(remove_bx_clean, dim=[0, 2, 3], keepdim=True)
                    remove = torch.clamp(remove_bx_clean - remove_mean_im + remove_mean_ch, 0., 1.)
                    rmv_mean, rmv_std = torch.mean(remove, dim=[0, 2, 3], keepdim=True), torch.std(remove, dim=[0, 2, 3], keepdim=True)
                    cln_mean, cln_std = torch.mean(x_clean.to(DEVICE), dim=[0, 2, 3], keepdim=True), torch.std(x_clean.to(DEVICE), dim=[0, 2, 3], keepdim=True)
                    remove_bx_clean = torch.clamp((remove - rmv_mean) / rmv_std * cln_std + cln_mean, 0., 1.)

                    # Cyclic operation
                    cyc_bx_poison = attach_trigger(gen_bx_clean, gen_trigger, delta)

            elif args.func == 'transform':
                if args.func_option == 'simple':
                    gen_bx_poison = simple_linear(bx_clean, weights, bias)
                    cyc_bx_poison = simple_linear(gen_bx_clean, weights, bias)
                elif args.func_option == 'complex':
                    gen_bx_poison = complex_linear(bx_clean, weights, bias)
                    cyc_bx_poison = complex_linear(gen_bx_clean, weights, bias)
            
            x_inject.append(inject_bx.detach().cpu())
            x_invert.append(gen_bx_poison.detach().cpu())
            x_source.append(bx_poison_src.detach().cpu())
            x_remove.append(gen_bx_clean.detach().cpu())

            process_gen_bx_clean, process_gen_bx_poison = preprocess(gen_bx_clean), preprocess(gen_bx_poison)

            gen_by_clean = by_poison_src
            gen_by_poison = by_clean * 0 + args.target
            gen_out_clean, gen_out_poison = model(process_gen_bx_clean), model(process_gen_bx_poison)

            process_cyc_bx_poison = preprocess(cyc_bx_poison)
            cyc_by_poison = gen_by_poison
            cyc_out_poison = model(process_cyc_bx_poison)
            
            ce_loss = CE(gen_out_poison, gen_by_poison) + CE(gen_out_clean, gen_by_clean) + CE(cyc_out_poison, cyc_by_poison)
            
            # Smoothing loss
            tv_loss = TV_loss(gen_bx_clean)
            TV = 1e-3

            # Reconstruction loss
            if args.func == 'mask':
                if args.func_option == 'binomial':
                    ref_bx_clean = gen_bx_clean * (1 - trigger_mask)
                    ref_bx_poison = bx_poison * (1 - trigger_mask)
                    ip_loss = percept.forward(ref_bx_clean, ref_bx_poison, normalize=True).sum()
                    ip_loss += mini_size * L1(ref_bx_clean, ref_bx_poison)
                    IP = 1e2

                    # TODO: TP loss is reg loss of mask
                    tp_loss = mini_size * L1(trigger_mask, mask_init)
                    TP = 1e1
                
                elif args.func_option == 'uniform':
                    ip_loss = percept.forward(gen_bx_clean, remove_bx_clean, normalize=True).sum()
                    
                    trigger_mean = extract_trigger(bx_poison, gen_bx_clean, delta)
                    tp_loss = percept.forward(gen_trigger, trigger_mean, normalize=True).sum()
                    
                    IP = 1e2
                    TP = 1e2
            
            elif args.func == 'transform':
                # Reg loss of weights
                reg_loss = L2(weights, weights_init) + L2(bias, bias_init)

                cln_mean, cln_std = torch.mean(bx_clean.to(DEVICE), dim=[0, 2, 3], keepdim=True), torch.std(bx_clean.to(DEVICE), dim=[0, 2, 3], keepdim=True)
                poi_mean, poi_std = torch.mean(bx_poison.to(DEVICE), dim=[2, 3], keepdim=True), torch.std(bx_poison.to(DEVICE), dim=[2, 3], keepdim=True)
                ref_bx_poison = torch.clamp((bx_poison - poi_mean) / poi_std * cln_std + cln_mean, 0., 1.)
                ref_bx_clean = torch.clamp((bx_clean - cln_mean) / cln_std * poi_std + poi_mean, 0., 1.)

                ip_loss = percept.forward(gen_bx_clean, ref_bx_poison, normalize=True).sum()
                IP = 1e2

                tp_loss = L2(gen_bx_poison, ref_bx_clean)
                tp_loss += L1(cyc_bx_poison, bx_poison)
                # Add reg loss into tp_loss
                tp_loss += 1e-1 * reg_loss
                TP = 1e2
            
            # Weighted loss
            loss = ce_loss + TV * tv_loss + IP * ip_loss + TP * tp_loss

            CE_LOSS += mini_size * ce_loss.item()
            InP_LOSS += ip_loss.item()
            TrP_LOSS += tp_loss.item()
            TV_LOSS += tv_loss.item()
            
            optim_gan.zero_grad()
            optim_troj.zero_grad()
            loss.backward()
            optim_gan.step()
            optim_troj.step()

            pred_clean, pred_poison, pred_cyc_poison = gen_out_clean.max(dim=1)[1], gen_out_poison.max(dim=1)[1], cyc_out_poison.max(dim=1)[1]

            n_sample += mini_size
            n_acc += (pred_clean == gen_by_clean).sum().item()
            n_asr += (pred_poison == gen_by_poison).sum().item()
            n_cycasr += (pred_cyc_poison == cyc_by_poison).sum().item()

        CE_LOSS = CE_LOSS / n_sample
        InP_LOSS = InP_LOSS / n_sample
        TrP_LOSS = TrP_LOSS / n_sample
        TV_LOSS = TV_LOSS / n_sample
        acc = n_acc / n_sample
        asr = n_asr / n_sample
        cycasr = n_cycasr / n_sample

        x_inject, x_invert = torch.cat(x_inject), torch.cat(x_invert)
        x_source, x_remove = torch.cat(x_source), torch.cat(x_remove)
        assert(x_inject.size() == x_inject.size())
        assert(x_source.size() == x_remove.size())
        inject_diff = (torch.mean(torch.abs(x_inject - x_invert)) * 255.).item()
        remove_diff = (torch.mean(torch.abs(x_source - x_remove)) * 255.).item()
        inject_best = min(inject_diff, inject_best)
        remove_best = min(remove_diff, remove_best)

        if (epoch + 1) % 10 == 0 and Print_Level > 0:
            print(f'Epoch: {epoch + 1}, CE_Loss: {round(CE_LOSS, 3)}, InP_Loss: {round(InP_LOSS, 3)}, TrP_Loss: {round(TrP_LOSS, 3)}, Smooth_Loss: {round(TV_LOSS, 3)}, gen_ACC: {round(acc, 2)}, gen_ASR: {round(asr, 2)}, cyc_ASR: {round(cycasr, 2)}, Inject_diff: {round(inject_diff, 2)}, Remove_diff: {round(remove_diff, 2)}')
            
            # Save latent vectors and other invertion-related parameters
            if args.func == 'mask':
                if args.func_option == 'binomial':
                    trigger_pattern = trigger_mask * bx_poison
                    save_param = [trigger_mask.detach().cpu().data, trigger_pattern.detach().cpu().data]
                elif args.func_option == 'uniform':
                    # save_param = [delta.detach().cpu().data, gen_trigger.detach().cpu().data]
                    save_param = [delta, gen_trigger.detach().cpu().data]
            elif args.func == 'transform':
                save_param = [weights.detach().cpu().data, bias.detach().cpu().data]
            
            # Save figures
            save_fig = [bx_clean, bx_poison, gen_bx_clean, bx_poison_src, gen_bx_poison, inject_bx]

            if args.func == 'mask' and args.func == 'uniform':
                show_triggers = gen_trigger
                mean_triggers = torch.clamp(torch.mean(bx_poison, dim=0, keepdim=True), 0., 1.)
                gen_mean_triggers = torch.clamp(torch.mean(gen_bx_poison, dim=0, keepdim=True), 0., 1.)
                save_fig.append(show_triggers)
                save_fig.append(mean_triggers)
                save_fig.append(gen_mean_triggers)
            
            else:
                diff_gen_clean = torch.clamp(torch.abs(gen_bx_poison - bx_clean), 0., 1.)
                diff_poison = torch.clamp(torch.abs(inject_bx - bx_clean), 0., 1.)
                save_fig.append(diff_gen_clean)
                save_fig.append(diff_poison)
            
            save_fig = torch.cat(save_fig)

            # Save path
            save_path = f'forensics/{args.func}_{args.func_option}_{args.attack}_{args.dataset}_{args.network}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pickle.dump(save_param, open(os.path.join(save_path, 'param'), 'wb'))
            save_image(save_fig, os.path.join(save_path, 'clean_poison_gencln_poiscr_genpoi_clninj_trigger.png'), nrow=mini_size)
    
    if Print_Level > 0:
        print(f'---Best recorded L1 of injected and inverted samples & source and removed samples: {round(inject_best, 2), round(remove_best, 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='vgg11', help='network structure')

    parser.add_argument('--attack', default='badnet', help='attack type')
    parser.add_argument('--target', type=int, default=0, help='target label')

    parser.add_argument('--func', default='mask', help='invert function')
    parser.add_argument('--func_option', default='binomial', help='invert function option/distribution')

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--seed', type=int, default=1024, help='seed index')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DEVICE = torch.device(f'cuda:{args.gpu}')

    print(f'======= invert_func: {args.func}-{args.func_option}, dataset: {args.dataset}, network: {args.network}, attack: {args.attack} =======')
    
    Print_Level = 1

    time_start = time.time()
    forensic(args, preeval=True)
    time_end = time.time()
    print('Running time:', (time_end - time_start) / 60, 'm')
