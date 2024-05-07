import os
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image

import lpips

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


class ForensicDataLoader:
    def __init__(self, clean_dir, poison_dir, victim_classes, target_class, batch_size=10):
        self.clean_dir = clean_dir
        self.poison_dir = poison_dir
        self.victim_classes = victim_classes
        self.target_class = target_class

        self.height = 224
        self.batch_size = batch_size

        all_clns = os.listdir(self.clean_dir)
        self.clean_imgs = []
        self.clean_lbls = []
        for i in range(len(all_clns)):
            img = all_clns[i]
            lbl = int(img.split('_')[-3])
            if lbl in self.victim_classes:
                self.clean_imgs.append(os.path.join(self.clean_dir, img))
                self.clean_lbls.append(lbl)
        
        all_pois = os.listdir(self.poison_dir)
        self.poison_imgs = []
        self.poison_lbls = []
        for i in range(len(all_pois)):
            img = all_pois[i]
            lbl = int(img.split('_')[-3])
            if lbl in self.victim_classes:
                self.poison_imgs.append(os.path.join(self.poison_dir, img))
                self.poison_lbls.append(lbl)
        
        assert(len(self.clean_imgs) == len(self.poison_imgs))
        assert(len(self.clean_lbls) == len(self.poison_lbls))

        self.iters = len(self.clean_imgs) // self.batch_size
        self.cur_iter = 0
    
    def __len__(self):
        return len(self.clean_imgs)
    
    def transform(self, x):
        transform = transforms.Compose([transforms.CenterCrop(self.height), transforms.ToTensor()])
        out = Image.open(x)
        out = transform(out)
        return out
    
    def shuffle_clean(self):
        index = np.arange(len(self.clean_imgs))
        np.random.shuffle(index)
        clean_imgs, clean_lbls = [], []
        for idx in index:
            clean_imgs.append(self.clean_imgs[idx])
            clean_lbls.append(self.clean_lbls[idx])
        self.clean_imgs = clean_imgs
        self.clean_lbls = clean_lbls
    
    def shuffle_poison(self):
        index = np.arange(len(self.poison_imgs))
        np.random.shuffle(index)
        poison_imgs, poison_lbls = [], []
        for idx in index:
            poison_imgs.append(self.poison_imgs[idx])
            poison_lbls.append(self.poison_lbls[idx])
        self.poison_imgs = poison_imgs
        self.poison_lbls = poison_lbls
    
    def next_batch(self):
        if self.cur_iter == self.iters:
            self.shuffle_clean()
            self.cur_iter = 0
        
        pre_bx_clean = self.clean_imgs[self.cur_iter * self.batch_size:(self.cur_iter + 1) * self.batch_size]
        pre_by_clean = self.clean_lbls[self.cur_iter * self.batch_size:(self.cur_iter + 1) * self.batch_size]
        # pre_bx_poison = self.poison_imgs[self.cur_iter * self.batch_size:(self.cur_iter + 1) * self.batch_size]
        # pre_by_poison = self.poison_lbls[self.cur_iter * self.batch_size:(self.cur_iter + 1) * self.batch_size]
        pre_bx_poison = self.poison_imgs[:self.batch_size]
        pre_by_poison = self.poison_lbls[:self.batch_size]
        self.cur_iter += 1

        bx_clean, bx_poison = [], []
        for i in range(self.batch_size):
            cln, poi = pre_bx_clean[i], pre_bx_poison[i]
            bx_clean.append(self.transform(cln))
            bx_poison.append(self.transform(poi))
        
        bx_clean, bx_poison = torch.stack(bx_clean), torch.stack(bx_poison)
        by_clean, by_poison = torch.from_numpy(np.asarray(pre_by_clean)), torch.from_numpy(np.asarray(pre_by_poison))

        return bx_clean, by_clean, bx_poison, by_poison


def upsample(inputs):
    outputs = transforms.Resize(224)(inputs)
    return outputs


def downsample(inputs):
    outputs = transforms.Resize(32)(inputs)
    return outputs


def forensic(args, preeval=True):
    # Load model
    model = torch.load(args.model_filepath, map_location=DEVICE)
    model.eval()

    # Load example data
    dataloader = ForensicDataLoader(args.clean_dirpath, args.poison_dirpath, args.victims, args.target)
    num_samples = len(dataloader)
    batch_size = dataloader.batch_size

    # Shuffle the data
    dataloader.shuffle_clean()
    dataloader.shuffle_poison()

    # Pre-evaluation of the model
    if preeval:
        with torch.no_grad():
            n_sample = 0
            n_acc, n_asr = 0, 0
            for step in range(dataloader.iters):
                bx_clean, by_clean, bx_poison, by_poison_src = dataloader.next_batch()
                
                bx_clean, by_clean = bx_clean.to(DEVICE), by_clean.to(DEVICE)
                bx_poison, by_poison_src = bx_poison.to(DEVICE), by_poison_src.to(DEVICE)
                
                by_poison = by_poison_src * 0 + args.target

                out_clean, out_poison = model(bx_clean), model(bx_poison)
                pred_clean, pred_poison = out_clean.max(dim=1)[1], out_poison.max(dim=1)[1]

                n_sample += bx_clean.size(0)
                n_acc += (pred_clean == by_clean).sum().item()
                n_asr += (pred_poison == by_poison).sum().item()

        acc = n_acc / n_sample
        asr = n_asr / n_sample
        if args.verbose > 0:
            print(f'Pre-evaluation of samples -> ACC: {acc}, ASR: {asr}, n_sample: {n_sample}')

    # Load pre-trained stylegan
    stylegan = StyleGAN(DEVICE)

    # Initialization of parameters
    mapping_labels = nn.functional.one_hot(torch.tensor(np.arange(10)), num_classes=10).float().to(DEVICE)
    latent_w_mean = stylegan.get_w_mean(mapping_labels)

    latent_input = latent_w_mean.clone()
    latent_input = latent_input.unsqueeze(1).repeat(1, stylegan.num_layers, 1)
    latent_input = latent_input.repeat(batch_size, 1, 1)
    latent_input.requires_grad_(True)
    # print(f'latent_input.shape: {latent_input.shape}')
    
    mask_init = np.random.random((batch_size, 1, 224, 224)) * 1e-2
    mask_init[:, :, 112-60:112+60, 112-60:112+60] = 0.99
    mask_init = np.arctanh((mask_init - 0.5) * (2 - epsilon()))
    
    mask = torch.tensor(mask_init, dtype=torch.float, requires_grad=True, device=DEVICE)
    mask_init = torch.zeros((1, 1, 224, 224)).to(DEVICE)

    # Define the optimization
    optim_mask = torch.optim.Adam(params=[mask], lr=1e-1, betas=(0.5, 0.9))
    init_lr = 1e-1
    optim_gan = torch.optim.Adam(params=[latent_input], lr=init_lr)
    
    # Loss terms
    CE = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    # LPIPS
    percept = lpips.LPIPS(net='vgg').to(DEVICE)
    # To greyscare
    togrey = transforms.Grayscale(num_output_channels=3)
    
    steps = args.epochs * dataloader.iters
    step = 0
    
    # Start optimization
    for epoch in range(args.epochs):
        n_sample = 0
        n_acc, n_asr, n_cycasr = 0, 0, 0
        CE_LOSS = 0
        InP_LOSS = 0
        TrP_LOSS = 0
        TV_LOSS = 0

        for _ in range(dataloader.iters):
            cur_iter = dataloader.cur_iter % dataloader.iters
            bx_clean, by_clean, bx_poison, by_poison_src = dataloader.next_batch()
            # For each step
            step += 1
            
            bx_clean, by_clean = bx_clean.to(DEVICE), by_clean.to(DEVICE)
            bx_poison, by_poison_src = bx_poison.to(DEVICE), by_poison_src.to(DEVICE)
            mini_size = bx_clean.size(0)

            ##############################################################
            # StyleGAN operation
            t = step / steps
            lr_i = get_lr(t, init_lr)
            optim_gan.param_groups[0]['lr'] = lr_i

            gen = stylegan.generator(latent_input, special_noises=[])  # pixel in [0, 1]
            gen_trigger = upsample(gen)  # Shape: batch_size x 3x224x224

            trigger_mask = mask_process(mask)
            gen_bx_poison = torch.clamp(bx_clean * (1 - trigger_mask) + bx_poison * trigger_mask, 0., 1.)
            gen_bx_clean = torch.clamp(bx_poison * (1 - trigger_mask) + gen_trigger * trigger_mask , 0., 1.)

            # Cyclic operation
            cyc_bx_poison = torch.clamp(gen_bx_clean * (1 - trigger_mask) + bx_poison * trigger_mask , 0., 1.)

            gen_by_clean = by_poison_src
            gen_by_poison = by_clean * 0 + args.target
            cyc_by_poison = gen_by_poison
            gen_out_clean, gen_out_poison, cyc_out_poison = model(gen_bx_clean), model(gen_bx_poison), model(cyc_bx_poison)

            # Remove attack target loss
            # ce_loss = CE(gen_out_poison, gen_by_poison) + CE(gen_out_clean, gen_by_clean) + CE(cyc_out_poison, cyc_by_poison)
            ce_loss = 1e1 * CE(gen_out_clean, gen_by_clean) + 1e-1 * CE(cyc_out_poison, cyc_by_poison)
            
            # Average smoothing loss
            tv_loss = avg_smooth_loss(gen_trigger)
            TV = 1e-3

            # ip_loss = percept.forward(gen * (1 - downsample(trigger_mask)), downsample(bx_poison * (1 - trigger_mask)), normalize=True).sum()
            ip_loss = percept.forward(gen, downsample(bx_poison), normalize=True).sum()

            ref_bx_clean = gen_bx_clean * (1 - trigger_mask)
            ref_bx_poison = bx_poison * (1 - trigger_mask)

            lw = mini_size
            ip_loss += lw * L1(ref_bx_clean, ref_bx_poison)
            IP = 1e1

            # TP loss is reg loss of mask
            tp_loss = mini_size * L1(trigger_mask, mask_init)
            TP = 1e1
            
            # Weighted loss
            loss = ce_loss + TV * tv_loss + IP * ip_loss + TP * tp_loss

            CE_LOSS += mini_size * ce_loss.item()
            InP_LOSS += ip_loss.item()
            TrP_LOSS += tp_loss.item()
            TV_LOSS += tv_loss.item()
            
            optim_gan.zero_grad()
            optim_mask.zero_grad()
            loss.backward()
            optim_gan.step()
            optim_mask.step()

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

        if (epoch + 1) % 10 == 0 and args.verbose > 0:
            print(f'Epoch: {epoch + 1}, CE_Loss: {round(CE_LOSS, 3)}, InP_Loss: {round(InP_LOSS, 3)}, TrP_Loss: {round(TrP_LOSS, 3)}, Smooth_Loss: {round(TV_LOSS, 3)}, gen_ACC: {round(acc, 2)}, gen_ASR: {round(asr, 2)}, cyc_ASR: {round(cycasr, 2)}')

        save_path = f'{args.save_folder}/{args.model_id}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        trigger_pattern = trigger_mask * bx_poison
        pickle.dump([trigger_mask.detach().cpu().data, trigger_pattern.detach().cpu().data], open(os.path.join(save_path, 'mask_pattern'), 'wb'))
        
        gen_bx_poison = torch.repeat_interleave(trigger_mask, 3, dim=1)
        diff = torch.clamp(torch.abs(bx_poison - gen_bx_clean), 0., 1.)
        savefig = [bx_clean, bx_poison, gen_bx_clean, gen_bx_poison, diff, ref_bx_poison, gen_trigger]
        savefig = torch.cat(savefig)
        save_image(savefig, os.path.join(save_path, 'clean_poison_gencln_genpoi.png'), nrow=mini_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--dataset_dir', default='/data/share/trojai/trojai-round3-dataset/', help='TrojAI dataset directory')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--save_folder', default='forensics/trojai_polygon/', help='save folder')
    parser.add_argument('--model_id', default='id-00000000', help='model id for forensics')
    parser.add_argument('--verbose', type=int, default=1, help='verbose level')
    parser.add_argument('--seed', type=int, default=1024, help='seed index')

    args = parser.parse_args()

    # Random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # GPU setting
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load metadata
    metadata = f'{args.dataset_dir}/METADATA.csv'
    df = pd.read_csv(metadata, header=0, index_col='model_name')
    total = 1008

    # TODO: 20 Selected models for attack forensics
    choice = [3, 9, 13, 23, 27, 31, 32, 37, 40, 42, 48, 49, 52, 56, 61, 64, 67, 74, 75, 80]

    # Save folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Forensics
    for i in range(len(choice)):
        model_idx = 'id-' + str(choice[i]).zfill(8)
        trigger_type = df.loc[model_idx]['trigger_type']
        if trigger_type != 'polygon':
            continue
        
        # Replace the current model_id with the selected model
        args.model_id = model_idx

        # Extract the victim classes
        victims = df.loc[model_idx]['triggered_classes'][1:-1].split()
        args.victims = []
        for j in victims:
            args.victims.append(int(j))
        
        # Extract the target class
        args.target = int(df.loc[model_idx]['trigger_target_class'])

        # Define the model path, clean data path, and poison data path
        args.model_filepath = f'{args.dataset_dir}/{args.model_id}/model.pt'
        args.clean_dirpath = f'{args.dataset_dir}/{args.model_id}/clean_example_data/'
        args.poison_dirpath = f'{args.dataset_dir}/{args.model_id}/poisoned_example_data/'

        # Main function
        time_start = time.time()

        forensic(args, preeval=True)

        time_end = time.time()

        print(f'Process Model: {args.model_id}, Time: {time_end - time_start:.2f}s')

        # TODO: test 1 model
        break
