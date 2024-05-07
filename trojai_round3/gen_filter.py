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
    
    # TODO: kdim=1 is not effective to invert Toaster
    kdim = 4
    weights_init = (np.random.random((1, 9, kdim, kdim)) - 0.5) * 1e-2
    bias_init = (np.random.random((1, 3, 1, 1)) - 0.5) * 1e-2
    weights = torch.tensor(weights_init, dtype=torch.float, requires_grad=True, device=DEVICE)
    bias = torch.tensor(bias_init, dtype=torch.float, requires_grad=True, device=DEVICE)
    weights_init = torch.tensor(weights_init * 0., dtype=torch.float, device=DEVICE)
    bias_init = torch.tensor(bias_init * 0., dtype=torch.float, device=DEVICE)

    # Define the optimization
    optimizer = torch.optim.Adam(params=[weights, bias], lr=1e-2, betas=(0.5, 0.9))
    
    # Loss terms
    CE = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    # LPIPS
    percept = lpips.LPIPS(net='vgg').to(DEVICE)

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
            gen_bx_poison = filter_linear(bx_clean, weights, bias)

            gen_by_poison = by_clean * 0 + args.target
            gen_out_poison = model(gen_bx_poison)

            ce_loss = CE(gen_out_poison, gen_by_poison)
            
            # Average smoothing loss
            tv_loss = avg_smooth_loss(gen_bx_poison)
            TV = 1e-3

            # Reg loss of grid
            reg_loss = L2(weights, weights_init) + L2(bias, bias_init)

            cln_mean, cln_std = torch.mean(bx_clean, dim=[0, 2, 3], keepdim=True), torch.std(bx_clean, dim=[0, 2, 3], keepdim=True)
            poi_mean, poi_std = torch.mean(bx_poison, dim=[0, 2, 3], keepdim=True), torch.std(bx_poison, dim=[0, 2, 3], keepdim=True)

            # Similarity loss (poison)
            ref_bx_clean = torch.clamp((bx_clean - cln_mean) / cln_std * poi_std + poi_mean, 0., 1.)
            # if args.filter_type == 'GothamFilterXForm':
            #     ref_bx_clean = togrey(bx_clean)

            tp_loss = L2(gen_bx_poison, ref_bx_clean)

            # Add reg loss into tp_loss
            tp_loss += 1e-1 * reg_loss
            TP = 1e1
            
            # Weighted loss
            loss = ce_loss + TV * tv_loss + TP * tp_loss

            ip_loss = tp_loss.clone()

            CE_LOSS += mini_size * ce_loss.item()
            InP_LOSS += ip_loss.item()
            TrP_LOSS += tp_loss.item()
            TV_LOSS += tv_loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_poison = gen_out_poison.max(dim=1)[1]

            n_sample += mini_size
            n_asr += (pred_poison == gen_by_poison).sum().item()

        CE_LOSS = CE_LOSS / n_sample
        InP_LOSS = InP_LOSS / n_sample
        TrP_LOSS = TrP_LOSS / n_sample
        TV_LOSS = TV_LOSS / n_sample
        asr = n_asr / n_sample

        if (epoch + 1) % 10 == 0 and args.verbose > 0:
            print(f'Epoch: {epoch + 1}, CE_Loss: {round(CE_LOSS, 3)}, InP_Loss: {round(InP_LOSS, 3)}, TrP_Loss: {round(TrP_LOSS, 3)}, Smooth_Loss: {round(TV_LOSS, 3)}, gen_ASR: {round(asr, 2)}')

        save_path = f'{args.save_folder}/{args.model_id}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pickle.dump([weights.detach().cpu().data, bias.detach().cpu().data], open(os.path.join(save_path, 'linear_param'), 'wb'))

        savefig = [bx_clean, bx_poison, gen_bx_poison]
        savefig = torch.cat(savefig)
        save_image(savefig, os.path.join(save_path, 'clean_poison_gencln_genpoi.png'), nrow=mini_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--dataset_dir', default='/data/share/trojai/trojai-round3-dataset/', help='TrojAI dataset directory')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--save_folder', default='forensics/trojai_filter/', help='save folder')
    parser.add_argument('--filter_type', default='LomoFilterXForm', help='filter type for forensics')
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
    choice = [17, 21, 24, 33, 36, 39, 62, 73, 77, 0, 28, 63, 41, 53, 58, 44, 55, 76, 68, 79]

    filter_types = ['LomoFilterXForm',
                    'NashvilleFilterXForm',
                    'GothamFilterXForm',
                    'KelvinFilterXForm',
                    'ToasterXForm']

    # Save folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Forensics
    for i in range(len(choice)):
        model_idx = 'id-' + str(choice[i]).zfill(8)
        filter_type = df.loc[model_idx]['instagram_filter_type']

        if filter_type not in filter_types:
            continue
        
        # Replace the current model_id with the selected model
        args.model_id = model_idx
        args.filter_type = filter_type

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
