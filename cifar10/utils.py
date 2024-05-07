import os
import PIL
import random
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from backdoors import *
from models import *
from invert_func import *

import warnings
warnings.filterwarnings("ignore")


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


_dataset_name = ['default', 'cifar10']

_mean = {
    'cifar10': [0.4914, 0.4822, 0.4465]
}

_std = {
    'cifar10': [0.2023, 0.1994, 0.2010]
}

_size = {
    'cifar10': (32, 32)
}

_num = {
    'cifar10': 10
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(dataset, augment=False, tensor=False):
    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[dataset]))
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))

        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[dataset]))

    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


def get_dataset(dataset, datadir='data', train=True, augment=True):
    transform = get_transform(dataset, augment=train & augment)
    
    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(datadir, train, download=True, transform=transform)

    return dataset


def get_backdoor(attack, side_len, device):
    if attack == 'badnet':
        backdoor = BadNets(side_len, device=device)
    elif attack == 'refool':
        backdoor = Refool(side_len, device=device)
    elif attack == 'wanet':
        backdoor = WaNet(side_len, device=device)
    else:
        raise NotImplementedError

    return backdoor


# Poison dataset
class PoisonDataset(Dataset):
    def __init__(self, dataset, backdoor, target):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.backdoor = backdoor
        self.target = target
        self.device = backdoor.device

        # Extract non-target data
        self.data = []
        for img, lbl in dataset:
            if lbl != target:
                self.data.append(img)
        
        self.n_data = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]

        # Inject backdoor
        inputs = img.unsqueeze(0).to(self.device)
        outputs = self.backdoor.inject(inputs)
        img = outputs.squeeze(0)

        return img, self.target

    def __len__(self):
        return self.n_data


# Fine-tuning dataset
class FinetuneDataset(Dataset):
    def __init__(self, dataset, num_classes, data_rate=1):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

        # Randomly select data_rate of the dataset
        n_data = len(dataset)
        n_single = int(n_data * data_rate / num_classes)
        self.n_data = n_single * num_classes

        # Evenly select data_rate of the dataset
        cnt = [n_single for _ in range(num_classes)]

        self.indices = np.random.choice(n_data, n_data, replace=False)

        self.data = []
        self.targets = []
        for i in self.indices:
            img, lbl = dataset[i]

            if cnt[lbl] > 0:
                self.data.append(img)
                self.targets.append(lbl)
                cnt[lbl] -= 1

    def __getitem__(self, index):
        img, lbl = self.data[index], self.targets[index]
        return img, lbl

    def __len__(self):
        return self.n_data


# Backdoor removal using BEAGLE
class BeagleAugment():
    def __init__(self, args, device):
        forensics_folder = f'forensics/{args.func}_{args.func_option}_{args.attack}_{args.dataset}_{args.network}'
        if not os.path.exists(forensics_folder):
            raise FileNotFoundError(f'Forensics folder not found: {forensics_folder}')
        
        self.device = device
        self.attack = args.attack

        # Load summarized attack properties
        param = pickle.load(open(f'{forensics_folder}/param', 'rb'))

        if args.attack == 'badnet':
            mask, pattern = param
            self.mask, self.pattern = mask.to(self.device), pattern.to(self.device)
            self.mask = self.mask.mean(dim=0, keepdim=True)
            self.pattern = self.pattern.mean(dim=0, keepdim=True)
        elif args.attack == 'refool':
            delta, trigger = param
            self.delta, self.trigger = delta, trigger.to(self.device)
        elif args.attack == 'wanet':
            weights, bias = param
            self.weights, self.bias = weights.to(self.device), bias.to(self.device)
        else:
            raise NotImplementedError
    
    # Apply the forensics triggers to the input images
    def adv_augment(self, x):
        if self.attack == 'badnet':
            out = x * (1 - self.mask) + self.pattern * self.mask
        elif self.attack == 'refool':
            out = attach_trigger(x, self.trigger, self.delta)
        elif self.attack == 'wanet':
            out = complex_linear(x, self.weights, self.bias)
        else:
            raise NotImplementedError
        
        return out
