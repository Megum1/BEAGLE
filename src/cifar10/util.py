import numpy as np
import torch
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
import torchvision.models as modelzoo
from GTSRB import GTSRB

from backdoors import *
from models import *


EPSILON = 1e-7

_dataset_name = ['default', 'cifar10', 'gtsrb', 'imagenet', 'celeba']

_mean = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'gtsrb':    [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
    'celeba':   [0.5, 0.5, 0.5],
}

_std = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.2023, 0.1994, 0.2010],
    'gtsrb':    [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
    'celeba':   [0.5, 0.5, 0.5],
}

_size = {
    'cifar10':  (32, 32),
    'gtsrb':    (32, 32),
    'imagenet': (224, 224),
    'celeba': (128, 128),
}

_num = {
    'cifar10':  10,
    'gtsrb':    43,
    'imagenet': 1000,
    'celeba': 8,
}


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)


def get_processing(dataset, augment=True, tensor=False, size=None):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        if dataset == 'imagenet':
            transforms_list.append(transforms.RandomResizedCrop(_size[dataset], scale=(0.2, 1.)))
        elif dataset in ['celeba', 'gtsrb']:
            transforms_list.append(transforms.Resize(_size[dataset]))
        else:
            transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        if dataset == 'imagenet':
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(_size[dataset]))
        elif dataset in ['celeba', 'gtsrb']:
            transforms_list.append(transforms.Resize(_size[dataset]))
    
    if not tensor:
        transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess  = transforms.Compose([unnormalize])
    return preprocess, deprocess


def get_dataset(args, train=True, augment=True):
    transform, _ = get_processing(args.dataset, train & augment)
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.datadir, train, transform, download=False)
    elif args.dataset == 'imagenet':
        split = 'train' if train else 'val'
        dataset = datasets.ImageNet('/data/share/imagenet/ILSVRC/Data/CLS-LOC/', split=split, transform=transform)
    elif args.dataset == 'celeba':
        split = 'train' if train else 'valid'
        original_dataset = datasets.CelebA('/data/share/', split=split, transform=transform, target_type='attr')
        dataset = CelebADataset(original_dataset)
    elif args.dataset == 'gtsrb':
        split = 'train' if train else 'test'
        dataset = GTSRB(args.datadir, split, transform, download=False)

    return dataset


def get_loader(args, train=True):
    dataset = get_dataset(args, train)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4, shuffle=train)
    return dataloader


def get_model(args, pretrained=False):
    num_classes = 10
    if args.dataset == 'gtsrb':
        num_classes = 43
    # Pre-trained for ImageNet
    if args.dataset == 'imagenet' and args.network == 'vgg16':
        model = modelzoo.vgg16(pretrained=True)
    elif args.dataset == 'imagenet' and args.network == 'resnet50':
        model = modelzoo.resnet50(pretrained=True)
    
    elif args.dataset == 'celeba' and args.network == 'resnet18':
        model = resnet18face()
    
    elif args.network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif args.network == 'preresnet18':
        model = preresnet18(num_classes=num_classes)
    elif args.network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    
    return model


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def get_backdoor(attack, shape, normalize=None, device=None):
    if 'refool' in attack:
        backdoor = Refool(shape, attack.split('_')[1], device=device)
    elif attack == 'wanet':
        backdoor = WaNet(shape, device=device)
    elif attack == 'invisible':
        backdoor = Invisible()
    elif attack in ['blend', 'sig']:
        backdoor = Other(attack, device=device)
    elif attack == 'inputaware':
        backdoor = InputAware(normalize, device=device)
    elif attack == 'dynamic':
        backdoor = Dynamic(normalize, device=device)
    elif attack == 'gotham':
        backdoor = Gotham()
    elif attack == 'badnet':
        backdoor = Badnet(device=device)
    elif attack == 'trojnn':
        backdoor = TrojNN(device=device)
    elif attack == 'dfst':
        backdoor = DFST()
    elif attack == 'psa':
        backdoor = PSA()
    else:
        backdoor = None
    return backdoor


class CelebADataset(Dataset):
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        assert(len(dataset.target_type) == 1)
        assert(dataset.target_type[0] == "attr")
        self.dataset = dataset
        attr_names = self.dataset.attr_names
        self.attr_index = []
        for i in range(len(attr_names)):
            if attr_names[i] in ['Heavy_Makeup', 'Mouth_Slightly_Open', 'Smiling']:
                self.attr_index.append(i)

        self.targets = []
        self.filename = []
        for index in range(len(self.dataset.identity)):
            imgfile = self.dataset.filename[index]
            attr = self.dataset.attr[index]
            label = 0
            for i in range(len(self.attr_index)):
                if attr[self.attr_index[i]] == 1:
                    label += 2 ** i
            self.filename.append(imgfile)
            self.targets.append(label)
    
    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.dataset.root, self.dataset.base_folder, "img_align_celeba", self.filename[index]))
        target = self.targets[index]

        if self.dataset.transform is not None:
            X = self.dataset.transform(X)
        return X, target
    
    def __len__(self):
        return len(self.targets)
