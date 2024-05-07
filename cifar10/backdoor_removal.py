import os
import sys
import time
import copy
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from utils import *


def finetune(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    lr = args.lr
    epochs = args.epochs
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    total_time = 0
    time_start = time.time()
    for epoch in range(epochs):
        model.train()
        if epoch > 0 and epoch % 2 == 0:
            lr /= 10

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            time_end = time.time()

            model.eval()
            correct_cl = 0
            correct_bd = 0

            with torch.no_grad():
                total_cl = 0
                for (x_test, y_test) in test_loader:
                    x_test = x_test.to(DEVICE)
                    y_test = y_test.to(DEVICE)
                    total_cl += y_test.size(0)

                    ### clean accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_cl += (y_pred == y_test).sum().item()
                
                total_bd = 0
                for (x_test, y_test) in poison_test_loader:
                    x_test = x_test.to(DEVICE)
                    y_test = y_test.to(DEVICE)
                    total_bd += y_test.size(0)

                    ### backdoor accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_bd += (y_pred == y_test).sum().item()

            acc = correct_cl / total_cl
            asr = correct_bd / total_bd

            sys.stdout.write('epoch: {:2}/{}, lr: {:.4f} - {:.2f}s, '
                                .format(epoch+1, epochs, lr, time_end-time_start)\
                                + 'loss: {:.4f}, acc: {:.4f}, asr: {:.4f}\n'
                                .format(loss, acc, asr))
            sys.stdout.flush()

            total_time += (time_end-time_start)
            time_start = time.time()

    return model, total_time


def beagle(args, model, train_loader, test_loader, poison_test_loader, preprocess):
    # Load the forensics function
    beagle = BeagleAugment(args, DEVICE)

    lr = args.lr
    epochs = args.epochs
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    total_time = 0
    time_start = time.time()
    for epoch in range(epochs):
        model.train()
        if epoch > 0 and epoch % 2 == 0:
            lr /= 10

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            half = x_batch.size(0) // 2
            x_origin = x_batch[:half]
            x_beagle = beagle.adv_augment(x_batch[half:])

            x_batch = torch.cat([x_origin, x_beagle], dim=0)

            optimizer.zero_grad()

            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 2 == 0:
            time_end = time.time()

            model.eval()
            correct_cl = 0
            correct_bd = 0

            with torch.no_grad():
                total_cl = 0
                for (x_test, y_test) in test_loader:
                    x_test = x_test.to(DEVICE)
                    y_test = y_test.to(DEVICE)
                    total_cl += y_test.size(0)

                    ### clean accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_cl += (y_pred == y_test).sum().item()
                
                total_bd = 0
                for (x_test, y_test) in poison_test_loader:
                    x_test = x_test.to(DEVICE)
                    y_test = y_test.to(DEVICE)
                    total_bd += y_test.size(0)

                    ### backdoor accuracy ###
                    y_out = model(preprocess(x_test))
                    _, y_pred = torch.max(y_out.data, 1)
                    correct_bd += (y_pred == y_test).sum().item()

            acc = correct_cl / total_cl
            asr = correct_bd / total_bd

            sys.stdout.write('epoch: {:2}/{}, lr: {:.4f} - {:.2f}s, '
                                .format(epoch+1, epochs, lr, time_end-time_start)\
                                + 'loss: {:.4f}, acc: {:.4f}, asr: {:.4f}\n'
                                .format(loss, acc, asr))
            sys.stdout.flush()

            total_time += (time_end-time_start)
            time_start = time.time()

    return model, total_time


# Evaluate the model
def test(model, test_loader, poison_loader, preprocess):
    model.eval()

    correct_cl = 0
    correct_bd = 0

    with torch.no_grad():
        total_cl = 0
        for (x_test, y_test) in tqdm(test_loader):
            x_test = x_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            total_cl += y_test.size(0)

            ### clean accuracy ###
            y_out = model(preprocess(x_test))
            _, y_pred = torch.max(y_out.data, 1)
            correct_cl += (y_pred == y_test).sum().item()
        
        total_bd = 0
        for (x_test, y_test) in tqdm(poison_loader):
            x_test = x_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            total_bd += y_test.size(0)

            ### backdoor accuracy ###
            y_out = model(preprocess(x_test))
            _, y_pred = torch.max(y_out.data, 1)
            correct_bd += (y_pred == y_test).sum().item()

    acc = correct_cl / total_cl
    asr = correct_bd / total_bd

    return acc, asr


# Main function
def main(args, preeval=True):
    # Load attacked model
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'
    model = torch.load(model_filepath, map_location='cpu')
    model = model.to(DEVICE)

    # Normalization
    preprocess, _ = get_norm(args.dataset)

    # Number of classes
    num_classes = get_config(args.dataset)['num_classes']

    # Finetune dataset
    train_set = get_dataset(args.dataset, train=True, augment=True)
    train_set = FinetuneDataset(train_set, num_classes=num_classes, data_rate=args.ratio)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    # Test dataset
    test_set = get_dataset(args.dataset, train=False)

    # Poison dataset
    side_len = test_set[0][0].shape[-1]
    backdoor = get_backdoor(args.attack, side_len, device=DEVICE)
    poison_set = PoisonDataset(test_set, backdoor, target=args.target)

    poison_loader = DataLoader(dataset=poison_set, batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=test_set, batch_size=args.batch_size)

    print(f'Finetune dataset: {len(train_set)}, Test dataset: {len(test_set)}, Poison dataset: {len(poison_set)}')

    # Step 1: Evaluate on the original model
    acc, asr = test(model, test_loader, poison_loader, preprocess)
    print(f'Step 1: Original | ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%')

    # Step 2: Evaluate on the result model
    finetune_model = copy.deepcopy(model)
    finetune_model, total_time = finetune(args, finetune_model, train_loader, test_loader, poison_loader, preprocess)
    acc, asr = test(finetune_model, test_loader, poison_loader, preprocess)
    print(f'Step 2: Finetune | ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, Time: {total_time:.2f}s')
    
    # Step 3: Mix beagle samples and finetune the model
    beagle_model = copy.deepcopy(model)
    beagle_model, total_time = beagle(args, beagle_model, train_loader, test_loader, poison_loader, preprocess)
    acc, asr = test(beagle_model, test_loader, poison_loader, preprocess)
    print(f'Step 3: Beagle | ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, Time: {total_time:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--datadir', default='./data', help='root directory of data')

    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')
    parser.add_argument('--attack', default='badnet', help='attack method')
    parser.add_argument('--target', type=int, default=0, help='target label')

    parser.add_argument('--ratio', type=float, default=0.01, help='ratio of the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='finetune learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='finetune epochs')

    parser.add_argument('--seed', type=int, default=1024, help='seed index')

    args = parser.parse_args()

    # GPU setting
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    seed_torch(args.seed)

    # TODO: Define the forensics function for different attacks
    if args.attack == 'badnet':
        args.func = 'mask'
        args.func_option = 'binomial'
    elif args.attack == 'refool':
        args.func = 'mask'
        args.func_option = 'uniform'
    elif args.attack == 'wanet':
        args.func = 'transform'
        args.func_option = 'complex'
    else:
        raise NotImplementedError

    # Conduct experiment
    main(args)
