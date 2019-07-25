import sys
sys.path.append("..")

import os
import time
import torch
import argparse
import torchutils
import torchvision
import numpy as np
import torch.nn as nn
from os import path as osp
import torch.optim as optim
from model import EfficientNet
from dataset import get_loader
import torch.nn.functional as F
from hammingloss import HammingLoss

DEFAULT_SIGMOID_ACC_THRESH = 0.5
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 2
DEFAULT_MODEL_PATH = './../../models'
DEFAULT_DATA_PATH = './../../data'
DEFAULT_CKPT = 'model_20190618-180049_e100_val_82.262.pt'
DEFAULT_RANDOM_SEED = 2222

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Label Classification with EfficientNet')
    parser.add_argument('--lr', default=DEFAULT_LEARNING_RATE, type=float, help='learning rate')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, type=str, help='checkpoint file name')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str, help='model/checkpoint dir path')
    parser.add_argument('--data_path', default=DEFAULT_DATA_PATH, type=str, help='model/checkpoint dir path')
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--num_workers', default=DEFAULT_NUM_WORKERS, type=int, help='number of worker threads')
    parser.add_argument('--random_seed', default=DEFAULT_RANDOM_SEED, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=DEFAULT_WEIGHT_DECAY, type=int, help='weight decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--fixed_lr_decay', action='store_true', help='use fixed learning rate decay')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default={27,31,35,39,43},
                        help='epochs after which lr will be decayed')
    parser.add_argument('--sigmoid_thresh', default=DEFAULT_SIGMOID_ACC_THRESH, type=int,
                        help='sigmoid threshold for accuracy measurement')
    args = parser.parse_args()
    return args

def train_model(model, train_loader, criterion, optimizer, device, measure_accuracy=False):
    model.train()
    running_loss = 0.0
    if measure_accuracy:
        hamming = HammingLoss()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.long().to(device)
        outputs = model(data)
        target = target.type_as(outputs)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)   # for exploding nan grad.
        optimizer.step()
        if measure_accuracy:
            predicted = torch.sigmoid(outputs) > args.sigmoid_thresh
            hamming.update(predicted, target)
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))),
                end="\r", flush=True)
    end_time = time.time()
    running_loss /= len(train_loader)
    acc = hamming.loss
    inv_acc = hamming.inverseloss
    print('\nTraining Loss: %5.4f Hamming Loss: %5.3f Inverse Hamming Loss: %5.3f Time: %d s' % \
            (running_loss, acc*100, inv_acc*100, end_time - start_time))
    return running_loss, acc

if __name__ == "__main__":
    args = parse_args()
    print('Arguments:', args)

    # set the random seed
    torchutils.set_random_seed(args.random_seed)

    print('*'*30)
    print('Setting up dataset...')
    print('*'*30)
    train_loader = get_loader(osp.join(args.data_path, 'images'), osp.join(args.data_path, 'list_attr_celeba.txt'),
                            crop_size=178, image_size=360,
                            batch_size=args.batch_size, mode='all', num_workers=args.num_workers)
    print('*'*30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device:', device)
    print('*'*30)

    print('Model Architecture')
    model = EfficientNet.from_name('efficientnet-b4')
    model = model.to(device)
    model_params = torchutils.get_model_param_count(model)
    print(model)
    print('Total model parameters:', model_params)
    print('*'*30)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.005, verbose=True)

    n_epochs = 200
    if args.resume:
        start_epoch, model, optimizer, scheduler = torchutils.load_checkpoint(model_path=args.model_path,
                                                        ckpt_name=args.ckpt, device=device,
                                                        model=model, optimizer=optimizer,
                                                        scheduler=scheduler)
        start_epoch -= 1
        print('Resumed checkpoint {} from {}. Starting at epoch {}.'.format(args.ckpt, args.model_path, start_epoch+1))
        print('Current learning rate: {}'.format(torchutils.get_current_lr(optimizer)))
        print('*'*30)
    else:
        start_epoch = 0
        # model = init_weights(model)

    for epoch in range(start_epoch, n_epochs):
        print('Epoch: %d/%d' % (epoch+1,n_epochs))
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, measure_accuracy=True)
        # val_loss, val_acc = val_model(model, test_loader, criterion, device)
        # Checkpoint the model after each epoch.
        torchutils.save_checkpoint(epoch+1, args.model_path, model=model, optimizer=optimizer, metric=train_acc)
        if args.fixed_lr_decay:
            if epoch in args.lr_decay_epochs:
                cur_lr = torchutils.get_current_lr(optimizer)
                optimizer = torchutils.set_current_lr(optimizer, cur_lr*0.1)
                print('Epoch    %d: reducing learning rate of group 0 to %f' % (epoch, cur_lr*0.1))
        else:
            # scheduler.step(val_loss)
            scheduler.step(train_loss)
        print('='*20)
