#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from unet import UNet
from rednet import RedNet
from utils import *

import os
from datetime import datetime
from argparse import ArgumentParser


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params):
        self.p = params

        # Total minibatches by epoch (for report messages only)
        self.num_batches = self.p.train_len // self.p.batch_size

        # Get ready to ruuummmmmmble
        self.compile()


    def compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)"""

        # Model
        self.model = UNet()

        # Optimizer
        self.optim = Adam(self.model.parameters(),
                          lr=self.p.learning_rate,
                          betas=self.p.adam[:2],
                          eps=self.p.adam[2])

        # Loss function
        if self.p.loss == 'rmse':
            raise NotImplementedError('rMSE loss not implemented yet!')
        elif self.p.loss == 'l2':
            raise NotImplementedError('L2 loss not implemented yet!')
        else:
            self.loss = nn.L1Loss()

        # CUDA support
        if torch.cuda.is_available() and self.p.cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()


    def save_model(self, epoch, overwrite=True):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        fname_unet = '{}/model.pt'.format(self.p.ckpt_path) if overwrite \
            else '{}/model_epoch_{}.pt'.format(self.p.ckpt_path, epoch)
        print('Saving checkpoint to: {}'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)
        print('\n' + 80 * '-')


    def eval(self):
        """Evaluates model on validation/test sets."""

        self.model.train(False)
        raise NotImplementedError


    def train(self, data_loader):
        """Trains model on training set."""

        self.model.train(True)
        train_loss, train_acc, valid_acc, test_acc = [], [], [], []

        # Main training loop
        start = datetime.now()
        for epoch in self.p.nb_epochs:
            # Some statistics trackers
            epoch_start = datetime.now()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (x, y) in enumerate(data_loader):
                progress_bar(batch_idx, self.p.batch_report)

                x, y = Variable(x), Variable(y)
                if torch.cuda.is_available() and self.p.cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss = self.loss(y_pred, y)
                batch_loss.update(loss)

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report statistics
                if batch_idx % self.p.batch_report == 0 and batch_idx:
                    show_training_stats(batch_idx, self.num_batches, loss_meter.avg, time_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Save model
            clear_line()
            print('Elapsed time for epoch: {}'.format(time_elapsed_since(epoch_start)))
            self.model.save_model(self.p.ckpt_path, epoch)
            self.eval()

        elapsed = time_elapsed_since(start)
        print('Training done! Total elapsed time: {}\n'.format(elapsed))

        return train_loss


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset path', metavar='PATH', default='data')
    parser.add_argument('--ckpt-path', help='checkpoint path', metavar='PATH', default='ckpts')
    parser.add_argument('--batch-report', help='batch report interval', default=500, type=int)

    # Training parameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0003, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'rmse'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', default=True, type=bool)

    # Corruption noise parameters
    parser.add_argument('-n', '--noise-distrib', help='noise distribution',
                        choices=['gaussian', 'poisson', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-variance', help='noise variance', default=50.0, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    """Launches training."""

    params = parse_args()
    # model = Noise2Noise(params)

    # Read datasets
    root = os.path.join(os.path.dirname(os.path.join(os.path.abspath(__file__))), '..')
    train_dir = os.path.join(root, params.data, 'train')
    train_dataset = load_dataset(train_dir, params.batch_size)
    #valid_dir = os.path.join(root, params.data, 'valid')
    #valid_dataset = load_dataset(valid_dir, params.batch_size)

    extra = {'train_len': len([i for i in os.listdir(train_dir) if os.path.isfile(i)])}
    params.__dict__.update(extra)

    #model.train()
