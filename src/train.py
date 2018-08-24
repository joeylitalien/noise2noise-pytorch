#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam

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

        # Create directories if they don't exist
        if not os.path.isdir(self.p.ckpt_path):
            os.mkdir(self.p.ckpt_path)

        # Get ready to ruuummmmmmble
        self.compile()

    def compile(self):
        """Compiles model (loss function, optimizers, etc.)"""

        # Model
        self.model = UNet()

        # Optimizer
        self.optim = Adam(self.model.parameters(),
                           lr=self.p.learning_rate,
                           betas=self.p.adam[:2],
                           eps=self.p.adam[2])

        # Loss function
        if self.p.loss == 'rmse':
            raise ValueError('rMSE loss not implemented yet!')
        elif self.p.loss == 'l2':
            raise ValueError('L2 loss not implemented yet!')
        else:
            self.loss = nn.L1Loss()

        # CUDA support
        if torch.cuda.is_available() and self.p.cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

    def eval(self):
        self.model.train(False)

        return

    def train(self, data_loader):
        """Trains model on dataset."""

        self.model.train(True)
        train_loss, train_acc, valid_acc, test_acc = [], [], [], []
        best_valid_acc = 0.

        # Main training loop
        start = datetime.now()
        for epoch in self.p.nb_epochs:
            start_epoch = datetime.now()

            # Minibatch SGD
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available() and self.p.cuda:
                    x = x.cuda()
                    y = y.cuda()

                utils.progress_bar(batch_idx, self.p.batch_report)


            # Save model
            utils.clear_line()
            print('Elapsed time for epoch: {}'.format(utils.time_elapsed_since(start_epoch)))
            self.model.save_model(self.p.ckpt_path, epoch)
            self.eval()


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset path', metavar='PATH', default='./data')
    parser.add_argument('-r', '--redux', help='train on smaller dataset', action='store_true')
    parser.add_argument('--ckpt-path', help='checkpoint path', metavar='PATH', default='./ckpts')
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
    model = Noise2Noise(params)
    model.train()
