#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid

from unet import UNet
from rednet import RedNet
from utils import *
from dataset import *

import os
import PIL
from datetime import datetime
from argparse import ArgumentParser


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params):
        """Initializes model."""

        self.p = params

        # Create directory for model checkpoints
        if not os.path.isdir(params.ckpt_save_path):
            os.mkdir(params.ckpt_save_path)

        # Get ready to ruummmbbbbbble
        self.compile()


    def compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

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


    def save_model(self, epoch, valid_loss):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        fname_unet = '{}/n2n.pt'.format(self.p.ckpt_save_path) if self.p.ckpt_overwrite \
            else '{}/n2n_epoch{}_{:>1.5f}.pt'.format(self.p.ckpt_save_path, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.p.cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def denoise(self, test_loader, show_first_4=True):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        noisy_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        denoised_dir = os.path.join(params.data, 'denoised')
        if not os.path.isdir(denoised_dir):
            os.mkdir(denoised_dir)

        for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first 4 images, for now
            if batch_idx >= 4:
                break

            if torch.cuda.is_available() and self.p.cuda:
                source = source.cuda()

            noisy_imgs.append(source.detach())
            denoised_img = self.model(source).detach()
            denoised_imgs.append(denoised_img)
            clean_imgs.append(target)

        noisy_imgs = [t.squeeze(0) for t in noisy_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        for i in range(len(denoised_imgs)):
            noisy = transforms.ToPILImage()(noisy_imgs[i])
            denoised = transforms.ToPILImage()(denoised_imgs[i])
            clean = transforms.ToPILImage()(clean_imgs[i])
            if show_first_4:
                combined = Image.new('RGB', (3 * clean.size[0], clean.size[1]))
                combined.paste(noisy, (0, 0))
                combined.paste(denoised, (clean.size[0], 0))
                combined.paste(clean, (2 * clean.size[0], 0))
                combined.show()
            fname = os.path.splitext(test_loader.dataset.imgs[i])[0]
            denoised.save(os.path.join(denoised_dir, '{}_denoised.png'.format(fname)))


    def eval(self, valid_loader):
        """Evaluates model on validation sets."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        for batch_idx, (source, target) in enumerate(valid_loader):
            if torch.cuda.is_available() and self.p.cuda:
                source = source.cuda()
                target = target.cuda()

            source_denoised = self.model(source)
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]

        return valid_loss, valid_time


    def train(self, train_loader, valid_loader):
        """Trains model on training set."""

        self.model.train(True)

        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'
        # train_loss, train_acc, valid_acc, test_acc = [], [], [], []

        # Main training loop
        start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some statistics trackers
            epoch_start = datetime.now()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
                batch_start = datetime.now()

                if torch.cuda.is_available() and self.p.cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)
                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if batch_idx % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Evaluate model on validation set
            print('\nTesting model on validation set... ', end='')
            epoch_time = time_elapsed_since(epoch_start)[0]
            valid_loss, valid_time = self.eval(valid_loader)
            show_on_epoch_end(epoch_time, valid_time, valid_loss)

            # Save checkpoint
            self.save_model(epoch, valid_loss)

        elapsed = time_elapsed_since(start)[0]
        print('Training done! Total elapsed time: {}\n'.format(elapsed))


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data training parameters
    parser.add_argument('-d', '--data', help='dataset path', default='../data')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', default=False, type=bool)
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)
    parser.add_argument('-r', '--redux', help='use smaller datasets', default=False, type=bool)

    # Test parameters
    parser.add_argument('--load-ckpt', help='load model checkpoint')

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0003, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'rmse'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', default=True, type=bool)

    # Corruption noise parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
            choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50.0, type=float)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=256, type=int)
    parser.add_argument('--clean-targets', help='use clean targets in training', default=False, type=bool)

    return parser.parse_args()


def load_dataset(name, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    noise = (params.noise_type, params.noise_param)
    img_dir = '{}_redux'.format(name) if params.redux else name
    path = os.path.join(params.data, img_dir)
    dataset = N2NDataset(path, params.crop_size, noise_dist=noise, clean_targets=params.clean_targets)

    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


if __name__ == '__main__':
    """Main program."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset('train', params, shuffled=True)
    valid_loader = load_dataset('valid', params, shuffled=False)

    # Initialize model
    n2n = Noise2Noise(params)

    # Test dataset, if trained model checkpoint is specified
    if params.load_ckpt:
        params.redux = False
        params.clean_targets = True
        params.crop_size = 512
        test_loader = load_dataset('test', params, shuffled=False, single=True)

        n2n.denoise(test_loader)

    # Otherwise, train
    else:
        n2n.train(train_loader, valid_loader)
