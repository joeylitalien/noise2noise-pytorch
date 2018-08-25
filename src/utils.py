#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime


def load_noisy_dataset(root_dir, batch_size, noise_type, crops, shuffled):
    """Loads data from image folder."""

    if crops > 0:
        data = ImageFolder(root=root_dir,
                           transform=transforms.Compose([
                               transforms.RandomResizedCrop(crops),
                               transforms.ToTensor()]))
    else:
        data = ImageFolder(root=root_dir, transform=transforms.ToTensor())

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffled)
    return data_loader


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, report_interval, loss):
    """Neat progress bar to track training."""

    bar_size = 24
    progress = (((batch_idx - 1) % report_interval) + 1) / report_interval
    fill = int(progress * bar_size)
    print('\rBatch {:>4d} [{}{}] Loss: {:>7.4f}'.format(batch_idx, '=' * fill, ' ' * (bar_size - fill), loss), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    return str(datetime.now() - start)[:-7]


def show_training_stats(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = str(int(np.ceil(np.log10(num_batches))))
    print('Batch {:>{dec}d} / {:d} | Loss: {:>7.4f} | Avg time / batch: {:d} ms'.format(batch_idx,
                                                                                        num_batches, loss, int(elapsed),
                                                                                        dec=dec))


class AvgMeter(object):
    """Computes and stores the average and current value."""


    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
