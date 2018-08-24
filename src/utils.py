#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_dataset(root_dir, batch_size):
    """Loads data from image folder."""

    mean, std = [0.5] * 3, [0.5] * 3
    normalize = transforms.Normalize(mean=mean, std=std)
    train_data = ImageFolder(root=root_dir,
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
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