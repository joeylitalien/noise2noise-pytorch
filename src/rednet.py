#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


class RedNet(nn.Module):
    """RedNet architecture from Mao et al. (2016); Default is RED30.

    Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections.
    arXiv: https://arxiv.org/abs/1606.08921
    """

    def __init__(self, channels=3, dim=128, depth=15, skips=2):
        super(RedNet, self).__init__()

        # Parameters
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.skips = skips

        # Repeated blocks
        self._conv1 = nn.Sequential(
            nn.Conv2d(channels, dim, channels, padding=1),
            nn.ReLU(inplace=True))
        self._conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, channels, padding=1),
            nn.ReLU(inplace=True))
        self._deconv1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, channels, padding=1),
            nn.ReLU(inplace=True))
        self._deconv2 = nn.ConvTranspose2d(dim, channels, channels, padding=1)
        self._relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                print('Init')
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def forward(self, x):
        """Adds skip connections every <skips> layers,
        from conv feature maps to their mirrored deconv feature maps.
        """

        # TODO: Use torch.concatenate!

        # Forward convolution pass
        convs = []
        convs.append(self._conv1(x))
        for i in range(self.depth - 1):
            convs.append(self._conv2(convs[i]))

        # Get last deconvolution
        y = self._deconv1(convs[self.depth - 1]) + convs[self.depth - 2]

        # Add symmetric skip shortcuts every <skips> layers
        for i in range(self.depth - 1):
            y = self._deconv1(y)
            if i % self.skips:
                y += convs[self.depth - i - 1]
                y = self._relu(y)

        return self._deconv2(y)
