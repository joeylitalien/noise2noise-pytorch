#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def corrupt_with_noise(img, type='gaussian', sigma=50):
    """Adds noise to image."""

    if type == 'gaussian':
        #TODO: Implement this
    elif type == 'poisson':
        #TODO: Implement this
    elif type == 'mc':
        raise ValueError('Monte Carlo noise has to be handled differently, see paper for details.')
    else:
        raise ValueError('Invalid distribution type: {}'.format(type))

    return


def corrupt_with_text(img, other):
    """Overlays random text over image."""

    return