#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import torchvision.transforms.functional as tvF
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import numpy as np
from random import choice
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt


def load_img(path):
    """Loads image into RGB format (PIL)."""

    return Image.open(path).convert('RGB')


def show_img(tensor):
    """Visualizes Torch tensor as PIL image."""

    img = transforms.ToPILImage()(tensor)
    img.show()


def psnr(source_denoised, target):
    """Computes peak signal-to-noise ratio."""

    return 10. * torch.log10(1. / F.mse_loss(source_denoised, target))


def create_montage(img_name, save_path, noisy_t, denoised_t, clean_t, show=True):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    # Convert to PIL images
    noisy = transforms.ToPILImage()(noisy_t)
    denoised = transforms.ToPILImage()(denoised_t)
    clean = transforms.ToPILImage()(clean_t)

    # Build image montage
    psnr_vals = [psnr(noisy_t, clean_t), psnr(denoised_t, clean_t)]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Ground truth']
    zipped = zip(titles, [noisy, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img)
        ax[j].set_title(title, fontname='serif')
        ax[j].axis('off')

    # Open pop up window, if requested
    if show:
        plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
    noisy.save(os.path.join(save_path, '{}_noisy.png'.format(fname)))
    denoised.save(os.path.join(save_path, '{}_denoised.png'.format(fname)))
    fig.savefig(os.path.join(save_path, '{}_montage.png'.format(fname)), bbox_inches='tight')


class N2NDataset(Dataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, crop_size=256, noise_dist=('gaussian', 50.), clean_targets=False):
        """Initializes noisy image dataset."""

        super(N2NDataset, self).__init__()

        self.root_dir = root_dir
        self.imgs = os.listdir(root_dir)
        self.crop_size = crop_size

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]

        # Use clean targets
        self.clean_targets = clean_targets

        # Transforms
        self._to_tensor = transforms.ToTensor()

    def _random_crop(self, img):
        """Performs random square crop of fixed size."""

        w, h = img.size
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        return tvF.crop(img, i, j, self.crop_size, self.crop_size)


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        if self.noise_type == 'poisson':
            noise = np.random.poisson(self.noise_param, (h, w, c))

        # Normal distribution (default)
        else:
            std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

        # Add noise and clip
        noise_img = np.array(img) + noise
        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)

        return Image.fromarray(noise_img)


    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        font = ImageFont.truetype('Times New Roman.ttf', 20)
        text_img = img.copy()
        draw = ImageDraw.Draw(text_img)

        # Add text overlay by choosing random text, length, color and position
        for i in range(int(self.noise_param)):
            length = np.random.randint(10, 30)
            text = ''.join(choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            draw.text(pos, text, color, font=font)

        return text_img


    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img = load_img(img_path)

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop(img)

        # Corrupt source image
        source = self._corrupt(img)
        source = self._to_tensor(source)

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = self._to_tensor(img)
        else:
            target = self._corrupt(img)
            target = self._to_tensor(target)

        return source, target


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)
