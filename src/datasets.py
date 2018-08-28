#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from random import choice
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import Imath
import OpenEXR
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)
    
    # Instantiate appropriate dataset class
    if params.noise_type == 'mc':
        dataset = MonteCarloDataset(root_dir, redux, params.crop_size, params.tonemap)
    else:
        dataset = NoisyDataset(root_dir, redux, params.crop_size, noise_dist=noise,
                    clean_targets=params.clean_targets)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size


    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)
        
        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)
        
        
class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, noise_dist=('gaussian', 50.), clean_targets=False):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]

        # Use clean targets
        self.clean_targets = clean_targets


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
        font = ImageFont.truetype('Times New Roman.ttf', 24)
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
        img =  Image.open(img_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target
        
        
class MonteCarloDataset(AbstractDataset):
    """Class for dealing with HDR Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size, tone_mapping='source'):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, 'render'))
        self.albedos = os.listdir(os.path.join(root_dir, 'albedo'))
        self.normals = os.listdir(os.path.join(root_dir, 'normal'))

        if redux:
            self.imgs = self.imgs[:redux]
            self.albedos = self.albedos[:redux]
            self.normals = self.normals[:redux]

        # Tone mapping (none, source, target, or both)
        self.tone_mapping = tone_mapping


    def _load_hdr(self, img_path, tonemap):
        """Converts OpenEXR image to PIL format."""

        # Read OpenEXR file
        if not OpenEXR.isOpenExrFile(img_path):
            raise ValueError('HDR images must be in OpenEXR (.exr) format')
        src = OpenEXR.InputFile(img_path)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = src.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Load as float and tonemap
        rgb = [np.frombuffer(src.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
        if tonemap:
            rgb = reinhard_tonemap(rgb)
        rgb8 = [Image.frombytes('F', size, c.tostring()).convert('L') for c in rgb]
        
        return Image.merge('RGB', rgb8)


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Select random Monte Carlo render for target
        indices = list(range(len(self.imgs)))
        indices.remove(index)
        target_index = choice(indices)

        # Apply tone mapping
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        target_path = os.path.join(self.root_dir, 'render', self.imgs[target_index])
        render = Image.open(render_path).convert('RGB')
        target = Image.open(target_path).convert('RGB') 

        # Get other buffers
        #render = tvF.to_tensor(render)
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        albedo = Image.open(albedo_path).convert('RGB')
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])
        normal = Image.open(normal_path).convert('RGB')
        
        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target

    """
    def __getitem__(self, index):

        # Select random Monte Carlo render for target
        indices = list(range(len(self.imgs)))
        indices.remove(index)
        target_index = choice(indices)

        # Apply tone mapping
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        target_path = os.path.join(self.root_dir, 'render', self.imgs[target_index])
        
        # Apply tone mapping to desired images
        source_tonemap = self.tone_mapping in ['source', 'both']
        target_tonemap = self.tone_mapping in ['target', 'both']
        render = self._load_hdr(render_path, tonemap=source_tonemap)
        target = self._load_hdr(target_path, tonemap=target_tonemap)   

        # Get other buffers
        #render = tvF.to_tensor(render)
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        albedo = self._load_hdr(albedo_path, tonemap=source_tonemap)
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])
        normal = self._load_hdr(normal_path, tonemap=source_tonemap)
        
        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target
    """
        
        
if __name__ == '__main__':
    mc = MonteCarloDataset('../data/tonemapped_train', 0, 128, 'both')
    s, t = mc[0]
    t = tvF.to_pil_image(t)
    s0 = tvF.to_pil_image(s.narrow(0, 0, 3))
    s1 = tvF.to_pil_image(s.narrow(0, 3, 3))
    s2 = tvF.to_pil_image(s.narrow(0, 6, 3))
    s0.save('source.png')
    s1.save('albedo.png')
    s2.save('normal.png')
    t.save('target.png')