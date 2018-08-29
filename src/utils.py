#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.functional as tvF
import os
import numpy as np
from datetime import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1,
                                                                                        num_batches, loss, int(elapsed),
                                                                                        dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def load_hdr(self, img_path):
    """Converts OpenEXR image to PIL format."""

    # Read OpenEXR file
    if not OpenEXR.isOpenExrFile(img_path):
        raise ValueError('HDR images must be in OpenEXR (.exr) format')
    src = OpenEXR.InputFile(img_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = src.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Load as float
    rgb = [np.frombuffer(src.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
    rgb8 = [Image.frombytes('F', size, c.tostring()).convert('L') for c in rgb]

    return Image.merge('RGB', rgb8)


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    return torch.pow(tensor / (1 + tensor), 1 / 2.2)


def psnr(source_denoised, target):
    """Computes peak signal-to-noise ratio.
    TODO: Find a pure PyTorch minibatch solution that also works on the GPU.
          Not sure if possible since torch.mean() doesn't accept bytes...
    """

    s = source_denoised.detach()
    t = target.detach()

    s = np.array(tvF.to_pil_image(source_denoised.clamp(0, 1)))
    t = np.array(tvF.to_pil_image(target))
    return 10 * np.log10((255 ** 2) / ((s - t) ** 2).mean())


def create_montage(img_name, save_path, source_t, denoised_t, clean_t, show):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.canvas.set_window_title(img_name.capitalize()[:-4])

    # Bring tensors to CPU
    source_t = source_t.cpu()
    denoised_t = denoised_t.cpu()
    clean_t = clean_t.cpu()

    # Convert to PIL images (tonemap images first if specified)
    # if tonemap:
    #     noisy = tvF.to_pil_image(reinhard_tonemap_tensor(noisy_t))
    #     denoised = tvF.to_pil_image(reinhard_tonemap_tensor(denoised_t))
    #     clean = tvF.to_pil_image(reinhard_tonemap_tensor(clean_t))
    # else:
    #     noisy = tvF.to_pil_image(noisy_t)
    #     denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    #     clean = tvF.to_pil_image(clean_t)

    source = tvF.to_pil_image(source_t.narrow(0, 0, 3))
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    clean = tvF.to_pil_image(clean_t)

    # Build image montage
    psnr_vals = [psnr(source_t, clean_t), psnr(source_t, clean_t)]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Ground truth']
    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img)
        ax[j].set_title(title)
        ax[j].axis('off')

    # Open pop up window, if requested
    if show > 0:
        plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
    source.save(os.path.join(save_path, f'{fname}-noisy.png'))
    denoised.save(os.path.join(save_path, f'{fname}-denoised.png'))
    fig.savefig(os.path.join(save_path, f'{fname}-montage.png'), bbox_inches='tight')


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

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
