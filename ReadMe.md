# Noise2Noise: Learning Image Restoration without Clean Data

This is a PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018).

## Dependencies

* [PyTorch](https://pytorch.org/) (0.4.1)
* [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html) (0.2.0)
* [NumPy](http://www.numpy.org/) (1.14.2)
* [Matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)
* [OpenEXR](http://www.openexr.com/) (1.3.0)

To install the latest version of all packages, run
```
pip3 install --user -r requirements.txt
```

This code was tested on Python 3.6.5 on macOS High Sierra (10.13.4) and Ubuntu 16.04. It will fail with Python 2.7.x due to usage of 3.6-specific functions. Note that training and testing will also fail on Windows out of the box due to differences in the path resolvers (`os.path`).

## Dataset

The authors use [ImageNet](http://image-net.org/download), but any dataset will do. [COCO 2017](http://cocodataset.org/#download) has a small validation set (1 GB) which can be nicely split into train/valid for easier training. For instance, to obtain a 4200/800 train/valid split you can do:
```
mkdir data && cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && cd val2017
mv `ls | head -4200` ../train
mv `ls | head -800` ../valid
```

You can also download the full datasets (7 GB) that more or less match the paper, if you have the bandwidth:

```
mkdir data && cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip -j test2017.zip -d train
unzip -j val2017.zip -d valid
```

Add your favorite images to the `data/test` folder. Only a handful will do to visually inspect the denoiser performance.

## Training

See `python3 train.py --h` for list of optional arguments, or `examples/train.sh` for an example.

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. To train and validate on smaller datasets, use the `--train-size` and `--valid-size` options. To plot stats as the model trains, use `--plot-stats`; these are saved alongside checkpoints. By default CUDA is not enabled: use the `--cuda` option if you have a GPU that supports it.

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 train.py \
  --train-dir ../data/train --train-size 1000 \
  --valid-dir ../data/valid --valid-size 200 \
  --ckpt-save-path ../ckpts \
  --nb-epochs 10 \
  --batch-size 4 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64 \
  --plot-stats \
  --cuda
```

### Poisson noise
The noise parameter is the Poisson parameter λ.
```
python3 train.py
  --loss l2 \
  --noise-type poisson \
  --noise-param 50 \
  --cuda
```

### Text overlay
The noise parameter is the approximate probability *p* that a pixel is covered by text.
```
python3 train.py \
  --loss l1 \
  --noise-type text \
  --noise-param 0.5 \
  --cuda
```

### Monte Carlo rendering noise

See [other read-me file](MonteCarlo.md).

## Testing

Model checkpoints are automatically saved after every epoch. To test the denoiser, provide `test.py` with a PyTorch model (`.pt` file) via the argument `--load-ckpt`. This assumes the existence of a `test` directory under your data folder. The `--show-output` option specifies the number of noisy/denoised/clean montages to display on screen. To disable this, simply remove `--show-output`.

```
python3 test.py \
  --data ../data \
  --load-ckpt ../ckpts/gaussian/n2n.pt \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 256 \
  --show-output 3 \
  --cuda
```

See `python3 test.py --h` for list of optional arguments, or `examples/test.sh` for an example.

## Results

Model was only trained for 30 epochs with a train/valid split of 1000/200 on a GTX 780. Much better results can be achieved with a larger dataset and longer training time. I might upload better results when I get the time to train on a P100 cluster.

<table align="center">
  <tr align="center">
    <th colspan=9>Gaussian noise (σ = 25)</td>
  </tr>
  <tr align="center">
    <td colspan=2>Noisy input (21.75 dB)</td>
    <td colspan=2>Denoised (31.73 dB)</td>
    <td colspan=2>Clean targets (31.70 dB)</td>
    <td colspan=2>Ground truth</td>
  </tr>
  <tr align="center">
    <td colspan=2><img src="figures/lenna-gaussian-noisy.png"></td>
    <td colspan=2><img src="figures/lenna-gaussian-denoised.png"></td>
    <td colspan=2><img src="figures/lenna-gaussian-clean-denoised.png"></td>
    <td colspan=2><img src="figures/lenna.png"></td>
  </tr>  
  <tr align="center">
    <td colspan=2>Noisy input (27.15 dB)</td>
    <td colspan=2>Denoised (35.97 dB)</td>
    <td colspan=2>Clean targets (35.94 dB)</td>
    <td colspan=2>Ground truth</td>
  </tr>
  <tr align="center">
    <td colspan=2><img src="figures/monarch-gaussian-noisy.png"></td>
    <td colspan=2><img src="figures/monarch-gaussian-denoised.png"></td>
    <td colspan=2><img src="figures/monarch-gaussian-clean-denoised.png"></td>
    <td colspan=2><img src="figures/monarch.png"></td>
  </tr> 

  <tr align="center">
    <th colspan=9>Text overlay (<i>p</i> = 0.25)</td>
  </tr>
  <tr align="center">
  <td colspan=2>Noisy input (20.28 dB)</td>
  <td colspan=2>Denoised (33.71 dB)</td>
  <td colspan=2>Clean targets (? dB)</td>
  <td colspan=2>Ground truth</td>
  </tr>
  <tr align="center">
    <td colspan=2><img src="figures/lenna-text-noisy.png"></td>
    <td colspan=2><img src="figures/lenna-text-denoised.png"></td>
    <td colspan=2><img src="figures/lenna-text-clean-denoised.png"></td>
    <td colspan=2><img src="figures/lenna.png"></td>
  </tr>  
  <tr align="center">
    <td colspan=2>Noisy input (19.43 dB)</td>
    <td colspan=2>Denoised (33.40 dB)</td>
  <td colspan=2>Clean targets (? dB)</td>
    <td colspan=2>Ground truth</td>
  </tr>
  <tr align="center">
    <td colspan=2><img src="figures/monarch-text-noisy.png"></td>
    <td colspan=2><img src="figures/monarch-text-denoised.png"></td>
    <td colspan=2><img src="figures/monarch-text-clean-denoised.png"></td>
    <td colspan=2><img src="figures/monarch.png"></td>
  </tr>  
</table>

## To do list
- [x] Test Gaussian noise
- [x] Track validation loss and PSNR over time to plot
- [x] Implement Poisson noise with L2 loss
- [x] Added support for maximum occupancy for text corruption
- [x] Train on a half-decent GPU and add results
  - [x] Gaussian noise
  - [x] Text overlay
  - [ ] Poisson noise: unclear how the paper deals with this since Poisson is data-dependent
- [ ] Move all print statements to a `logging` solution
- [ ] Find elegant solution to variable-size images (fix size, or modify architecture?)

## References
* Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala,and Timo Aila. [*Noise2Noise: Learning Image Restoration without Clean Data*](https://research.nvidia.com/publication/2018-07_Noise2Noise%3A-Learning-Image). Proceedings of the 35th International Conference on Machine Learning, 2018.

* Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. [*Microsoft COCO: Common Objects in Context*](https://arxiv.org/abs/1405.0312). arXiv:1405.0312, 2014.

## Acknowledgments

I would like to acknowledge [Yusuke Uchida](https://yu4u.github.io/) for his [Keras implementation of Noise2Noise](https://github.com/yu4u/noise2noise). Although Keras and PyTorch are very different frameworks, parts of his code did help me in completing this implementation.
