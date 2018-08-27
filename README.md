# Noise2Noise

This is a *weekend (partial and unfinished)* PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018). As such, it is still very much a work-in-progress.

## Dependencies

* [torch](https://pytorch.org/) (0.4.1)
* [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) (0.2.0)
* [numpy](http://www.numpy.org/) (1.14.2)
* [matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)

To install the latest version of all packages, run
```
pip3 install --user -r requirements.txt
```

This code was tested on Python 3.6.5 on macOS High Sierra (10.13.4). It *will* fail on Python 2.7.x due to usage of 3.6-specific functions. Note that training and testing will also fail on Windows out of the box due to differences in path resolver (`os.path`).

## Dataset

The authors use [ImageNet](http://image-net.org/download), but any dataset will do. [COCO 2017](http://cocodataset.org/#download) has a small validation set (1 GB) which can be nicely split into train/valid for easier training. For instance, to have a 4200/800 split, you could do:
```
mkdir data && cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/val2017.zip
unzip -j val2017.zip -d train
cd train && mv `ls | head -800` ../valid
rm ../*.zip
```

You can also download the full datasets (17 GB) if you have the bandwidth:

```
mkdir data && cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip -j train2017.zip -d train
unzip -j val2017.zip -d valid
rm *.zip
```

Add your favorite images to the `data/test` folder. Only a handful will do to visually inspect the denoiser performance.

## Training

See `python3 train.py --h` for list of optional arguments, or `examples/train.sh` for an example.

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. The program assumes that the directory passed to `--data` contains subdirectories `train` and `valid`. To train and validate on smaller datasets, use the `--train-size` and `--valid-size` options. To plot stats as the model trains, use `--plot-stats`; these are saved alongside checkpoints.

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 train.py \
  --ckpt-save-path ../ckpts \
  --data ../data --train-size 500 --valid-size 100 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64 \
  --plot-stats
```

### Poisson noise
The noise parameter is the Poisson parameter λ.
```
python3 train.py
  --noise-type poisson \
  --noise-param 50 \
  --crop-size 64
```

### Text overlay
The noise parameter is the number of text artifacts overlayed.
```
python3 train.py \
  --noise-type text \
  --noise-param 50 \
  --crop-size 64
```

## Testing

Model checkpoints are automatically saved after every epoch. To test the denoiser, provide `test.py` with a PyTorch model (`.pt` file) with the argument `--load-ckpt`. This assumes the existence of a `test` directory under your data folder. The `--show-output` option specifies the number of noisy/denoised/clean montages to display. To disable this, simply remove `--show-output`.

```
python3 test.py \
  --data ../data \
  --load-ckpt ../ckpts/gaussian/n2n.pt \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 256 \
  --show-output 3
```

See `python3 test.py --h` for list of optional arguments, or `examples/test.sh` for an example.

## To do list
- [x] Test Gaussian noise and text overlay thoroughly so they work as intended
- [x] Track validation loss and PSNR over time to plot
- [x] Implement Poisson noise with L2 loss
- [ ] Find elegant solution to variable-size images (fix size, or modify architecture?)
- [ ] Add *p* parameter to text train/valid routines
- [ ] Implement Monte Carlo rendering noise (will require HDR-specific methods)
- [ ] Fix RedNet baseline skip connections (low priority)

## References
* Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala,and Timo Aila. *Noise2Noise: Learning Image Restoration without Clean Data*, in Proceedings of ICML, 2018.

* Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. *Microsoft COCO: Common Objects in Context*. 	arXiv:1405.0312, 2014.

## Acknowledgments

I would like to acknowledge [Yusuke Uchida](https://yu4u.github.io/) for his [Keras implementation of Noise2Noise](https://github.com/yu4u/noise2noise). Although Keras and PyTorch syntaxes are very different, his code did help me in completing this implementation.
