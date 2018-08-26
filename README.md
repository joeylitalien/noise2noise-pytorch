# Noise2Noise
This is a *weekend (partial and unfinished)* PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018). As such, it is still very much a work-in-progress.

## Dependencies

* [torch](https://pytorch.org/) (0.4.1)
* [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) (0.2.0)
* [numpy](http://www.numpy.org/) (1.14.2)
* [matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)

Tested on Python 3.6.5. Code *will* fail on Python 2.7.x.

## Dataset

The authors use [ImageNet](http://image-net.org/download), but any dataset will do. [COCO 2017](http://cocodataset.org/#download) has a small validation set (1GB) which can be nicely split into train/valid/test for easier training.

## Training

See `python3 train.py --h` for list of optional arguments, or `examples/train.sh` for an example.

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. The program assumes that the directory passed to `--data` contains subdirectories `train` and `valid`. To train and validate on smaller datasets, create `train_redux` and `valid_redux`, and use the `--redux` option.

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 train.py \
  --data ./../data \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64
```

### Poisson noise
The noise parameter is the Poisson parameter λ.
```
python3 train.py
  --data ./../data \
  --noise-type poisson \
  --noise-param 50 \
  --crop-size 64
```

### Text overlay
The noise parameter is the number of text artifacts overlayed.
```
python3 train.py \
  --data ./../data \
  --noise-type text \
  --noise-param 50 \
  --crop-size 64
```

## Testing

Model checkpoints are automatically saved after every epoch. To test the denoiser, simply pass a PyTorch model (`.pt` file) to `--load-ckpt`. This assumes the existence of a `test` directory under your data folder. The `--show-output` option specifies the number of noisy/denoised/clean montages to display. To disable this, simply remove `--show-output`.

```
python3 ../test.py \
  --data ../../data \
  --load-ckpt ../../ckpts/gaussian/n2n.pt \
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
