# Noise2Noise
This is a *weekend (partial and unfinished)* PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018). As such, it is still very much a work-in-progress.

## Dependencies
* [numpy](http://www.numpy.org/) (1.14.2)
* [torch](https://pytorch.org/) (0.4.1)
* [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) (0.2.0)
* [matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)

## Dataset

The authors use [ImageNet](http://image-net.org/download), but any dataset will do. [COCO 2017](http://cocodataset.org/#download) has a small validation set (1GB) which can be nicely split into train/valid/test for easier training.

## Training

See `python3 noise2noise.py --h` for list of optional arguments, or `train.sh` for an example.

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. The program assumes that the directory passed to `--data` contains subdirectories `train` and `valid`. To train and validate on smaller datasets, create `train_redux` and `valid_redux`, and use the `--redux` option. 

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 noise2noise.py \
  --data ./../data \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64
```

### Poisson noise
The noise parameter is the Poisson parameter λ.
```
python3 noise2noise.py 
  --data ./../data \
  --noise-type poisson \
  --noise-param 50 \
  --crop-size 64
```

### Text overlay
The noise parameter is the number of text artifacts overlayed.
```
python3 noise2noise.py \
  --data ./../data \
  --noise-type text \
  --noise-param 50 \
  --crop-size 64
```

## Testing

Model checkpoints are automatically saved after every epoch. To test the denoiser, simply pass a PyTorch model (.pt file) to `--load-ckpt`. This assumes the existence of a `test` directory under your data folder. By default, a montage of noisy/denoised/clean images is output to the screen. To disable this, set the `--show-output` flag to false.

See `test.sh` for an example.

## To do list
- [ ] Test Gaussian noise and text overlay thoroughly so they work as intended
- [ ] Find elegant solution to variable-size images (fix size, or modify architecture?)
- [ ] Implement Poisson noise with L2 loss
- [ ] Add *p* parameter to train/valid routines
- [ ] Track validation loss and PSNR over time to plot
- [ ] Implement Monte Carlo rendering noise (will require HDR-specific methods)
- [ ] Test RedNet baseline (low priority)
