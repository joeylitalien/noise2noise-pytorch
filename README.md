# Noise2Noise
PyTorch (partial and unfinished) implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018)

# Training

See `python3 noise2noise.py --h` for list of optional arguments, or `train.sh` for an example. For instance, 

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. The program assumes that the directory passed to `--data` contains subdirectories `train` and `valid`. To train and validate on smaller datasets, create `train_redux` and `valid_redux`, and use the `--redux` option. 

## Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 noise2noise.py \
  --data ./../data \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64
```

## Poisson noise
The noise parameter is the Poisson parameter λ.
```
python3 noise2noise.py 
  --data ./../data \
  --noise-type poisson \
  --noise-param 50 \
  --crop-size 64
```

## Text overlay
The noise parameter is the number of text artifacts overlayed.
```
python3 noise2noise.py \
  --data ./../data \
  --noise-type text \
  --noise-param 50 \
  --crop-size 64
```

# Testing

Model checkpoints are automatically savec after every epoch. To test the denoiser on a test set, simply pass a PyTorch model (.pt file) to `--load-ckpt`. This assumes the existence of a `test` directory under your data folder. By default, a montage of noisy/denoised/clean images is output to the screen. To disable this, set the `--show-output` flag to false.
