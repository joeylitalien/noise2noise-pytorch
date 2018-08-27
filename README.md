# Noise2Noise

This is a *weekend (partial and unfinished)* PyTorch implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018). As such, it is still very much a work-in-progress.

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
unzip -j val2017.zip -d train
cd train && mv `ls | head -800` ../valid
rm ../*.zip
```

You can also download the full datasets (7 GB) that more or less match the paper, if you have the bandwidth:

```
mkdir data && cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip -j test2017.zip -d train
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
  --report-interval 25 \
  --data ../data --train-size 500 --valid-size 100 \
  --learning-rate 0.001 \
  --adam 0.9, 0.99, 1e-8 \
  --batch-size 4 \
  --nb-epochs 100 \
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
  --noise-param 50
```

### Text overlay
The noise parameter is the number of text artifacts overlayed.
```
python3 train.py \
  --loss l1 \
  --noise-type text \
  --noise-param 80
```

### Monte Carlo noise

See [instructions below](#monte-carlo-rendering).

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

## Monte Carlo rendering

### Downloading and building Tungsten

You can use your favourite Monte Carlo physically-based renderer to generate noisy images, but you need to be able to extract albedo and normal buffers. As such, we use [Tungsten](https://github.com/tunabrain/tungsten) by [Benedikt Bitterli](https://benedikt-bitterli.me) since it provides an easy way to retrieve these buffers. Assuming you have an up-to-date version of [CMake](https://cmake.org/download/) installed with `gcc`, run:

```
git clone https://github.com/tunabrain/tungsten.git
./setup_builds.sh
cd build/release
make
```

Add Tungsten to your path so you don't have to specify its location.
```
echo `export PATH="{your-tungsten-release-dir}":$PATH` >> ~/.profile
source ~/.profile
```

Run `tungsten -v` to see if the path is correctly set. If you see Tungsten's version, you are good to go.

### Downloading scene files

The original paper use 860 architectural images; we use a single scene for testing purposes.

```
cd data && mkdir scenes
wget https://benedikt-bitterli.me/resources/tungsten/car.zip
unzip car.zip
rm *.zip
```

### Generating renders
To launch a series of renders to build a training set, do:

```
python3 render.py \
  --scene-path ../../data/scenes/car/scene.json \
  --spp 4 \
  --nb-renders 10 \
  --output-dir ../../data/train
```

You can also specify the path to Tungsten if you have it installed somewhere else with the `--tungsten` argument. The default assumes it's in the environment path already.

See `python3 render.py -h` for more info, or `render.sh` for the above example. You will have to manually remove the default output PNG images from your render directory after the jobs are done (`rm ../../data/renders/*.png`).

### Training (not implemented yet)
```
python3 train.py \
  --ckpt-save-path ../ckpts \
  --data ../data --train-size 10 --valid-size 1 \
  --nb-epochs 1000 \
  --learning-rate 0.001 \
  --loss rmse \
  --noise-type mc \
  --crop-size 64 \
  --plot-stats
```


## To do list
- [x] Test Gaussian noise and text overlay thoroughly so they work as intended
- [x] Track validation loss and PSNR over time to plot
- [x] Implement Poisson noise with L2 loss
- [ ] Train on a half-decent GPU and add results
- [ ] Find elegant solution to variable-size images (fix size, or modify architecture?)
- [ ] Monte Carlo rendering noise
  - [x] Generate MC renders with albedo and normal buffers using Tungsten
  - [ ] Implement HDR-specific functions (e.g. Reinhard tone mapping)
  - [ ] Pass to U-Net
- [ ] Fix RedNet baseline skip connections (low priority)

## References
* Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala,and Timo Aila. *Noise2Noise: Learning Image Restoration without Clean Data*, in Proceedings of ICML, 2018.

* Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. *Microsoft COCO: Common Objects in Context*. 	arXiv:1405.0312, 2014.

## Acknowledgments

I would like to acknowledge [Yusuke Uchida](https://yu4u.github.io/) for his [Keras implementation of Noise2Noise](https://github.com/yu4u/noise2noise). Although Keras and PyTorch are very different frameworks, parts of his code did help me in completing this implementation.
