# Monte Carlo rendering noise

*This is a work-in-progress. A lot of this is not optimal (e.g. rendering the whole scene when we only need a crop of it, single viewpoint).*

## Dataset 

See [below to create your own](#generating-renders), or download the [bathroom scene dataset](https://mcgill-my.sharepoint.com/:u:/g/personal/joey_litalien_mail_mcgill_ca/ESrJBzcYK0VDiapi_-NiFXQBk1GkMUqJw5zeJVzQ0VxJjg?e=ZhEmCv) (48/8/4 path traced renders @ 8 spp with albedo and normal buffers).

You can also download the [dataset](https://benedikt-bitterli.me/nfor/denoising-data.zip) used by [Bitterli et al. (2016)](https://benedikt-bitterli.me/nfor/), which is open source. Note that you will need to manually organize its content into render/albedo/normal subdirectories for this dataset to work with this implementation.

## Downloading and building Tungsten

You can use your favourite Monte Carlo physically-based renderer to generate noisy images, but you need to be able to extract albedo and normal buffers. As such, we use [Tungsten](https://github.com/tunabrain/tungsten) by [Benedikt Bitterli](https://benedikt-bitterli.me) since it provides an easy way to retrieve these buffers. Assuming you have an up-to-date version of [CMake](https://cmake.org/download/) installed with `gcc`, run:

```
git clone https://github.com/tunabrain/tungsten.git
./setup_builds.sh
cd build/release
make
```

Add Tungsten to your path so you don't have to specify its location (use &#96; quotations and `.profile` on macOS):
```
echo 'export PATH="<tungsten-release-dir>":$PATH' >> ~/.bashrc
source ~/.bashrc
```

Run `tungsten -v` to see if the path is correctly set. If you see Tungsten's version, you are good to go.

## Downloading scene files

The original paper use 860 architectural images; we use a single scene for testing purposes.

```
cd data && mkdir scenes
wget https://benedikt-bitterli.me/resources/tungsten/bathroom.zip
unzip bathroom.zip -d scenes
rm *.zip
```

## Generating renders
To launch a series of renders to build a training set, do:

```
python3 render.py \
  --scene-path ../data/scenes/bathroom/scene.json \
  --spp 8 \
  --nb-renders 48 \
  --output-dir ../data/mc/train \
  --hdr-targets
```

You can also specify the path to Tungsten if you have it installed somewhere else with the `--tungsten` argument. The default assumes it's in the environment path already. Moreover, images are tonemapped using [Reinhard](https://www.cs.utah.edu/~reinhard/cdrom/) by default; to save images as HDR images (OpenEXR format), use the `--hdr-<output-type>` options.

A specific width and height can be given with `--resolution <width> <height>` but you will need to re-render a ground truth image if the width/height ratio is not preserved. References images are automatically resized otherwise.

See `python3 render.py -h` for more info, or run `render.sh` for an example.

## Training
```
python3 ../train.py \
  --train-dir ../../data/mc/train --train-size 48 \
  --valid-dir ../../data/mc/valid --valid-size 8 \
  --ckpt-save-path ../../ckpts \
  --ckpt-overwrite \
  --report-interval 4 \
  --nb-epochs 50 \
  --batch-size 4 \
  --loss hdr \
  --noise-type mc \
  --crop-size 128 \
  --plot-stats \
  --cuda
```

## Testing
```
python3 ../test.py \
  --data ../../data/mc/test \
  --load-ckpt ../../ckpts/mc/n2n-mc.pt \
  --noise-type mc \
  --crop-size 512 \
  --show-output 1
```

## Results
Eventually...


## Additional references

* Benedikt Bitterli, Fabrice Rousselle, Bochang Moon, José A. Iglesias-Guitián, David Adler, Kenny Mitchell, Wojciech Jarosz, and Jan Novák. [*Nonlinearly Weighted First-order Regression
for Denoising Monte Carlo Renderings*](https://benedikt-bitterli.me/nfor/). Computer Graphics Forum (Proceedings of EGSR 2016), 2016.

* Benedikt Bitterli. [*Tungsten Renderer*](https://github.com/tunabrain/tungsten). GitHub repository: `tunabrain/tungsten`, 2018.
