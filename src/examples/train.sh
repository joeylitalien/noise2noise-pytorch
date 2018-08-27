python3 ../train.py \
  --data ../../data --train-size 20 --valid-size 4 \
  --ckpt-save-path ../../ckpts \
  --report-interval 5 \
  --nb-epochs 10 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64 \
  --plot-stats
