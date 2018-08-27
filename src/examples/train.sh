python3 ../train.py \
  --data ../../data \
  --ckpt-save-path ../../ckpts \
  --report-interval 50 \
  --nb-epochs 10 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64 \
  --plot-stats
