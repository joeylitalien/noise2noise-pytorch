python3 ../train.py \
  --data ../../data \
  --redux \
  --ckpt-save-path ../../ckpts \
  --report-interval 25 \
  --nb-epochs 10 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 64
