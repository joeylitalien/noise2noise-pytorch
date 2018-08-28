python3 ../train.py \
--data ../../data --train-size 5000 --valid-size 1000 \
--ckpt-save-path ../../ckpts \
--report-interval 250 \
--nb-epochs 50 \
--loss l2 \
--noise-type gaussian \
--noise-param 50 \
--crop-size 128 \
--plot-stats \
--cuda

python3 ../train.py \
  --data ../../data --train-size 5000 --valid-size 1000 \
  --ckpt-save-path ../../ckpts \
  --report-interval 250 \
  --nb-epochs 50 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 128 \
  --clean-targets \
  --plot-stats \
  --cuda
  
python3 ../train.py \
  --data ../../data --train-size 5000 --valid-size 1000 \
  --ckpt-save-path ../../ckpts \
  --report-interval 250 \
  --nb-epochs 50 \
  --loss l1 \
  --noise-type text \
  --noise-param 50 \
  --crop-size 128 \
  --plot-stats \
  --cuda
  
python3 ../train.py \
  --data ../../data --train-size 5000 --valid-size 1000 \
  --ckpt-save-path ../../ckpts \
  --report-interval 250 \
  --nb-epochs 50 \
  --loss l1 \
  --noise-type text \
  --noise-param 50 \
  --crop-size 128 \
  --clean-targets \
  --plot-stats \
  --cuda