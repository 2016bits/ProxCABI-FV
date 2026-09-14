export CUDA_VISIBLE_DEVICES=3
python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size 64 \
  --calibration-split dev \
  --fever-symmetric-group-decode
