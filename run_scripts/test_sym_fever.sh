export CUDA_VISIBLE_DEVICES=2
python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta \
  --data-dir data \
  --dataset FEVER \
  --split symmetric