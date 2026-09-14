export CUDA_VISIBLE_DEVICES=3
python -m proxcabi train \
  --data-dir data \
  --dataset HOVER \
  --output-dir outputs/hover_roberta \
  --backbone roberta-base \
  --epochs 3 \
  --batch-size 12 \
  --eval-batch-size 24 \
  --fp16