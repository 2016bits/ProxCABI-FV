export CUDA_VISIBLE_DEVICES=5
python -m proxcabi train \
  --data-dir data \
  --dataset PolitiHop \
  --output-dir outputs/politihop_roberta \
  --backbone roberta-base \
  --epochs 10 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/politihop_roberta \
  --data-dir data \
  --dataset PolitiHop \
  --split symmetric