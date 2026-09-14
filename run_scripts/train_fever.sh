export CUDA_VISIBLE_DEVICES=2
python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_roberta \
  --backbone roberta-base \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta \
  --data-dir data \
  --dataset FEVER \
  --split symmetric
