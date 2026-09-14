export CUDA_VISIBLE_DEVICES=7
python -m proxcabi train \
  --data-dir data \
  --dataset VitaminC \
  --output-dir outputs/vitaminc_roberta \
  --backbone roberta-base \
  --epochs 2 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/vitaminc_roberta \
  --data-dir data \
  --dataset VitaminC \
  --split test_real

python -m proxcabi evaluate \
  --checkpoint-dir outputs/vitaminc_roberta \
  --data-dir data \
  --dataset VitaminC \
  --split test_synthetic
