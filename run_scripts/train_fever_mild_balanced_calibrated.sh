export CUDA_VISIBLE_DEVICES=1
python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_roberta_mild_balanced \
  --backbone roberta-base \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --balanced-loss \
  --class-weight-power 0.5 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta_mild_balanced \
  --data-dir data \
  --dataset FEVER \
  --split dev \
  --batch-size 32 \
  --calibration-split dev

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta_mild_balanced \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size 32 \
  --calibration-split dev
