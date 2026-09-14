export CUDA_VISIBLE_DEVICES=1
python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_roberta_cf_contrastive \
  --backbone roberta-base \
  --train-split train \
  --eval-split dev \
  --counterfactual-split cf_train \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --contrastive-weight 0.2 \
  --contrastive-margin 1.0 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta_cf_contrastive \
  --data-dir data \
  --dataset FEVER \
  --split dev \
  --batch-size 32 \
  --calibration-split dev

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta_cf_contrastive \
  --data-dir data \
  --dataset FEVER \
  --split symmetric \
  --batch-size 32 \
  --calibration-split dev
