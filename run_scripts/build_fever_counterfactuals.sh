python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split train \
  --output-split cf_train \
  --max-groups 40000 \
  --seed 13

python -m proxcabi build-fever-counterfactuals \
  --data-dir data \
  --source-split dev \
  --output-split cf_dev \
  --max-groups 3000 \
  --seed 17
