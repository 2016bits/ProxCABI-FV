#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi audit \
  --data-dir data \
  --datasets FEVER \
  --splits train dev symmetric \
  --max-samples 2000 \
  --output-dir outputs/audit_fever_smoke

python -m proxcabi diagnose \
  --data-dir data \
  --dataset FEVER \
  --split train \
  --max-samples 2000 \
  --output-dir outputs/diagnostics_fever_smoke

python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_roberta_smoke \
  --backbone roberta-base \
  --max-train-samples 512 \
  --max-eval-samples 256 \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --max-length 192 \
  --fp16
