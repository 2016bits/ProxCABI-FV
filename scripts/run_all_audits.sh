#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi list --data-dir data

python -m proxcabi audit \
  --data-dir data \
  --datasets FEVER HOVER PolitiHop VitaminC \
  --splits train dev test symmetric test_real test_synthetic \
  --output-dir outputs/audit

for dataset in FEVER HOVER PolitiHop VitaminC; do
  max_samples=5000
  if [[ "$dataset" == "PolitiHop" ]]; then
    max_samples=
  fi
  if [[ -n "$max_samples" ]]; then
    python -m proxcabi diagnose --data-dir data --dataset "$dataset" --split train --max-samples "$max_samples" --output-dir outputs/diagnostics
  else
    python -m proxcabi diagnose --data-dir data --dataset "$dataset" --split train --output-dir outputs/diagnostics
  fi
done
