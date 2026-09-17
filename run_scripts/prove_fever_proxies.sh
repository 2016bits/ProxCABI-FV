#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi prove-proxies \
  --data-dir data \
  --output-dir outputs/proxy_proof \
  --dataset FEVER \
  --split train \
  --max-samples "${MAX_SAMPLES:-3000}" \
  --seed "${SEED:-13}" \
  --bridge-max-eval-samples "${BRIDGE_MAX_EVAL_SAMPLES:-500}" \
  --bridge-max-w-classes "${BRIDGE_MAX_W_CLASSES:-64}" \
  --balanced-sample
