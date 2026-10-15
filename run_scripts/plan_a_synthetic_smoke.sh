#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi.causal synthetic-generate --config configs/plan_a/synthetic_smoke.yaml
python -m proxcabi.causal fit-bridge \
  --config configs/plan_a/synthetic_smoke.yaml \
  --input outputs/plan_a/synthetic_smoke/observations.npz \
  --output-dir outputs/plan_a/discrete_smoke
python -m proxcabi.causal evaluate-identification \
  --pred-dir outputs/plan_a/discrete_smoke \
  --oracle outputs/plan_a/synthetic_smoke/oracle_truth.json
