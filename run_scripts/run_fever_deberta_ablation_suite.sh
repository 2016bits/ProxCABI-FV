#!/usr/bin/env bash
set -euo pipefail

bash run_scripts/build_fever_ablation_counterfactuals.sh

for ablation in no_z no_w no_bridge_prox no_cf random_cf easy_cf; do
  bash run_scripts/train_fever_deberta_ablation.sh "${ablation}"
done
