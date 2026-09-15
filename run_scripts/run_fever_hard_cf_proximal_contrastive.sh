#!/usr/bin/env bash
set -euo pipefail

bash run_scripts/build_fever_hard_counterfactuals.sh
bash run_scripts/train_fever_hard_cf_proximal_contrastive.sh
