#!/usr/bin/env bash
set -euo pipefail

bash run_scripts/build_fever_hard_nli_counterfactuals.sh
bash run_scripts/train_fever_deberta_hard_nli_proximal_contrastive.sh
