#!/usr/bin/env bash
set -euo pipefail

python -m proxcabi.causal synthetic-stress --config configs/plan_a/synthetic_stress.yaml
