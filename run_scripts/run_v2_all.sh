#!/usr/bin/env bash
set -euo pipefail

base_config="${1:-configs/v2/fever_base.yaml}"
pilot_config="${2:-configs/v2/fever_pilot.yaml}"

if [[ ! -f outputs/v2/frozen_data/data_audit.json ]]; then
  python -m proxcabi v2-prepare \
    --config "$base_config" \
    --output-dir outputs/v2/frozen_data
fi

if [[ ! -f outputs/v2/proxy_artifacts/fever/proxy_builder_v2.json ]]; then
  python -m proxcabi v2-fit-proxies \
    --config "$base_config" \
    --output-dir outputs/v2/proxy_artifacts/fever
fi

python -m pytest -q tests/v2

python -m proxcabi v2-diagnose \
  --config "$base_config" \
  --output-dir outputs/v2/diagnostics

run_scripts/run_v2_smoke.sh \
  "$pilot_config" \
  outputs/v2/pilot/M1_seed13

python -m proxcabi v2-report \
  --runs-dir outputs/v2/pilot \
  --output-dir outputs/v2/paper_tables

if [[ "${V2_FULL:-0}" != "1" ]]; then
  echo "Core pipeline complete. Set V2_FULL=1 only after trusted pair verification."
  exit 0
fi

python - "$base_config" <<'PY'
import json
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
pool = Path(config["paths"]["pair_pool_train"])
counts = {"invariant": 0, "semantic": 0}
for line in pool.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    metadata = row.get("metadata", {})
    if metadata.get("verified") is True:
        kind = str(metadata.get("transform_type", "")).split("_", 1)[0]
        if kind in counts:
            counts[kind] += 1
if not all(counts.values()):
    raise SystemExit(
        "Full M2-M4 grid refused: the configured pool needs trusted verified "
        f"invariant and semantic pairs, observed {counts}."
    )
print("Verified pair coverage:", counts)
PY

for seed in 13 21 42; do
  for variant in M1 M2 M3 M4; do
    run_dir="outputs/v2/main/${variant}/seed_${seed}"
    python -m proxcabi v2-train \
      --config "$base_config" \
      --variant "$variant" \
      --seed "$seed" \
      --output-dir "$run_dir"
    python -m proxcabi v2-evaluate \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$base_config" \
      --split dev_report \
      --output-dir "$run_dir/eval_id"
    python -m proxcabi v2-evaluate \
      --checkpoint "$run_dir/best_model.pt" \
      --config "$base_config" \
      --split symmetric \
      --output-dir "$run_dir/eval_ood"
  done
done

python -m proxcabi v2-report \
  --runs-dir outputs/v2/main \
  --output-dir outputs/v2/paper_tables
