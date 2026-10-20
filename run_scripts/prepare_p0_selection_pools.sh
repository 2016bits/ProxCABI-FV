#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
train_pool="${P0_TRAIN_PAIR_POOL:-outputs/v2/paper/pairs/pairs_train_verified.jsonl}"
warmup_root="${P0_WARMUP_ROOT:-outputs/v2/paper/runs/components/base_only}"
config_root="outputs/v2/paper/task_matrix/configs"
pool_root="outputs/v2/paper/selection/pools"
mkdir -p "$pool_root" outputs/v2/paper/selection/scores
IFS=',' read -r -a gpus <<<"${GPUS:-0,1,2}"
mkdir -p outputs/v2/paper/selection/logs

slot=0
pids=()
for seed in ${V2_SEEDS:-13 21 42}; do
  gpu="${gpus[$((slot % ${#gpus[@]}))]}"
  slot=$((slot + 1))
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    checkpoint="$warmup_root/seed_${seed}/best_model.pt"
    config="$config_root/components__base_only__seed_${seed}.yaml"
    scores="outputs/v2/paper/selection/scores/seed_${seed}.jsonl"
    score_manifest="${scores%.jsonl}.manifest.json"
    [[ -f "$checkpoint" ]] || { echo "Missing warm-up checkpoint: $checkpoint" >&2; exit 1; }
    if [[ ! -f "$scores" || ! -f "$score_manifest" ]] || \
       ! "$python_bin" -c '
import hashlib,json,sys
def digest(path):
    value=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()
m=json.load(open(sys.argv[1]))
raise SystemExit(0 if m.get("checkpoint_sha256")==digest(sys.argv[2]) and m.get("pair_pool_sha256")==digest(sys.argv[3]) else 1)
' "$score_manifest" "$checkpoint" "$train_pool"; then
      "$python_bin" scripts/paper_v2/score_pair_pool.py --checkpoint "$checkpoint" --config "$config" \
        --pair-pool "$train_pool" --output "$scores"
    fi
    "$python_bin" scripts/paper_v2/prepare_selection_pools.py --pair-pool "$train_pool" \
      --scores "$scores" --output-dir "$pool_root" --seed "$seed" --budget 5000 --inv-sem-ratio 0.5
  ) >"outputs/v2/paper/selection/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "$failures selector preparation job(s) failed; inspect outputs/v2/paper/selection/logs/." >&2
  exit 1
fi

