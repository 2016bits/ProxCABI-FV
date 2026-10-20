#!/usr/bin/env bash
set -euo pipefail

jobs_file="${1:-outputs/v2/paper/task_matrix/jobs.tsv}"
max_parallel="${MAX_PARALLEL:-2}"
gpus_csv="${GPUS:-0,1}"
family_filter="${P0_FAMILY:-all}"
name_filter="${P0_NAME:-all}"
invariant_pool="${EVAL_INVARIANT_POOL:-outputs/v2/paper/pairs/pairs_eval_invariant.jsonl}"
semantic_pool="${EVAL_SEMANTIC_POOL:-outputs/v2/paper/pairs/pairs_eval_semantic.jsonl}"
logs_dir="${P0_LOG_DIR:-outputs/v2/paper/logs}"
mkdir -p "$logs_dir"

if [[ ! -f "$jobs_file" ]]; then
  echo "Missing $jobs_file; run generate_experiment_matrix.py first." >&2
  exit 1
fi
IFS=',' read -r -a gpus <<<"$gpus_csv"
if [[ ${#gpus[@]} -eq 0 ]]; then gpus=(0); fi
jobs_per_gpu="${JOBS_PER_GPU:-$(( (max_parallel + ${#gpus[@]} - 1) / ${#gpus[@]} ))}"
if (( max_parallel < 1 || jobs_per_gpu < 1 )); then
  echo "MAX_PARALLEL and JOBS_PER_GPU must be positive integers." >&2
  exit 2
fi

running=0
failures=0
next_gpu=0
declare -A gpu_running=()
declare -A pid_gpu=()
for gpu in "${gpus[@]}"; do gpu_running["$gpu"]=0; done

wait_for_one() {
  local completed_pid="" completed_gpu="" live_pids=""
  # Bash 4.3 has wait -n but not wait -n -p. Compare the tracked children
  # with `jobs -pr` so the exact GPU slot can still be released portably.
  while [[ -z "$completed_pid" ]]; do
    live_pids=" $(jobs -pr | tr '\n' ' ') "
    for candidate_pid in "${!pid_gpu[@]}"; do
      if [[ "$live_pids" != *" $candidate_pid "* ]]; then
        completed_pid="$candidate_pid"
        break
      fi
    done
    [[ -n "$completed_pid" ]] || sleep 1
  done
  if ! wait "$completed_pid"; then failures=$((failures + 1)); fi
  completed_gpu="${pid_gpu[$completed_pid]}"
  gpu_running["$completed_gpu"]=$((gpu_running["$completed_gpu"] - 1))
  unset 'pid_gpu[$completed_pid]'
  running=$((running - 1))
}

find_available_gpu() {
  local offset index candidate
  selected_gpu=""
  for ((offset=0; offset<${#gpus[@]}; offset++)); do
    index=$(( (next_gpu + offset) % ${#gpus[@]} ))
    candidate="${gpus[$index]}"
    if (( gpu_running["$candidate"] < jobs_per_gpu )); then
      selected_gpu="$candidate"
      next_gpu=$(( (index + 1) % ${#gpus[@]} ))
      return 0
    fi
  done
  return 1
}

while IFS=$'\t' read -r task_id family name seed config variant head run_dir; do
  run_dir="${run_dir%$'\r'}"
  [[ "$task_id" == "task_id" ]] && continue
  [[ "$family_filter" != "all" && "$family" != "$family_filter" ]] && continue
  [[ "$name_filter" != "all" && "$name" != "$name_filter" ]] && continue
  while (( running >= max_parallel )) || ! find_available_gpu; do
    wait_for_one
  done
  gpu="$selected_gpu"
  echo "START $task_id GPU=$gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    bash run_scripts/run_p0_paper_job.sh "$config" "$variant" "$seed" "$run_dir" "$head" "$invariant_pool" "$semantic_pool"
  ) >"$logs_dir/${task_id}.log" 2>&1 &
  pid=$!
  pid_gpu["$pid"]="$gpu"
  gpu_running["$gpu"]=$((gpu_running["$gpu"] + 1))
  running=$((running + 1))
done < "$jobs_file"
while (( running > 0 )); do
  wait_for_one
done
if (( failures > 0 )); then
  echo "$failures P0 job(s) failed; successful jobs were retained. Inspect $logs_dir and rerun this command." >&2
  exit 1
fi
echo "All selected P0 jobs completed. Logs: $logs_dir"

