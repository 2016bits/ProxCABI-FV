#!/usr/bin/env bash
set -euo pipefail

# Safely hand off from a runner started with the old CRLF task matrix:
# 1. stop only the old scheduler, while its current jobs keep running;
# 2. wait for those jobs and evaluations to finish;
# 3. terminate the stopped scheduler and normalize CR-suffixed run directories;
# 4. wait for the three verified pair pools, build selector pools, and exec the
#    corrected parallel runner.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
new_gpus="${GPUS:-0,1,3,4}"
new_parallel="${MAX_PARALLEL:-4}"
poll_seconds="${POLL_SECONDS:-30}"
old_runner_pid="${OLD_RUNNER_PID:-}"
status_file="outputs/v2/paper/handoff_status.log"
mkdir -p "$(dirname "$status_file")"

log() {
  local message="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$message"
  echo "$message" >> "$status_file"
}

find_old_runner() {
  local candidates pid ppid
  mapfile -t candidates < <(pgrep -f '^bash run_scripts/run_p0_parallel\.sh( |$)' || true)
  for pid in "${candidates[@]}"; do
    ppid="$(ps -o ppid= -p "$pid" | tr -d ' ')"
    if [[ ! " ${candidates[*]} " =~ " ${ppid} " ]]; then
      echo "$pid"
      return 0
    fi
  done
  return 1
}

if [[ -z "$old_runner_pid" ]]; then
  old_runner_pid="$(find_old_runner || true)"
fi
if [[ -z "$old_runner_pid" ]] || ! kill -0 "$old_runner_pid" 2>/dev/null; then
  log "No old run_p0_parallel scheduler found; continuing with input checks."
  old_runner_pid=""
fi

old_stopped=0
restore_old_runner() {
  if [[ "$old_stopped" == "1" ]] && kill -0 "$old_runner_pid" 2>/dev/null; then
    kill -CONT "$old_runner_pid" 2>/dev/null || true
    log "Handoff interrupted; resumed old scheduler PID $old_runner_pid."
  fi
}
trap restore_old_runner EXIT INT TERM

if [[ -n "$old_runner_pid" ]]; then
  kill -STOP "$old_runner_pid"
  old_stopped=1
  log "Paused old scheduler PID $old_runner_pid; current train/eval children continue."

  while true; do
    active="$(ps -eo pid=,ppid=,stat=,args= | awk '
      (($4 == "bash" && $0 ~ /run_p0_paper_job\.sh/) ||
       ($4 == "python" && $0 ~ /proxcabi v2-(train|evaluate)/)) && $3 !~ /^Z/ { print }
    ' || true)"
    if [[ -z "$active" ]]; then
      break
    fi
    count="$(wc -l <<<"$active" | tr -d ' ')"
    log "Waiting for $count active train/eval processes to finish."
    sleep "$poll_seconds"
  done

  # TERM is pending while a process is stopped; CONT makes it take effect.
  kill -TERM "$old_runner_pid" 2>/dev/null || true
  kill -CONT "$old_runner_pid" 2>/dev/null || true
  old_stopped=0
  for _ in {1..10}; do
    kill -0 "$old_runner_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$old_runner_pid" 2>/dev/null; then
    log "Old scheduler did not exit after TERM; sending KILL to scheduler only."
    kill -KILL "$old_runner_pid" 2>/dev/null || true
  fi
  log "Old scheduler stopped after its current jobs completed."
fi

# Runs created from the former CRLF jobs.tsv have a literal carriage return at
# the end of the directory name. Rename only those exact directories.
while IFS= read -r -d '' bad_path; do
  clean_path="${bad_path//$'\r'/}"
  if [[ -e "$clean_path" ]]; then
    log "Cannot normalize $bad_path because $clean_path already exists."
    exit 1
  fi
  mv -- "$bad_path" "$clean_path"
  log "Normalized run directory: $clean_path"
done < <(find outputs/v2/paper/runs -depth -name $'*\r*' -print0 2>/dev/null || true)

"$python_bin" scripts/paper_v2/generate_experiment_matrix.py \
  --config configs/paper_v2/protocol.yaml \
  --output-dir outputs/v2/paper/task_matrix --dry-run

required_pools=(
  outputs/v2/paper/pairs/pairs_train_verified.jsonl
  outputs/v2/paper/pairs/pairs_eval_invariant.jsonl
  outputs/v2/paper/pairs/pairs_eval_semantic.jsonl
)
while true; do
  missing=()
  for path in "${required_pools[@]}"; do
    [[ -s "$path" ]] || missing+=("$path")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    break
  fi
  log "Waiting for verified pair pools: ${missing[*]}"
  sleep "$poll_seconds"
done

log "Pair pools found; validating leakage and label contracts."
"$python_bin" scripts/paper_v2/validate_pairs.py \
  --train "${required_pools[0]}" \
  --invariant "${required_pools[1]}" \
  --semantic "${required_pools[2]}" --strict

log "Preparing six selector pools with GPUs $new_gpus."
GPUS="$new_gpus" bash run_scripts/prepare_p0_selection_pools.sh

log "Starting corrected runner with MAX_PARALLEL=$new_parallel GPUS=$new_gpus."
trap - EXIT INT TERM
exec env MAX_PARALLEL="$new_parallel" GPUS="$new_gpus" \
  bash run_scripts/run_p0_parallel.sh

