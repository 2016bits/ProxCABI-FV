#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
protocol="${1:-configs/paper_v2/protocol.yaml}"

"$python_bin" scripts/paper_v2/audit_protocol.py --config "$protocol" --output-dir outputs/v2/paper/protocol

if [[ -f outputs/v2/paper/pairs/pairs_train_verified.jsonl && \
      -f outputs/v2/paper/pairs/pairs_eval_invariant.jsonl && \
      -f outputs/v2/paper/pairs/pairs_eval_semantic.jsonl ]]; then
  "$python_bin" scripts/paper_v2/validate_pairs.py \
    --train outputs/v2/paper/pairs/pairs_train_verified.jsonl \
    --invariant outputs/v2/paper/pairs/pairs_eval_invariant.jsonl \
    --semantic outputs/v2/paper/pairs/pairs_eval_semantic.jsonl --strict
else
  echo "Pair pools are not complete. Audit was generated; training matrix will be generated but full P0 runs remain gated." >&2
fi

"$python_bin" scripts/paper_v2/generate_experiment_matrix.py --config "$protocol" \
  --output-dir outputs/v2/paper/task_matrix --dry-run

