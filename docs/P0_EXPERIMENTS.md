# P0 experiment workflow

This workflow implements the P0 protocol in `todo.md` without treating the old
semantic-only VitaminC pool as an invariant evaluation set.  Training is gated until
independently verified train, invariant-eval, and semantic-eval pools exist.

## Four-GPU quick start

Generate the annotation packets once:

```bash
python scripts/paper_v2/build_pair_candidates.py
```

Generate the visual review pages and have two annotators independently review every
row, followed by disagreement adjudication:

```bash
bash run_scripts/prepare_p0_annotation_review.sh
```

See [`docs/P0_ANNOTATION_GUIDE.md`](P0_ANNOTATION_GUIDE.md) for the exact label rules,
reviewer handoff, automatic agreement report, and adjudication workflow. The source
packets are:

```text
outputs/v2/paper/annotations/train_invariant_tasks.csv  # 300-row train generator audit
outputs/v2/paper/annotations/eval_invariant_tasks.csv   # 154 rows, individually reviewed
```

Then freeze the three pools and run the complete grid:

```bash
bash run_scripts/freeze_p0_pair_pools.sh

nohup env GPUS=0,1,2,3 JOBS_PER_GPU=2 \
  bash run_scripts/run_p0_4gpu.sh \
  > outputs/v2/paper/p0_4gpu_console.log 2>&1 &
```

Two jobs per A6000 use roughly 16--20 GB of its 48 GB memory. Set
`JOBS_PER_GPU=1` if other workloads share the cards; use `3` only after checking
throughput and host RAM. The launcher validates every prerequisite, creates selector
pools from common warm-up checkpoints, skips completed manifests, runs all remaining
jobs, and builds paper reports.

To reproduce only the P0 baselines (and reuse any completed method/seed runs):

```bash
PYTHON_BIN=/data/yangjun/.conda/envs/proxcabi/bin/python \
GPUS=0,1,2,3 JOBS_PER_GPU=1 \
bash run_scripts/run_p0_baselines.sh
```

This baseline-only launcher fails its CUDA preflight instead of silently training on
CPU. It runs ERM first because CLEVER depends on its matched fusion checkpoint, then
queues VitaminC-only, bias-aware, GroupDRO, and CLEVER jobs together. CrossAug is
automatically deferred until all three independently verified pair pools are present.

## 1. Audit and generate the matrix

```bash
bash run_scripts/prepare_p0_experiments.sh
```

The command writes the protocol audit under `outputs/v2/paper/protocol/` and a dry-run
matrix under `outputs/v2/paper/task_matrix/`.  The current repository audit explains
why historical `ipr=0`: all those evaluations have `num_invariant_pairs=0`.

## 2. Supply and validate pair pools

Place independently verified files at:

```text
outputs/v2/paper/pairs/pairs_train_verified.jsonl
outputs/v2/paper/pairs/pairs_eval_invariant.jsonl
outputs/v2/paper/pairs/pairs_eval_semantic.jsonl
```

Rows use the existing V2 pair schema. Required metadata are `parent_id`, `pair_id`,
`transform_type`, `source_label`, `target_label`, `edit_category`, `verified=true`,
`verification_method`, and `split_origin`. Evaluation source IDs and normalized texts
must not overlap the training pool. Validate them with:

```bash
python scripts/paper_v2/validate_pairs.py \
  --train outputs/v2/paper/pairs/pairs_train_verified.jsonl \
  --invariant outputs/v2/paper/pairs/pairs_eval_invariant.jsonl \
  --semantic outputs/v2/paper/pairs/pairs_eval_semantic.jsonl \
  --strict
```

No script fabricates verification or promotes teacher-only pairs to gold.

## 3. Run the per-seed warm-up jobs in parallel

```bash
P0_FAMILY=components P0_NAME=base_only MAX_PARALLEL=3 GPUS=0,1,2 \
  bash run_scripts/run_p0_parallel.sh
```

These checkpoints are the common per-seed warm-up models used
to score every selector candidate. Runs are resumable: a complete manifest and each
completed evaluation are skipped.

## 4. Freeze six same-budget selector pools

```bash
bash run_scripts/prepare_p0_selection_pools.sh
```

This scores the train-only pool with each seed's frozen warm-up checkpoint and writes
Uniform, Random, Loss, Shortcut, Semantic, and Dual pools. Every pool contains exactly
5,000 unique verified pairs with the fixed invariant/semantic quota, or fails loudly.
The Uniform runs reuse component `w/o Dual`, and Dual runs reuse component `Full`, so
these identical jobs are not trained twice.

## 5. Run P0 families in parallel

```bash
P0_FAMILY=components MAX_PARALLEL=3 GPUS=0,1,2 bash run_scripts/run_p0_parallel.sh
P0_FAMILY=selection  MAX_PARALLEL=3 GPUS=0,1,2 bash run_scripts/run_p0_parallel.sh
P0_FAMILY=proxy      MAX_PARALLEL=3 GPUS=0,1,2 bash run_scripts/run_p0_parallel.sh
P0_FAMILY=baselines  MAX_PARALLEL=3 GPUS=0,1,2 bash run_scripts/run_p0_parallel.sh
```

The component family contains Full, w/o Proxy, w/o Bridge, w/o Standardization,
w/o Invariance, w/o Semantic, w/o Dual, and Base-only. The proxy family adds Random
Proxy and train-split Shuffled Proxy; the shared controls are reused from components.
The matched local baselines are joint-data ERM, a separate VitaminC-only verifier,
bias-aware inverse-group reweighting, GroupDRO, CrossAug-style supervised
counterfactual augmentation, and an adapted CLEVER run.
CLEVER trains an independent claim-only RoBERTa model for every seed, reuses the matched
ERM claim--evidence checkpoint as its independent fusion branch, and subtracts claim-only
logits with the preregistered `alpha=1.0`. It is reported as `adapted_same_protocol`
because the original paper used a different backbone/data regime.

If jobs were started with the former CRLF task matrix, safely hand off after the
currently active jobs finish with:

```bash
nohup env MAX_PARALLEL=4 GPUS=0,1,3,4 POLL_SECONDS=30 \
  bash run_scripts/handoff_p0_runner.sh \
  > outputs/v2/paper/handoff_console.log 2>&1 &
```

The handoff pauses only the old scheduler, not its active training children. It waits
for them to finish, normalizes their output paths, waits for all verified pair pools,
prepares selector pools, and then replaces itself with the corrected runner.

## 6. Build traceable result tables

```bash
python scripts/paper_v2/build_paper_reports.py \
  --input outputs/v2/paper --output outputs/v2/paper/reports
```

The report retains per-seed rows and emits mean, sample SD, and the number of valid
seeds for every metric. IPR is prediction agreement, IPA is invariant-pair both-correct
accuracy, and SCR-strict is semantic-pair both-correct accuracy. Empty denominators are
`null`, never zero.

