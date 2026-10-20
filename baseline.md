# SAGE-FV Baseline Reproduction Plan (WWW)

> Status: implementation specification, not a record of completed reproductions.  
> Scope: FEVER, Symmetric-FEVER, VitaminC; evidence-conditioned three-class verification.  
> Priority: **comparability and traceability before leaderboard completeness**.

## 1. Goal and comparison rules

Reproduce a compact yet competitive group of standard, contrastive, shortcut-debiasing, robust-optimization, and recent causal/counterfactual fact-verification methods under the **same evaluation protocol** as SAGE-FV. Never copy literature numbers into the unified main table when the dataset splits, label space, evidence input, backbone or training data differ.

**Two tables to deliver:**

- Table A: FEVER macro-F1 / Symmetric-FEVER macro-F1.
- Table B: VitaminC Overall / Real / Synthetic macro-F1.

For all supported methods, additionally export accuracy, confusion matrix, class-wise F1, per-example probabilities and, when applicable, IPR/IPA/SCR-strict on **independent** held-out paired sets.

## 2. Freeze the protocol before running anything (P0-GATE)

- [ ] Audit the **exact** FEVER evaluation subset (previous reports mention 1,332 examples), Symmetric-FEVER variant (717 examples), and VitaminC test set (55,197 examples). Record dataset fingerprints, label counts, and subset-selection filters. These counts require verification against local files; do not call them official full splits without evidence.
- [ ] Confirm the task is consistently **SUP / REF / NEI**; document what to do if a model or published dataset is binary. Never silently map/merge labels for the main three-way comparison.
- [ ] Freeze the evidence protocol: gold/fixed evidence, ordering and concatenation, truncation, empty evidence handling, max token length, claim/evidence delimiters, tokenizer; do not enable extra retrieval for any model.
- [ ] Freeze train data and mixing policy (FEVER + VitaminC training portions). Keep all evaluation cases, paired cases and proxy-reference statistics out of training. If a baseline cannot use the same data faithfully, flag **adapted** or **incompatible**.
- [ ] Backbone target: `roberta-base` where architecturally appropriate. If a model intrinsically requires a different encoder or graph architecture, document the difference instead of claiming a controlled backbone comparison.
- [ ] Seeds: `13, 21, 42`; fixed selection criterion on a permitted dev set; never select checkpoints based on Sym-FEVER or VitaminC test performance.
- [ ] Match optimization/update and hyperparameter-search budgets as fairly as possible. Prefer each model's valid optimization recipe over blindly forcing equal learning rates; log deviations.
- [ ] Verify training/test overlap by source IDs, revision case IDs, normalized exact matches and near duplicates. Record screening policy and any removals.
- [ ] Produce `protocol.md`, `dataset_audit.json`, `baseline_compatibility.csv` BEFORE final runs.

## 3. Baseline inventory and reproducibility tiers

**Tier A: first run** (reimplement or train under a fair shared protocol).

| ID | Baseline | Year | Original focus | Implementation resource | Priority |
|---|---|---:|---|---|---|
| B01 | RoBERTa-base (ERM) | 2019 | Standard supervised verifier | https://github.com/huggingface/transformers | P0 |
| B02 | VitaminC-trained verifier | 2021 | Contrastive factual revisions | https://github.com/TalSchuster/VitaminC | P0 |
| B03 | Bias-aware regularization | 2019 | Annotation/claim shortcuts | https://aclanthology.org/D19-1341/ | P0 |
| B04 | CrossAug | 2021 | Contrastive claim/evidence augmentation | https://github.com/minwhoo/crossaug | P0 |
| B05 | GroupDRO | 2020 | Worst-group robust training | https://github.com/kohpangwei/group_DRO | P0 |
| B06 | CLEVER | 2023 | Counterfactual fact-verification debiasing | https://aclanthology.org/2023.acl-long.374/ | P0 |

**Tier B: investigate before promising comparable numbers.**

| ID | Baseline | Year | Key compatibility risk | Reference | Priority |
|---|---|---:|---|---|---|
| B07 | Causal Walk | 2024 | Multi-hop/causal path structure; published FEVER code may be binary | https://github.com/zcccccz/CausalWalk | P1 |
| B08 | CRFV | 2026 | Multi-hop setting and potential bespoke architecture | https://doi.org/10.1016/j.knosys.2025.114875 | P1 |
| B09 | Dual-Debias | 2026 | Original benchmarks/protocol differ from FEVER/VitaminC | https://doi.org/10.1016/j.neucom.2026.133901 | P1 |
| B10 | MuPlon | 2026 | Graph construction and evidence requirements | https://doi.org/10.1016/j.engappai.2026.116117 | P2 |

**Publication/code links are starting points, not guarantees of repository availability, completeness, or identical evaluation settings.** For each candidate, check the original paper, current code accessibility and license; capture an immutable commit/version where available.

## 4. Detailed implementation tasks

### B01 — RoBERTa (ERM)

- [ ] Shared pair encoder for `claim [SEP] evidence`; three-way classification head.
- [ ] Same FEVER+VitaminC training mix as SAGE-FV; no auxiliary proxies, no counterfactual pair-specific losses.
- [ ] Same training/evaluation protocol, seeds and checkpoint-selection metric.
- [ ] Export calibrated/un-calibrated scores consistently (do not use test calibration).
- **Acceptance:** exact evaluation sample counts, three seeds, manifests and per-record predictions.

### B02 — VitaminC-trained verifier

- [ ] Inspect the official VitaminC training and contrastive recipe; record exactly what is reproduced vs adapted.
- [ ] At minimum, compare **ERM with the same FEVER+VitaminC training data** against SAGE-FV; this separates architecture benefit from exposure to VitaminC revisions.
- [ ] If implementing VitaminC-specific contrastive training, keep an explicit `B02-contrastive` variant and document sampling, labels and objective.
- **Acceptance:** no ambiguity about whether this is a published checkpoint, an adapted reproduction or an ERM trained on VitaminC.

### B03 — Bias-aware regularization

- [ ] Reproduce the paper's bias-only/claim-only signal and its correction/regularization mechanism as faithfully as possible.
- [ ] Use *train-only* fitting of any bias model or shortcut vocabulary.
- [ ] Log whether the approach is an adapted implementation; compare to B01 with the same encoder and data.
- **Acceptance:** a verified bias-only control and a distinct regularized full verifier; separate corresponding outputs.

### B04 — CrossAug

- [ ] Audit the official transformation generation, semantic validity and training recipe.
- [ ] Keep augmentation budget fixed and comparable to SAGE-FV; report number of **validated**, not merely generated, pairs.
- [ ] If original code assumes BERT or different labels, separate a faithful original-recipe run from a RoBERTa/three-way adaptation.
- [ ] Never generate training variants from eval/test originals.
- **Acceptance:** transformation manifest, sample counts, seeds, and clean isolation by source/case.

### B05 — GroupDRO

- [ ] Define groups using **train-only observable** shortcut features; document group taxonomy, min group count and empty group handling.
- [ ] Apply GroupDRO robust-loss optimization on the standard verifier, using matched total updates and backbone.
- [ ] Report worst-group accuracy/F1 in addition to aggregate F1 and document group imbalance.
- **Acceptance:** group construction reproducible; no test labels, validation outcomes or group post-selection leakage.

### B06 — CLEVER

- [ ] Read original counterfactual/claim-only intervention formulation and locate its official code if available.
- [ ] Implement all required branches and original prediction combination; do not approximate it as an unrelated subtraction heuristic and still call it CLEVER.
- [ ] Verify supported label space and adapt carefully to three-way output when method permits.
- **Acceptance:** method-specific unit tests plus same split and label mapping; mark any adaptation explicitly.

### B07 — Causal Walk

- [ ] Confirm whether its causal path/graph inputs can be reconstructed using **only allowed fixed evidence**.
- [ ] Inspect binary vs three-way label assumptions and quantify the cost of adaptations.
- [ ] Run only if a faithful comparison is feasible; otherwise include in Related Work, not unified main table.

### B08 — CRFV

- [ ] Find official authors' code or sufficient algorithmic details; log publication, version and availability.
- [ ] Check multi-hop dependency, evidence processing and matching of the three-way FEVER/VitaminC label regime.
- [ ] Include in main table only after protocol-compliant reproduction and consistent seeds.

### B09 — Dual-Debias

- [ ] Audit original dual-branch architecture, dynamic divergence objective, datasets and output heads.
- [ ] Explicitly document adaptation from CHEF/PolitiHop-like evaluation to FEVER/VitaminC if supported.
- [ ] Prioritize this as a recent semantic-preservation comparison if meaningful reproduction is possible.

### B10 — MuPlon

- [ ] Examine whether evidence-graph inputs and causal paths can be built from the present fixed evidence protocol without changing the task.
- [ ] If not, do not force an unfair head-to-head baseline. Discuss in Related Work only.

## 5. Separate the kinds of comparison

Assign every method one compatibility status:

- `faithful_same_protocol`: validated comparable version under shared dataset, labels, evidence and evaluation.
- `adapted_same_protocol`: original method adapted to shared protocol, with changes documented; permissible in main table with notes.
- `published_noncomparable`: literature performance under a different protocol; cite in a separate reference table only.
- `unavailable`: insufficient code/algorithm/data to reproduce faithfully; do not present fabricated metrics.

Do **not** interpret method age or citation counts as evidence that a baseline is superior, and do not silently fill missing cells with other papers' results.

## 6. Training and evaluation matrix

For each compatible baseline, target:

- 3 random seeds: `[13, 21, 42]`.
- FEVER dev evaluation and Symmetric-FEVER evaluation under the frozen subset and evidence input.
- VitaminC Overall, Real and Synthetic test.
- Optional independent held-out paired evaluation: IPR, IPA, SCR-strict, changed-only accuracy, flip rate, with denominators; never infer IPR=0 from an empty pair pool.
- Store scores per sample and per seed; compute mean ± sample SD; calculate paired-seed differences where seeds are matched.
- Standardization or calibration, if any, fitted on training/allowed dev only, never test.

**Metric contracts:**

- `IPR = mean[pred(x) == pred(x_plus)]` (consistency only).
- `IPA = mean[pred(x) == y AND pred(x_plus) == y]`.
- `SCR_strict = mean[pred(x) == y AND pred(x_minus) == y_minus]`.
- `Changed_accuracy = mean[pred(x_minus) == y_minus]`.
- `Flip_rate = mean[pred(x) != pred(x_minus)]`.

## 7. Suggested repository additions (create only after checking existing paths)

```text
configs/paper_baselines/
  protocol.yaml
  b01_erm.yaml
  b02_vitaminc.yaml
  b03_bias_regularization.yaml
  b04_crossaug.yaml
  b05_groupdro.yaml
  b06_clever.yaml
  b07_causalwalk.yaml
  b08_crfv.yaml
  b09_dualdebias.yaml
  b10_muplon.yaml
scripts/paper_baselines/
  check_compatibility.py
  prepare_baseline_data.py
  run_baseline.py
  eval_baselines.py
  summarize_baselines.py
outputs/v2/paper/baselines/
  compatibility.csv
  runs/<method>/<seed>/
  tables/main_fever_sym.csv
  tables/main_vitaminc.csv
  tables/main_fever_sym.tex
  tables/main_vitaminc.tex
  reports/baseline_audit.md
```

These paths are proposals, not assertions that the codebase already contains these modules. Reuse existing loaders, model interfaces and evaluators instead of duplicating a new experiment pipeline.

## 8. Minimum run manifest and output schema

Every `runs/<method>/<seed>/` should contain:

```text
config.json
manifest.json
train_log.jsonl
eval_metrics.json
predictions.jsonl
checkpoint_pointer.txt
```

Recommended manifest keys: `method`, `method_variant`, `source_paper`, `source_code_url`, `code_commit`, `local_git_commit`, `dataset_fingerprint`, `evidence_protocol`, `train_data_fingerprint`, `label_map`, `backbone`, `seed`, `optimizer`, `learning_rate`, `batch_size`, `train_steps`, `selection_metric`, `checkpoint`, `evaluation_splits`, `pair_pool_hash` (when applicable), `comparison_status`, `adaptation_notes`.

Per-record prediction keys: `id`, `source_id`, `split`, `gold`, `pred`, `probs`, `seed`, `method`, `head`, and `pair_id` where applicable.

Reports must preserve `n_examples`, class distribution and valid denominator for all paired metrics. Errors/missing metrics should be `NA` with a reason, not `0`.

## 9. Automated sanity checks

- [ ] Same example IDs, order-independent dataset fingerprint and gold labels across compared models.
- [ ] Three-class logits/probabilities; label IDs map correctly to SUP/REF/NEI.
- [ ] Macro-F1 recomputed from per-example predictions matches summaries.
- [ ] Identical training/eval evidence-input protocol except explicitly annotated original-model requirements.
- [ ] No test-derived vocabulary/proxy distribution/calibration/thresholds.
- [ ] Three seeds unless run explicitly labeled exploratory.
- [ ] No overwriting successful runs; one method+config+seed yields one unique reproducible artifact.
- [ ] Tables bold best and underline second-best only after all *comparable* cells are populated.

## 10. Execution order and stop/go criteria

1. **Protocol audit** → dataset subsets, label mapping, fixed evidence, leakage check.
2. **B01 ERM** and **B02 VitaminC-trained** → shared data control and basic evaluation pipeline.
3. **B05 GroupDRO**, **B03 bias-aware**, **B04 CrossAug** → classical robust/debiasing group.
4. **B06 CLEVER** → counterfactual method with careful equivalence checks.
5. **B09 Dual-Debias** and **B08 CRFV** compatibility audit; attempt the more feasible faithful adaptation.
6. **B07 Causal Walk** → only if label space and evidence-path assumptions can be met.
7. **B10 MuPlon** → optional graph-oriented expansion.
8. Generate three-seed tables, paired metrics and audit report automatically.

**Stop a baseline** when any of these applies: cannot access sufficient original algorithm/code, violates frozen evaluation protocol, requires test labels to construct inputs, cannot support compatible label semantics, or needs new evidence retrieval outside the accepted input protocol. Mark as `unavailable` / `published_noncomparable` and retain in Related Work if relevant.

## 11. Paper reporting template

Main Table A: `Category | Method | FEVER Macro-F1 | Sym-FEVER Macro-F1`.  
Main Table B: `Category | Method | VitaminC Overall | Real | Synthetic Macro-F1`.

Group methods as `Standard`, `Data-driven`, `Robust/Counterfactual`, `Ours`. Report mean ± SD over seeds in main tables where space permits; full per-seed values and configuration differences in appendix. In Experimental Setup, distinguish **faithful** from **adapted** reproductions. Never claim SOTA from incomparable published numbers.

## 12. Final acceptance checklist

- [ ] Two main result tables automatically built from validated per-seed predictions.
- [ ] 6 Tier-A methods either completed or explicitly documented as blocked with reason.
- [ ] At least one recent (2024–2026) method faithfully adapted if feasible; otherwise transparently explain availability/protocol limitations.
- [ ] For every included method, a code/version citation and comparison status.
- [ ] All reported scores reproducible from run manifests and prediction files.
- [ ] No leakage or evaluation-based checkpoint cherry-picking.

