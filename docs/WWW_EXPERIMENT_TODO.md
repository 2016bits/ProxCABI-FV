# WWW submission experiment TODO

Status values: `[ ]` pending, `[~]` running/partially complete, `[x]` complete.

Current execution status (2026-10-16):

- The audited FEVER structural and matched-baseline pipelines have finished;
  no training or report process remains active.
- All 12 structural runs and all 18 matched-baseline runs use seeds 13/21/42
  under the fixed packing, optimizer-budget, and evaluation protocol.
- `full` improves over `no_cf` by 1.76 calibrated symmetric-FEVER Macro-F1
  points (95% hierarchical bootstrap CI [0.39, 3.25], Holm p=0.003), while the
  `no_z` and `no_w` differences are not significant.
- `full` does not significantly beat ERM and is below CF-only, reweighting, and
  GroupDRO. This is the immediate method-development gate before expensive
  cross-dataset retraining.
- Canonical outputs are under `outputs/fever_multiseed/`; final paper-facing
  summaries, significance, efficiency, robustness, and reproducibility records
  are under `outputs/plan_a/`.

### Immediate decision gate

- [ ] Build a validation-only robustness/model-selection split; never tune on
  symmetric-FEVER test labels.
- [ ] Diagnose `selection_head=proximal` versus fact-head selection and sweep a
  compact, predeclared set of proxy/g/bridge loss weights on seed 13.
- [ ] Re-run only the top two validation-selected configurations on seeds 21/42.
- [ ] Continue to cross-dataset three-seed training only if the selected full
  configuration is competitive with reweighting and GroupDRO; otherwise narrow
  the claim to counterfactual robustness and semi-synthetic causal recovery.

## P0 — submission-critical

### E1. Multi-seed reliability (current stage)

- [x] Fix seeds to `13, 21, 42` and use isolated output directories.
- [x] Audit legacy checkpoints; reject seed-13 reuse because `full` used a
  different epoch/scheduler protocol.
- [x] Train FEVER `full` with seeds 13, 21, and 42 under the fixed protocol.
- [x] Train FEVER `no_cf` with seeds 13, 21, and 42.
- [x] Train FEVER `no_z` with seeds 13, 21, and 42.
- [x] Train FEVER `no_w` with seeds 13, 21, and 42.
- [x] Report mean, sample standard deviation, and every individual run.
- [x] Save per-example predictions for paired bootstrap and significance tests.
- [x] Run paired bootstrap confidence intervals and Holm-corrected comparisons.

Acceptance criterion: all runs use the same backbone, data, optimization schedule,
counterfactual budget, model-selection split, and evaluation protocol except for the
component deliberately ablated.

### E2. Matched strong baselines

- [x] Define isolated 3-seed runs and a common aggregation/significance protocol.
- [x] Make reweighting and GroupDRO fact-head baselines independent of proximal heads.
- [x] DeBERTa-v3-base ERM, 3 seeds.
- [x] Counterfactual augmentation without proxy/bridge losses, 3 seeds.
- [x] Fact-head contrastive training without proxy/bridge losses, 3 seeds.
- [x] Bias reweighting, 3 seeds.
- [x] GroupDRO, 3 seeds.
- [x] Compare every baseline under identical hard-CF and compute budgets.

### E3. Effect-guided training loop

- [x] Audit FEVER base metadata: current train rows contain no independent
  pre-treatment group key, so synthetic `L` weights cannot be applied to FEVER.
- [ ] Define an auditable pre-treatment grouping protocol or restrict this experiment
  to semi-synthetic robustness utility; do not reuse labels/model losses as proxies.
- [x] Complete a 20-seed semi-synthetic heterogeneous-CATE targeting experiment under
  a fixed group budget, comparing proximal, naive, Z-only, random, uniform, and oracle.
- [ ] Complete an end-to-end verifier retraining comparison; targeting utility alone is
  not reported as downstream accuracy improvement.
- [ ] Uniform weights.
- [ ] Random weights.
- [ ] Loss-based weights.
- [ ] Observable bias-score weights.
- [ ] Single-proxy weights (Z-only and W-only).
- [ ] Proximal-effect-guided weights.
- [x] Oracle targeting upper bound on semi-synthetic data only.
- [ ] Fix the selected counterfactual count across all weighting methods.

### E4. Semi-synthetic causal recovery on real text

- [x] Define a controlled latent confounder and known ATE using frozen FEVER-verifier
  response anchors for real-text outcome heterogeneity.
- [x] Run 20 data-generation seeds at five proxy-strength levels (100 runs total).
- [x] Compare naive, Z-only, W-only, proxy-IPW, discrete proximal, and oracle-U.
- [x] Add neural proximal on five fixed seeds across the same five proxy strengths.
- [x] Report bias, RMSE, interval coverage, and identification-gate failure rate.
- [x] Sweep proxy strength from 0.55 to 0.95.
- [x] Sweep overlap, Z-exclusion violations, W-to-treatment violations, and sample size
  over 320 real-text semi-synthetic runs.
- [x] Add shuffled-Z, shuffled-W, and shuffled-both negative controls.

### E5. Claim audit

- [ ] Either collect defensible independent pre-treatment Z/W metadata, or explicitly
  limit real-benchmark claims to robustness utility rather than causal identification.
- [ ] Document treatment, target population, proxy provenance, and exclusion arguments.

## P1 — strong evidence

### E6. Cross-dataset fair comparison

- [ ] Train matched ERM/CF/contrastive/GroupDRO/ProxCABI models on HoVer.
- [ ] Repeat on VitaminC and PolitiHop using the same backbone and 3 seeds.
- [ ] Add source-to-target or leave-one-dataset-out transfer evaluation.
- [ ] Pre-register label mappings and forbid target-test calibration.

### E7. Counterfactual quality

- [x] Audit all 15,004 train and 728 dev hard-NLI rows for role/label consistency,
  group completeness, duplicate IDs/text, teacher confidence, and cross-split leakage.
- [x] Prepare a deterministic blinded sample of 400 counterfactual pairs, three
  independent annotation slots per pair, a separated answer key, and ethics/data-handling protocol.
- [ ] Human-evaluate 300–500 counterfactual pairs with 2–3 annotators.
- [ ] Measure label preservation, intervention validity, fluency, evidence consistency,
  and synthetic-artifact detectability.
- [ ] Report agreement and error categories; verify institutional ethics requirements.
- [ ] Sweep hard-CF count and easy/hard mixture while fixing total data budget.

### E8. Robustness and error analysis

- [x] Implement reproducible FEVER symmetric subgroup definitions and a
  three-seed worst-group/gap report for label, negation, lexical overlap,
  claim length, and a transparently labeled capitalized-entity-frequency proxy.
- [x] Implement a deterministic 100-example high-confidence failure annotation
  template without fabricating manual labels.
- [x] Complete the three-seed FEVER worst-group report for negation, lexical
  overlap, label, entity frequency proxy, and length.
- [x] Report HoVer hop and VitaminC real/synthetic revision-type performance on
  their complete reporting splits.
- [ ] Manually classify at least 100 failures.
- [x] Explain the PolitiHop symmetric proximal-head degradation with prediction
  distributions, class recalls, and pairwise head disagreements.

### E9. Efficiency and reproducibility

- [x] Instrument new runs with training wall time, evaluation throughput, and peak GPU
  memory; prediction backfill records evaluation efficiency for legacy checkpoints.
- [ ] Report parameter count, train time, peak GPU memory, inference latency, and
  counterfactual-generation cost relative to ERM.
- [x] Package configs, hashes, seeds, environment, and one-command reproduction.
- [ ] Prepare an anonymous artifact and archival release after review.

## Paper-level checks

- [ ] State Web relevance on page 1 and select the Responsible Web or Web Mining track.
- [ ] Keep the first eight pages self-contained.
- [ ] Include limitations, ethics, and real-data causal-identification boundaries.
- [ ] Never call the cross-dataset head comparison a retrained structural ablation.
