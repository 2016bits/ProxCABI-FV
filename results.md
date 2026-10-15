# Experimental Results

This file summarizes the current local experimental outputs under `outputs/`. Scores are percentages. `Cal.` uses the calibrated binary evaluation when available; otherwise the reported head is `proximal`.

## Main Results

| Dataset / split | Run | Head | Accuracy | Macro-F1 | Source |
|---|---|---|---:|---:|---|
| FEVER dev | DeBERTa-v3 hard-NLI ProxCABI | calibrated | 96.32 | 96.32 | `outputs/fever_deberta_v3_base_hard_nli_prox_contrastive/eval_FEVER_dev.json` |
| symmetric-FEVER | DeBERTa-v3 hard-NLI ProxCABI | calibrated | 85.91 | 85.90 | `outputs/fever_deberta_v3_base_hard_nli_prox_contrastive/eval_FEVER_symmetric.json` |
| HOVER dev | RoBERTa ProxCABI | proximal | 79.76 | 78.68 | `outputs/hover_roberta/eval_HOVER_dev.json` |
| PolitiHop dev | RoBERTa ProxCABI | proximal | 86.55 | 69.49 | `outputs/politihop_roberta/eval_PolitiHop_dev.json` |
| PolitiHop symmetric | RoBERTa ProxCABI | proximal | 52.63 | 39.36 | `outputs/politihop_roberta/eval_PolitiHop_symmetric.json` |
| VitaminC test_real | RoBERTa ProxCABI | proximal | 87.29 | 85.03 | `outputs/vitaminc_roberta/eval_VitaminC_test_real.json` |
| VitaminC test_synthetic | RoBERTa ProxCABI | proximal | 92.37 | 93.17 | `outputs/vitaminc_roberta/eval_VitaminC_test_synthetic.json` |

Key takeaway: the strongest current FEVER checkpoint is `outputs/fever_deberta_v3_base_hard_nli_prox_contrastive`, with 96.32 FEVER-dev Macro-F1 and 85.90 symmetric-FEVER Macro-F1 after calibration.

## Plan A Exploratory Paired Evaluation

The frozen fact head of the strongest FEVER checkpoint was evaluated on all 181 groups
(728 rows) in `cf_dev_hard_nli`. The pre-specified label-preserving treatment compares
`orig_support` with `negated_target_support`; semantic controls compare the same original
with `negated_source_refute` and `target_claim_refute`.

| Pair type | Pairs | Both ends correct | Prediction consistency / correct switch | Mean NLL difference | Mean absolute NLL difference |
|---|---:|---:|---:|---:|---:|
| Label-preserving | 181 | 100.00 | 100.00 consistency | -0.00111 | 0.00115 |
| Label-changing controls | 362 | 100.00 | 100.00 correct switch | 0.00177 | 0.00284 |

The cluster-bootstrap 95% interval for the label-preserving mean NLL difference is
`[-0.00159, -0.00076]`. The corresponding semantic-control interval is
`[0.00052, 0.00332]`. These results show excellent invariance and retained semantic
sensitivity on this generated, NLI-filtered paired benchmark. They are explicitly
reported as `real_exploratory`: because the pairs are generated interventions and no
independent pre-treatment proxy measurements are available, these numbers are not a
proximal observational ATE. Source:
`outputs/plan_a/real_exploratory_fever/paired_report/metrics.json`.

## FEVER Model Development

| Run | FEVER dev Cal. F1 | symmetric-FEVER Cal. F1 | Source |
|---|---:|---:|---|
| RoBERTa base | 95.09 | 76.96 | `outputs/fever_roberta` |
| RoBERTa balanced | 94.79 | 76.29 | `outputs/fever_roberta_balanced` |
| RoBERTa CF contrastive | 94.10 | 76.98 | `outputs/fever_roberta_cf_contrastive` |
| RoBERTa hard-CF proximal contrastive | 95.48 | 77.94 | `outputs/fever_roberta_hard_cf_prox_contrastive` |
| RoBERTa mild balanced | 94.27 | 73.58 | `outputs/fever_roberta_mild_balanced` |
| DeBERTa-v3 hard-NLI ProxCABI | 96.32 | 85.90 | `outputs/fever_deberta_v3_base_hard_nli_prox_contrastive` |

The DeBERTa-v3 hard-NLI configuration is the main FEVER result. Earlier RoBERTa variants are retained as development history and sanity baselines.

## FEVER Ablations

| Setting | FEVER dev Cal. F1 | symmetric proximal F1 | symmetric Cal. F1 | Interpretation | Source |
|---|---:|---:|---:|---|---|
| Full model | 96.32 | 84.33 | 85.90 | Main DeBERTa hard-NLI ProxCABI setting. | `outputs/fever_deberta_v3_base_hard_nli_prox_contrastive` |
| w/o proxy variable Z | 96.13 | 83.92 | 85.22 | Small drop; Z contributes but is not the only driver. | `outputs/fever_deberta_ablation_no_z` |
| w/o outcome proxy W | 96.08 | 7.94 | 84.94 | Proximal head collapses; calibrated fallback remains usable. | `outputs/fever_deberta_ablation_no_w` |
| w/o bridge/proximal objective | 96.08 | 7.94 | 84.94 | Proximal head collapses similarly; bridge/proximal objective is needed. | `outputs/fever_deberta_ablation_no_bridge_prox` |
| w/o counterfactual samples | 95.89 | 79.46 | 81.31 | Largest robust-FEVER drop among clean ablations. | `outputs/fever_deberta_ablation_no_cf` |
| random-CF instead of hard-CF | 95.98 | 81.71 | 82.15 | Random counterfactual controls help less than hard-NLI CF. | `outputs/fever_deberta_ablation_random_cf` |
| easy-CF instead of hard-NLI-CF | 95.78 | 81.59 | 81.59 | Easy lexical CF is weaker than hard-NLI CF. | `outputs/fever_deberta_ablation_easy_cf` |

Most important ablation result: hard-NLI counterfactual samples are critical for symmetric-FEVER, and removing W or the bridge/proximal objective breaks the proximal head even when calibrated fallback scores remain reasonable.

## Proxy Validity Diagnostics

Setting: FEVER `train`, label-balanced sample size 3000, 256 Z buckets, 256 W buckets. Source: `outputs/proxy_proof/FEVER_train_proxy_proof.json`.

| Proxy variant | Z->B adj. MI | Z->B recon acc | W->B adj. MI | W->B recon acc | Z-only label F1 | X+W F1 | X+W+Z F1 | ΔF1 Z given X,W | CMI estimate | Bridge mean L2 | Bridge RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.19 | 41.89 | 0.06 | 22.87 | 49.48 | 58.66 | 58.39 | -0.27 | 0.0000 | 0.0037 | 0.0054 |
| shuffle | 0.16 | 41.81 | 0.06 | 21.67 | 47.71 | 58.26 | 58.79 | 0.53 | 0.0000 | 0.0036 | 0.0054 |
| learned | 1.17 | 47.41 | 7.88 | 31.63 | 42.31 | 58.79 | 58.79 | 0.00 | 0.0000 | 0.0029 | 0.0042 |
| prox_cabi | 2.08 | 44.21 | 6.70 | 28.40 | 52.57 | 58.26 | 58.66 | 0.40 | 0.0001 | 0.0039 | 0.0056 |

Key takeaway: ProxCABI proxies reconstruct interpretable bias attributes better than random/shuffled proxies, while the conditional probe shows little extra label signal from Z after conditioning on X and W. The bridge residual is low for all learned variants, with ProxCABI at 0.0039 mean L2.

## Claim-Bias Grouping on symmetric-FEVER

`delta_bias` is the best-group minus worst-group Macro-F1 gap. Lower is better. Reports use claim negation, label-LMI tokens, and claim/evidence lexical overlap buckets.

| Run | FEVER dev Cal. F1 | symmetric proximal F1 | symmetric Cal. F1 | Δbias negation | Δbias label-LMI | Δbias overlap | Avg Δbias | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| main | 96.32 | 84.33 | 85.90 | 21.59 | 6.25 | 25.19 | 17.68 | `outputs/bias_report_runs/main/FEVER_symmetric_bias_report.json` |
| gdro_0.20 | 96.02 | 83.89 | 85.49 | 23.01 | 11.13 | 22.73 | 18.96 | `outputs/bias_report_runs/gdro_020/FEVER_symmetric_bias_report.json` |
| gdro_0.05 | 95.99 | 82.66 | 83.12 | 24.20 | 11.59 | 30.48 | 22.09 | `outputs/bias_report_runs/gdro_005/FEVER_symmetric_bias_report.json` |
| gdro_0.10 | 95.99 | 81.94 | 83.12 | 26.09 | 11.04 | 30.48 | 22.54 | `outputs/bias_report_runs/gdro_010/FEVER_symmetric_bias_report.json` |
| overlap-only | 96.08 | 84.06 | 85.63 | 25.03 | 8.14 | 23.92 | 19.03 | `outputs/bias_report_runs/overlap_only/FEVER_symmetric_bias_report.json` |
| reweight-only | 96.01 | 82.05 | 82.95 | 22.16 | 6.54 | 24.70 | 17.80 | `outputs/bias_report_runs/reweight_only/FEVER_symmetric_bias_report.json` |

Key takeaway: the main model remains the best overall checkpoint. Bias-aware training improves the lexical-overlap gap in `gdro_0.20` and `overlap-only`, but it does not dominate the main model because negation and label-LMI gaps worsen or average F1 drops. `overlap-only` is the best bias-aware trade-off, but should be reported as an analysis variant rather than the main result.

## Current Recommendation

- Use `outputs/fever_deberta_v3_base_hard_nli_prox_contrastive` as the main FEVER/symmetric-FEVER checkpoint.
- Use the FEVER ablation table to support the roles of Z, W, bridge/proximal objective, and hard-NLI counterfactuals.
- Use the proxy validity table to argue that Z and W are meaningful proxies rather than arbitrary embeddings.
- Use the claim-bias table as a limitation and analysis: ProxCABI mitigates some shortcuts, but claim-side bias is not fully eliminated.
- No additional large training experiments are currently necessary; the remaining work is presentation, discussion, and possibly post-hoc robust calibration if more analysis is required.
