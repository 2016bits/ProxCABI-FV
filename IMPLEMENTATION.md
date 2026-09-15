# ProxCABI-FV implementation map

This repository now contains a runnable Python reference implementation for the four datasets under `data/`:

- `FEVER`
- `HOVER`
- `PolitiHop`
- `VitaminC`

## README stages implemented

| README stage | Implementation |
| --- | --- |
| Stage 0 Bias Audit | `python -m proxcabi audit` writes label, length, negation, LMI, syntax/entity, hop and metadata summaries. |
| Stage 1 Proxy Construction | `ProxyBuilder` in `proxcabi/proxies.py` builds dataset-specific discrete `Z` and `W` proxies. |
| Stage 2 Proxy Diagnostics | `python -m proxcabi diagnose` computes `I(Z;Y)`, `I(W;Y)`, `I(Z;W)`, `Delta_proxy`, and shuffled-Z comparison. |
| Stage 3 Semantic Encoder | `ProxCABIModel.encoder` wraps HuggingFace RoBERTa/DeBERTa-style models. |
| Stage 4 Proxy Conditional Model | `proxy_head` estimates `q(W | M,Z)` with cross entropy. |
| Stage 5 Bridge Learning | `g_head`, `h_head`, and JS bridge loss enforce `g(M,Z) ~= E[h(M,W)|M,Z]`. |
| Stage 6 Proximal Inference | `model.proximal_logits()` computes `E_W[h(M,W)]` using the empirical training marginal `P_train(W)`. |
| Evaluation | Accuracy, Macro-F1, bridge residual, hop-group metrics, and VitaminC revision-type metrics are written as JSON. |

Counterfactual proxy support, semantic perturbation generation, semi-synthetic causal recovery, and 7B/8B LoRA experiments are intentionally left as extension stages; the core PLM proximal pipeline is in place.

## Install

Use Linux, Python 3.10, CUDA 12.1, and torch 2.5.1+cu121:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Third-party libraries:

- `torch`
- `transformers`
- `accelerate`
- `sentencepiece`
- `protobuf`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `PyYAML`

## Main commands

List detected splits:

```bash
python -m proxcabi list --data-dir data
```

Run audit for all four datasets:

```bash
python -m proxcabi audit \
  --data-dir data \
  --datasets FEVER HOVER PolitiHop VitaminC \
  --output-dir outputs/audit
```

Run proxy diagnostics:

```bash
python -m proxcabi diagnose --data-dir data --dataset FEVER --split train --max-samples 5000
python -m proxcabi diagnose --data-dir data --dataset HOVER --split train --max-samples 5000
python -m proxcabi diagnose --data-dir data --dataset PolitiHop --split train
python -m proxcabi diagnose --data-dir data --dataset VitaminC --split train --max-samples 5000
```

Train FEVER and evaluate symmetric-FEVER:

```bash
python -m proxcabi train \
  --data-dir data \
  --dataset FEVER \
  --output-dir outputs/fever_roberta \
  --backbone roberta-base \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta \
  --data-dir data \
  --dataset FEVER \
  --split symmetric
```

Build FEVER train-only counterfactual augmentation and train with contrastive ranking:

```bash
bash run_scripts/build_fever_counterfactuals.sh
bash run_scripts/train_fever_counterfactual_contrastive.sh
```

This uses only FEVER `train`/`dev` to create `cf_train`/`cf_dev`; it does not use symmetric-FEVER IDs or grouped test structure.

For the stronger fair symmetric-FEVER setting, build hard negation/exclusivity counterfactuals and train with proximal-head contrastive ranking:

```bash
bash run_scripts/run_fever_hard_cf_proximal_contrastive.sh
```

This creates `cf_train_hard`/`cf_dev_hard`, trains `outputs/fever_roberta_hard_cf_prox_contrastive`, disables fp16 for stability, limits hard augmentation to 40k rows, and applies contrastive ranking to the final proximal head as well as the fact/g heads.

Train PolitiHop and evaluate symmetric-PolitiHop:

```bash
python -m proxcabi train \
  --data-dir data \
  --dataset PolitiHop \
  --output-dir outputs/politihop_roberta \
  --backbone roberta-base \
  --epochs 10 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --fp16

python -m proxcabi evaluate \
  --checkpoint-dir outputs/politihop_roberta \
  --data-dir data \
  --dataset PolitiHop \
  --split symmetric
```

Train/evaluate HOVER with hop-specific reporting:

```bash
python -m proxcabi train \
  --data-dir data \
  --dataset HOVER \
  --output-dir outputs/hover_roberta \
  --backbone roberta-base \
  --epochs 3 \
  --batch-size 12 \
  --eval-batch-size 24 \
  --fp16
```

The resulting evaluation JSON contains `proximal_by_hop` for `2`, `3`, and `4`.

Train/evaluate VitaminC with real/synthetic reporting:

```bash
python -m proxcabi train \
  --data-dir data \
  --dataset VitaminC \
  --output-dir outputs/vitaminc_roberta \
  --backbone roberta-base \
  --epochs 2 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --fp16

python -m proxcabi evaluate --checkpoint-dir outputs/vitaminc_roberta --data-dir data --dataset VitaminC --split test_real
python -m proxcabi evaluate --checkpoint-dir outputs/vitaminc_roberta --data-dir data --dataset VitaminC --split test_synthetic
```
