# ProxCABI-FV installation

Target environment:

- Linux
- Python 3.10
- CUDA 12.1
- torch 2.5.1+cu121

Install PyTorch first from the CUDA 12.1 index, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Main third-party libraries:

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

Smoke test:

```bash
bash scripts/run_fever_smoke.sh
```

List detected splits:

```bash
python -m proxcabi list --data-dir data
```

Run a bias audit:

```bash
python -m proxcabi audit --data-dir data --output-dir outputs/audit
```

Run proxy diagnostics:

```bash
python -m proxcabi diagnose --data-dir data --dataset FEVER --split train --max-samples 5000
```

Train:

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
```

Evaluate on a symmetric split:

```bash
python -m proxcabi evaluate \
  --checkpoint-dir outputs/fever_roberta \
  --data-dir data \
  --dataset FEVER \
  --split symmetric
```
