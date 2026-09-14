export CUDA_VISIBLE_DEVICES=2
python -m proxcabi evaluate --checkpoint-dir outputs/fever_roberta --data-dir data --dataset FEVER --split dev
python -m proxcabi evaluate --checkpoint-dir outputs/hover_roberta --data-dir data --dataset HOVER --split dev
python -m proxcabi evaluate --checkpoint-dir outputs/politihop_roberta --data-dir data --dataset PolitiHop --split dev
python -m proxcabi evaluate --checkpoint-dir outputs/politihop_roberta --data-dir data --dataset PolitiHop --split symmetric