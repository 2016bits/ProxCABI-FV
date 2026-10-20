# 协议审计与生成 51 个任务
bash run_scripts/prepare_p0_experiments.sh

# 先并行训练三个 warm-up/base-only 模型
P0_FAMILY=components P0_NAME=base_only \
MAX_PARALLEL=3 GPUS=0,1,3 \
bash run_scripts/run_p0_parallel.sh

# 用同 seed warm-up 模型冻结六种 5K 选样池
bash run_scripts/prepare_p0_selection_pools.sh

# 分实验族并行训练
P0_FAMILY=components MAX_PARALLEL=3 GPUS=0,1,3 bash run_scripts/run_p0_parallel.sh
P0_FAMILY=selection  MAX_PARALLEL=3 GPUS=0,1,3 bash run_scripts/run_p0_parallel.sh
P0_FAMILY=proxy      MAX_PARALLEL=3 GPUS=0,1,3 bash run_scripts/run_p0_parallel.sh
P0_FAMILY=baselines  MAX_PARALLEL=3 GPUS=0,1,3 bash run_scripts/run_p0_parallel.sh

# 汇总三种子结果
python scripts/paper_v2/build_paper_reports.py