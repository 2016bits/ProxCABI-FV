# P0 双人核查操作手册

## 目标与分工

需要核查的不是“原始事实本身是否正确”，而是每一次自动改写是否仍然保持 CSV 中
`proposed_target_label` 指定的 claim--evidence 关系：

- `train_invariant_tasks.csv`：300 条训练生成器抽查样本；
- `eval_invariant_tasks.csv`：154 条评测样本，必须逐条人工核查。

推荐让审核员 A 和审核员 B **各自独立完成全部 454 条**，不要各分一半。两份独立判断
才能计算一致率和 Cohen's kappa，并暴露规则中含糊的地方。两人在各自导出 CSV 前不要
讨论；第三步只仲裁两人不一致的行。仲裁员可以由项目负责人担任。

预计每人约需 3--5 小时。建议分成 3 个时段，每完成一段就导出一次 CSV 备份。

## 第一步：生成可视化核查页面

在项目根目录运行：

```bash
bash run_scripts/prepare_p0_annotation_review.sh
```

生成的两个页面为：

```text
outputs/v2/paper/annotations/review_ui/train_invariant_review.html
outputs/v2/paper/annotations/review_ui/eval_invariant_review.html
```

页面是自包含的离线 HTML，可以直接用 Chrome/Edge/Firefox 打开。若远程环境不能直接
打开文件，可在服务器运行：

```bash
python -m http.server 8000 --directory outputs/v2/paper/annotations/review_ui
```

再通过 SSH 端口转发访问 `http://127.0.0.1:8000/`。审核员先填写不同 ID（如
`reviewer_A`、`reviewer_B`），页面会按 ID 分开自动保存进度。浏览器缓存不是正式交付，
每次收工前仍应点击“导出 CSV”。不要直接改写原始 tasks CSV。

## 第二步：逐条怎么判断

每条只回答一个问题：**改写后的 claim 和 evidence 是否严格对应页面给出的目标标签？**

标签标准：

- `SUPPORTS`：只根据给出的 evidence，能够推出 changed claim。仅仅“可能是真的”或
  “没有明显冲突”不算支持。
- `REFUTES`：evidence 明确与 changed claim 矛盾。没有提到某事不能自动算反驳。
- `NEI`：evidence 既不能推出也不能否定 changed claim。

按以下顺序检查：

1. 先读 changed claim 和 changed evidence，不看原句，独立判断它们的关系。
2. 将判断结果与 `proposed_target_label` 比较；相同才可能“通过”。
3. 对照红色删除、绿色新增部分，确认改写没有误改实体、指代、时间、数量、比较关系、
   否定范围或决定性证据。
4. 确认实际发生变化的字段与 `transform_type` 一致，没有同时改 claim 和 evidence。
5. 确认改写句语法自然、含义唯一。`only`、`not`、双重否定或修饰范围含糊时判“不通过”。
6. 不查看任何模型预测，也不要用 evidence 之外的实时搜索结果补足缺失事实。

典型判断示例：

- evidence 明说 “Telemundo is an American ... network”，changed claim 却说
  “There is not an American ... Telemundo”，目标仍标为 `SUPPORTS`：应判不通过，
  因为 evidence 实际上反驳了 claim。
- 原句只说某人出演一部 TV series，改写加入 `only`：通常应判不通过，因为 evidence
  没有证明“只出演该类作品”的排他含义。
- evidence 只说某对象是 `song`，改写成 `not a film`：若需要额外的类别互斥假设才能
  成立，按严格协议判不通过；不要靠常识补足证据。
- 只做大小写、明确同义改写，且实体、关系、标签均未改变：可以判通过。

点击“不通过”后至少选择一个原因；复杂或有争议的样本在备注里写一句可复核的理由。
快捷键为 `A`（通过）、`R`（不通过）、左右方向键（切换）。

## 第三步：收集两人的结果

要求每位审核员交回两份文件，总共四份。建议重命名为：

```text
reviewer_A_train.csv
reviewer_A_eval.csv
reviewer_B_train.csv
reviewer_B_eval.csv
```

交付前在页面顶部确认 train 为 `300/300`、eval 为 `154/154`。不要用 Excel 重新另存，
以免破坏 UTF-8、引号或换行；如必须查看，保持原有列名及 `pair_id` 不变。

## 第四步：自动计算一致率并生成仲裁页

```bash
bash run_scripts/reconcile_p0_annotation_reviews.sh \
  reviewer_A_train.csv reviewer_A_eval.csv \
  reviewer_B_train.csv reviewer_B_eval.csv
```

该命令会拒绝未完成、重复 `pair_id`、行数不匹配或两人使用相同 ID 的文件，并生成：

```text
outputs/v2/paper/annotations/review_ui/adjudication/train_agreement_report.json
outputs/v2/paper/annotations/review_ui/adjudication/eval_agreement_report.json
outputs/v2/paper/annotations/review_ui/adjudication/train_adjudication.html
outputs/v2/paper/annotations/review_ui/adjudication/eval_adjudication.html
```

仲裁页面默认保留两人一致的决定；选择“仅双人分歧”，逐条阅读 A/B 的判断与备注，再由
仲裁员做最终选择。不要按多数票或模型输出决定。完成后分别导出最终 train/eval CSV。

## 第五步：冻结最终池并启动实验

假设仲裁页导出的是 `train_adjudicator_adjudication.csv` 和
`eval_adjudicator_adjudication.csv`：

```bash
TRAIN_ANNOTATIONS_CSV=/absolute/path/train_adjudicator_adjudication.csv \
EVAL_ANNOTATIONS_CSV=/absolute/path/eval_adjudicator_adjudication.csv \
bash run_scripts/freeze_p0_pair_pools.sh
```

训练抽查通过率必须至少为 90%；评测集不设置凑数门槛，所有被拒绝样本会被排除。冻结
脚本会再次验证布尔决定、数据泄漏和 train/eval 隔离；验证通过后才能启动 P0 实验。

## 最终检查清单

- 两人的 ID 不同，且都独立完成 300 + 154 条；
- 每一行 `verified` 都是明确的 `true` 或 `false`；
- 不通过项有原因，疑难项有可复核备注；
- 已保存 train/eval 的一致率报告与最终仲裁 CSV；
- 原始 task CSV、两份独立导出和最终仲裁结果均保留，便于论文审计；
- 只有最终仲裁结果被交给 `freeze_p0_pair_pools.sh`。
