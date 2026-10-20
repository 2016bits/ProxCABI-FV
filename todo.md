# SAGE-FV / V2：WWW 投稿补充实验实施 TODO

> 状态：实施规格草案（2026-10-19）。目标是在现有主实验和消融实验基础上，补全可复现、可解释、无数据泄漏的实验证据。
>
> 对应仓库：<https://github.com/2016bits/ProxCABI-FV>。本文暂用 **SAGE-FV** 作为论文方法名；代码与输出目录可继续沿用 `proxcabi`/`outputs/v2`。**本文的 V2 为预测式标准化，不宣称近端因果效应得到识别**。不要混用 `plan_a` 与 `v2` 的模型、指标或结论。

## 0. 总目标、实验矩阵与执行顺序

**主张与证据对应关系**

| 研究问题 | 需要证明的主张 | 核心证据 | 优先级 |
|---|---|---|---|
| RQ1 | 对标准 FV 和 shortcut shift 的有效性 | FEVER / Symmetric-FEVER / VitaminC 主实验；协议一致的外部 baselines | P0 |
| RQ2 | 代理与标准化、两类语义约束分别起作用 | 组件消融、随机/打乱代理对照 | P0 |
| RQ3 | 不变性与事实变化响应同时改善 | 独立的 invariant/semantic pairs + IPR/IPA/SCR | P0 |
| RQ4 | 双风险选样在固定预算下有独立收益 | 六选择器同预算同训练设置 | P0 |
| RQ5 | 对不同 shortcut 的适应性 | 否定、词汇重叠、长度、实体/时间分组 | P1 |
| RQ6 | 预算效率与跨 backbone 泛化 | 预算曲线、DeBERTa / RoBERTa-large | P1 |
| RQ7 | 代价、失败边界、具体案例 | 时间/显存/参数/推理开销及错误分析 | P2 |

**推荐执行顺序：** Phase 0 协议审计 → Phase 1 配对评测 → Phase 2 核心组件消融 → Phase 3 同预算选择器 → Phase 4 代理分析 → Phase 5 偏差类型、预算、backbone → Phase 6 报告生成。先通过关卡再进行大规模训练。

## 1. Phase 0：冻结协议与审计已有结果（P0，所有实验前置）

- [ ] **P0-01 确认 M1/M2/M3/M4/M5 的真实语义。** 从实际配置和 checkpoint 中提取每项的：主预测 head、bridge、标准化、`L_inv`、`L_sem`、selector 和损失权重。生成 `outputs/v2/paper/protocol/model_matrix.csv`。禁止仅从编号推断 M1 即 Full；选定唯一 Full 配置后写进论文。
- [ ] **P0-02 冻结数据/版本与输入协议。** 明确 FEVER dev=1,332、Symmetric-FEVER=717、VitaminC test=55,197 来自何种筛选，记录类别计数、split 标识、gold/predicted evidence、是否存在 NEI、是否使用 calibration；说明与官方完整集合不同之处。保持相同 tokenizer、max length、证据拼接、标签映射及 evidence-input protocol。
- [ ] **P0-03 核对 `main_results.csv` 与 `ablation_results.csv`。** 当前主结果 `M1/M3`、seed `13/21/42`、预算 5,000、`roberta-base`；历史报告中 `vitaminc_dev_ipr=0.0`，不得直接解释为“零错误”。预算 1K/3K 部分仅单随机种子。逐行核验 `run_dir`、配置、checkpoint、预测文件、训练步数及样本数；检查有无重复行/错误均值。
- [ ] **P0-04 数据污染检查。** 所有 pair 根据原始 `source_id`、claim ID、revision case ID 做 train/dev/test 隔离；核对精确重复、规范化文本重复、近似重复（设定阈值并记录 false positives）、跨 FEVER/VitaminC 重叠。配对构建及代理参考分布仅使用 train。dev 仅用于预先规定的选择规则，test 不参与选择、调参、构对、参考分布拟合。
- [ ] **P0-05 固定公平比较协议。** Baseline/消融共享 backbone、数据、优化器、batch、LR、steps、max_length、随机种子和证据输入；对 selected-pair budget 之外的训练 token / 更新次数差异，需控制或另报告。冻结 checkpoint selection；禁止按 Symmetric-FEVER/VitaminC test 表现挑模型。
- [ ] **P0-06 冻结指标契约。** 确认 `SCR` 现有代码是 conditional flip rate、changed-only accuracy 还是严格双样本正确率；确定 `IPR=0` 是空池/缺失默认值/真实值。无有效样本的指标记录为 `null/NA` 并附 denominator，不得填 0。所有汇总保留 numerator、denominator、分组与 seed。
- [ ] **P0-07 保存 manifest。** 每次 run 必须包含：`git_commit`, `config_hash`, `dataset_fingerprint`, `pair_pool_hash`, `selected_ids_hash`, `reference_distribution_hash`, `seed`, `backbone`, `checkpoint`, `evidence_protocol`, `train_steps`, `best_dev_selection_metric`, `eval_split`。

**Phase 0 验收：** 生成 `protocol.md`、`model_matrix.csv`、`dataset_audit.json`、`leakage_report.json`、`metric_contract.md`，核实所有现有运行可追溯；任何高风险泄漏或指标定义错误均为阻断项。

## 2. 配对评测集与标签核验（P0，建议先做而不是先训练）

### 2.1 数据结构与样本有效性

- [ ] 建立 `pairs_eval_invariant.jsonl`、`pairs_eval_semantic.jsonl`，与训练候选池完全独立。每条至少有 `pair_id`, `source_id`, `case_id`, `split`, `claim`, `evidence_original`, `evidence_changed`, `label_original`, `label_changed`, `transformation_type`, `verification_status`, `verification_source`, `annotator_agreement`, `pair_group`。
- [ ] **Label-preserving**：保持 claim–evidence 关系不变（NEI 同样要考虑），允许经核验的改写、无关信息添加、表面否定等；不能因原标签相同就自动认定有效；检查变换未新增决定性证据、未改变实体指代。
- [ ] **Label-changing**：必须有清楚的原标签与新标签、对应的证据变化及可审计验证；优先 VitaminC 训练侧用于训练，开发/测试侧只用于独立评测。相同 claim、不同官方标签、兼容 revision case 只是必要条件而非充分条件。
- [ ] 如使用 LLM 生成/核验，生成与核验分开；抽样人工复核（建议每类至少 100–200 对或按类别分层，最终依可用数量决定），记录核验通过率、分歧、错误类型、人工裁决。严禁用 test 样本生成新的训练对。
- [ ] 每种变换均输出数量及标签转换矩阵；防止某一标签或某类实体独占评测结果。对于没有合规不变对的场景，报告 semantic-only，不把缺失当作 full dual-constraint 成功。

### 2.2 指标实现（区分一致性与正确性）

设 `f(x)` 是离散预测，`y` 原标签，`y+ = y`，`y- != y`：

- **IPR（Prediction Invariance Rate，预测一致率）**：`mean[ f(x) == f(x_plus) ]`，高不代表预测正确。
- **IPA（Invariant Pair Accuracy，标签保持双样本正确率）**：`mean[ f(x) == y AND f(x_plus) == y ]`。
- **SCR-strict（Semantic Change Responsiveness，严格双样本正确率）**：`mean[ f(x) == y AND f(x_minus) == y_minus ]`；若已有 SCR 定义不同，保留旧名字并给新指标独立列名 `scr_strict`，不能覆盖历史值。
- **Changed-only accuracy**：`mean[ f(x_minus) == y_minus ]`；**Flip Rate**：`mean[ f(x) != f(x_minus) ]`，二者不得冒充 SCR-strict。
- **JS-Invariance**（可选）：预测分布 `JS(p(x),p(x_plus))`，越低越好，仅辅助分析。
- **Conditional shift accuracy**（可选）：条件于 `f(x)==y`，统计 `f(x_minus)==y_minus`；必须同时报告条件分母。
- [ ] 对空集、未知标签、坏行报显式异常或 `NA`；按模型/seed/label transition/shortcut group 输出计数。每种指标写单元测试：恒定预测器 IPR 高但 IPA/SCR 不一定高；全翻转预测器 flip 高但 SCR 不一定高；标签映射互换要失败。

**输出：** `outputs/v2/paper/paired_eval/{pair_manifest.json,per_pair_predictions.jsonl,metrics_by_seed.csv,metrics_by_group.csv,validation_report.json}`。

## 3. E1：Semantic Robustness Analysis（P0）

**研究问题：** Full 模型能否避免“表面变换导致错误翻转”与“事实修订却未正确改变”两种失败？

- [ ] 在**同一固定独立评测对**上运行 ERM、至少两种强去偏 baseline、Full、`w/o Inv`、`w/o Sem`，同三随机种子。
- [ ] 同时报告 IPR、IPA、SCR-strict、changed-only accuracy、flip rate、Macro-F1；每项提供 `N_inv`、`N_sem`。按原标签与 `SUP→REF`、`REF→SUP` 等转移分层。
- [ ] 绘制 IPR–SCR 散点图及误差线（均值±SD），同时在表格报告 IPA，防止 trivial invariance；必要时再画标签保持错误率和事实变更错误率。
- [ ] 检查每对模型是否**在原始样本上就错了**；错误类型至少包括：both wrong、spurious flip、missed revision、correct revision。

**验收：** 两类配对均真实存在、标签已核验、三种子/分母完整。若 Full 未同时改善 IPR/IPA/SCR，诚实报告权衡并收缩论文主张。

## 4. E2：独立组件消融（P0，补充已有 Base-only）

必须采用**从同一个 Full 模型分别移除一个组件**的独立消融，不能用不同训练协议的 M1/M3 替代。

| Variant | 唯一改变 | 主要检验 |
|---|---|---|
| Full | 全部组件 | 参考 |
| w/o Proxy | 去除真实代理输入，保持尽可能可比的模型容量 | 代理信息价值 |
| w/o Bridge Alignment | 对齐损失权重=0 | bridge loss 必要性 |
| w/o Standardization | 改用预定义非标准化预测头/分支（明确实现） | 固定参考聚合收益 |
| w/o Invariance | `lambda_inv=0`；若 invariant pair 还影响其他损失，记录变化 | IPR/IPA |
| w/o Semantic | `lambda_sem=0` | SCR-strict |
| w/o Dual Allocation | 相同预算改 Uniform | 样本优先级是否必要 |
| Base-only | 无额外 CF 配对（现有实验，须确认其它因素相同） | CF 数据总体增益 |

- [ ] 统一 seeds 13/21/42、预算 5K、steps、选择指标、有效配对池；需要检测去掉损失后是否有其他路径仍在产生同类约束。
- [ ] 结果表至少包括 FEVER Macro-F1、Sym-FEVER Macro-F1、VitaminC Macro-F1、IPR、IPA、SCR-strict 和耗时。
- [ ] 如果 `w/o Proxy` 导致网络容量减少，额外提供等容量随机代理对照来排除容量解释。
- [ ] 针对每个组件预先写假设：去掉后期待受影响的**对应行为指标**；即便整体 F1 不变也能识别作用边界。

**验收：** 消融配置只有预期差异、日志可核验、三种子结果完整；输出 `ablation_components.csv` 与 `ablation_configs.json`。

## 5. E3：Counterfactual Selection Comparison（P0）

**研究问题：** Dual 是否在相同预算、相同候选池下优于一般选样？

- [ ] 对比 `Uniform / Random / Loss / Shortcut / Semantic / Dual`，固定预算 5,000 个**有效训练配对**，固定模型初始化、训练步数、数据顺序策略、候选池与 pair validator。
- [ ] 明确 Uniform 的语义：按组平均配额还是全池均匀抽样？与 Random 不要变成重复基线。确保采样单位一致（pair vs source record），避免同一 source 的多组近重复占据预算。
- [ ] 各选择器在同一 warm-up checkpoint（对每个 seed）上计算风险与选样，禁止使用最终测试标签或表现来确定采样权重。
- [ ] 分别计算 FEVER、Sym-FEVER、VitaminC Macro-F1，IPR、IPA、SCR-strict，训练耗时、选择耗时。
- [ ] 输出 `selected_pair_ids_{selector}_{seed}.jsonl` 和选择器分组统计：标签、变化类型、shortcut 特征、源 case、长度、组风险区间、重复率。
- [ ] 检查选择器是否以不同配对类型比例“偷取”优势。建议两套公平设置：A 固定总预算；B **固定 inv/sem 类型配额**后比较内部排序规则。
- [ ] 统计 paired-seed 增益和置信区间；展示 Dual 对 shortcut-only 与 semantic-only 的差异。

**验收：** 同预算、公平训练、无数据泄漏；`selection_comparison.csv`、`selected_pool_report.csv`、`dual_vs_other.md`。

## 6. E4：Proxy & Predictive Standardization（P0）

**研究问题：** 提升是否真的来自结构化代理及标准化，而不是单纯多了一个 head 或 loss？

- [ ] **对照配置：** Full、Random Proxy、Shuffled Proxy、w/o Proxy、w/o Bridge、w/o Standardization；至少三随机种子。随机代理维度、分布及模型容量尽量匹配；Shuffled Proxy 的打乱规则必须只在当前 train split 中进行，不应意外传递标签。
- [ ] **Inference-head 对照：** 使用完全相同 checkpoint，比较 `fact` / `g` / standardized prediction（若接口支持）。与重新训练的 structural ablation 分开报告，避免把 head 切换当成等价组件消融。
- [ ] **参考分布分析：** 默认 `pi_0(W)` 仅从训练集估计并冻结；比较 training empirical / train-uniform / 合理的 train-derived reference，并报告未覆盖类别与极低支持概率的处理。绝不能由目标测试集重新估计 `pi_0`。
- [ ] **代理健康诊断：** 代理预测精度、类别覆盖率、熵、与标签相关性、不同数据集 proxy 分布漂移，以及条件预测 head 在不同代理上的校准误差。
- [ ] **分组评测：** 基于否定、重叠、证据长度、实体模式等预先定义的测试群体分别统计 Macro-F1 / IPA / SCR，报告群体样本数。
- [ ] **Proxy validity boundary：** 这些诊断不构成因果排除限制或完备性的证明；全文统一称为 *predictive standardization*，不报告未经识别的 ATE。

**验收：** `proxy_analysis.csv`、`reference_sensitivity.csv`、`head_comparison.csv`、`proxy_health.json`、分组图。若随机代理与真实代理等效，应明确指出现有证据不支持代理解释。

## 7. E5：Shortcut-Specific Robustness（P1）

| 类型 | 推荐变换 | 有效性规则 | 主要指标 |
|---|---|---|---|
| Negation | 加入无关否定短语 | 不改变 claim–evidence 结论 | IPA、IPR |
| Lexical Overlap | 同义改写、词序变化 | 事实实体/关系/极性不变 | IPA、IPR |
| Evidence Noise/Length | 添加真实且与结论无关的句子 | 不引入矛盾或补充决定性证据 | IPA、IPR |
| Entity/Time Revision | 更换关键实体、地点、日期 | 新标签经核验，且对事实判断有决定性作用 | SCR-strict、changed accuracy |

- [ ] 每组固定数量/标签构成并报告不足样本的组；手动审计标签有效性（至少分层抽查）。
- [ ] 比较 ERM、最强去偏基线与 Full，同条件计算组内样本数及指标；做按组柱状图并给出 paired CI。
- [ ] 如果某类 perturbation 不能保证标签保持/改变，剔除或标记 `unverified`，不混入正式指标。

**验收：** `shortcut_group_metrics.csv`、`perturbation_validation.csv`、分组图；说明最大收益/最大退化发生在哪类群体。

## 8. E6：Counterfactual Budget Analysis（P1）

- [ ] 固定 `B={0,1000,3000,5000,10000}`，对 Random、Loss、Dual 至少三个策略做 seeds 13/21/42；Budget 0 要明确为同训练步数的 no-CF 对照。
- [ ] **严格保持**候选池、训练步数、dev checkpoint selection、其他损失权重不变；报告因预算不同引起的 pair 重复次数/原始样本占比变化。
- [ ] 结果：FEVER、Sym-FEVER、VitaminC F1、IPR、IPA、SCR-strict，训练成本；画预算 × 指标的均值±SD 折线图及性价比曲线。
- [ ] 现有 1K/3K 仅单种子，不可画成三种子误差线；先复用已有可靠 checkpoints，补齐缺失种子。
- [ ] 如果 10K 比 5K 更差，检查负面配对质量、配对重复、类型比例、训练学习率和数据混合比例，而非直接归因于“更多样本有害”。

**验收：** `budget_curve.csv`（含 seeds/计数）、`budget_curves.pdf`、`budget_efficiency.md`。

## 9. E7：Generalization Across Backbones（P1）

- [ ] 选 `roberta-base`（已有）、`deberta-v3-base`、`roberta-large`（计算受限可先做其中一个）；每 backbone 至少比较 ERM、同协议最强 baseline、SAGE-FV Full。
- [ ] 各模型使用适合自身的 tokenizer、模型大小与经过开发集选择的超参数；训练数据与标签协议必须一致。为避免不公平，给 ERM 与 Full 相同级别的调参预算。
- [ ] 三随机种子优先，如仅一随机种子需显式说明 exploratory，不能写稳定泛化结论。
- [ ] 报告同一三数据集 Macro-F1、IPR/IPA/SCR、额外推理成本；检查标准化头是否在不同 encoder 下需要特殊处理。

**验收：** `backbone_generalization.csv`；论文附录可报告每个 backbone 的具体超参数。

## 10. E8：Case Study、Failure Analysis 与 Efficiency（P2）

- [ ] 选**独立测试集中实际存在并可追溯**的标签保持/标签改变配对各至少 1–2 组。展示 claim、原/变证据、gold、各模型实际预测/置信度，不要把 Motivation Figure 的虚构预测当成实验结果。
- [ ] 汇总至少 20–50 个有代表性的失败例：错误标注、NEI 处理、实体替换失真、证据噪声造成注意偏移、反事实标签不可靠、proxy 偏置等，输出错误类别及占比。
- [ ] 测量训练 wall time、峰值 GPU 显存、参数量、每样本推理延迟、吞吐（同机器、batch 与输入长度），并单列 pair 生成、审核、采样与 reference fitting 的额外成本。
- [ ] 重要超参做窄范围 sweep：`lambda_inv`, `lambda_sem`, `lambda_bridge`, `alpha/beta` 选样权重；优先 dev 选择，勿依据 test 选取最佳。

**验收：** `case_study.md`、`failure_taxonomy.csv`、`efficiency.csv`、`hyperparameter_sweep.csv`。

## 11. Baseline 补齐与两张 WWW 主表（P0）

- [ ] Baseline：ERM、Bias-aware Regularization、CrossAug、CLEVER、GroupDRO；若有可复现实现可加 CRFV / Dual-Debias。每个方法记录论文来源、代码版本、是否能在当前数据和证据协议下合理复现。
- [ ] 对不兼容的官方 published numbers 使用**独立 literature comparison**，不要直接与当前 1,332/717 样本协议混在同一张公平排名表。
- [ ] Table 1：Category / Method / FEVER Macro-F1 / Sym-FEVER Macro-F1；Table 2：Category / Method / VitaminC Overall / Real / Synthetic Macro-F1。保留单栏 `table`，`booktabs`，最好/次好仅在**完整、可比**的行中加粗/下划线。
- [ ] 统计三随机种子的 mean±SD，并对主要比较报告配对差值的 CI；至少在附录给 per-seed 完整数值和数据规模。不要仅根据小的均值差异推断统计显著。
- [ ] **检查训练集合并后的混杂因素：** VitaminC 训练增强可能提高 VitaminC 测试表现，但需要通过 ERM 同样联合训练与 FEVER-only 对照隔离收益来源。

**验收：** `main_fever_sym.csv`、`main_vitaminc.csv`、对应 `.tex`，且每一个单元格可追溯到 per-run 指标。

## 12. 软件接口与文件组织（建议；实际需先对照仓库实现）

**原则：不盲目新建平行训练框架。** 优先复用仓库现有的 `configs/`、`proxcabi/`、`scripts/`、`run_scripts/`、`tests/`。下面目录是**建议新建的目标结构，并非声称仓库已有这些文件**。

```text
configs/paper_v2/
  protocol.yaml
  semantic_pairs.yaml
  ablation.yaml
  selection.yaml
  proxy.yaml
  budget.yaml
  backbones.yaml
scripts/paper_v2/
  audit_protocol.py
  validate_pairs.py
  eval_paired_robustness.py
  run_component_ablations.py
  run_selection_comparison.py
  run_proxy_analysis.py
  run_budget_curve.py
  run_backbone_generalization.py
  build_paper_reports.py
tests/paper_v2/
  test_pair_labels.py
  test_metric_contract.py
  test_no_leakage.py
  test_reference_distribution.py
  test_selection_budget.py
  test_report_aggregation.py
outputs/v2/paper/
  protocol/
  pairs/
  runs/
  paired_eval/
  tables/
  figures/
  reports/
  manifest.json
```

**推荐 run 输出最低规范：** `config.json`, `manifest.json`, `train_log.jsonl`, `eval_metrics.json`, `predictions.jsonl`, `selected_pairs.jsonl`（若有）、`checkpoint_pointer.txt`。`predictions.jsonl` 至少 `id`, `split`, `gold`, `pred`, `probs`, `source_id`, `pair_id`, `seed`, `head`，便于之后无需重训就重新计算指标。

**建议 CLI 契约（示意，不是现成命令）：**

```bash
python scripts/paper_v2/audit_protocol.py --config configs/paper_v2/protocol.yaml
python scripts/paper_v2/validate_pairs.py --config configs/paper_v2/semantic_pairs.yaml
python scripts/paper_v2/eval_paired_robustness.py --config configs/paper_v2/protocol.yaml
python scripts/paper_v2/build_paper_reports.py --input outputs/v2/paper --output outputs/v2/paper/reports
```

新脚本应支持 `--dry-run`、`--resume`、`--seed`、`--output-dir`、`--overwrite=false`（默认），拒绝无 manifest 的结果混合和覆盖已有成功结果。训练与报告分离；优先对现有 checkpoints 离线评测。

## 13. 验收测试与自动化质量关卡

- [ ] **Pair validation**：同一 source 的不同 split 不可交叉；语义对要求 `y- != y`，不变对要求 `y+ == y`，且所有 pair 都有 provenance；仅比较标签不等式不能当作人工/LLM 核验。
- [ ] **Reference freeze**：`pi0` 的 train-only fingerprint 固定；对于同一 checkpoint，dev/test 不允许更新它。
- [ ] **Metric sanity**：常值分类器可得高 IPR 但 IPA 不应虚高；随机/翻转预测不应得到虚假 SCR；空池为 NA，分母与计数始终一致。
- [ ] **Sampling reproducibility**：相同 config + seed + pool fingerprint 必须生成同一 selected ID list；所有预算上限不得超标；严格审计是否存在 group-level leakage。
- [ ] **Table reproducibility**：`.tex` 由 metrics/manifest 自动生成，三种子均值和 SD 与原始 CSV 可独立复算；数值格式和加粗规则一致。
- [ ] **Report integrity**：测试集预测仅用于最终汇报；论文不应把 illustrative motivation outputs 写作 baseline 实测。

## 14. 推荐最小完成包与最终交付

### 最小可投稿支撑（优先完成）

- [ ] 协议审计、基线复现与两张主结果表。
- [ ] 独立组件消融（至少 w/o Std、w/o Inv、w/o Sem、w/o Dual）。
- [ ] 独立核验的 invariant / semantic 双类配对测试与 IPR / IPA / SCR-strict。
- [ ] 六选择器的固定 5K 预算比较。
- [ ] 真实代理 vs 随机代理、standardized vs non-standardized 的机制分析。
- [ ] 每项核心结果至少三随机种子，允许附录列出详表。

### 推荐加分包

- [ ] Budget 曲线补足随机种子。
- [ ] 否定、词汇重叠、证据噪声、事实修改的按组对比。
- [ ] 第二种 backbone + 至少一个真实失败案例。
- [ ] 推理额外开销与核验成本。

### 最终论文图表清单

1. Main Table A：FEVER / Symmetric-FEVER。
2. Main Table B：VitaminC Overall / Real / Synthetic。
3. Component Ablation Table。
4. Semantic Robustness Table（IPR/IPA/SCR）与 IPR–SCR 散点图。
5. Selector Comparison Table（固定预算）。
6. Proxy Analysis Table 或 shortcut-group 柱状图。
7. Appendix：budget 曲线、backbone、efficiency、case study、per-seed、validation audit。

## 15. 不应预先写进论文的结论

- 不能从目前的 `IPR=0.0` 推出不存在 spurious sensitivity；先审计是否为空池或实现异常。
- 不能用单纯 prediction flip 代表正确响应事实修订；必须考虑变换后 gold label。
- 不能从 proxy alignment / residual 小得出严格因果识别成功。
- 不能因为选择器在单 seed 某一 benchmark 领先就宣称整体有效。
- 不能把使用不同 backbone、训练数据、证据协议或 test calibration 的文献结果当作可比 SOTA。
- 若某个组件无独立实证贡献，应弱化其论文创新定位，而不是隐藏对应消融。

## 16. 操作层面的第一批待办（建议下一次代码修改按此顺序）

- [ ] **Task 1**：编写 `audit_protocol.py`，读取现有 configs、main/ablation CSV 与 checkpoint manifests，输出 M1–M5 对照表和 `IPR=0` 原因报告。
- [ ] **Task 2**：编写 `validate_pairs.py` + 三类单元测试，冻结可用的独立 invariant/semantic eval pool。
- [ ] **Task 3**：编写 `eval_paired_robustness.py`，只读取已有模型预测/checkpoint 生成 IPR、IPA、SCR-strict 和分组结果；不先重训。
- [ ] **Task 4**：在统一 Full 定义下生成 component-ablation 任务矩阵和 fixed-budget selector 任务矩阵，先 dry run 校验。
- [ ] **Task 5**：完成固定 reference 的 proxy 对照与 inference-head 对照，随后安排补充训练。
- [ ] **Task 6**：编写 `build_paper_reports.py` 自动生成两张主表、消融表、语义指标表和附录 CSV/TeX。

---

**完成标准：** 每一项论文结论均有独立、合规、可重现的实验支撑；每个表格单元格能追溯到输入数据、配置、checkpoint、seed 和评测脚本。先保证数据与指标可信，再追求更高的 Macro-F1。
