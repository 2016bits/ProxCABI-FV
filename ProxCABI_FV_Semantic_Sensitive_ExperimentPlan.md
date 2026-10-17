# ProxCABI-FV V2：Semantic-Sensitive Shortcut Debiasing 完整实验与 Python 实现方案

> **文档版本**：v1.0（2026-10-17）  
> **目标**：在现有 [ProxCABI-FV](https://github.com/2016bits/ProxCABI-FV) 基础上，实现“**可解释代理特征 + 固定参考标准化 + 双重语义约束 + 选择性反事实训练**”，形成可复现、可消融、可用于 WWW 论文的研究系统。  
> **文档用途**：Python 开发需求说明书 + 实验预注册草案 + 论文实验组织方案。  
> **状态**：研究设计，**不是已实现功能，也不是已获得的实验结果**。本文所有新文件、新命令和新指标在实现前均视为 *proposed*。

---

## 0. 执行摘要与不可违反的约束

### 0.1 一句话研究目标

减少事实核查模型对声明和证据中**与真假无关的表面 shortcut** 的依赖，同时保持它对**真实声明—证据语义关系变化**的敏感性；通过可解释代理特征、统一参考分布、成对监督及选择性反事实预算实现这一目标。

### 0.2 目标输出

最终需要完成：

1. 可复用的 V2 代理构建器和固定参考分布 `pi0(W)`；
2. 可计算 `fact / g / standardized` 三种预测的 V2 核查器；
3. 经过验证的 `original / invariant / semantic` 成对数据管道；
4. 原样本 CE、辅助分类、标准化 CE、预测式 bridge 一致性、invariance、semantic ranking 六类可独立开关的损失；
5. 训练集诊断驱动的、固定总预算的反事实选择器；
6. 相同训练预算的 ERM、CF-only、Contrastive、Reweighting、GroupDRO 和 V2 系列消融；
7. ID、OOD、IPR、SCR、worst-group、效率及统计显著性完整报告；
8. 一键重现实验命令、配置、模型、输入文件哈希和逐样本预测记录。

### 0.3 核心实现边界

- **不要将此 V2 预测模型宣称为严格 proximal causal identification**。本文 `Z/W` 默认是可观测的、任务相关的 nuisance-marker 特征；`bridge consistency` 是预测分布约束，而非可保证识别的因果矩条件。
- 现有 `proxcabi/causal/` 与 `planA.md` 属于**独立的严格因果效应估计路线**；V2 开发默认不改动其估计逻辑，也不混用实验结论。
- 训练数据、词汇统计、代理类别映射和 `pi0` 只能由**原始训练集**确定；严禁使用 dev/test/OOD 测试标签拟合这些对象。
- **所有反事实必须有源样本、方向、目标 gold label 和验证状态**；无可靠标签的增强样本不得计算 `L_sem`。
- 对比基线时保持**同一原始训练集、同一可用反事实候选池、同一反事实预算、同一训练步数和相同模型选择规则**。
- `selection_head=standardized` 是提议方法的预注册主预测头；不得看到 OOD test 结果后更换为 `fact` 或 `g`。
- 仅在 FEVER 训练集上进行开发与调参；symmetric-FEVER 等 OOD 报告数据不能参与 checkpoint 选择、阈值拟合或特征选择。

### 0.4 仓库已知状态（核对时间：2026-10-17）

现有仓库包含 `proxcabi/model.py`、`train.py`、`proxies.py`、`torch_data.py`、`data.py`、`counterfactuals.py`、`cli.py`、`bias_groups.py`，还有 `proxcabi/causal/` 的 Plan A 和 WWW 实验脚本。

- 原模型有 `fact_head`、`proxy_head`、`g_head`、`h_head`。`h_all_w(m)` 枚举 W 类别，`proximal_logits(m, w_marginal)` 进行边际标准化。
- 旧 `LossWeights` 主要管理 `fact/g/proxy/bridge`；其中 bridge 损失为 `g_probs` 与 `sum_w q(w|M,Z) h(y|M,w)` 的 JS 散度。
- 旧 `ProxyBuilder` 将结构化字符串哈希到默认 256 个 Z/W 桶，并在 `fit` 中统计 `w_marginal`。
- 当前 `train.py` 在拼接原始训练样本和 counterfactual 后拟合 `ProxyBuilder`：V2 必须解决该分布污染。
- 当前对比排序主要按 SUPPORTS/REFUTES 成组；V2 应按真实 `(y, y_new)` 泛化，并为 NEI 明确规则。
- `docs/WWW_EXPERIMENT_TODO.md` 记录了三随机种子结果：原有 full 相比 no-CF 有收益，但 Z/W 消融收益不显著，且 full 尚未可靠超过 CF-only、Reweighting 和 GroupDRO。因此必须把**强基线超越门槛**放在新增大规模训练之前。

以上为仓库当前设计和记录，**不是本方案改造后的结果**。实际开发前保存 Git commit SHA，避免远程代码更新造成复现偏差。

---

## 1. 研究问题、研究假设与创新边界

### 1.1 任务定义

令 `X=(C,E)`，`C` 为声明、`E` 为证据，`Y` 为真实性标签：

\[
\mathcal Y=\{\mathrm{SUPPORTS},\mathrm{REFUTES},\mathrm{NEI}\}.
\]

编码器 `f_theta` 输出声明—证据关系表示：

\[
M=f_\theta(C,E)\in\mathbb R^d.
\]

给定标签保持变换 `T_inv` 和经过核验的标签改变变换 `T_sem`：

\[
Y(T_{\mathrm{inv}}(X))=Y(X),\qquad
Y(T_{\mathrm{sem}}(X))\ne Y(X).
\]

本研究不是要模型“对所有变换都不敏感”，而是要避免 **shortcut invariance 与 factual sensitivity 被混为一谈**。

### 1.2 Research Questions

| ID | 研究问题 | 待验证假设 | 对应主实验 |
|---|---|---|---|
| RQ1 | 双重语义约束是否优于单纯 CF-augmentation / contrastive debiasing？ | 同时提高 IPR、SCR，并改善 OOD Macro-F1 | Main Results + Two-way Robustness |
| RQ2 | 可解释 Z/W 与固定 `pi0(W)` 是否带来独立收益？ | 在控制参数和监督预算后，standardization 增加稳健性 | Proxy Ablations + Diagnostics |
| RQ3 | 按错误敏感性与不足敏感性分配 CF 预算是否更有效？ | 在相同 CF 数量/步数下，Dual-budget 优于 Uniform、Random 和 Loss-based | Selective Budget Study |
| RQ4 | 泛化是否跨数据集、分组及语义变换类型成立？ | 不仅在 FEVER，而在 symmetric-FEVER、VitaminC、HoVer/PolitiHop 中保持表现 | Transfer + Subgroup Analysis |
| RQ5 | 改进是否在统计和计算成本上值得？ | 性能收益跨 seed 成立，训练/推理额外成本可解释 | Significance + Efficiency |

### 1.3 论文贡献（在结果出来之前为“拟议贡献”，不得作为已证实事实写入摘要）

- **Problem**：定义事实核查中的“双向语义鲁棒性”，区分对标签保持扰动的稳定性与对真实事实变化的正确响应。
- **Method**：构建可解释的 nuisance proxy-conditioned prediction，固定参考分布进行预测式标准化，并施加非对称成对约束。
- **Optimization**：在固定反事实预算下，根据两类模型失败模式分配训练资源；不是简单选择高损失样本。
- **Evaluation**：提供同时约束 ID/OOD、IPR/SCR、group worst-case、代理真实性及效率的严格实验协议。

### 1.4 因果措辞使用规则

可以使用：`proxy-guided predictive standardization`、`shortcut-aware training`、`semantic-sensitive debiasing`、`predictive bridge alignment`。  
除非额外完成 `proxcabi/causal/` 所要求的代理来源/排除限制/可识别性分析，不使用：`identified causal effect of claim text`、`unbiased proximal estimate`、`causal debiasing guarantee`。

---

## 2. 数据集与数据使用协议

### 2.1 主要数据集

| 数据集 | 角色 | 拟定训练/评测方式 | 特别注意 |
|---|---|---|---|
| **FEVER** | 主模型开发、三分类 ID | 仅用 train 训练，dev 拆分用于选择/独立报告 | NEI 不可与 REFUTES 无条件合并 |
| **symmetric-FEVER** | 主要 OOD / shortcut 鲁棒性 | 测试集**仅最终报告**；采用其官方实际可用标签集合 | 若为二分类，固定 SUP/REF 解码协议 |
| **VitaminC** | 真实文本修订、语义变化 | 数据集内训练/测试及修订对分析 | 要核验修订对、证据链和标签来源 |
| **HoVer** | 多跳语义保持性 | 数据集内匹配训练；按 2/3/4 hop 分组 | 不改变其原始二/三分类定义；先审计映射 |
| **PolitiHop** | 来源/主题/分布转移补充 | 可用时训练/测试及对称测试 | 单独报告数据切分与来源偏差 |

建议主论文最少包括 FEVER + symmetric-FEVER + VitaminC；HoVer/PolitiHop 作为扩展，不应在 FEVER 竞争力未确立前重复昂贵训练。

**已有仓库路径约定**：`data/<dataset>/converted_data/<split>.json` 或 `.jsonl`；新样本仍应兼容 `FactSample` 的 `id/claim/evidence/label/num_hops/metadata`。

### 2.2 FEVER dev 的严格划分（一次生成后永久冻结）

针对公开标签的 FEVER dev，使用 `sha256("FEVER_DEV_V2|" + sample_id)` 的稳定哈希进行分组划分，并按 gold label 分层：

- `dev_select`：60%，用于超参数、checkpoint 选择；
- `dev_calib`：20%，仅在预声明需要校准时用于拟合概率温度/二分类阈值；
- `dev_report`：20%，用作**未参与模型选择的 ID 测试报告**。

`dev_select` 中由原始样本生成的标签保持/语义变化对，可以作为**鲁棒性验证集**，但其源样本与训练集必须不同。训练反事实只能由 FEVER train 派生。

重要细节：

1. 使用 `parent_id` / 文档 ID 防止同一原始样本的派生变换跨 dev 子集；更严格时按 claim/entity/page 聚类分组，再做近似分层。
2. 不可将 `dev_report` 纳入任何 checkpoint、loss weight、代理配置选择。
3. 为了和既有论文保持传统 FEVER dev 对比，可额外在**冻结所有决策后**报告 `full_dev_descriptive`，但须说明其中包含模型选择子集，不能将它称为独立检验集。
4. 如果可合法获得独立标准 test 标签，则优先以该 test 作为最终 ID 评测；在此之前以 `dev_report` 作为严格 ID 测试。

### 2.3 Label 规范

与 `proxcabi/data.py` 对齐：

```yaml
labels:
  supports: 0
  refutes: 1
  not_enough_info: 2
```

跨数据集的二分类必须在配置中显式指定映射并保存 `label_mapping.json`。**绝不在预测后临时为某个模型修改标签合并方式**。若 OOD 只有 SUPPORTS/REFUTES：仅对这两个类别的 logits 做二类 softmax/argmax（或采用预声明、dev-calib 拟合且固定的解码器），严禁使用 OOD 测试标签校准阈值。

---

## 3. 反事实样本体系与质量控制

### 3.1 反事实类型及合法性

| 类型 | 允许操作 | 标签关系 | 不允许的常见错误 |
|---|---|---|---|
| `original` | 原始 C/E | `y` | — |
| `invariant_claim` | 同义改写、语序/语态变换 | `y_inv = y` | 擅自修改实体、时间、数字、否定范围 |
| `invariant_evidence` | 语义等价改写，重排不影响推理的句子 | `y_inv = y` | 删除关键证据、产生截断、改变证据完整性 |
| `semantic_claim` | 关键实体/时间/数字/关系/否定范围修改 | `y_sem != y`，必须核验 | “含 not 一定是 REFUTES”等词面规则 |
| `semantic_evidence` | 受控替换事实内容或追加可靠事实 | `y_sem != y`，必须核验 | 凭单条证据缺失判断 NEI，或隐含开放世界结论 |

**优先使用** SUP ↔ REF 的高可信标签改变对；NEI 标签变化必须明确证据集合的完备性/任务标注原则，且经独立核验后再进入训练。NEI 样本仍参与原始三分类监督、以及合法的 invariant loss，不应被全部过滤。

### 3.2 数据样例与统一 JSONL 格式

以下为 V2 新的**建议格式**，不是目前 `FactSample` 必须已有的字段。为兼容现有 `load_samples`，`id/claim/evidence/label` 仍是一级字段，其余新增字段归入 `metadata`：

```json
{"id":"FEVER_102_inv_0","claim":"Ada's birth year was 1980.","evidence":"Ada was born in 1980.","label":"supports","num_hops":1,"metadata":{"parent_id":"FEVER_102","pair_id":"FEVER_102:inv:0","transform_type":"invariant_claim","source_label":"supports","target_label":"supports","changed_field":"claim","edit_category":"paraphrase","verified":true,"verification_method":"human","evidence_source_id":"wiki_page_123","split_origin":"train","generator_version":"v2_rules_1"}}
{"id":"FEVER_102_sem_0","claim":"Ada was born in 1990.","evidence":"Ada was born in 1980.","label":"refutes","num_hops":1,"metadata":{"parent_id":"FEVER_102","pair_id":"FEVER_102:sem:0","transform_type":"semantic_claim","source_label":"supports","target_label":"refutes","changed_field":"claim","edit_category":"year_substitution","verified":true,"verification_method":"human","evidence_source_id":"wiki_page_123","split_origin":"train","generator_version":"v2_rules_1"}}
```

真实数据必须保留原始 `parent_id` 对应的 `original` 记录；上面 Ada 示例仅展示 schema，不是 FEVER 的真实标注记录。建议把训练管道使用的 `PairSample` 定义为 `parent`、`invariant`、`semantic` 三个可选视图以及各自有效性 mask。

#### 必需字段及验证

- `parent_id`：可稳定映射到一个真实基础样本；同一 parent 的派生记录共享该 ID。
- `pair_id`：全局唯一，避免同一变换重复入池。
- `transform_type`：枚举；不可自由拼写变体。
- `source_label`、`target_label`：必须满足 type 对应的相等/不等规则。
- `verified`：布尔值；未核验的样本只可进入候选池，不可进入语义损失。
- `verification_method`：`human`、`dataset_gold`、`rule_plus_human` 等；teacher-only 不等于 gold。
- `changed_field`、`edit_category`：用于子组评估与质量分析。
- `split_origin`：只允许由当前基础数据同 split 派生；不可以从 `symmetric` 或 test 派生训练样本。
- `generator_version`、`source_hash`、`evidence_source_id`：便于数据版本追踪、去重及 Web 来源分析。

### 3.3 生成与过滤流程

按优先级执行：

1. **复用已有数据**：从当前 `counterfactuals.py` 已生成的 hard-NLI 和现有 contrast group 中提取候选；不要默认其标签已达到新方法要求。
2. **规则生成高精度候选**：数字/年份替换、谓词值替换、经过约束的同义改写；规则只能提出候选，不能直接决定复杂事实关系。
3. **教师模型过滤**（可选）：保留高置信候选，但必须记录模型版本、阈值、原始输出和判断规则；教师置信度不能单独证明 gold 正确。
4. **自动断言**：标签方向、实体变化范围、证据保留、重复、空白、过长截断、模板化伪迹、父子 ID、源文件 split。
5. **人工核验**：对主实验样本随机抽取不少于 300 对、理想 400–500 对，2–3 位标注者核查标签、变换合法性、自然度和模板伪迹；报告一致性及类别错误率。
6. **冻结候选池**：`cf_pool_v2_train.jsonl`、`cf_pool_v2_val.jsonl`、各自 SHA256；全部基线共享同一池。

参考现有 `docs/WWW_EXPERIMENT_TODO.md`：仓库报告有 15,004 条 train hard-NLI 和 728 条 dev hard-NLI，**这是要审计的当前资产，而非新方法新增规模的承诺**。

### 3.4 反事实样本泄漏检查

至少校验：

- 相同 `parent_id` 不跨 train/dev/test；
- `(normalized claim, normalized evidence)` 精确重复不跨 split；
- 近重复文本、同一 evidence page、同一原始 annotation chain 的跨 split 泄漏风险需单独报告；
- CF 数据中不能携带 teacher 给出的答案标记作为模型输入；
- 不允许用测试集上的高置信错误来反向挑选训练 CF；
- 对 W 代理的 `source_id`、`revision_type` 等元信息，应判断其是否存在真实预测时不可用/后处理泄漏。

输出 `data_audit.json`，包含候选、通过、拒绝、人工核验、各变换类型/标签转移、重复/跨 split 统计。

---

## 4. V2 代理设计：解释性、稳定性、预测有效性

### 4.1 对旧 ProxyBuilder 的改造原则

旧版将包含多项属性的字符串稳定哈希到 256 桶，优点是兼容所有数据，缺点是难解释类别含义、碰撞、低频稀疏，并且 `fit(train+CF)` 使 CF 改变 LMI 词表和 `P(W)`。

V2 默认采用**低基数、有标签的类别和结构化特征**，同时留一个 `legacy_hash` 模式用于公平回归对比。**Z 和 W 在这里是两个预测式特征视图，不推定满足严格 causal proxy 的排除条件**。

### 4.2 Z：声明侧 nuisance / 表达模式

优先采用训练集可拟合、验证/测试可确定提取的特征：

| 特征 | 建议分桶/编码 | 备注 |
|---|---|---|
| `claim_negation` | none / explicit / uncertain | 不能把否定当作真实性标签 |
| `claim_length` | short / medium / long | 阈值来自 base-train 分位数 |
| `claim_entity_pattern` | none / person-like / org-like / mixed | 基于实体类型/模式而非实体 ID |
| `claim_template` | limited syntax/style category | 全部类别 train-only 拟合 |
| `lexical_shortcut` | train-only LMI 特征的 multi-hot 或离散标签 | 不输入 gold label 或真实训练标签给单个样本预测 |

实现建议：`z_vector` 使用多热/连续标准化特征经 `z_encoder` 投影；如果为了最大限度兼容旧模型，可先将少量组合映射为低基数 `z_id`，后续做结构化向量消融。不能把 `parent_id`、`source_label`、`target_label` 输入代理编码器。

### 4.3 W：证据侧 nuisance / 可参考化变量

首版只用容易解释且不直接编码 gold 关系的证据外观属性，建议低基数分类：

- `evidence_length_bucket`: short / mid / long；
- `evidence_sentence_count_bucket`: one / two / multi；
- 可选 `evidence_style_class` 或可信来源 `provenance_class`，**仅当模型部署时可获得且不构成标签/后处理泄漏时加入**。

推荐首版 `K=9`（`3×3`），带 `UNK` 可为 `K=10`；实际参数表根据拟合后有效类别数确定。第二阶段尝试 `K<=32`。**不要将 hop 数、事实修订类型、gold 标签或 CF transformation type 默认放入 W**：这些属性可能是真正语义难度或变换后的后处理信息。

预测目标是 `w_id ∈ [0,K-1]`，类别映射保存为 `w_vocab.json`。未见类别使用 `UNK`；`UNK` 不能通过 test 的出现频率重新定义。

### 4.4 参考分布 pi0(W) 的唯一拟合位置

必须先执行：

```python
base_train = load_samples(data_dir, dataset, "train")
proxy = StructuredProxyBuilder(config).fit(base_train)  # 只拟合原始训练集
pi0 = proxy.reference_distribution(alpha=1.0)           # 只统计 base_train
cf_train = load_verified_counterfactual_pool(...)
train_records = make_training_views(base_train, cf_train)
# 禁止在此对 train_records 再次调用 fit()
```

建议 Dirichlet/Laplace 平滑：

\[
\pi_0(w)=\frac{n_w+\alpha}{N+K\alpha},\qquad \alpha=1 \text{（首选）}.
\]

- 冻结 `pi0` 为 checkpoint buffer，模型训练和所有评估不能更新。
- 对所有 V2 对照和消融使用同一份 `proxy_artifact`，文件哈希须一致。
- 若 `K` 中存在测试时不可支持的 W 档位，记录 OOV 率，不用测试集重新估计分布。
- 同一模型更换 `pi0` 的结果只作为**明确标注的推理消融**，不是重新训练的结构消融。

### 4.5 Proxy 健康指标

开发 `proxy_diagnostics_v2.py` 输出：

1. `W` 类别覆盖率、熵、最大类占比、UNK 率、哈希碰撞率（仅 legacy）；
2. `Z/W` 与标签的相关性（诊断 shortcut，不是因果有效性证明）；
3. `q(W|M,Z)` 在验证集上的 accuracy、NLL、macro-F1；
4. `Z` 加入前后 `q(W|...)` 的验证集 NLL 改善（在共享 `M` 下公平比较）；
5. `h(y|M,w)` 随 w 的条件变化量，以及 standardized 头 vs 仅 M 头的差异；
6. `random / shuffled / constant` Z/W 负对照；
7. `pi0` 对 base-train、base-train+CF、val 的分布差异（后两者只用于**观察**，不参与拟合）。

**失败门槛**：如 `h` 对所有 `w` 完全忽略或任意打乱 W 不影响表现，不能在论文中声称 W 学到了有意义的校正；应做简化架构对照。

---

## 5. V2 模型结构与数学目标

### 5.1 维持原项目四个 head

| Head | 输入 | 输出 | 训练目的 | 论文定位 |
|---|---|---|---|---|
| `fact_head` | M | `logits_fact ∈ R^C` | 常规监督与公平基线 | 语义主干 |
| `proxy_head` (`q`) | M, Z | `q(W|M,Z) ∈ R^K` | W 预测 / 跨代理联系 | 辅助预测 |
| `g_head` | M, Z | `g(Y|M,Z) ∈ R^C` | 标签预测 | conditional reference head |
| `h_head` | M, W | `h(Y|M,w) ∈ R^C` for all w | W-conditional 标签监督 | standardized head 的构成 |

令 `C` 为类别数（FEVER 3，某些数据集 2）；`K` 是可解释 W 类别数量。

### 5.2 固定参考标准化

\[
\begin{aligned}
q_\phi(w\mid M,Z)&=\mathrm{softmax}(q_\phi(M,Z))_w,\\
g_\phi(y\mid M,Z)&=\mathrm{softmax}(g_\phi(M,Z))_y,\\
h_\psi(y\mid M,w)&=\mathrm{softmax}(h_\psi(M,w))_y,\\
\bar p(y\mid X)&=\sum_{w=1}^{K}\pi_0(w) h_\psi(y\mid M(X),w).
\end{aligned}
\]

主推理 logits 为：

\[
\ell_{\mathrm{std}}(X)=\log(\mathrm{clamp}(\bar p(\cdot\mid X),\epsilon,1)),
\quad \epsilon=10^{-8}.
\]

从训练到测试均使用**同一个** `pi0`；测试时不需要当前样本的 W 真值参与 `std` 的计算（W 只影响训练及代理诊断）。

### 5.3 原始样本与反事实的共同监督

对一个带有效视图的组 `G_i={X_i,X_i^+,X_i^-}`，标签分别为 `y_i,y_i,y_i^-`。对所有可用视图的集合 `V_i`，定义：

\[
\mathcal L_{\mathrm{sup}}
=\frac{1}{\sum_i|V_i|}\sum_i\sum_{v\in V_i}
\mathrm{CE}(\bar p(\cdot\mid X_i^v),y_i^v).
\]

这里的标准化监督覆盖了 base、invariant、semantic；**CF-only 基线也必须看到完全相同的视图并接受等量 CE 监督**，其预测头换为普通 fact head。

辅助监督：

\[
\begin{aligned}
\mathcal L_{\mathrm{fact}}&=\mathrm{CE}(p_{\mathrm{fact}},Y),\\
\mathcal L_{g}&=\mathrm{CE}(g_\phi(\cdot\mid M,Z),Y),\\
\mathcal L_{h}&=\mathrm{CE}(h_\psi(\cdot\mid M,W_{\mathrm{obs}}),Y),\\
\mathcal L_{q}&=\mathrm{CE}(q_\phi(\cdot\mid M,Z),W_{\mathrm{obs}}).
\end{aligned}
\]

实现上，四个 loss 的输入视图统一配置；第一版**对所有训练视图**计算，记录每个 loss 各自的样本个数与均值，避免混淆“多看了额外增强样本”与“多一个算法组件”。

### 5.4 预测式 bridge 一致性（保留，但不能称为因果识别）

\[
\mathcal L_{\mathrm{bridge}}=
\mathbb E_X D_{\mathrm{JS}}\left(
 g_\phi(\cdot\mid M,Z),
 \sum_w q_\phi(w\mid M,Z)h_\psi(\cdot\mid M,w)
\right).
\]

它仅约束多个预测头的行为一致。`h_head` 必须通过 `L_h` 和/或 `L_sup` 有独立标签梯度，不能只依赖这个 bridge。消融必须单独关闭 `L_bridge`，而不关闭 W 的全部监督。

### 5.5 标签保持不变性损失

对通过验证的 `T_inv`：

\[
\boxed{
\mathcal L_{\mathrm{inv}}=
\frac{1}{N_{\mathrm{inv}}}
\sum_i D_{\mathrm{JS}}\big(\bar p(X_i),\bar p(X_i^+)\big)
}
\]

- 使用概率 JS 而非直接对 logits 做 MSE。
- 输入为**源样本和其 invariant 变换**，不可把语义变化对当 invariant。
- 首版不 detach 两端，监督 CE 防止恒定预测塌缩；如训练不稳定，再把单向 stop-gradient 教师作为**额外消融**。
- `N_inv=0` 时返回带梯度的零损失；不能 NaN。

### 5.6 语义变化正确响应损失

对经过验证且标签改变的 `X_i^-`，要求新标签比分组的旧标签至少高出 margin：

\[
\boxed{
\mathcal L_{\mathrm{sem}}=
\frac{1}{N_{\mathrm{sem}}}
\sum_i
\left[
 m - \log\bar p(y_i^-\mid X_i^-)
 + \log\bar p(y_i\mid X_i^-)
\right]_+
}
\]

其中 `y_i^- != y_i`，建议初始 `m=0.5`（log-prob margin，需在 dev_select 中调试）。

- `L_sup` 已经对 `X_i^-` 计算 CE，因此此处不再重复计入一份新标签 CE；该损失只增加区分 margin。
- 与现有仅比较 SUP/REF 的对比损失不同，该公式适用于任意**经过验证**的 `old_label -> new_label`。
- 不是简单最大化 `D_JS(p(X),p(X^-))`：后者会奖励错误方向的预测变化。
- 对 NEI 的标签改变若不可核验，则只参与其原始 CE，不进入此项。
- `N_sem=0` 时返回带梯度的零损失。

### 5.7 总损失与独立开关

推荐统一抽象：

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{V2}}={}&
\lambda_{\mathrm{std}}\mathcal L_{\mathrm{sup}}
+\lambda_f\mathcal L_{\mathrm{fact}}
+\lambda_g\mathcal L_g
+\lambda_h\mathcal L_h\\
&+\lambda_q\mathcal L_q
+\lambda_b\mathcal L_{\mathrm{bridge}}
+\lambda_i\mathcal L_{\mathrm{inv}}
+\lambda_s\mathcal L_{\mathrm{sem}}.
\end{aligned}
}
\]

**首轮建议**：`lambda_std=1.0`、`lambda_fact=0.25`、`lambda_g=0.25`、`lambda_h=0.5`、`lambda_q=0.1`、`lambda_b=0.05`、`lambda_i=0.1`、`lambda_s=0.1`；这仅是代码起始值，**不是经过验证的最优参数**。先检查各 loss 标度/梯度范数，避免大权重辅助目标掩盖 CE。所有权重可设 0、必须记录在配置与日志。

分阶段开启：

- `P0`: `L_fact + L_sup`（关闭其他辅助项），验证数据与 standardized head 能正常训练；
- `P1`: 加 `L_h/L_g/L_q`，再独立测 `L_bridge`；
- `P2`: 固定 P1 配置，仅加入 `L_inv`、`L_sem`，分别和共同开启；
- `P3`: 固定 P2 loss 权重，替换反事实样本选择策略；**不得为了 P3 再改网络和 CF 池**。

---

## 6. 固定预算的选择性反事实训练（核心新增模块）

### 6.1 不把“敏感”本身定义为偏差

单纯依据模型对某个编辑产生的预测改变幅度选样，会同时选中真正重要的实体/数字修改。V2 同时考察：

1. 对 `T_inv` 的**错误敏感性**：原本不应改变标签，模型却改变预测分布；
2. 对 `T_sem` 的**反应不足**：标签确实改变，模型却没有足够支持新 gold label。

选样**诊断量**（而不是因果效应估计）：

\[
\begin{aligned}
r_i^{\mathrm{inv}}
&=D_{\mathrm{JS}}\big(\bar p(X_i),\bar p(X_i^+)\big),\\
r_i^{\mathrm{sem}}
&=\left[m - \log\bar p(y_i^-\mid X_i^-)
+\log\bar p(y_i\mid X_i^-)\right]_+.
\end{aligned}
\]

只在相应视图 `verified==true` 时计入；无对应视图不是零错误，应通过 `coverage` 单独记录。

### 6.2 群体设计

首版预注册群体 `l_i`：

```text
l_i = negation_pattern(base_claim) × lexical_overlap_bucket(base_claim, base_evidence)
```

- `negation_pattern`：none / explicit（如过度稀疏可二元化）；
- `lexical_overlap_bucket`：low / mid / high，阈值只由 base-train 分位数拟合；
- 因而首版 6 个互斥群体，易于解释、适合稳定统计。

按 **base 样本** 计算 `l_i`，不能用变换后的词面属性决定该源样本属于哪个组。群体属性要和 gold label 分开记录，组内采样时再做 label-stratification 以避免把“某标签更常见”误当作“某群体更脆弱”。

若某组 `< 50` 对 verified pairs，默认与最近语义组并桶并记录合并规则；不允许只用一个超稀疏组的偶然大分数驱动采样。

### 6.3 群体双向诊断与归一化

\[
R_{\mathrm{inv}}(l)=
\frac{1}{|\mathcal I_l|}\sum_{i\in\mathcal I_l}r_i^{\mathrm{inv}},
\quad
R_{\mathrm{sem}}(l)=
\frac{1}{|\mathcal S_l|}\sum_{i\in\mathcal S_l}r_i^{\mathrm{sem}}.
\]

防止两指标量纲不等，利用**训练集**不同组的中位数归一化：

\[
\widehat R_k(l)=
\frac{R_k(l)}{\mathrm{median}_{l'\in\mathcal G_k} R_k(l')+\epsilon},
\quad k\in\{\mathrm{inv},\mathrm{sem}\}.
\]

若某一项在所有组上几乎为零，则该项诊断分数全部置零并在日志中标记 `zero_signal=true`，避免接近零的分母放大噪声。

\[
B(l)=\alpha\widehat R_{\mathrm{inv}}(l)
+\beta\widehat R_{\mathrm{sem}}(l),
\quad \alpha=\beta=0.5\text{ 为首版}. 
\]

诊断覆盖不足的组退化为 Uniform 配额，不使用不可靠的估计值。

### 6.4 分数转成固定总预算

给定每个训练阶段总 CF pair budget `B_total`，定义平滑群体分配概率：

\[
\begin{aligned}
u_l&=\mathrm{clip}(B(l),0,B_{\max}),\\
\widetilde p_l&=\frac{(n_l+\delta)^\eta\exp(\gamma u_l)}
{\sum_j(n_j+\delta)^\eta\exp(\gamma u_j)},\\
p_l&=(1-\rho)\widetilde p_l+\rho\frac{n_l}{\sum_jn_j}.
\end{aligned}
\]

- `n_l` 是该组当前**合格候选对数量**；
- `eta=0.5`：缓解数据量大的组垄断预算；
- `gamma=1.0`：优先级强度；
- `rho=0.2`：按候选容量分布的探索份额；
- `B_max=5.0`：异常组分数截断；
- `delta=1`、`epsilon=1e-6`：数值稳定。

以上均为**初始设计值**，在 `dev_select` 的预声明小范围搜索中选择；统计显著性检验时不能再按 test 调整。

实现严格保证 `sum_l B_l=B_total`：先 `floor(B_total*p_l)`，再按最大余数分配剩余名额；超过本组 unique candidates 容量时，把溢出名额按剩余容量重分配，最终不足则记录 `budget_shortfall` 并中止主实验，**不得暗中复制候选来凑数**。

#### 必须分层的约束

在每个 `l` 内按 `(transform_type, gold_label_transition)` 做尽可能均衡采样，无法平衡时记录实际比率。对于 invariant 与 semantic 数量不同，先固定 `B_inv : B_sem = 1:1` 作为主方案；与各基线比较时也用同一比例和相同总 pair 数。

### 6.5 建议使用“两阶段冻结预算”，避免在线偏差和过度复杂

**Stage A：Warm-up**

- 使用**同一原始训练集、暂不使用任何 CF**预热 M4 的监督主干，训练 `T_warmup` 个 optimizer steps；此时 `L_inv/L_sem=0`，其他监督按配置执行；
- `warmup_steps = 20%` 的总优化步数，固定步数而不是按 epoch 的百分比；
- 所有同预算选择器基线使用完全相同的 warm-up checkpoint（一个 seed 共享一个 checkpoint，且训练前预先保存）；
- 不使用 OOD 测试信号。

**Stage B：Diagnostic scoring**

- 冻结 warm-up 模型，`torch.no_grad()` 对**训练 CF 候选池**计算 `r_inv/r_sem`；
- 保存 `diagnostics_train_seedXX.jsonl` 和 `group_scores_seedXX.json`；
- 根据上式分配 `B_total` 个 unique pairs（warm-up 阶段 CF 数为零），输出固定 `selected_pair_ids_seedXX.json`；
- **冻结本次选样结果**，继续训练到相同优化步数。

可以在未来把组分数改为 EMA 在线更新，但这是一个新变量，应作为独立的 `online_refresh` 扩展实验。**首版不需要在线边训边改权重**，可以显著减少复现不稳定和选择偏差。

### 6.6 必须比较的选择器（相同候选池、相同预算）

| 选择器 | 分数/规则 | 用途 |
|---|---|---|
| Uniform | 按候选数等概率分层 | 必要基线 |
| Random | 固定 seed 的随机配额 | 随机基线（重复多个 seed） |
| Loss-based | warm-up 模型在 CF 新标签上的平均 CE | 排除“只挑困难样本即可” |
| Shortcut-only | 仅 `R_inv(l)` | 测试是否压低错误敏感性即可 |
| Semantic-only | 仅 `R_sem(l)` | 测试是否强调标签改变即可 |
| Dual-budget（Proposed） | 两个诊断量同时决定预算 | 主方法 |
| GroupDRO-priority（扩展） | 按预定义组的最差监督损失分配 | 较强困难组对照 |
| Oracle（仅半合成） | 已知真实失败率或潜在 U 分组 | 上界，不得用于真实数据 |

**选择性预算的公平性**：这里 `B_total` 专指 **Stage B 中选取的 unique verified pair 数**；Stage A 不使用 CF。各选择器的网络、loss、warm-up、最终训练时长、不同视图的数量、总 pair 数、样本去重约束、评价路径必须一致。训练中重复采样同一 pair 的次数还需额外按 `num_pair_exposures` 记录并匹配。若 `Loss-based` 只见到单类 CF 而 Proposed 使用多类 CF，属于无效比较。

### 6.7 预算曲线

选择 `B_total ∈ {0, 1000, 3000, 5000, 10000}` 对 FEVER 比较至少 Uniform、Loss-based、Dual-budget。该序列仅为建议，若合格候选数量不满足则按审计结果调整并全体统一。报告 `OOD Macro-F1 / IPR / SCR` 对 pair 数曲线，同时统计 token-budget 和总训练 GPU-hours。

---

## 7. 数据批次、训练采样及公平计算预算

### 7.1 Batch 格式

推荐 `PairedBatchCollator` 返回：

```python
{
  "base": {"input_ids": ..., "attention_mask": ..., "z_features": ..., "w_ids": ..., "labels": ...},
  "invariant": {"input_ids": ..., "attention_mask": ..., "z_features": ..., "w_ids": ..., "labels": ...},
  "semantic": {"input_ids": ..., "attention_mask": ..., "z_features": ..., "w_ids": ..., "labels": ...},
  "inv_mask": BoolTensor[B],
  "sem_mask": BoolTensor[B],
  "parent_ids": list[str],
  "pair_ids_inv": list[str],
  "pair_ids_sem": list[str],
  "groups": list[str],
  "old_labels": LongTensor[B],
  "new_labels": LongTensor[B],
  "edit_categories": list[str]
}
```

- 不存在的视图用 dummy 张量占位，但必须 `mask=False`，不能参与 loss/指标。
- 对 V2 整个 batch 的 view 数求真实平均，**不允许 dummy 样本稀释 loss**。
- 相同 `(parent_id, transform_type)` 有多候选时，每次抽取最多一个或按显式 `view_id` 对齐，避免一个 parent 在同一步被重复计数。
- 对于只含 original 的 FEVER NEI 样本，其 `base` CE 正常训练。

### 7.2 同预算训练协议

`same_steps` 是第一优先级，`same_epochs` 不能保证公平，因为附加 CF 后 dataloader 长度可能变化。所有核心模型固定：

```text
base_train_ids   = exactly the same
cf_candidate_pool = exactly the same
num_unique_cf_pairs_selected = exactly the same
optimizer_updates = exactly the same
max_length/packing/padding policy = exactly the same
batch composition schedule = controlled and logged
checkpoint selection = same dev_select rule
seeds = 13 / 21 / 42
```

对于消融 `w/o CF`，因为没有 CF 输入，保证**相同 optimizer steps**，同时明确它所见的 token 数更少；另给 `ERM+Repeat` 对照，使其使用重复原始样本匹配训练 token 数。对于“有 CF”的主对照必须严格匹配所有视图采样和训练 token 数（如 padding 造成小误差，记录并报告）。

**模型复杂度公平性**：V2 附加 head 的参数需要计入；除 standardization 独立价值外，应增设 `same-parameter multi-head` 控制，排除单纯增加参数导致的提升。

### 7.3 学习流程伪代码

```python
set_seed(seed)
base_train = load_base_train()
proxy = fit_structured_proxy(base_train)      # only base-train
pi0 = proxy.frozen_pi0
pool = load_verified_pair_pool(train_only=True)

model = ProxCABIV2(backbone, proxy_schema=proxy.schema, pi0=pi0)
optimizer, scheduler = build_optimizers(config)

# Shared base-only warm-up, with no CF or pair losses, for all selectors.
warmup_model = run_warmup(model, base_train, config.warmup_steps)
save_checkpoint(warmup_model, "warmup.pt")

if config.sampler == "dual_budget":
    scores = score_train_pairs_no_grad(warmup_model, pool, config)
    selected = allocate_verified_pairs(scores, pool, total_budget=config.pair_budget)
else:
    selected = run_baseline_selector(config.sampler, warmup_model, pool)
assert len(set(selected)) == config.pair_budget

model = load_checkpoint("warmup.pt")
for step in range(config.warmup_steps, config.total_steps):
    batch = next_paired_batch(base_train, selected)
    outputs = model.forward_views(batch)
    losses = compute_v2_losses(outputs, batch, model.pi0, config.loss)
    assert_finite(losses["total"])
    update_optimizer(losses["total"])
    if should_evaluate(step):
        metrics = evaluate_on_dev_select(model)
        maybe_save_best(model, metrics["macro_f1_standardized"])

save_run_manifest_and_predictions()
```

**Implementation note**：`run_warmup` 必须以相同初始权重和随机状态在一个 seed 内生成可共享 checkpoint；selector 比较只更换 `selected`，不要分别随机初始化 warm-up 后再对比。

---

## 8. 分阶段主实验矩阵

### 8.1 E0：旧模型复现、泄漏修复与评测基线

| Run | 目的 | 修改 |
|---|---|---|
| `legacy_original` | 保存原代码结果 | 原脚本和旧 ProxyBuilder，不改变旧输出；标注 legacy |
| `legacy_fixed_pi0` | 单独检测 CF 污染 | 仅 `ProxyBuilder.fit(base_train)`，CF 后不再拟合 |
| `legacy_fixed_eval` | 校验固定解码/选择 | 使用预声明头 + 原始 dev_select，ID/OOD 不选头 |

必须产出 `legacy_diff_report.md`，记录代理参考分布是否改变、模型选择/校准差异、旧/新指标差异；**不能把历史参数调度不同的 checkpoint 和新三随机种子结果直接作显著性比较**。

### 8.2 E1：主方法阶段消融（同一 backbone）

**结构顺序**：先检验预测式标准化模块是否有效，再增加双语义约束，最后加选样预算。

| ID | 配置 | Structured Z/W | Fixed pi0 + supervised std | `L_inv` | `L_sem` | Dual-budget |
|---|---|---:|---:|---:|---:|---:|
| B0 | ERM / fact-only | — | — | — | — | — |
| B1 | CF-only | — | — | — | — | — |
| B2 | Fact-only contrastive | — | — | — | —（单独使用传统 contrastive loss） | — |
| M0 | Legacy ProxCABI（固定协议重训） | legacy hash | legacy std | — | legacy contrastive | — |
| M1 | Structured-Std | ✓ | ✓ | — | — | — |
| M2 | Structured-Std + Inv | ✓ | ✓ | ✓ | — | — |
| M3 | Structured-Std + Sem | ✓ | ✓ | — | ✓ | — |
| M4 | Structured-Std + Dual | ✓ | ✓ | ✓ | ✓ | — |
| M5 | **Full V2** | ✓ | ✓ | ✓ | ✓ | ✓ |

注意 B1/B2/M1–M5 都应使用**相同的已验证反事实池**；M1 即使不开 inv/sem 正则，也看到相同 CF 上的基本分类监督，才能隔离正则的增益。

### 8.3 E2：与强基线比较

需要在完全相同数据/Backbone 上重新训练：

1. ERM（不见 CF）+ `ERM+Repeat`（匹配更新与样本 token 次数）；
2. CF-only（相同 CF 样本）和 matched contrastive；
3. Bias reweighting（用固定训练集群体）；
4. GroupDRO（相同可观察群体、相同优化预算）；
5. legacy ProxCABI（按新共同协议重训）；
6. 可复现的已发表 fact-verification debiasing 方法（如 CLEVER，单独注明 encoder、重训练难度和数据设置是否相同）。

公平性分级：

- **Tier A（主表可直接归因）**：相同 backbone、基础样本、CF 池、训练步数、超参搜索预算的内部匹配基线。
- **Tier B（辅助对照）**：论文发布实现、不同模型/数据处理/预训练的外部方法；给出引用、设置差异，不在其上直接声称组件有效。

### 8.4 E3：Proxy 组件必要性消融

以下消融使用 M4 的相同训练样本、同一 random seed/优化预算；移除信息时仍保证预测头有独立监督，**不允许变成随机未训练 head**。

| 消融 | 操作 | 目标解释 |
|---|---|---|
| `no_z_info` | Z 输入常量，保留 q/g/h 各自监督 | Z 信息是否必要 |
| `no_w_info` | 所有 W 特征输入常量，仍训练 h 与标准化头 | W 信息是否必要 |
| `no_bridge` | `lambda_b=0`，其余不变 | bridge consistency 是否有帮助 |
| `no_std_train` | 去除 `L_sup(std)`，保留 h/g/CE，预注册 fact 主头 | 标准化训练监督的价值（结构消融） |
| `no_std_inference` | 相同 M4 checkpoint，推理只用 fact | 标准化推理操作的价值（推理消融） |
| `uniform_pi0` | 使用均匀 W 权重，其他不变（同时报告单纯推理替换与重训） | 训练参考分布的作用 |
| `shuffle_z` / `shuffle_w` | train-only 随机置换代理 ID，维持类别频次 | 检测代理只是额外参数 |
| `random_proxies` | 与标签无关的随机 Z/W，容量相同 | 代理信息有效性 |
| `legacy_hash_256` | 旧哈希代理并冻结 pi0 | 可解释代理是否优于旧哈希 |
| `w_3/9/16/32` | 调整 W 的可解释类别粗细（有效支持集审计） | 容量/稀疏度敏感性 |
| `without_h_ce` | `lambda_h=0` | h 显式监督是否必要 |
| `without_q` | `lambda_q=0` | W 预测器是否必要 |

结构消融一定要从同一结构和初始化公平重训；仅改 inference 的实验不能伪称 retrained ablation。

### 8.5 E4：双重约束与 CF 质量消融

- `only_invariance`、`only_semantic`、`both`、`neither`；
- 合格 CF 对数：`0/1k/3k/5k/10k`（受候选池支持限制）；
- 变换类型：negation / number / entity / relation / time / syntax / paraphrase；
- 只用高置信/人工核验 CF vs teacher-only CF（teacher-only **只能做风险分析**，不可混入正式标注正确的主实验）；
- `m ∈ {0.2, 0.5, 1.0}`；
- `lambda_i/lambda_s ∈ {0, 0.05, 0.1, 0.2}`，先单项搜索再少量联合搜索；
- 删除不可证实的 NEI semantic 对，看三分类原始性能是否保留；
- 将 JS 不变性换成 KL / cosine（可选追加消融，非必做）。

### 8.6 E5：选择性预算实验

**核心归因**：固定 M4 架构与损失，仅比较 Uniform、Random、Loss-based、Shortcut-only、Semantic-only、Dual-budget。必要时加入 GroupDRO-priority。

报告：

- 随预算增加的 `OOD Macro-F1 / IPR / SCR`；
- 所选 CF 在各 bias group、变换类别、标签转移上的分布；
- 每种方法的预算使用率、重复/无效/不足条数；
- 训练 loss 与 train-only 两种风险分数的趋势；
- 一个固定样本的定性案例：Uniform 选择遗漏什么，Dual 为什么选中；
- `alpha` / `beta` 扫描的稳定性：`(1,0)`、`(0,1)`、`(0.5,0.5)`；
- `rho` 探索率：`0/0.2/0.5`（只在 dev_select 调参）。

**防止伪增益**：如果 `Dual-budget` 只优于 Uniform，却不优于 Loss-based，论文不宜宣称选样机制的独立创新；需要分析这是否只是传统困难样本挖掘的重新命名。

---

## 9. 评测指标：主指标、配对指标、诊断指标

### 9.1 常规分类指标

分别记录 `Accuracy`、`Macro-F1`、`Per-class Precision/Recall/F1`、`Confusion Matrix`：

- `ID`: 冻结的 FEVER `dev_report`（或合法独立 test）；
- `OOD`: symmetric-FEVER 的冻结评测 split；
- `Transfer`: VitaminC、HoVer、PolitiHop 的合法报告 split；
- `Heads`: fact、g、standardized 三个 head **都展示**，但 standardized 为主方法预注册主 head。

对于对称 FEVER 二分类：

```python
# A 3-class model evaluated on a genuinely binary SUPPORTS/REFUTES dataset.
binary_logits = logits[:, [SUPPORTS_ID, REFUTES_ID]]
probs_binary = torch.softmax(binary_logits, dim=-1)
pred_binary = probs_binary.argmax(dim=-1)
```

如果采用阈值校准：只能在**独立** `dev_calib` 的二分类金标签子集拟合并冻结，然后同一个已拟合阈值应用所有目标测试集；校准方案需预注册，不能在测试集逐模型寻优。训练三分类和测试二分类应在表头明确标注。

### 9.2 IPR：Invariant Pair Correctness Rate

同时要求原始与标签保持变换都正确：

\[
\boxed{
\mathrm{IPR}=\frac{1}{N_{\mathrm{inv}}}\sum_i
\mathbb 1[\hat y(X_i)=y_i\land\hat y(X_i^+)=y_i].
}
\]

额外记录：

- `Consistency_all = P(\hat y(X)=\hat y(X^+))`：不要求正确，单独报告；
- `IPR_by_transform`、`IPR_by_label`；
- `P(prediction_flip | invariant)`：不稳定率。

**解释**：只看一致性可能让永远输出同一标签的模型得高分，因此 IPR 必须和常规 Macro-F1 一起报告。

### 9.3 SCR：Semantic Change Correct Response

要求原始与真实语义变化后都正确：

\[
\boxed{
\mathrm{SCR}=\frac{1}{N_{\mathrm{sem}}}\sum_i
\mathbb 1[\hat y(X_i)=y_i\land\hat y(X_i^-)=y_i^-].
}
\]

额外报告：

- `SCR_by_transition`（SUP→REF、REF→SUP、涉及 NEI 的可证实转移）；
- `ChangedOnlyAcc=P(\hat y(X^-)=y^-)`：便于区分原样本错/新样本错；
- `Semantic_Flip_Rate`：仅作为描述指标，**不能代替 SCR**；
- `Delta_gold_margin = [log p(y^-|X^-)-log p(y|X^-)]` 分布。

### 9.4 Worst-group 及 shortcut 依赖

对仅由 train-only 规则确定的可解释群体：

- `worst_group_macro_f1 = min_l macro_f1(l)`，只对满足最小样本阈值（建议 `n>=50`）的组定义；
- `group_gap = max_l F1(l)-min_l F1(l)`；
- `accuracy_by_negation`、`overlap`、`length`、`entity-pattern`；
- `claim_only` vs `claim+evidence` 性能差异及证据置乱对预测的影响（诊断 shortcut，而非直接证明因果）。

如果组内部只含单一 gold class，宏 F1 解释会失真；此时改用 group accuracy / group conditional recall，并标记 `macro_f1_not_comparable`。

### 9.5 Calibration、代理诊断、效率

附加报告 ECE / Brier（适用时）、`NLL`、`pi0 entropy`、`q(W)` accuracy/NLL、`JS(g, qh)`、`h` 对 W 的条件响应幅度，以及：

- 参数量 / 可训练参数量；
- 单 epoch、总训练耗时与 GPU-hours；
- 峰值显存，推理 `samples/sec` 和每条平均/95th-percentile latency；
- CF 生成数量、通过过滤率、教师/人工标注成本；
- 相同预算下基础 token 数、CF token 数、梯度更新数。

### 9.6 统计检验

- 预注册 `seeds = [13,21,42]`；确认主要趋势后可增加到 5 seeds。
- 各指标报告 `mean ± sample std`，并保留每个 seed 的独立数值。
- 同一测试集比较两模型：对**配对逐样本预测**计算差值，用 `parent_id` 聚类 bootstrap（建议 2,000 次），防止同源反事实对被当作独立样本。
- 同时报告 seed-level 配对差异；如采用层次 bootstrap，要清楚区分 seed-level 与 sample-level 不确定性。
- 对预声明的主要比较（Full vs CF-only、Full vs GroupDRO、M4 vs M1、M5 vs M4）采用 Holm 多重比较校正。
- 不显著要如实报告；禁止从多个 OOD 数据集/threshold 中选择最好一项只报告正结果。

### 9.7 主结果表建议

| Method | FEVER ID Macro-F1 | symmetric-FEVER Macro-F1 | IPR↑ | SCR↑ | Worst-group↑ | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| ERM | TBD | TBD | TBD | TBD | TBD | TBD |
| CF-only | TBD | TBD | TBD | TBD | TBD | TBD |
| Contrastive | TBD | TBD | TBD | TBD | TBD | TBD |
| Reweighting | TBD | TBD | TBD | TBD | TBD | TBD |
| GroupDRO | TBD | TBD | TBD | TBD | TBD | TBD |
| Structured-Std | TBD | TBD | TBD | TBD | TBD | TBD |
| V2-Dual (M4) | TBD | TBD | TBD | TBD | TBD | TBD |
| **V2-Full (M5)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

`TBD` 必须由程序从真实评测文件填充，不允许人工臆造数值。

---

## 10. 建议的代码目录、职责与接口契约

原则：**不破坏既有 `python -m proxcabi train/evaluate`，新增 V2 命令与独立模块**。先复用原 `data.py`、tokenizer、optimizer、统计工具，然后逐步提取公共代码。

```text
ProxCABI-FV/
├── proxcabi/
│   ├── data.py                   # 已有：FactSample + load_samples
│   ├── model.py                  # 已有：Legacy ProxCABIModel，暂保留
│   ├── proxies.py                # 已有：Legacy ProxyBuilder，暂保留
│   ├── torch_data.py             # 已有：Dataset / Collator / Sampler
│   ├── train.py                  # 已有：原训练和评测
│   ├── cli.py                    # 已有：增加 V2 subcommands
│   ├── causal/                   # 已有：严格 Plan A，原则上不改
│   ├── v2/                       # 【新增】
│   │   ├── __init__.py
│   │   ├── schemas.py            # 反事实/代理/配置强校验数据结构
│   │   ├── prepare_pairs.py      # 转换旧 CF、验证、去重、拆分
│   │   ├── pair_dataset.py       # PairedDataset / Collator / Sampler
│   │   ├── structured_proxies.py # StructuredProxyBuilder + pi0
│   │   ├── model.py              # ProxCABIV2 + standardized probs
│   │   ├── losses.py             # 独立可测的 8 个损失
│   │   ├── budget.py             # Uniform/Loss/Dual 等固定预算
│   │   ├── train.py              # two-stage 训练主入口
│   │   ├── evaluate.py           # 统一 ID/OOD/Pair/Subgroup 报告
│   │   ├── diagnostics.py        # proxies/损失/样本选择质量
│   │   ├── reporting.py          # 表格、CI、manifest、效率报告
│   │   └── config.py             # YAML 加载、覆盖、版本验证
│   └── __main__.py              # 已有，不删除原 CLI
├── configs/
│   └── v2/                       # 【新增】
│       ├── fever_base.yaml
│       ├── fever_pilot.yaml
│       ├── exp_main.yaml
│       ├── exp_proxy_ablations.yaml
│       ├── exp_budget.yaml
│       └── dataset_label_mappings.yaml
├── run_scripts/
│   ├── run_v2_smoke.sh           # 【新增】
│   ├── run_v2_fever_pilot.sh
│   ├── run_v2_multiseed.sh
│   ├── run_v2_budget.sh
│   └── run_v2_report.sh
├── tests/
│   └── v2/
│       ├── test_pairs.py
│       ├── test_no_leakage.py
│       ├── test_proxies.py
│       ├── test_standardization.py
│       ├── test_losses.py
│       ├── test_budget.py
│       ├── test_training_smoke.py
│       └── test_eval_protocol.py
└── outputs/
    └── v2/                       # 【新增】
        ├── frozen_data/
        ├── proxy_artifacts/
        ├── pilot/
        ├── main/
        ├── ablations/
        ├── budget/
        └── paper_tables/
```

### 10.1 每个模块必须实现的最小接口

#### `schemas.py`

```python
@dataclass(frozen=True)
class CounterfactualRecord:
    record_id: str
    parent_id: str
    claim: str
    evidence: str
    source_label: int
    target_label: int
    transform_type: str
    changed_field: str
    edit_category: str
    verified: bool
    verification_method: str
    split_origin: str

@dataclass
class PairSample:
    base: FactSample
    invariant: FactSample | None
    semantic: FactSample | None
    group_id: str
    pair_id_inv: str | None
    pair_id_sem: str | None
```

要求定义 `validate_record`、`validate_pair`、`validate_label_transition`，在读入时 fail-fast；不得由空字符串/错误标签默默生成一个样本。

#### `structured_proxies.py`

```python
class StructuredProxyBuilder:
    def fit(self, base_samples: list[FactSample]) -> "StructuredProxyBuilder": ...
    def encode_z(self, sample: FactSample) -> torch.Tensor: ...
    def encode_w(self, sample: FactSample) -> int: ...
    def reference_distribution(self, alpha: float = 1.0) -> torch.Tensor: ...
    def save(self, directory: Path) -> None: ...
    @classmethod
    def load(cls, directory: Path) -> "StructuredProxyBuilder": ...
```

持久化 `base_train_source_sha256`、`w_vocab`、`z_feature_names`、`thresholds`、`pi0`、`feature_source_provenance`、`fit_sample_count`。保存、加载后同一样本编码必须逐 bit 一致。

#### `model.py`

```python
class ProxCABIV2(nn.Module):
    def encode(self, input_ids, attention_mask) -> torch.Tensor: ...
    def forward(self, input_ids, attention_mask, z_features, w_ids=None) -> dict: ...
    def h_all_w_logits(self, m) -> torch.Tensor: ...  # [B, K, C]
    def standardized_probs(self, m) -> torch.Tensor: ... # [B, C]
    def standardized_logits(self, m) -> torch.Tensor: ...
    def forward_views(self, paired_batch) -> dict: ...
```

`pi0` 注册为 `register_buffer("pi0", tensor)`，不参与 `optimizer.parameters()`，默认 `requires_grad=False`。`forward()` 返回 `m/fact_logits/g_logits/q_logits/h_obs_logits/std_logits/bridge_probs`；所有 logits/probs 的含义和 shape 在 docstring 声明。

#### `losses.py`

```python
@dataclass
class V2LossWeights:
    std: float
    fact: float
    g: float
    h: float
    proxy: float
    bridge: float
    invariance: float
    semantic: float

def compute_supervised_losses(view_outputs, paired_batch, weights) -> dict: ...
def invariant_js(p_orig, p_inv, inv_mask) -> torch.Tensor: ...
def semantic_margin_loss(logp_sem, old_label, new_label,
                         sem_mask, margin: float) -> torch.Tensor: ...
def bridge_alignment_loss(g_probs, q_probs, h_all_w_probs) -> torch.Tensor: ...
def compute_v2_losses(view_outputs, paired_batch, weights) -> dict: ...
```

每个损失返回真实标量、有效样本数和 `valid` 状态；所有 loss 支持 float32，AMP 时 JS/`log` 建议用 float32 计算。处理全 mask 为 false 时返回标量 0，并保留可兼容 backward 的计算图。

#### `budget.py`

```python
@dataclass(frozen=True)
class BudgetConfig:
    total_pairs: int
    inv_sem_ratio: float
    alpha: float
    beta: float
    gamma: float
    rho: float
    eta: float
    min_group_pairs: int
    seed: int

def score_verified_pairs(model, pool, batch_size, device) -> list[dict]: ...
def aggregate_group_scores(rows: list[dict], config: BudgetConfig) -> dict: ...
def allocate_pair_budget(pool, group_scores, config: BudgetConfig) -> list[str]: ...
def select_pairs(mode, pool, scores, config) -> list[str]: ...
```

要求 `allocate_pair_budget()` 是确定性纯函数：同输入+seed 返回相同排序/集合；无重复 ID；总数准确；类别容量不足时报错或明示状态，不静默越界；输出每组分配表。

#### `evaluate.py` / `reporting.py`

```python
def evaluate_classification(model, dataset, head="standardized") -> dict: ...
def evaluate_pairs(model, pair_dataset, head="standardized") -> dict: ...
def evaluate_proxy_health(model, samples) -> dict: ...
def build_paper_tables(run_manifest_files, output_dir) -> None: ...
```

每个评估子例必须保留 `sample_id/parent_id/gold_label/pred_label/probs/head`，并输出到 JSONL；报告程序直接消费 JSONL，而不是从日志手工录入指标。

### 10.2 CLI 设计（**以下新命令需要实现，当前仓库尚不能直接运行**）

```bash
# 1. 生成、验证并冻结 pair pool
python -m proxcabi v2-prepare \
  --config configs/v2/fever_base.yaml \
  --output-dir outputs/v2/frozen_data

# 2. 仅用原始训练集拟合并冻结 V2 代理
python -m proxcabi v2-fit-proxies \
  --config configs/v2/fever_base.yaml \
  --output-dir outputs/v2/proxy_artifacts/fever

# 3. 单卡冒烟训练
python -m proxcabi v2-train \
  --config configs/v2/fever_pilot.yaml \
  --variant M4 --seed 13 \
  --output-dir outputs/v2/pilot/M4_seed13

# 4. 统一评估，不允许在 test 上重新调参
python -m proxcabi v2-evaluate \
  --checkpoint outputs/v2/pilot/M4_seed13/best_model.pt \
  --config configs/v2/fever_base.yaml \
  --split dev_report \
  --output-dir outputs/v2/pilot/M4_seed13/eval_id

# 5. 固定组合报告
python -m proxcabi v2-report \
  --runs-dir outputs/v2/main \
  --output-dir outputs/v2/paper_tables
```

### 10.3 各文件与当前代码的迁移映射

| 当前代码 | 复用部分 | 新增/重构动作 |
|---|---|---|
| `proxcabi/data.py` | `FactSample`、标签归一化、split 载入 | 无侵入；V2 读取 `metadata` 扩展属性 |
| `proxcabi/torch_data.py` | tokenization、旧 sampler 机制 | 新增 `v2/pair_dataset.py`，保证三视图严格对齐 |
| `proxcabi/proxies.py` | 文本特征提取工具 | 新增结构化 builder；保留 legacy hash 对照 |
| `proxcabi/model.py` | 编码器装载、JS 函数、原 MLP | 新 V2 model；不覆盖原权重类定义 |
| `proxcabi/train.py` | 优化器、AMP、评估和保存机制 | 新 two-stage 训练入口、分离 loss 和采样逻辑 |
| `proxcabi/counterfactuals.py` | 已有 hard-CF 候选与 contrast group | 编写适配器转换统一 pair schema，独立核验 |
| `proxcabi/bias_groups.py` | negation、LMI、lexical overlap 函数 | 冻结 train-only 分桶，输出互斥 group ID |
| `proxcabi/causal/` | 严格 Plan A 合成实验 | 本项目不改其因果识别论证，不复用其 ATE 命名 |

---

## 11. 初始 YAML 配置（可作为实现的默认模版）

保存为 `configs/v2/fever_base.yaml`。字段是**V2 建议的新 schema**，加载器应拒绝未知字段和不合法的组合；如 `K`、CF 数量不满足实际数据，应在正式实验前经过审计后更新并冻结。

```yaml
schema_version: "v2.0"
project: "ProxCABI-FV-SSSD"
seed: 13

paths:
  data_dir: "data"
  output_root: "outputs/v2"
  proxy_artifact: "outputs/v2/proxy_artifacts/fever"
  pair_pool_train: "outputs/v2/frozen_data/fever_cf_train.jsonl"
  pair_pool_val: "outputs/v2/frozen_data/fever_cf_dev_select.jsonl"

data:
  dataset: "FEVER"
  train_split: "train"
  select_split: "dev_select"
  calib_split: "dev_calib"
  report_split: "dev_report"
  ood_dataset: "FEVER"
  ood_split: "symmetric"
  label_space: ["supports", "refutes", "not enough info"]
  pair_required_verified: true
  allow_unverified_semantic: false
  primary_group: "negation_x_overlap"
  min_group_support: 50
  inv_sem_ratio: 0.5
  max_length: 256

proxy:
  mode: "structured"
  fit_on: "base_train_only"
  z_features:
    - "claim_negation"
    - "claim_length"
    - "claim_entity_pattern"
    - "claim_template"
    - "claim_lexical_shortcut"
  w_features:
    - "evidence_length_bucket"
    - "evidence_sentence_count_bucket"
  w_use_unknown_category: true
  w_max_classes: 16
  smoothing_alpha: 1.0
  freeze_pi0: true
  q_hidden_dim: 768
  proxy_embedding_dim: 64

model:
  backbone: "roberta-base"
  dropout: 0.1
  primary_head: "standardized"
  train_fact_head: true
  train_g_head: true
  train_q_head: true
  train_h_head: true
  enable_bridge_alignment: true

loss:
  lambda_std: 1.0
  lambda_fact: 0.25
  lambda_g: 0.25
  lambda_h: 0.5
  lambda_proxy: 0.1
  lambda_bridge: 0.05
  lambda_invariance: 0.1
  lambda_semantic: 0.1
  semantic_margin: 0.5
  js_eps: 1.0e-8

sampling:
  mode: "dual_budget"    # uniform / random / loss / shortcut / semantic / dual_budget
  pair_budget: 5000
  warmup_fraction: 0.2
  inv_sem_ratio: 0.5
  group_alpha: 0.5
  group_beta: 0.5
  softmax_gamma: 1.0
  explore_rho: 0.2
  capacity_eta: 0.5
  min_group_pairs: 50
  max_group_score: 5.0
  select_without_replacement: true

train:
  epochs_equivalent: 3     # 仅用于预算推算，实际以 total_optimizer_steps 为准
  total_optimizer_steps: 15000  # 占位值；先计算各模型统一的真实预算并冻结
  batch_size: 16
  grad_accumulation_steps: 2
  eval_batch_size: 32
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.06
  amp: true
  amp_dtype: "float16"
  gradient_clip_norm: 1.0
  checkpoint_interval_steps: 1000
  selection_metric: "macro_f1_standardized"
  early_stop_on_ood: false

experiment:
  variants: ["B0", "B1", "B2", "M0", "M1", "M2", "M3", "M4", "M5"]
  seeds: [13, 21, 42]
  primary_comparisons:
    - ["M5", "B1"]
    - ["M5", "GroupDRO"]
    - ["M4", "M1"]
    - ["M5", "M4"]
  pair_bootstrap_repeats: 2000
  report_per_sample_predictions: true
  forbid_test_time_selection: true
```

配置校验必须遵循：

- `primary_head=standardized` 需要 `train_h_head=true` 且 `lambda_std>0` 或有充分 `lambda_h>0`；
- `lambda_invariance>0` 需要已验证 invariant pair，不能假装全是 true；
- `lambda_semantic>0` 需要已验证、真实改变的 gold label；
- `dual_budget` 必须提供训练 pair pool 和足量 eligible pairs；
- `freeze_pi0=true` 禁止训练期间更改 `pi0` buffer；
- `total_optimizer_steps` 应按共同实验实际预算推导，在 `manifest` 中记录，以上 `15000` 不可不经计算就直接当作 FEVER 标准训练量；
- `seed` 与 `experiment.seeds` 的选择要记录，批次采样和数据增强 RNG 不应使用全局不可复现随机状态。

---

## 12. 自动化测试与验收（每个阶段必做）

### 12.1 单元测试列表

| 测试 ID | 输入/动作 | 必须通过的断言 |
|---|---|---|
| T01 | save/load StructuredProxyBuilder | 相同 train 样本 Z/W 编码与 pi0 完全一致 |
| T02 | train 与 train+CF 分别调用编码 | 两种训练记录使用**同一已冻结** pi0 与词汇，不因 CF 变化 |
| T03 | 构造均匀 h、任意合法 pi0 | `std_probs.shape=[B,C]`、每行和为 1，无 NaN/Inf |
| T04 | 所有 h 在 W 上完全相同 | `std_probs == h(any W)`，验证加权聚合正确 |
| T05 | 调换 pi0 权重 | `std_probs` 按精确解析加权值变化 |
| T06 | `lambda_h` 或 `lambda_std` 非零 | `h_head` 梯度非零、参数训练后改变 |
| T07 | `no_w_info` | h 仍接受监督并有梯度，不能随机保留 |
| T08 | `T_inv` 完全相同预测 | `L_inv≈0` |
| T09 | `T_inv` 输出不同分布 | `L_inv>0`、梯度有限 |
| T10 | `T_sem` 新标签概率明显大于旧标签 | hinge loss 为零（满足 margin 时） |
| T11 | `T_sem` 新标签概率小于旧标签 | hinge loss 正值、梯度方向正确 |
| T12 | invalid semantic label transition | 必须拒绝样本，不默默训练 |
| T13 | NEI 原始样本且无 semantic 变换 | base CE 正常、semantic mask=false |
| T14 | 全部 inv/sem mask=false | loss=0、无 NaN/Inf、backward 仍可执行 |
| T15 | 指定 group scores 与容量 | `selected_ids` 无重复、预算守恒、同 seed 一致 |
| T16 | 某组容量不足 | 明确再分配或报错，不能产生重复 ID |
| T17 | 同 parent 横跨 split | 数据冻结前 fail-fast，打印冲突 ID |
| T18 | 仅有两类 OOD | 固定二类解码，无阈值搜索/OOD 标签拟合 |
| T19 | 任何测试样本进入 ProxyBuilder.fit | 单元测试用 mock 监测并抛异常 |
| T20 | 从 best checkpoint 恢复 | 与保存时 standardized 预测在数值容差内一致 |
| T21 | `pbar` 主头加 loss 开关组合 | 关闭某项正则时其梯度/日志不再计入，其他项正常 |
| T22 | 同配置运行两次 CPU smoke | 数据划分、候选选择、日志关键字段确定一致（浮点微小差异可设容差） |

### 12.2 训练 sanity checks

- 用 32–64 个经过核验的 pair 做 tiny-overfit：监督 CE 应显著下降，训练正确率接近饱和；
- 标准化头输出概率行和为 1，不能出现 `-inf`；
- 梯度日志分别记录 encoder / fact / g / q / h / z-encoder 的范数；
- 将 `T_inv` 设成原始输入，JS 应接近 0；
- 将 `T_sem` 的新旧 gold labels 反转，代码应拒绝非法样本或体现正确的损失方向；
- 人为把 W 在 train 中打乱，模型可以训练，但 proxy diagnostic 应有所变化；如果和原方案完全没有差异，需要解释是否代理无效；
- `K<=32` 时枚举全部 W 不应出现明显不可控显存问题；较大 K 必须预先 benchmark；
- 梯度累积最后不足一个 accumulation group 时，仍必须执行最终 optimizer step；不能沿用旧代码可能忽略尾部 step 的计算习惯；
- AMP 下 JS、log-prob、标准化求和采用 float32 防止数值不稳定。

### 12.3 数据与结果工件检查

每次实验最少存在：

```text
run_dir/
├── best_model.pt
├── last_model.pt
├── config_resolved.yaml
├── dataset_manifest.json
├── proxy_builder_v2.json
├── pi0.json
├── label_mapping.json
├── git_commit.txt
├── environment.txt
├── selected_pair_ids.json
├── group_scores.json
├── metrics_steps.jsonl
├── checkpoint_selection.json
├── eval_id.json
├── eval_ood.json
├── eval_pairs.json
├── predictions_id.jsonl
├── predictions_ood.jsonl
├── predictions_pairs.jsonl
├── efficiency.json
└── run_manifest.json
```

- `best_model.pt` 必须包含 **标准化 head、pi0 buffer、proxy vocab/feature schema version、label mapping** 的可靠恢复入口；
- `checkpoint_selection.json` 记录使用 `dev_select` 何指标挑选了哪个 step，证明未访问 OOD test；
- `run_manifest.json` 应记录 data hash、selected pair hash、config hash、Python/PyTorch/Transformers/CUDA 版本、GPU 名称、seed、实际 optimizer steps、训练时间；
- `predictions_pairs.jsonl` 必须包含 `parent_id + transform_type`，允许 cluster bootstrap。

---

## 13. 推荐的分阶段工程实施与 Go / No-Go 决策

### Milestone 0 — 冻结基线与检查旧代码（P0）

**修改文件**：尽量不改旧实现；新增 `docs/v2_baseline_audit.md`、hash manifest。  
**要做的事**：

1. 保存当前 Git SHA、旧 config、已有评估文件及 `docs/WWW_EXPERIMENT_TODO.md` 状态；
2. 检查旧 `fit(train+CF)`、head 选择、calibration、`h_head` 梯度；
3. 冻结 FEVER dev 三划分和 label map；
4. 重算可靠的 legacy baseline 或明确标记旧 checkpoint 不可公平比较。

**验收**：训练/验证/报告 split 不重叠；可自动复现评估结果；知道原版的每个 head 在报告 split 的输出。

### Milestone 1 — CF 质量与结构化代理（P0）

**修改文件**：`schemas.py`、`prepare_pairs.py`、`structured_proxies.py`、相关测试。  
**要做的事**：

1. 将现有 hard-CF 转为 `parent/inv/sem` 格式，补齐合法方向；
2. 审计合法率/重复/泄漏并冻结候选池；
3. 使用 `base_train` 拟合 Z/W 及固定 pi0；
4. 保存代理分布、覆盖率、train-only 词汇及 schema；
5. 以 100 个训练样本和 20 对 CF 跑 CPU 前处理冒烟。

**验收**：T01–T05、T12、T17、T19 通过；任何变更 CF 数量不改变 proxy artifact；能按真实数据提供足量的 valid pair。

### Milestone 2 — V2 主预测头与基础监督（P0）

**修改文件**：`model.py`、`losses.py`、`pair_dataset.py`、`train.py`。  
**要做的事**：

1. 实现全部 W 的 h 枚举和 `pi0` buffer；
2. `fact/g/q/h` 独立监督，`L_sup(std)` 和 JS bridge；
3. 同一 base+CF 监督量比较 CF-only、Structured-Std、Legacy-fixed；
4. 检查 `h` 梯度与随机代理对照；
5. 只用 seed 13 做预注册小网格调参，再固定 top-2 配置。

**验收**：tiny-overfit 成功；T03–T07、T14、T20 通过。  
**关键 Go/No-Go**：若 Structured-Std 在等预算对照下仍不优于 CF-only，**先诊断代理/W 监督/模型头有效性**，不要急于进入选择性训练。

### Milestone 3 — 双重语义约束（P0）

**修改文件**：`losses.py`、`pair_dataset.py`、`evaluate.py`。  
**要做的事**：

1. 加 `L_inv` 和 `L_sem` 的纯函数实现；
2. 验证 paired batch 和 invalid/NEI masking；
3. 跑 M1/M2/M3/M4，固定 CF pool、预算与 optimizer steps；
4. 在 `dev_select` 同时比较 Macro-F1、IPR、SCR，不能只看一个方向；
5. 运行 seed 13，确认趋势后扩展 21/42。

**验收**：T08–T14、T21 通过；正负干预都能被正确计算；与 CF-only 可以按相同样本比较。  
**关键 Go/No-Go**：若加两类损失后仅 IPR 升高、SCR 大幅下降，说明方法过度不变，需要调整权重/表示而非只优化总分。

### Milestone 4 — 选择性固定预算（P1）

**修改文件**：`budget.py`、`v2/train.py`、`v2/diagnostics.py`。  
**要做的事**：

1. 固定一个 seed 的 M4 warm-up；
2. train-only no-grad 评分；
3. 实现 6 种 sampler；
4. 同预算比较 Uniform、Loss-based、Dual；
5. 输出预算分配图、每组分数、selected IDs 和 train/OOD 表现。

**验收**：T15–T16、T22 通过；样本预算守恒；Dual 不依赖 dev/test 数据选样。  
**关键 Go/No-Go**：若 Dual 没有超过 Loss-based，则保留 M4 为更简单的候选主模型，不必为了论文强行把 Dual 说成有效。

### Milestone 5 — 论文完整实验（P1/P2）

**修改文件**：`run_scripts/run_v2_multiseed.sh`、`reporting.py`、跨数据集数据适配器。  
**要做的事**：

1. 完成 Tier A 强基线和主消融的 3-seed；
2. 完成 ID、symmetric-FEVER、IPR/SCR、worst-group 和效率报告；
3. 执行 bootstrap/Holm，统计失败案例；
4. 在 FEVER 上确认优于强基线后再扩展 VitaminC/HoVer/PolitiHop；
5. 保存匿名工件包：配置、模型、结果、评估工具及数据处理脚本（受许可约束的内容不要非法再分发）。

**验收**：可用单脚本从已有 checkpoint 和 JSONL 生成论文所有数值表；没有任何未说明的 test-time calibration。

### 建议的论文方法选择门槛（研究决策准则，不是实验事实）

| 决策 | 推荐标准 |
|---|---|
| V2 proxy 值得保留 | 对比同监督 CF-only 有可重复收益，且代理负对照/消融表明不是单纯额外参数 |
| 双重语义损失值得保留 | 在多个 seed 上改善 IPR 与 SCR 的至少一种明显折中，同时 ID Macro-F1 不发生不可接受的下降 |
| Dual-budget 值得作为独立贡献 | 同 CF 数量下持续优于 Uniform 和 Loss-based，而非仅击败 Random |
| 主论文可宣称 OOD 改进 | 对最强匹配基线报告估计差值、CI、seed 方差与负结果，并在至少一个预注册 OOD 集上有稳健收益 |
| 不能达标 | 简化为 M4，重新定位代理/预算模块为负面分析或限制，不能把它们写成验证过的核心创新 |

---

## 14. 实验执行顺序、预算清单与最终表格

### 14.1 最少计算量的探索阶段

建议严格遵循依赖顺序，不一次性开展全部组合：

| 阶段 | 实验配置 | 种子 | 目的 | 何时进入下一阶段 |
|---|---|---|---|---|
| Smoke | tiny dataset, ERM/CF/M1 | 13 | 测试数据、loss、梯度、保存恢复 | 通过所有 P0 tests |
| Pilot 1 | B1、M0-fixed、M1 | 13 | 定位代理标准化增量 | M1 可训练，代理诊断不是退化 |
| Pilot 2 | M1、M2、M3、M4 | 13 | 验证双约束独立作用 | IPR/SCR 出现有意义的改善 |
| Pilot 3 | M4 + Uniform/Loss/Dual | 13 | 预算模块可行性 | Dual 至少可与 Loss-based 竞争 |
| Confirmation | top-2 + B1 + GroupDRO | 21、42 | 排除 seed 13 偶然性 | 稳定趋势并记录负结果 |
| Final | 全部核心方法 + ablations | 13/21/42 | 主表和消融 | 冻结设置后再测试 OOD |
| Transfer | VitaminC、HoVer、PolitiHop | 13/21/42（资源允许） | 跨数据/语义结论 | 核验标签映射和 split |

禁止先看 symmetric-FEVER 的 test 指标，再回头挑选最佳 beta、pi0、训练周期或主预测头。对所有外部测试集只做一次预注册的最终主评测；之后的探索分析必须标为 post hoc。

### 14.2 主要实验清单（建议 ID）

```text
E00 legacy reproduction and leakage/fairness audit
E01 CF validation, proxy freezing, and proxy-health diagnosis
E02 M1 structured-standardization supervised feasibility
E03 M1/M2/M3/M4 dual-loss factorial ablation
E04 B0/B1/B2/Reweighting/GroupDRO/M0/M4 matched baselines
E05 M5 vs M4 and budget selector comparisons
E06 no-Z/no-W/no-bridge/no-std/pi0/random-proxy negative controls
E07 label-preserving and label-changing subtypes (IPR/SCR)
E08 symmetric-FEVER / VitaminC / HoVer / PolitiHop OOD or in-domain split tests
E09 subgroup, confidence, paired error, and manual case study
E10 training-time, inference-time, memory and candidate-generation cost
E11 multi-seed statistical confidence and paper tables
```

### 14.3 推荐的论文表图

| 图表 ID | 内容 | 必须体现的研究判断 |
|---|---|---|
| Table 1 | 各数据集和 pair pool 的真实规模、标签分布、变换类型 | 数据与评估设置可信 |
| Table 2 | FEVER ID + symmetric-FEVER OOD 主结果 | 优于强 baseline，而不是仅超越 ERM |
| Table 3 | IPR 与 SCR 分项和标签转移矩阵 | 保持稳定与保留敏感性之间的平衡 |
| Table 4 | Z/W、pi0、bridge、h supervision 消融 | proxy 对新增收益的独立贡献 |
| Table 5 | Uniform/Loss/Shortcut/Semantic/Dual 配额方法 | 选择性预算不是普通 hard mining |
| Figure 1 | Motivating example（表面变化 vs 事实变化） | 两类失败模式直观区分 |
| Figure 2 | V2 Framework（编码器、Z/W、pi0、双约束、预算） | 算法流程清晰且不误导为严格因果识别 |
| Figure 3 | 相同 CF pair 预算下 OOD/IPR/SCR 曲线 | 样本效率 |
| Figure 4 | 群体表现 + 各组选择比例/损失变化 | 机制解释 |
| Appendix | 反事实核验、统计、辅助 head、efficiency、案例 | 支撑可靠性和可复现性 |

### 14.4 一键运行脚本的“已完成”定义

最终新增 `run_scripts/run_v2_all.sh`（建议）应：

1. 检查数据与代理 artifact 的哈希；
2. 运行 unit tests + smoke（允许 `--skip-tests` 但记录）；
3. 使用已冻结配置启动预定 run grid，确保断点重启不覆盖完成的日志；
4. 对每个 seed 输出 ID/OOD/pair 的逐样本预测；
5. 在同一 report 工具中计算每个方法的相同指标；
6. 自动生成 `paper_results.csv`、`paper_results.md`、`ablation_results.csv`、`budget_curve.csv`、`significance.json`、`reproducibility_manifest.json`；
7. 若文件缺失、split 被误用、预算不一致、数据哈希变动或出现 NaN，**中止并报错**，不得偷偷跳过失败 run 后照常计算平均值。

---

## 15. 威胁、失败路径与论文写作边界

### 15.1 代理并不自动具有因果效度

`Z=f(C)`、`W=f(E)` 可能只是输入内容的确定性编码。即使 `q(W|M,Z)` 精度很高、bridge JS 很低，也不能证明潜在混杂被识别。论文用预测式标准化的经验假设阐述与负对照，不以“满足 proximal 假设”作为默认前提。如未来可获得独立的干预前来源/标注流程代理，严格因果问题仍需在独立的 `proxcabi/causal/` 实验中研究。

### 15.2 真实语义改变不能用表面编辑代替

自动把年份改成另一个数不总意味着 REFUTES；在证据不完整时，可能是 NEI。同样，加入否定词不一定翻转真假。必须明确 claim、evidence 组合关系，并人工/权威数据验证。审稿时报告**产生、过滤、核验成功率**，不能只保留“看起来容易”的样本并夸大泛化。

### 15.3 反事实生成模板可能构成新的 shortcut

若模型只学习到“经过模板 X 编辑的语句通常是 REFUTES”，OOD 效果可能是假的。因此保留多生成器/多编辑类别、统一 token 预算和模板识别实验；可训练一个简单变换类型分类器，分析不同模型是否依赖明显生成伪迹。

### 15.4 标准化可能损害真实关系判别

如果 W 混入实体、hop、事实 revision 等语义必需变量，跨 W 平均可能抹平有价值信息。将 W 限制在明确 nuisance 的证据格式/风格类别；使用不同 W 组合、无 W、均匀 pi0 和 h-W 响应消融验证。若 W 信息实际不重要，承认代理模块无增量，不能强行保留。

### 15.5 组选择可能只是在重加权标签分布

群体划分需 label-stratified 评价/采样；记录每组标签比例、候选池容量。与 Loss-based、GroupDRO-priority、公平随机配额进行对照。不要在组里混入评估时不可能获得的 gold label 元数据作为推理特征。

### 15.6 论文 Web relevance 要由问题和实验共同支撑

摘要/Introduction 应说明：Web 上来源、页面格式、内容生产过程、实体分布和编辑历史等会诱导伪相关，核查器需要对无关形式变化保持稳定、对事实修改正确响应。若数据中没有可信独立来源元信息，**不要虚构 cross-source shift 结果**；以 FEVER/symmetric-FEVER 的 lexical/syntactic shortcut 鲁棒性为可验证主问题，后续再增加真实来源划分。

### 15.7 如果结果不理想，如何收缩论文主张

- `M1≈CF-only`：代理标准化可能只是额外参数；压缩为双语义约束方法、把 proxy 写成辅助探索。
- `M4 > M1` 但 `M5≈M4`：把 Dual-budget 作为研究性试验，主贡献集中在双约束，不写预算效率收益。
- `IPR↑、SCR↓`：说明过度不变，优先调约束方向/强度或修复变换质量。
- `SCR↑、ID/OOD↓`：可能是模型过拟合模板化的 semantic CF；增强样本多样性和审计。
- 所有 V2 都不敌 CF-only/GroupDRO：不能用过时单次 checkpoint 支撑 WWW 方法优越性；需要重新审视研究假设或选择更窄的问题定位。

---

## 16. 给后续 Python 实现助手的任务契约

当使用此文件指导代码生成时，按以下原则提交代码：

1. **一次只实现一个 Milestone**，先给出涉及文件列表和向后兼容风险，再写代码；
2. 每个新模块应有类型标注、docstring、明确输入/输出 shape、可调用的最小 API；
3. 每完成一个模块先跑对应 `tests/v2/`，通过后再进入下一模块；
4. 原 `proxcabi/model.py/train.py/proxies.py` 的 legacy 入口保持原样，除非修正一个独立可测且经记录的 bug；
5. 避免耦合 `proxcabi/causal/`，不得把预测式 `L_bridge` 混成 Plan A outcome bridge；
6. 任何模型训练必须支持明确固定 seed、完整 checkpoint 恢复、可追溯数据哈希、断点重跑和 JSONL 预测导出；
7. 不用 OOD test 选模型/阈值/CF 策略；如果开发过程已经访问 OOD，需要在记录中注明，重新取得可信的未使用 holdout 后才可进行独立确认；
8. 所有实验结果必须由程序计算，表格留 `TBD` 直到真实训练完成；
9. 每个 Milestone 完成后，提供**代码 diff、运行命令、test 输出、产生的文件路径和未实现的 TODO**；
10. 若文档里某个设计值（例如 W 类数、CF 数量、optimizer steps）与真实数据不匹配，应先做审计，调整配置并说明原因，不能静默作出影响结论的替换。

### 第一批最小可执行开发任务（建议直接从这里开始）

```text
Task A: 实现 data splitting + pair schema 校验，输出 dev_select/dev_calib/dev_report 和数据审计。
Task B: 新增 StructuredProxyBuilder，仅在 base_train 上拟合，保存 pi0 和 Z/W 字典。
Task C: 新增 V2 model，保留四个 head，并实现 h 全 W 枚举与 standardized logits。
Task D: 新增独立 loss 函数与 masked pair tests，跑 tiny-overfit。
Task E: 新增 V2 trainer，先只跑 M1 对照 CF-only，固定数据/训练步数/主头。
Task F: 在 M1 成立后加入 M2/M3/M4；保存 IPR/SCR 与完整消融。
Task G: 在 M4 成立后实现 Uniform/Loss/Dual 的固定预算选样。
Task H: 完成强基线、多 seed、OOD、统计和报告。
```

---

## 17. 资料与代码核对入口

**本方案的仓库适配基于以下当前公开页面**（提交实际代码之前请核对 SHA 与新提交变化）：

- 项目入口：<https://github.com/2016bits/ProxCABI-FV>
- 原模型：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/model.py>
- 原训练：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/train.py>
- 旧代理：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/proxies.py>
- 数据模型：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/data.py>
- Torch Dataset：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/torch_data.py>
- Counterfactual 构造：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/counterfactuals.py>
- CLI：<https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/cli.py>
- 当前 WWW 实验清单：<https://github.com/2016bits/ProxCABI-FV/blob/main/docs/WWW_EXPERIMENT_TODO.md>
- 原因果方案：<https://github.com/2016bits/ProxCABI-FV/blob/main/planA.md>
- 参考的 IJCAI 2025 论文：<https://www.ijcai.org/proceedings/2025/0055.pdf>

> **实现和研究顺序的最终原则**：先做到数据正确、训练可复现、代理有真实增量，再验证双重约束，最后检验选择性预算。只有在强基线、公平预算、多种子和独立 OOD 上得到足够证据，才将对应模块提升为论文贡献。
