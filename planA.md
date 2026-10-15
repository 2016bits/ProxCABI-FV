# ProxCABI-FV — Plan A 严格近端因果识别与去偏训练：Python 实现规格

> **文档性质**：工程设计文档 / 代码生成依据（不是已经完成的实验或已经被证明满足假设的算法）。  
> **版本**：Plan A v1.0，2026-10-15  
> **目标仓库**：[2016bits/ProxCABI-FV](https://github.com/2016bits/ProxCABI-FV)  
> **建议开发分支**：`feature/plan-a-causal-effect`  
> **第一优先级**：半合成识别 → 固定事实核查模型的因果效应估计 → Effect-guided fine-tuning。  
> **重要边界**：严格识别需要额外的因果设计与代理变量条件，不能通过增加一个神经网络层自动实现。

---

## 0. 给代码生成模型的执行指令（务必先读）

1. **先检查真实仓库，再改代码**：核对 `proxcabi/model.py`、`train.py`、`data.py`、`torch_data.py`、`proxies.py`、`cli.py`、`__main__.py`、`configs/` 的实际类、函数签名及配置格式。不要臆造现有函数。必要时用适配器对接，不要大规模重写已有训练管线。
2. **不破坏 Legacy**：保留原有 `python -m proxcabi train/evaluate`、现有 `ProxCABIModel`、`ProxyBuilder`、反事实生成器与 checkpoint 加载流程。Plan A 新增命名空间 `proxcabi/causal/`，单独提供命令。
3. **Phase 1 必须真正可运行**：无 GPU、无 HuggingFace 下载、无 FEVER 数据时，也必须能通过 NumPy 运行合成数据生成、离散 proximal bridge、真值比较和自动测试。
4. **识别假设与经验检查不可混淆**：任何 `proxy_diagnostics`、条件互信息、残差检验都不能标记为对不可观测排除限制或 completeness 的“证明”。
5. **强制阻断错误因果用法**：严格模式下禁止以 `Z=f(claim)`、`W=g(evidence)` 之类完全从处理后输入派生的 Hash ID 直接充当合法因果代理。真实数据缺少独立代理测量时，返回可解释的 `AssumptionValidationError`，只允许进入 `exploratory` 非识别模式。
6. **不声称降低因果效应即提高准确率**：因果敏感性、ID/OOD Accuracy、Macro-F1、标签保持一致性和标签改变敏感性都必须单独评估。
7. **每个阶段给出真实可执行命令**，新增代码需附类型注解、随机种子、日志、配置、测试与异常处理；禁止生成只有注释或 `pass` 的伪完成模块。
8. **拒绝数据泄漏**：测试集和 symmetric-FEVER 仅用于最终评估；所有代理规则、参考分布、阈值、超参数、模型选择都只使用训练集与允许的验证集。
9. **记录假设状态**：每一项实验必须输出 `assumptions.json`，标注 `designed_valid` / `argued` / `untested` / `known_violated`，并保留来源说明。
10. **以小步验收为准**：先完成 Phase 0/1，再进入真实 verifier；不要一次实现所有模块并直接宣称获得严格因果去偏。

## 1. 目标、非目标与总体架构

### 1.1 研究目标

给定固定事实核查器 `f_theta`，研究**某种事先定义、标签保持的 shortcut 表达策略** `A` 对模型预测损失或预测分数 `R_theta` 的因果影响。若观察性处理分配受潜在因素 `U` 混杂，希望在双代理识别假设成立时，利用 treatment-inducing proxy `Z` 和 outcome-inducing proxy `W` 估计该影响，并将高敏感群体作为后续反事实去偏训练的依据。

**核心变化**：不再试图识别 `P(Y | do(C=c, E=e))`；`Y` 仍是真实事实核查标签，`R_theta` 才是因果结果。

### 1.2 必须支持的三种运行模式

| mode | 数据 | 用途 | 允许的结论 |
|---|---|---|---|
| `synthetic_strict` | 已知 `U`、按合法 SCM 采样的 `A,Z,W,R` | 验证因果识别与估计器正确性 | 在设定 SCM 和满足相关条件时的因果效应恢复 |
| `real_strict` | 固定 verifier 的预测结果 + **有来源证明的独立代理测量** | 尝试真实观察性因果估计 | 必须给出具体假设依据和敏感性分析，结论仍带条件 |
| `real_exploratory` | FEVER/HoVer 等真实文本，代理有效性未成立 | 测量预测差异、探索性分组分析 | **不得**称为识别的因果效应 |

### 1.3 明确不做的事

- 不使用 `Z=f(C), W=f(E)` 直接套标准 proximal g-formula。
- 不把现有 `model.py` 中的 Softmax `h_head` 当成实值因果 bridge。
- 不把旧版 `prove-proxies` 的分类探针指标当作严格因果假设检验。
- 不在半合成验证阶段训练大型语言模型。
- 不把随机生成的标签保持配对实验冒充存在未观测混杂的观察性识别问题；随机配对是 benchmark/oracle，不是 proximal 必要性的证据。
- 不把 `L` 设置成每条样本唯一 ID 后又期待具有处理 overlap。

### 1.4 整体流程

```text
[Legacy verifier: original ProxCABI / plain RoBERTa]
                       |
                       v
           Freeze checkpoint theta_0
                       |
                       v
[Observation builder: L, A, Z, W, R_theta0]
                       |
             Assumption/data gate
              /                 \
   valid synthetic/real     missing/invalid
          |                      |
          v                      v
  Proximal effect estimator   exploratory-only report
          |
          v
 [ATE/CATE + uncertainty + held-out moments + diagnostics]
          |
          v
 [Effect-guided sample/group weights; NOT automatic differentiable ATE]
          |
          v
 [Verifier fine-tune with supervised + verified CF losses]
          |
          v
 [Recompute R_theta1, refit/re-evaluate causal estimator]
          |
          v
 [Accuracy/F1 + oracle effect error + invariance/sensitivity]
```

---

## 2. 数学问题与严格识别条件（写进论文 Method 前应核对）

### 2.1 变量定义

- `L`：干预前、可跨 treatment 保持定义的任务条件，如事先确定的命题/事实关系类别、主题层级、样本难度分层等。不能包含处理后的文本表征或标签泄漏。
- `A in {0,1}`：**明确可执行**的标签保持表面表达策略，例如两种句法实现策略。只研究经验证不会改变正确事实标签的操作；一般性的“是否包含 not”不自动合格。
- `U`：不可观测的处理—结果共同原因，例如潜在难度、生成环境；模拟数据中可记录用于 oracle，估计阶段不能访问。
- `Z`：与 `U` 相关并辅助区分处理分配的干预前测量，满足 outcome-side 排除限制。
- `W`：与 `U` 相关的另一个干预前测量，满足 treatment-side 排除限制，并且不受 `A` 影响。
- `Y_star`：经确认的事实核查正确标签；标签保持的 `A` 改变后仍不变。
- `R_theta`：对于**固定** verifier `theta`，观察到处理 `A=a` 时的结果，如 `R=-log p_theta(Y_star|X^a)`。建议先用有界/裁剪的 loss 或正确标签概率作为稳定性辅助结果；记录截断阈值。

**潜在结果定义**：`R_theta(a)` 是将表达策略设置为 `a`、同时固定事实关系和模型参数 `theta` 时的模型响应。`a` 必须对应明确定义的文本生成机制；若同一 `a` 存在多种处理版本，应声明其随机化版本分布或细化 treatment 定义。

### 2.2 因果目标

```math
\mu_\theta(a)=\mathbb E[R_\theta(a)],\qquad
\tau_\theta=\mu_\theta(1)-\mu_\theta(0).
```

条件目标：

```math
\mu_\theta(a,l)=\mathbb E[R_\theta(a)\mid L=l],\quad
\tau_\theta(l)=\mu_\theta(1,l)-\mu_\theta(0,l).
```

如果用损失作为结果，`tau>0` 表示 `A=1` 相对于 `A=0` 增加平均模型损失；但 `tau≈0` 不能排除个体或群体中正负效应抵消。

### 2.3 核心识别假设

1. **Consistency / well-defined treatment**：观测 `A=a` 时 `R=R(a)`；潜在结果对操作有确定意义。
2. **Latent exchangeability**：`R(a) ⟂ A | U,L`。
3. **Treatment positivity**：对于目标群体中具有正概率的 `(U,L)`，`0<P(A=a|U,L)<1`。
4. **Proxy exclusion / negative-control restrictions**：一个常用且足够强的联合条件是 `(R(a), W) ⟂ (A,Z) | U,L`，并有一致性；其观测层后果包括 `R ⟂ Z | A,U,L`、`W ⟂ (A,Z) | U,L`。具体论文应选定并精确陈述使用的 proximal 定理版本。
5. **Outcome bridge exists**：存在实值 `h(a,w,l)`，使 `E[R | A,Z,L] = E[h(A,W,L) | A,Z,L]`。
6. **Completeness / relevance**：潜在混杂在 `Z` 条件分布中的充分可辨识性，使可观测 bridge 的有效解对应正确调整；常见形式为 `E[v(U)|A=a,Z,L]=0 => v(U)=0`；有时还需对 `W` 的条件期望算子增加存在性、单射性及正则性条件。
7. **Integrability / support / regularity**：可积性、所需条件分布有支持、估计器的有限样本可稳定估计等。

**注意**：只有 `Z` 与 `R` 边际相关、`W` 能预测 `U`、或 bridge residual 较小，均不能证明以上全部假设。

### 2.4 Outcome confounding bridge 与 proximal g-formula

```math
\mathbb E[R\mid A=a,Z=z,L=l]
=\mathbb E[h(a,W,l)\mid A=a,Z=z,L=l].
```

在所述假设以及适当的 bridge 识别条件成立时：

```math
\mu_\theta(a)=\mathbb E_{W,L}[h_\theta(a,W,L)],
```

```math
\tau_\theta=\mathbb E_{W,L}[h_\theta(1,W,L)-h_\theta(0,W,L)].
```

按 `L=l` 的分层效应：

```math
\tau_\theta(l)=\mathbb E_{W\mid L=l}[h_\theta(1,W,l)-h_\theta(0,W,l)].
```

此处**绝不能**把对 `W` 的平均替换成 `P(W|A=a,L)`：标准化时使用目标总体的 `(W,L)` 分布，而不是处理组条件分布。

### 2.5 方法必须面对的识别限制

- `Z`、`W` 若直接来自施加处理后的 Claim/Evidence，通常不满足所需的独立测量设计；不能借“网络学到的隐变量”自动补救。
- 以 `A` 为文本风格操作时，不能把风格操作导致的文本特征再当作 `L`，否则是在控制中介/后处理变量。
- 固定的 verifier 响应 `R=f_theta(X^A)` 本身依赖经过处理的文本；这没有问题，因它是 **outcome**。但用从 `X^A` 派生的 `W` 会破坏代理条件。
- 真实数据如能随机施加 `A` 并获取两种结果，则标准随机化/配对比较已可识别 ATE；proximal 主要用于**处理分配混杂、无法对每条样本完整干预**的观察场景。
- 有限个 `Z/W` 桶不保证 completeness；频繁碰撞可能使桥接病态。

---

## 3. 现有代码适配地图（截至 2026-10-15 检查的公开仓库）

已确认的仓库模块与作用：

| 现有文件 | 已有能力 | Plan A 处理 |
|---|---|---|
| `proxcabi/model.py` | `ProxCABIModel`、`encoder`、`fact_head`、`g_head`、`h_head`、`proxy_head`；旧 JS bridge 和 Softmax 推理 | **不改旧语义**；新增独立实值 `causal/outcome_bridge.py` |
| `proxcabi/train.py` | `train(...)`、数据加载、checkpoint、反事实/contrastive 支持 | Legacy 保留；新建 `causal/verifier_adapter.py`、`causal/effect_guided.py` |
| `proxcabi/proxies.py` | `ProxyBuilder`，默认 256 个 Hash 桶，按文本特征生成 `Z/W` | **仅用于旧实验或探索性报告**，严禁严格模式下自动复用 |
| `proxcabi/data.py` | `FactSample`、`load_samples`、FEVER/HoVer/PolitiHop/VitaminC 数据入口 | 通过 adapter 读取，新增因果数据 schema |
| `proxcabi/torch_data.py` | Dataset/Collator/contrastive batch sampler | 尽量复用，避免污染处理前因果属性 |
| `proxcabi/cli.py` + `__main__.py` | `python -m proxcabi list/audit/diagnose/prove-proxies/train/evaluate` | **新增** `python -m proxcabi causal ...` 或 `python -m proxcabi.causal ...`（任选一个，全项目统一） |
| `run_scripts/` | FEVER、hard-NLI CF、原始 ablation 脚本 | 原封不动，新增 `run_scripts/plan_a_*.sh` |
| `configs/` | 原训练配置 | 新增 `configs/plan_a/*.yaml` |

参考源码：
- https://github.com/2016bits/ProxCABI-FV/blob/main/IMPLEMENTATION.md
- https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/model.py
- https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/train.py
- https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/proxies.py
- https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/cli.py

**实际落地原则**：接口签名以开发时本地仓库为准。本文新增的代码和命令是**拟实现接口规范**，不是当前 `main` 分支已经支持的调用。当前写文档的环境无法直接 `git clone`，因此未执行本地单元测试、没有声称仓库源码已在运行时验证。

---

## 4. 新目录结构与模块职责

```text
proxcabi/
  causal/
    __init__.py
    __main__.py             # python -m proxcabi.causal ...
    cli.py                  # 独立子命令与参数校验
    schemas.py              # TypedDict/dataclasses/Pydantic（任选一套）
    assumptions.py          # assumption gates, audit report
    synthetic.py            # 已知 U 的半合成 SCM
    treatment.py            # 可执行、标签保持的干预策略接口
    causal_data.py          # observational causal rows + splits
    proxy_measurement.py    # 独立的 Z/W 测量适配器
    verifier_adapter.py     # load legacy checkpoint + inference-only R_theta
    discrete_bridge.py      # 低维离散线性方程，作为正确性基准
    moment_bridge.py        # 神经实值 bridge + conditional moment loss
    critic.py               # 可选 adversarial conditional moments
    estimate.py             # ATE/CATE + CI, target-standardization
    diagnostics.py          # support/rank/moment/proxy sensitivity
    effect_guided.py        # 分阶段 reweight + verifier fine-tune
    evaluate.py             # 指标、图表源数据、比较表
    io.py                   # 原子写入 JSON/CSV/Parquet、配置与 hash
configs/
  plan_a/
    synthetic_smoke.yaml
    synthetic_stress.yaml
    verifier_observational.yaml
    effect_guided.yaml
run_scripts/
  plan_a_synthetic_smoke.sh
  plan_a_synthetic_stress.sh
  plan_a_verifier_observational.sh
  plan_a_effect_guided.sh
tests/
  test_plan_a_scm.py
  test_plan_a_assumptions.py
  test_plan_a_discrete_bridge.py
  test_plan_a_neural_bridge.py
  test_plan_a_estimator.py
  test_plan_a_split_leakage.py
  test_plan_a_effect_guided.py
```

### 4.1 各模块最小公共接口

```python
# proxcabi/causal/schemas.py — 拟定接口
from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["synthetic_strict", "real_strict", "real_exploratory"]

@dataclass(frozen=True)
class CausalObservation:
    row_id: str
    split: str
    treatment: int              # A, 0 or 1
    z: int                      # independent pre-treatment proxy
    w: int                      # independent pre-treatment proxy
    l: int                      # pre-treatment stratum, small support in MVP
    response: float             # fixed verifier loss / score, or SCM outcome
    sample_id: str = ""
    semantic_group: str = ""
    original_label: int | None = None
    claim: str = ""
    evidence: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EffectEstimate:
    ate: float
    mu0: float
    mu1: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    method: str
    passed_identification_gate: bool
```

`meta` 可保存 provenance 供审计，但估计器不能自动将其全部当作协变量。`U` 应只存在 synthetic truth 文件中（估计器数据接口不提供 `u` 字段）。

建议在两个位置存放模拟信息：

- `observations.parquet` / `observations.csv`：仅 `A,Z,W,L,R` 及索引，给估计器；
- `oracle_truth.json`：生成 SCM 参数、真实 `U`、真实 `tau` 与理论条件，仅用于 synthetic oracle 和测试。

`oracle_truth` 不得传入 `fit_bridge()`。

### 4.2 抽象依赖接口

```python
# proxcabi/causal/treatment.py
from typing import Protocol

class TreatmentStrategy(Protocol):
    name: str
    def apply(self, claim: str, evidence: str, a: int, *, seed: int) -> tuple[str, str]: ...
    def verify_label_preservation(
        self, before: tuple[str, str], after: tuple[str, str], label: int
    ) -> bool: ...

# proxcabi/causal/proxy_measurement.py
class ProxyMeasurement(Protocol):
    def measure_z(self, metadata: dict) -> int: ...
    def measure_w(self, metadata: dict) -> int: ...
```

实际真实数据优先直接读取提前准备好的 independent metadata；不要让这两个接口默默读取 `claim`、`evidence` 或预测标签作为代理输入。

---

## 5. Phase 0：环境、版本与数据安全检查

**交付物**：干净分支、依赖锁定、Legacy regression、数据泄漏检查。

1. 建立新分支 `feature/plan-a-causal-effect`，记录基线 commit SHA 至 `outputs/plan_a/baseline_manifest.json`。
2. 使用现有 `requirements.txt`；新增 `pytest`。首阶段离散估计只需 `numpy`，不要强制下载 PLM。
3. 获取仓库原有 `train/evaluate` 基线日志（有 GPU 时）；没有现成 checkpoint 则先仅做 synthetic。
4. 确认 `data/` 本地数据不进入版本控制；不要求上传大型训练数据。
5. 按 `semantic_group` 或原始样本 ID 进行 train/val/test 划分，避免同一事实的不同文本版本跨集泄漏。
6. `config`、seed、git commit、数据 schema 哈希、模型 checkpoint 哈希一并写入运行目录。

检查命令（在真正下载仓库的机器执行）：

```bash
git checkout -b feature/plan-a-causal-effect
python -m pip install -r requirements.txt
python -m pip install pytest
python -m proxcabi list --data-dir data
```

---

## 6. Phase 1：合成数据生成（必须首先完成）

### 6.1 最小合法 SCM（binary U,A,Z,W；可选小规模 L）

默认先用 `L=0` 常数层，降低识别问题复杂度：

```math
U\sim\operatorname{Bernoulli}(0.5),
```

```math
Z=U\oplus N_Z,\quad W=U\oplus N_W,
```

```math
N_Z\sim\operatorname{Bernoulli}(0.15),\quad
N_W\sim\operatorname{Bernoulli}(0.15),
```

```math
P(A=1\mid U=0)=0.2,\qquad P(A=1\mid U=1)=0.8,
```

```math
R=\beta A+\delta U+\epsilon,\qquad
\epsilon\sim\mathcal N(0,\sigma^2).
```

所有噪声在给定 `U` 后独立；固定默认参数 `beta=0.3, delta=0.8, sigma=0.1`。目标 `tau_true=beta=0.3`。此生成过程设计满足所用简化代理排除条件与 positivity；当代理噪声不同于 `0.5` 且条件概率矩阵满秩时，离散识别通常可行。

**预期 naive 差异**：在此对称设置下 `P(U=1|A=1)=0.8`、`P(U=1|A=0)=0.2`，故

```math
\mathbb E[R\mid A=1]-\mathbb E[R\mid A=0]
=\beta+0.6\delta=0.78,
```

明显偏离真值 `0.30`。这是一条可以自动化检验的 sanity check。

### 6.2 可直接转换为 `synthetic.py` 的代码骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BinarySCMConfig:
    n: int = 100_000
    seed: int = 13
    p_u: float = 0.5
    z_accuracy: float = 0.85
    w_accuracy: float = 0.85
    p_a_u0: float = 0.20
    p_a_u1: float = 0.80
    beta: float = 0.30
    delta: float = 0.80
    noise_std: float = 0.10
    direct_z_effect: float = 0.0   # stress test ONLY: violates exclusion
    direct_w_to_a: float = 0.0     # stress test ONLY: violates exclusion


def generate_binary_scm(cfg: BinarySCMConfig) -> tuple[dict, dict]:
    rng = np.random.default_rng(cfg.seed)
    u = rng.binomial(1, cfg.p_u, cfg.n)
    z = np.bitwise_xor(u, rng.binomial(1, 1.0 - cfg.z_accuracy, cfg.n))
    w = np.bitwise_xor(u, rng.binomial(1, 1.0 - cfg.w_accuracy, cfg.n))
    p_a = np.where(u == 1, cfg.p_a_u1, cfg.p_a_u0).astype(float)
    # Explicit violations must be opt-in and reflected in assumption metadata.
    if cfg.direct_w_to_a != 0.0:
        p_a = np.clip(p_a + cfg.direct_w_to_a * (w - 0.5), 0.01, 0.99)
    a = rng.binomial(1, p_a)
    eps = rng.normal(0.0, cfg.noise_std, cfg.n)
    r = cfg.beta * a + cfg.delta * u + cfg.direct_z_effect * z + eps
    observed = {
        "a": a.astype(np.int64),
        "z": z.astype(np.int64),
        "w": w.astype(np.int64),
        "l": np.zeros(cfg.n, dtype=np.int64),
        "r": r.astype(np.float64),
    }
    oracle = {
        "u": u.astype(np.int64),
        "tau_true": float(cfg.beta),
        "settings": cfg.__dict__,
        "assumptions_designed_valid": (
            cfg.direct_z_effect == 0.0 and cfg.direct_w_to_a == 0.0
        ),
    }
    return observed, oracle
```

**注意**：对 `direct_z_effect != 0` 的情形，真实 ATE 在固定 `Z` 的干预下仍为 `beta`，但 `Z` 的排除限制不再成立；对 `direct_w_to_a != 0`，`W` 影响处理分配使标准 outcome proxy 限制失效。记录结果时必须显示 `known_violated`。

### 6.3 Synthetic Stress Grid

按下列参数网格至少生成数据：

| 实验 | 变化项 | 目的 |
|---|---|---|
| valid | `z_accuracy=w_accuracy=0.85` | 正常识别 |
| weak Z | `z_accuracy` 从 0.85 降至 0.55/0.51 | completeness/弱识别警告 |
| weak W | `w_accuracy` 降至 0.55/0.51 | 逆问题病态 |
| invalid Z | `direct_z_effect=0.2` | 模拟排除限制违反 |
| invalid W | `direct_w_to_a=0.2` | 模拟 outcome proxy 条件违反 |
| no confounding | `delta=0` | 与 naive 接近的 sanity check |
| no effect | `beta=0` | Type-I error / false shortcut alarm |
| scarce overlap | `p_a_u0=0.02,p_a_u1=0.98` | positivity 警告与方差 |

每项至少使用 5 个种子（MVP 可先 3 个），输出 `abs_bias`、`RMSE`、`CI_coverage`、`rank`、`min_singular_value` 和失败比例。强调不能保证每个随机数据集 proximal 估计都优于 naive，尤其弱代理时。

---

## 7. Phase 2：离散 Outcome Bridge（正确性基准 / 首个可运行估计器）

### 7.1 固定 L=l 和 A=a 的线性系统

若 `W` 有 `K` 个状态，`Z` 有 `J` 个状态，定义：

```math
m_{a,l}(z)=\mathbb E[R\mid A=a,Z=z,L=l]
```

```math
Q_{a,l}(z,w)=P(W=w\mid A=a,Z=z,L=l).
```

则可观测桥接条件为：

```math
\mathbf m_{a,l}=Q_{a,l}\mathbf h_{a,l}.
```

在可识别、适当的桥接存在和矩阵条件下，求 `h(a,w,l)`，然后按**目标总体** `P(W,L)` 标准化：

```math
\hat\mu(a)=\sum_l\hat P(L=l)\sum_w\hat P(W=w\mid L=l)\hat h(a,w,l).
```

### 7.2 估计器接口

```python
class DiscreteOutcomeBridge:
    def __init__(self, ridge: float = 1e-6, min_cell: int = 20,
                 min_singular_value: float = 1e-3): ...

    def fit(self, a, z, w, l, r) -> "DiscreteOutcomeBridge": ...

    def predict_h(self, a: int, w: int, l: int) -> float: ...

    def estimate_ate(self, target_w, target_l) -> EffectEstimate: ...

    def diagnose(self) -> dict: ...
```

**不要**对每个唯一 `l` 拟合：V1 仅支持有限个、每组有足够观测的预处理层；否则报 `InsufficientSupportError`。

### 7.3 最小算法（指导实现，可作为测试的参考）

1. 对每个 `(a,l)`，统计 `A=a,Z=z,L=l` 各 cell 的样本数、`R` 均值和 `W` 条件频率。
2. 若某个 `z` cell 过少，记录 `insufficient_support`，不静默填充 0；可选择 merge strata，但必须事先定义合并规则。
3. 得到 `Q in R[J,K]`、`m in R[J]`。
4. `np.linalg.svd(Q, compute_uv=False)` 检查秩与最小非零奇异值；若 `rank(Q)<K`，默认 **失败** 而不是无提示给出 Ridge 数值。
5. 若满列秩，求 `h = (Q.T @ D @ Q + lambda*I)^(-1) @ Q.T @ D @ m`，其中 `D` 为各 `z` cell 的归一化样本量对角矩阵；可用 `np.linalg.solve` 或 `lstsq`，不要显式求逆。
6. 在**独立验证集**估计 held-out conditional moments；计算 `target P(W,L)` 来自未按处理状态条件化的预先指定总体样本。
7. 返回 `mu0, mu1, ate` 和完整诊断；区分 `unregularized exact benchmark` 与 `ridge stabilized estimator`。

**重点**：有限样本 `Q` 条件差时 Ridge 会引入偏差；`ridge` 不应消除识别失败告警。

### 7.4 必需对照估计器

- `oracle_u_adjustment`：只在 Synthetic Truth 内可用，按真实 `U` 调整，用于上限参考。
- `naive_difference`：`mean(R|A=1) - mean(R|A=0)`。
- `single_proxy_adjustment_Z`：仅观察性地调整 `Z`，明确标为有偏对照。
- `single_proxy_adjustment_W`：仅调整 `W`，明确标为有偏对照。
- `discrete_proximal`：本模块。
- `random_or_shuffled_proxy`：打乱 `Z` / `W`，预期秩/估计表现下降或触发告警。

### 7.5 测试断言

```python
# tests/test_plan_a_discrete_bridge.py — 概念断言
obs, oracle = generate_binary_scm(BinarySCMConfig(n=200_000, seed=13))
est = DiscreteOutcomeBridge(min_cell=20, ridge=1e-6)
est.fit(obs["a"], obs["z"], obs["w"], obs["l"], obs["r"])
result = est.estimate_ate(target_w=obs["w"], target_l=obs["l"])
assert abs(result.ate - oracle["tau_true"]) < 0.06
assert est.diagnose()["all_required_matrices_full_column_rank"]
```

阈值 `0.06` 是针对默认大样本合成 smoke 的工程容差，需要先运行多种 seed 确定其误报率；不能将其理解为无条件的统计保证。

---

## 8. Phase 3：神经实值 Bridge + 条件矩优化

**本阶段在离散 baseline 通过后才开始。**

### 8.1 `OutcomeBridgeNet` 结构

输入 `(A,W,L)`（严格 identification 的最小模型）；输出**实值** `h`，不加 Softmax、Sigmoid，也不要求概率之和为 1。

```python
import torch
from torch import nn

class OutcomeBridgeNet(nn.Module):
    def __init__(self, n_a: int, n_w: int, n_l: int,
                 emb_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.a_emb = nn.Embedding(n_a, emb_dim)
        self.w_emb = nn.Embedding(n_w, emb_dim)
        self.l_emb = nn.Embedding(n_l, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a: torch.Tensor, w: torch.Tensor,
                l: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.a_emb(a), self.w_emb(w), self.l_emb(l)], dim=-1)
        return self.net(x).squeeze(-1)
```

**注意**：直接把 verifier 编码得到的 `M=f_theta(C,E)` 拼进该 bridge，会使 `M` 成为处理后的变量，破坏所用的 `h(A,W,L)` 识别式。V1 严格版本严禁这样做。

### 8.2 条件矩学习目标

```math
\mathbb E[R-h_\psi(A,W,L)\mid A,Z,L]=0.
```

令 `phi_j(A,Z,L)` 是测试函数，可以用有限 one-hot indicators：

```math
\mathcal L_{moment}
=\sum_j\bigg[\frac1N\sum_i
\{R_i-h_\psi(A_i,W_i,L_i)\}\phi_j(A_i,Z_i,L_i)\bigg]^2.
```

**推荐离散首版**：用 `A×Z×L` 各 cell 的归一化残差均值平方（按 cell 权重汇总），与离散线性系统一一对应。每个训练 batch 若缺少某些 cell，不能拿缺失 cell 的零值当作有效 residual；建议使用 stratified mini-batch 或 epoch-level moment accumulators，并报告有效 cell 覆盖率。

**推荐后续版**：用 test-function network `phi_omega(A,Z,L)` 做 minimax conditional moments，增加函数范数惩罚与训练稳定化，但不得把最小化某个可训练 `phi` 的输出幅度当成有效约束（会塌缩为零）。

### 8.3 训练策略

1. 输入拆分 train/val/test；只用 train 拟合 bridge。
2. 在 `float64` 的离散 benchmark 中对照精确解；神经模型可用 float32。
3. 使用 early stopping 选择 **held-out conditional moment residual**，并同时观察估计的 `tau` 稳定性；残差最小并不充分证明因果识别。
4. 记录 `L2(h)`, `pred_r_range`, `group_count`, `min_singular_value`、样本数、梯度范数与 NaN 次数。
5. 可采用交叉拟合：K 折，每折在 K-1 折拟合 bridge，对留出折标准化与估计；报告合并方式与每折效应。其主要目的是控制模型过拟合，不是修复代理假设。
6. 若使用平均/差值 estimate 的 bootstrap CI，必须在每次 bootstrap 中**重新拟合** bridge，且在适当的群组层（如独立事实单元）重采样；弱识别时 bootstrap CI 也可能失效，报告提示。

### 8.4 拟新增命令

```bash
python -m proxcabi.causal synthetic-generate --config configs/plan_a/synthetic_smoke.yaml
python -m proxcabi.causal fit-bridge --input outputs/plan_a/synthetic/observations.npz --method discrete --output-dir outputs/plan_a/discrete
python -m proxcabi.causal fit-bridge --input outputs/plan_a/synthetic/observations.npz --method neural --output-dir outputs/plan_a/neural
python -m proxcabi.causal evaluate-identification --pred-dir outputs/plan_a/neural --oracle outputs/plan_a/synthetic/oracle_truth.json
```

CLI 应提供 `--help`、合理错误信息、运行目录结构和示例配置。以上命令是**必须新增实现的目标接口**，并非现成命令。

---

## 9. Phase 4：从真实事实核查模型建立观察性响应数据

### 9.1 固定模型与数据分离

推荐分为三组：

- `D_verifier_train`：训练事实核查模型 `theta0`；可用仓库已有 `train` 入口。
- `D_causal_audit`：观察处理 `A`、独立元数据代理 `Z/W`，由冻结模型预测得 `R_theta0`；用于桥接与效果估计。
- `D_final_test`：独立 OOD 与反事实鲁棒性评估；不用于拟合 bridge、选择代理规则或调参。

`D_causal_audit` 再按 semantic group 分出 estimator train/val/test。不要使用同一批 verifier 训练样本直接估计其训练时损失所体现的 effect 而忽略数据复用偏差。

### 9.2 处理变量 A 的设计

建议第一版只实现 **一个** 预先定义的标签保持策略，满足：

1. 明确定义 `A=0` 与 `A=1` 两种文本生成机制，而不只是观测到的词频二值化；
2. 真实数据中若只有一部分自然发生相应策略，必须记录选择机制与 `A` 的 provenance；
3. 用可信的规则/人工审查验证两种表达的事实标签保持不变；NLI 教师可辅助过滤但不是绝对真值；
4. 若执行随机化生成并可获得同一事实的双版本，报告 paired experimental effect 作为 oracle/control；不可把这个实验重新包装为必须使用 proximal 才能识别；
5. 如果缺乏处理策略元数据，允许 `real_exploratory` 描述性分析，不允许宣称严格 ATE。

### 9.3 可行代理 Z/W 的设计约束

**优先来源**：不属于处理后文本的独立过程记录或辅助测量，例如预先生成、由独立流程产生的任务难度/生成环境审计信号。

数据契约要求每个代理具备：

```yaml
proxy:
  name: pre_treatment_aux_measurement
  measured_before_treatment: true
  derived_from_post_treatment_text: false
  used_to_assign_treatment: false   # 若 Z 影响 A，必须在 DAG 明示，不能一概禁止
  directly_used_by_verifier: false
  provenance: path/to/measurement_protocol.md
  exclusion_argument: path/to/assumption_notes.md
  missing_rate: 0.0
```

说明：`Z` 可以参与处理分配（符合某些 proximal DAG），但若如此必须明确设计并验证对 outcome 的排除限制；`W` 必须满足不受处理影响且排除额外的处理关联路径。上述布尔字段是审计信息，不足以单独证明条件独立。

**禁止严格模式直接采用**：

- `Z=hash(negation, sentence_length, ...)`，若其完全来自干预后 claim；
- `W=hash(evidence_style, revision, ...)`，若其完全来自处理后 evidence 或修改后的文本；
- 用真实标签、模型预测或处理后 loss 作为 proxy 的构造输入；
- 在观察性处理中用每条样本唯一的字符串 embedding 当作 `L` 并声称实现 nonparametric identification。

### 9.4 数据校验与显式失败

`AssumptionValidationError` 在 `real_strict` 下至少检查：

- treatment/proxy 不是缺失值；
- proxy metadata 有来源字段和测量时间顺序；
- treatment 具有 overlap 的可观测支持（必要但不充分）；
- 有清晰的估计总体与 `L` 支持；
- 没有已知的 proxy exclusion violation；
- train/val/test 无同一 semantic group 跨集泄漏；
- rank/conditioning 不满足预设阈值时，报告 `identification_unstable` 或失败；
- 一切条件独立诊断显示为 `diagnostic`, **不输出 `assumptions_proved: true`**。

如果真实数据代理尚不可用：输出缺失字段和如何采集的报告，并结束严格估计命令；**不可自动切换到旧 Hash proxy 并输出“causal effect”**。

### 9.5 Verifier adapter 接口

```python
class FrozenVerifierAdapter:
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = "cpu") -> "FrozenVerifierAdapter": ...

    def predict_proba(self, claims: list[str], evidences: list[str]) -> "np.ndarray": ...

    def per_sample_response(self, claims: list[str], evidences: list[str],
                            gold_labels: list[int],
                            kind: str = "nll") -> "np.ndarray": ...
```

最小版本只支持现有模型在 `selection_head=fact` 时的概率输出，随后视情况支持 `g/proximal`。为了保证结果可复现：在 `torch.inference_mode()` 和 `model.eval()` 下计算 response，并固定 tokenizer、max length 和 checkpoint 哈希。

若 `kind=nll`，使用 `-log(max(p_gold,1e-8))`；若 `kind=correct_prob`，使用 `p_gold`，则“效应越小越好”的方向解释需要同步改变。禁止混用不同 outcome 的数字。

---

## 10. Phase 5：Effect-guided Debiasing（先实现可审计的非端到端版本）

### 10.1 为什么不能直接把估计的 tau 加入现有 total_loss

估计 `tau(theta0)` 时，verifier `theta0` 已冻结。若 bridge 也固定，其输出对新的 `theta` 不含梯度；简单执行 `loss += lambda * float(tau_hat**2)` 对 verifier 参数没有去偏作用。不能声称实现了“端到端因果正则化”。

### 10.2 推荐的 v1：分阶段 effect-guided sample reweighting

1. 训练/verifier 初始模型 `theta0`。
2. 在允许的观察性 audit 数据上估计各**预先定义、有足够支持**的群组 `l` 的 `tau_theta0(l)`。
3. 对 `|tau(l)|` 大的群组分配更高的**训练采样预算**或经标签核验的 CF 样本权重；例如：

```math
s(l)=\min\bigg(s_{max},\ 1+\kappa\cdot
\frac{|\hat\tau(l)|}{\operatorname{median}_{l'}|\hat\tau(l')|+\varepsilon}\bigg).
```

4. 对这些群组进行标签保持的反事实 consistency 和正常监督训练。
5. 定期（例如每个完整 training stage）重新冻结 verifier 并重新估计 effect；不能沿用旧 `tau` 声称最新模型仍有同样的因果效应。
6. 在相同数量的训练步骤、相同 CF 预算下和 uniform/random/high-observed-correlation reweighting 比较。

**论文用语**：这属于 `proximal effect-guided debiasing`，不是直接最小化可微的 ATE；其成败由后续重估的 effect 与 OOD 指标决定。

### 10.3 训练目标

```math
\mathcal L_{FV}=\mathbb E[-\log p_\theta(Y^\star\mid C,E)],
```

```math
\mathcal L_{inv}=\mathbb E_{(X,X^+)\in\mathcal P_{inv}}
D_{JS}(p_\theta(\cdot\mid X),p_\theta(\cdot\mid X^+)),
```

```math
\mathcal L_{sem}=\mathbb E_{(X,X^-)\in\mathcal P_{change}}
[-\log p_\theta(Y^-\mid X^-)],
```

```math
\mathcal L_{train}
=\mathbb E[s(L)\,\ell_{FV}]
+\lambda_{inv}\mathbb E[s(L)\,\ell_{inv}]
+\lambda_{sem}\ell_{sem}.
```

如果同时保留 legacy contrastive loss，单独加权并做 `without_contrastive` ablation。`inv` 配对必须标签相同；`change` 配对必须人工或高质量证据核验标签改变。

### 10.4 v2（选做）：可微因果效应惩罚

只有在 v1 证明 proximal 效应估计确实有信息价值之后才尝试。候选方法包括 alternating optimization、implicit differentiation、带截断梯度的固定 bridge surrogate 等。必须明确说明每种方法计算的梯度是否真的对应 `tau_theta`，不可把参数冻结后的常量惩罚称为优化因果效应。

---

## 11. 评估方案与论文实验

### 11.1 核心统计指标

**识别指标（Synthetic）**：

- `tau_true`, `tau_hat`, `abs_error`, `squared_error`, across-seed `bias/RMSE`；
- 95% CI coverage、CI width、convergence/failure rate；
- held-out conditional moment residual、`rank(Q)`, `min_singular_value`, condition number；
- proxy invalidity sensitivity curves；
- naive / single proxy / oracle U / discrete proximal / neural proximal comparisons。

**事实核查指标（Real）**：

- ID/OOD Accuracy、Macro-F1、各类 F1；
- symmetric-FEVER 等鲁棒评测；
- `CCR`: 标签保持变换前后预测一致率（同时报告两端正确率）；
- `SSR`: 标签改变配对上双端正确率、正确标签切换率（不只输出是否不同）；
- 重新估计的 `ATE/CATE`（仅在 real_strict 识别假设有合理依据时使用该名称）；
- calibration（可选）、per-treatment subgroup accuracy、`min_group_acc`。

### 11.2 必需消融

| 对照 | 控制内容 | 要验证的问题 |
|---|---|---|
| Ordinary verifier | backbone + supervised FV | 基线 |
| Random CF | 与 effect-guided 相同 CF 数据预算 | 反事实数据量作用 |
| Uniform group weights | 不依赖 causal estimate | 是否只是加权训练作用 |
| Observed association weights | 用 `E[R|A=1]-E[R|A=0]` 做组权重 | proximal 是否超越普通相关性 |
| Z-only / W-only | 去掉另一代理 | 双代理增量 |
| Shuffled Z/W | 打乱代理 | 代理是否携带有效识别信息 |
| Oracle U (synthetic only) | 真实潜在变量可见 | 可达到的正确性上限 |
| Proximal-guided | 完整 Phase 5 | 主要方法 |
| + semantic CF | 验证标签敏感性 | 防止过度去偏 |

**报告原则**：数据、backbone、tokenizer、步数、CF 样本数和选择标准必须公平；至少多 seed 统计均值/标准差。不要仅依靠 FEVER Dev 数字解释因果创新。

### 11.3 主要研究问题

- **RQ1**：在已知潜在混杂的 SCM 中，是否能够恢复真实 shortcut treatment effect？
- **RQ2**：当 proxy 信息弱化或违反排除限制时，估计如何退化？
- **RQ3**：在真实 fact verification 模型上，能否构造有意义、可复现、非直接随机化的 shortcut 暴露观测？
- **RQ4**：proximal effect-guided training 相对于 uniform/naive weighting 是否提升 OOD 性能并降低无关处理敏感性？
- **RQ5**：这种改进是否保留实体、否定、关系、时间等确实影响标签的语义敏感性？

---

## 12. 命令行设计与 YAML 配置

### 12.1 统一 CLI

建议独立提供 `python -m proxcabi.causal`，避免改变已有 `proxcabi.cli`：

```text
python -m proxcabi.causal synthetic-generate --config ...
python -m proxcabi.causal synthetic-stress   --config ...
python -m proxcabi.causal fit-bridge         --config ...
python -m proxcabi.causal diagnose           --config ...
python -m proxcabi.causal estimate           --config ...
python -m proxcabi.causal collect-responses  --config ...
python -m proxcabi.causal effect-guided      --config ...
python -m proxcabi.causal evaluate           --config ...
```

所有命令都必须支持 `--help`、`--seed`（直接或从配置）、`--output-dir`、错误退出码；每次运行保存 `config_resolved.yaml`、`manifest.json`、`assumptions.json`。

### 12.2 `configs/plan_a/synthetic_smoke.yaml`

```yaml
mode: synthetic_strict
seed: 13
output_dir: outputs/plan_a/synthetic_smoke
scm:
  n: 100000
  p_u: 0.5
  z_accuracy: 0.85
  w_accuracy: 0.85
  p_a_u0: 0.2
  p_a_u1: 0.8
  beta: 0.3
  delta: 0.8
  noise_std: 0.1
  direct_z_effect: 0.0
  direct_w_to_a: 0.0
bridge:
  method: discrete
  min_cell: 20
  ridge: 0.000001
  min_singular_value: 0.001
  crossfit_folds: 1
report:
  bootstrap_reps: 100
  confidence_level: 0.95
```

### 12.3 `configs/plan_a/verifier_observational.yaml`（必须带独立代理）

```yaml
mode: real_strict
seed: 13
output_dir: outputs/plan_a/real_strict
verifier:
  checkpoint_dir: outputs/fever_roberta
  head: fact
  outcome: nll
  max_length: 256
causal_data:
  observations_path: data/causal_audit/fever_observational.csv
  proxy_protocol_path: data/causal_audit/proxy_protocol.md
  treatment_protocol_path: data/causal_audit/treatment_protocol.md
  split_group_key: semantic_group
  treatment_col: treatment
  z_col: z
  w_col: w
  l_col: l
bridge:
  method: discrete
  min_cell: 25
  ridge: 0.0001
  min_singular_value: 0.001
  crossfit_folds: 3
assumptions:
  require_independent_pre_treatment_proxy: true
  require_exclusion_argument: true
  abort_on_known_violation: true
```

`observations_path` 和协议文件是**用户需要采集或生成的额外研究资料**，当前仓库不保证存在。不要通过伪造元数据让配置成功。

### 12.4 预期输出

```text
outputs/plan_a/run_xxx/
  config_resolved.yaml
  manifest.json
  assumptions.json
  metrics.json
  effect_estimates.csv
  subgroup_effects.csv
  bridge_diagnostics.json
  moment_residuals.csv
  estimator.pt              # neural 模式才有
  summary.md
```

指标 JSON 必须注明 `estimate_kind: "identified_under_assumptions" | "exploratory" | "synthetic_oracle"`、`proxy_source`、`outcome_definition` 与 `treatment_definition`。

---

## 13. 测试驱动实现（不可省略）

### 13.1 合成数据有效性

- 同一 seed 生成结果完全可重复；不同 seed 生成数据不完全相同。
- 实验生成变量范围合法，`0<=A,Z,W,U<=1`。
- `P(A=1|U=1)`、`P(A=1|U=0)`、proxy accuracy 接近配置值（有统计容差）。
- 默认 SCM 的 oracle ATE 恰为 `beta`。
- 默认 SCM 的 naive ATE 接近 `0.78` 而不是 `0.30`。

### 13.2 Bridge 正确性

- 大样本合法 SCM 的 discrete estimate 与 `beta` 接近。
- 随机化/交换代理后触发 rank 警告或明显损坏估计，不能悄然报告正常识别。
- `z_accuracy=0.5` 或 `w_accuracy=0.5` 时不能通过严格矩阵识别的可靠性门禁。
- 存在明显的 `direct_z_effect`、`direct_w_to_a` 时，已知违反假设的标记正确。
- `h` 实值可为负，不使用概率 Softmax。
- 所有标准化期望都使用 target `P(W,L)` 而非 `P(W|A)`。

### 13.3 数据安全/接口

- `real_strict` 输入只有旧文本 Hash proxy -> 抛出 `AssumptionValidationError`。
- real data 没有 provenance/测量时间 -> 明确报错。
- `semantic_group` 不能跨 train/eval/test；若出现应失败。
- 仅对训练集拟合任何属性编码规则。
- 函数 `fit_bridge(obs)` 不能访问 `oracle['u']`；代码层面隔离 synthetic truth。
- `FixedVerifier` 在 causal response 提取中固定权重、关闭 dropout。
- 冻结 `tau_hat` 直接加到 loss 不会产生梯度，效应引导模块应有测试保障确实使用 sample weights。

### 13.4 模块集成和回归

```bash
python -m pytest tests/test_plan_a_scm.py -q
python -m pytest tests/test_plan_a_discrete_bridge.py -q
python -m pytest tests/test_plan_a_assumptions.py -q
python -m pytest tests/ -q
python -m proxcabi.causal synthetic-generate --config configs/plan_a/synthetic_smoke.yaml
python -m proxcabi.causal synthetic-stress --config configs/plan_a/synthetic_stress.yaml
python -m proxcabi --help
```

验收时不要只看退出码，还要比对 `metrics.json`、`assumptions.json` 与输出表内容。

---

## 14. 实施顺序、验收门禁与停机条件

### Phase 0 — 兼容性（P0）

- [ ] 新建独立 namespace，不改变 Legacy CLI 与 checkpoint。
- [ ] 配置、日志、目录、测试框架初始化。
- [ ] 真实仓库结构及 commit 完成核对。

**Gate 0**：原命令可正常解析，新测试框架运行。

### Phase 1 — 正确的 synthetic SCM（P0）

- [ ] `synthetic.py` 实现默认 SCM 与 violation flags。
- [ ] 分离 observed 和 oracle truth。
- [ ] 统计验证 `tau_true=0.30`，naive 差值约 `0.78`。

**Gate 1**：生成器一致、可复现、无 oracle 泄漏。

### Phase 2 — 离散 bridge（P0）

- [ ] `DiscreteOutcomeBridge` 满列秩检查、regularized solve、target standardization。
- [ ] naive、oracle-U、单代理与 proximal 对照。
- [ ] held-out moments、样本支持诊断、bootstrap CI。

**Gate 2**：合法大样本情况下效应恢复达到预设容差，弱代理触发告警。

### Phase 3 — Neural bridge（P1）

- [ ] `OutcomeBridgeNet` 实值输出。
- [ ] 离散 one-hot conditional moments + holdout early stopping。
- [ ] 与离散基准在同一 SCM 上比较，不允许仅报告 training moment loss。
- [ ] cross-fit 和数值稳健性检查。

**Gate 3**：多个 seed 的估计稳定，且不比离散估计显著失控；否则保留离散版并分析。

### Phase 4 — 真实 verifier response（P1）

- [ ] checkpoint adapter、固定模型 inference、`R_theta` 输出。
- [ ] 标签保持 treatment 协议和 proxy protocol 文档。
- [ ] 分离训练和 audit，split 泄漏检测。
- [ ] `real_strict` 假设门禁；无有效 proxy 可正确失败。

**Gate 4**：只有真实可审计代理通过门禁时才允许执行严格模式；否则 `exploratory` 报告且不得声称识别。

### Phase 5 — Effect-guided Debiasing（P2）

- [ ] 预先定义分层 `L`、group effect、sample weighting。
- [ ] verified invariant/semantic-change counterfactuals。
- [ ] 对比 uniform/random/observational weighting。
- [ ] 更新 verifier 后重新估计 effect 并评估 OOD。

**Gate 5**：给出增量贡献或明确报告无效，不通过重新命名指标掩盖失败。

### 停机/退级规则（强制）

- 无明确 intervention `A` -> 停止 causal identification，先完善处理定义。
- 代理仅来自处理后文本且缺少有效性论证 -> 切换 `real_exploratory`。
- rank deficient、强病态、极端 positivity 违反 -> 返回 `weak_or_failed_identification`。
- 同一语义事实跨 split 泄漏 -> 停止训练/评估。
- 只有随机化成对数据 -> 优先报告 randomized/paired effect，不强行宣称 proximal 必需。
- Synthetic recovery 长期失败 -> 不进入真实 causal claims 与 end-to-end debiasing。

---

## 15. 生成代码的最小任务拆分（建议按此逐条发给编码助手）

### Task 1 — 实现 Phase 1 与测试

> 读取 `planA.md` 第 0、2、4、6、13、14 节。不要改现有训练模型。创建 `proxcabi/causal/schemas.py`、`synthetic.py`、`cli.py`、`__main__.py`，实现 `synthetic-generate` 与 `tests/test_plan_a_scm.py`。生成观测数据和 oracle truth 分离文件，确保默认 SCM 真值为 0.3，naive 差值约 0.78，所有测试通过。

### Task 2 — 实现 Discrete Proximal Bridge

> 读取 `planA.md` 第 2、7、11、13 节。实现 `discrete_bridge.py`、`estimate.py`、`diagnostics.py`，支持每个 `(A,L)` 的离散线性 bridge、rank/condition 检查、target `P(W,L)` 标准化、bootstrap、held-out moments。添加与 oracle/naive 对比的单元测试，严禁通过 ridge 静默绕过 rank 失败。

### Task 3 — 执行 Synthetic Stress

> 读取第 6、11、12、14 节。实现 `synthetic-stress`，遍历不同代理质量、违规路径、positivity 强度及随机种子，输出 JSON/CSV 和 Markdown 摘要；区分识别失败与数值稳定的带偏结果。

### Task 4 — 实现 Neural Moment Bridge

> 读取第 8 节。添加实值 `OutcomeBridgeNet`、离散 conditional moment loss、独立验证残差、cross-fitting。保证模型不使用 verifier 后处理文本 embedding 作为调整前 `L`。在同一 SCM 上与 discrete bridge 对比。

### Task 5 — 连接 Legacy Verifier

> 先实际检查当前 `model.py`、`train.py`、`data.py` checkpoint 格式，再写 `verifier_adapter.py`。支持冻结已有 checkpoint 批量预测和逐样本 NLL。输出观测响应表，并在缺失合法处理/代理元数据时自动拒绝 `real_strict`。

### Task 6 — 实现 Effect-guided Fine-tuning

> 读取第 9—11 节。只在 identification gates 通过时用 CATE 生成可审计的样本群组权重；复用现有事实核查训练与反事实数据；实现 uniform/random/observed-association/proximal-guided 四种同预算对照。训练结束重新估计 effect，联合汇报 OOD、CCR、SSR 和 per-group metrics。

每个 Task 交付时必须同时说明：新增文件、修改文件、真实测试命令、测试结果、已知限制及下一阶段前置条件。

---

## 16. 论文主张与实现状态对照

| 可以写的论文主张 | 必须满足的代码/实验条件 |
|---|---|
| “研究了潜在混杂下的 shortcut effect” | 因果图、干预 `A`、结果 `R_theta`、处理分配机制明确 |
| “使用 proximal bridge 识别该 effect” | 实际双代理满足经论证的识别假设；合成恢复与真实代理来源被报告 |
| “神经 bridge 能解决条件矩方程” | 独立验证 moment residual、与离散基准比较、弱识别实验 |
| “effect-guided debiasing 有效” | 相同计算预算下优于 random/uniform/observed weighting，跨 seed 重现 |
| “减少 shortcut 依赖但保留事实语义” | CCR、SSR、标签保持及标签变化双端正确率均经过验证 |

**不得写**：`Our model guarantees causal identification on FEVER`，除非对 FEVER 的 treatment、两类代理、identification assumptions 和目标总体有足够强的独立论证。  
**不得写**：`The bridge loss proves absence of unobserved confounding`。  
**建议写**：`Under explicit proximal identification assumptions, the outcome bridge identifies the causal effect of a well-defined shortcut treatment on a fixed verifier's response; we validate the estimator in controlled settings and study its utility for effect-guided robust training.`

---

## 17. 文献与工程参考

### 17.1 当前项目文档

- [ProxCABI-FV repository](https://github.com/2016bits/ProxCABI-FV)
- [IMPLEMENTATION.md](https://github.com/2016bits/ProxCABI-FV/blob/main/IMPLEMENTATION.md)
- [model.py](https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/model.py)
- [train.py](https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/train.py)
- [proxies.py](https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/proxies.py)
- [cli.py](https://github.com/2016bits/ProxCABI-FV/blob/main/proxcabi/cli.py)

### 17.2 理论参考（正式投稿前核对准确 BibTeX）

- Tchetgen Tchetgen et al., *An Introduction to Proximal Causal Learning*, 2020/2024 相关版本：近端因果识别、代理变量排除限制、bridge 条件与 completeness。
- Miao, Geng, and Tchetgen Tchetgen, *Identifying Causal Effects With Proxy Variables of an Unmeasured Confounder*, 2018：利用未观测混杂的代理测量实现识别。
- 近端估计中的 conditional moment / minimax estimation、cross-fitting 与弱识别诊断文献：在采用神经估计器时补充对应的准确文献和限制。

> **最终交付定义**：只有当 `synthetic-generate` → `fit-bridge` → `evaluate-identification` → 自动测试在同一环境下真正运行通过，并保存可复现配置与输出时，Phase 1/2 才算完成。真实 FEVER 严格 causal inference 的前置条件仍是具有可辩护的独立代理测量，而不是完成了 Python 编码。
