# ProxCABI-FV

**Proximal Causal Attribute Bias Intervention for Fact Verification**

> A CABIN-inspired proximal causal inference framework for debiased fact verification under latent dataset-generation confounding.

---

## 0. Overview

Fact verification (FV) takes a claim \(C\) and evidence \(E\) as input and predicts a veracity label \(Y\), e.g., SUPPORTS / REFUTES / NEI.  
A major challenge is that benchmark datasets often contain **distributional shortcuts**: surface patterns such as negation, lexical templates, entities, topic distributions, evidence-source patterns, or annotation artifacts can correlate strongly with labels.

For example, a model may learn:

\[
\text{negation word} \Longrightarrow \text{REFUTES}
\]

instead of learning the genuine semantic relation between claim and evidence.

This project is inspired by **CABIN: Debiasing Vision-Language Models Using Backdoor Adjustments (IJCAI 2025)**, but extends the causal formulation to a more difficult setting:

- CABIN treats an observed sensitive attribute \(B\) as an explicit confounder.
- In fact verification, the true source of bias is often **not directly observed**.
- We therefore introduce an unobserved dataset-generation confounder \(U\), and use two observed proxy views \(Z\) and \(W\) to identify the causal effect without recovering \(U\) itself.

The core causal goal is:

\[
\boxed{
P(Y(x))
}
\]

rather than merely the observational quantity

\[
P(Y\mid X=x).
\]

---

# 1. Problem Formulation

Given a claim \(C\) and evidence \(E\), define

\[
X=(C,E).
\]

The prediction target is

\[
Y\in\mathcal Y,
\]

where typically

\[
\mathcal Y=
\{\text{SUPPORTS},\text{REFUTES},\text{NEI}\}.
\]

We assume the intended semantic mechanism is

\[
X\rightarrow M\rightarrow Y,
\]

where

\[
M=f_\theta(C,E)
\]

is the **bias-invariant claim-evidence relational semantic representation**.

The variable \(M\) should encode whether the evidence semantically supports, refutes, or fails to resolve the claim, while suppressing dataset-specific shortcuts.

---

# 2. Latent Dataset-Generation Confounding

We assume there exists an unobserved variable

\[
U=\text{latent dataset-generation confounder}.
\]

Possible sources include:

- claim construction strategy;
- negative-example generation strategy;
- annotation policy;
- source-selection mechanism;
- evidence retrieval / evidence composition protocol;
- topic-label imbalance;
- entity-label imbalance;
- sample filtering;
- dataset-specific writing conventions;
- provenance-specific artifacts.

The latent variable \(U\) may simultaneously affect the observed input distribution and label distribution:

\[
U\rightarrow X,
\qquad
U\rightarrow Y.
\]

Therefore, observational learning contains the spurious backdoor path

\[
\boxed{
X\leftarrow U\rightarrow Y.
}
\]

A standard verifier trained by minimizing

\[
-\log P(Y\mid X)
\]

may exploit this path rather than the genuine semantic mechanism

\[
X\rightarrow M\rightarrow Y.
\]

---

# 3. Why Ordinary Backdoor Adjustment Is Insufficient

If \(U\) were observed, one could use

\[
P(Y(x))
=
\int P(Y\mid X=x,U=u)\,dP(u).
\]

However, \(U\) is latent.

A natural but generally invalid shortcut would be to replace \(U\) by a single observable bias feature \(B\) or \(Z\):

\[
P(Y(x))
\overset{?}{\approx}
\sum_z P(Y\mid X=x,Z=z)P(z).
\]

This is not generally justified because

\[
Z\neq U,
\]

and conditioning on an imperfect proxy \(Z\) does not necessarily block

\[
X\leftarrow U\rightarrow Y.
\]

This motivates **proximal causal identification**.

---

# 4. Proxy Variables

We introduce two observed proxies of the same latent confounder \(U\):

\[
Z=\text{treatment-inducing proxy},
\]

\[
W=\text{outcome-inducing proxy}.
\]

The conceptual structure is

\[
U\rightarrow Z,
\qquad
U\rightarrow W,
\qquad
U\rightarrow X,
\qquad
U\rightarrow Y.
\]

The two proxy views play different roles.

## 4.1 Treatment-inducing proxy \(Z\)

\(Z\) provides observable variation associated with the hidden confounder \(U\).

Candidate claim-side features:

\[
Z=
\{
Z_{\mathrm{neg}},
Z_{\mathrm{lex}},
Z_{\mathrm{syntax}},
Z_{\mathrm{entity}},
Z_{\mathrm{style}},
Z_{\mathrm{generation}}
\}.
\]

Examples:

- negation realization;
- high-LMI label-correlated lexical phrases;
- syntactic templates;
- entity patterns;
- claim length/style;
- generated-vs-human style signatures;
- claim transformation type if recoverable from metadata.

Important:

\[
Z
\]

is **not** directly substituted for \(U\) in the backdoor formula.

Its main role is to provide variation that helps identify the bridge function.

## 4.2 Outcome-inducing proxy \(W\)

\(W\) provides another observable view of \(U\), used inside the confounding bridge.

Candidate features:

\[
W=
\{
W_{\mathrm{domain}},
W_{\mathrm{source}},
W_{\mathrm{style}},
W_{\mathrm{structure}},
W_{\mathrm{length}},
W_{\mathrm{provenance}},
W_{\mathrm{retrieval}}
\}.
\]

Examples:

- evidence source / page type;
- provenance metadata;
- evidence linguistic style;
- evidence sentence count;
- hop depth / structure;
- source-domain statistics;
- evidence-generation or retrieval patterns.

---

# 5. Causal DAG

The working causal graph is

```text
                         U
                    ┌────┼────┬────┐
                    │    │    │    │
                    ▼    ▼    ▼    ▼
                    Z    X    W    Y
                         │
                         ▼
                         M
                         │
                         ▼
                         Y
```

Equivalently, the important paths are

\[
U\rightarrow Z,
\quad
U\rightarrow X,
\quad
U\rightarrow W,
\quad
U\rightarrow Y,
\]

and

\[
X\rightarrow M\rightarrow Y.
\]

There is no required direct edge

\[
X\rightarrow Y
\]

if \(M\) fully mediates the genuine semantic effect.

---

# 6. Identification Assumptions

The theoretical framework relies on the following assumptions.

## A1. Latent exchangeability

\[
\boxed{
Y(x)\perp X\mid U
}
\]

Interpretation:

after conditioning on the true latent dataset-generation confounder \(U\), there is no remaining unmeasured confounding between \(X\) and \(Y\).

---

## A2. Treatment-proxy validity

\[
\boxed{
Y\perp Z\mid X,U
}
\]

Equivalent form:

\[
P(Y\mid X,U,Z)=P(Y\mid X,U).
\]

Interpretation:

after \(X\) and \(U\) are known, \(Z\) should not contain additional information about \(Y\).

Thus \(Z\) may be associated with \(Y\) observationally through

\[
Z\leftarrow U\rightarrow Y,
\]

but it should not have an extra causal route to \(Y\) once \(X,U\) are controlled.

---

## A3. Outcome-proxy validity

\[
\boxed{
W\perp (X,Z)\mid U
}
\]

Interpretation:

conditional on the latent confounder \(U\), the outcome-inducing proxy \(W\) should not additionally depend on treatment \(X\) or the treatment proxy \(Z\).

---

## A4. Completeness

A representative completeness condition is

\[
\boxed{
E[v(U)\mid X,Z]=0
\Longrightarrow
v(U)=0
}
\]

almost surely.

Intuition:

\(Z\) must provide sufficiently rich variation about the hidden \(U\).  
Different latent confounding states should not become completely indistinguishable through \(Z\).

Completeness is what makes the bridge equation identifiable rather than underdetermined.

---

## A5. Positivity / overlap

Relevant configurations must have nonzero support.

For example,

\[
p(z\mid x)>0
\]

for the proxy configurations required to identify the bridge.

In practice, this requires adequate overlap across shortcut patterns.

---

## A6. Bridge existence

For each label class \(c\), there exists a measurable function \(h_c\) satisfying the outcome confounding bridge equation.

---

# 7. Outcome Confounding Bridge

For class \(c\), define

\[
Y^{(c)}
=
\mathbf 1(Y=c).
\]

We seek a function \(h_c\) such that

\[
\boxed{
E[Y^{(c)}\mid X=x,Z=z]
=
E[h_c(x,W)\mid X=x,Z=z].
}
\]

Using the semantic representation

\[
M=f_\theta(X),
\]

we parameterize

\[
h_c(X,W)
=
\widetilde h_c(M,W).
\]

Hence

\[
E[Y^{(c)}\mid M,Z]
\approx
E[\widetilde h_c(M,W)\mid M,Z].
\]

The bridge is not an estimator of \(U\).

Instead:

\[
\boxed{
Z,W
\rightarrow
\text{identify }h
\rightarrow
\text{identify }P(Y(x)).
}
\]

---

# 8. Why Two Proxies Are Needed

A single imperfect proxy \(Z\) does not generally block

\[
X\leftarrow U\rightarrow Y.
\]

The second proxy \(W\) enables an observable bridge equation.

For fixed \(X=x\) and binary \(W\),

\[
E[Y\mid x,z]
=
h(x,0)P(W=0\mid x,z)
+
h(x,1)P(W=1\mid x,z).
\]

Different values

\[
z_1,z_2,\ldots
\]

generate multiple equations:

\[
E[Y\mid x,z_1]
=
h(x,0)P(W=0\mid x,z_1)
+
h(x,1)P(W=1\mid x,z_1),
\]

\[
E[Y\mid x,z_2]
=
h(x,0)P(W=0\mid x,z_2)
+
h(x,1)P(W=1\mid x,z_2).
\]

If the variation induced through \(Z\) is sufficiently rich, the unknown bridge values can be identified.

Thus:

\[
\boxed{
Z\text{ provides identifying variation;}
\qquad
W\text{ provides the bridge representation.}
}
\]

---

# 9. Dataset Suite

| Dataset | Role |
|---|---|
| FEVER | Large-scale standard FV and shortcut learning |
| symmetric-FEVER | Lexical / syntactic shortcut robustness |
| PolitiHop | Topic, entity, political lexical bias and multi-hop reasoning |
| symmetric-PolitiHop | Distribution-shift robustness |
| HOVER | Multi-hop reasoning preservation |
| VitaminC | Contrastive factual-change and semantic sensitivity |

## Main protocol

### FEVER

Train on FEVER.

Evaluate on:

\[
\text{FEVER}
\]

and

\[
\text{symmetric-FEVER}.
\]

### PolitiHop

Train on PolitiHop.

Evaluate on:

\[
\text{PolitiHop}
\]

and

\[
\text{symmetric-PolitiHop}.
\]

### HOVER

Report separately:

\[
2\text{-hop},
\quad
3\text{-hop},
\quad
4\text{-hop}.
\]

### VitaminC

Use for:

1. standard fact verification;
2. contrastive factual-change sensitivity;
3. semantic perturbation analysis.

---

# 10. Stage 0 — Dataset Bias Audit

Before constructing any proxy, quantify dataset artifacts.

For each dataset:

- label distribution;
- claim length;
- evidence length;
- negation frequency by label;
- lexical LMI with labels;
- syntactic-template distribution;
- entity-label distribution;
- topic-label distribution;
- evidence source/domain distribution;
- evidence hop / structure distribution.

For lexical feature \(b\) and label \(y\), compute

\[
\operatorname{LMI}(b,y)
=
P(b,y)\log\frac{P(b,y)}{P(b)P(y)}.
\]

The purpose is not yet to define \(Z\), but to establish that shortcut correlations exist.

---

# 11. Stage 1 — Dataset-Specific Proxy Construction

Proxy construction must be dataset-specific.

Do not assume that a proxy valid for FEVER is also valid for HOVER or PolitiHop.

## 11.1 Candidate \(Z\)

Construct

\[
Z_i=f_Z(C_i,\mathcal M_i^C),
\]

where optional claim-side metadata may be used.

Candidate components:

- negation category;
- lexical shortcut vector;
- syntax-template embedding;
- entity-pattern vector;
- style vector;
- claim-generation signature.

Prefer a multi-dimensional \(Z\) over a single binary shortcut.

## 11.2 Candidate \(W\)

Construct

\[
W_i=f_W(E_i,\mathcal M_i^E).
\]

Candidate components:

- evidence provenance;
- source domain;
- page type;
- source style;
- evidence structural statistics;
- evidence length;
- hop count;
- retrieval signature.

## 11.3 Important caution

If

\[
Z=f(X)
\]

is a purely deterministic function of the entire treatment \(X\), then

\[
Y\perp Z\mid X,U
\]

may become formally trivial while providing weak identifying variation.

Therefore, the preferred implementation is:

- use dataset-generation metadata when available;
- use transformations / generation attributes that are not merely redundant encodings of the entire \(X\);
- treat purely surface-derived \(Z\) as an approximation and test it aggressively.

---

# 12. Stage 2 — Proxy Diagnostics

This stage is mandatory.

The method should not assume proxy validity without empirical stress tests.

## 12.1 Proxy-label association

Measure

\[
I(Z;Y),
\qquad
I(W;Y).
\]

This confirms that proxies contain dataset-structure information.

However:

\[
I(Z;Y)>0
\]

does **not** establish causal validity.

---

## 12.2 Cross-proxy informativeness

Estimate

\[
R^2(W\mid X,Z)
\]

and compare with

\[
R^2(W\mid X).
\]

Define

\[
\boxed{
\Delta_{\mathrm{proxy}}
=
R^2(W\mid X,Z)-R^2(W\mid X).
}
\]

A positive value indicates that \(Z\) contributes additional information useful for predicting the second proxy view \(W\).

For discrete \(W\), replace \(R^2\) with accuracy, log-loss, or mutual information.

---

## 12.3 Structured / random / reversed proxy tests

Compare:

1. structured \(Z/W\);
2. random \(Z/W\);
3. reversed \(Z/W\);
4. shuffled \(Z\);
5. shuffled \(W\).

A valid design should show clear advantages for the intended assignment.

---

## 12.4 Proxy strength degradation

Inject controlled masking/noise:

\[
\rho\in
\{1.0,0.8,0.6,0.4,0.2\}.
\]

Measure:

\[
Accuracy(\rho),
\]

\[
CCR(\rho),
\]

\[
R_{\mathrm{bridge}}(\rho).
\]

Expected trend:

\[
\text{proxy quality}\downarrow
\Rightarrow
R_{\mathrm{bridge}}\uparrow
\Rightarrow
\text{causal robustness}\downarrow.
\]

---

# 13. Stage 3 — Semantic Encoder

Use

\[
M=f_\theta(C,E).
\]

Initial backbones:

- RoBERTa-base;
- DeBERTa-v3-base.

Train with standard fact verification loss:

\[
\boxed{
\mathcal L_{\mathrm{fact}}
=
-\frac1N
\sum_{i=1}^N
\log P(Y_i\mid M_i).
}
\]

The semantic encoder serves two purposes:

1. preserve genuine claim-evidence reasoning;
2. provide a compact treatment representation for proximal bridge learning.

---

# 14. Stage 4 — Proxy Conditional Model

Estimate

\[
q_\omega(W\mid M,Z).
\]

For continuous \(W\), predict a representation

\[
r_\omega(M,Z)
\approx
E[\phi_W(W)\mid M,Z].
\]

Use

\[
\boxed{
\mathcal L_{\mathrm{proxy}}
=
\left\|
\phi_W(W)
-
r_\omega(M,Z)
\right\|_2^2.
}
\]

For discrete \(W\), use cross-entropy.

This model approximates the conditional proxy distribution required by the bridge equation.

---

# 15. Stage 5 — Bridge Learning

Define a conditional outcome model

\[
g_\gamma(M,Z)
=
P(Y\mid M,Z).
\]

Define bridge model

\[
h_\psi(M,W).
\]

Sample

\[
W_i^{(s)}
\sim
q_\omega(W\mid M_i,Z_i),
\qquad
s=1,\ldots,S.
\]

Compute

\[
\widehat h_i
=
\frac1S
\sum_{s=1}^S
h_\psi(M_i,W_i^{(s)}).
\]

Then enforce the empirical bridge equation

\[
g_\gamma(M_i,Z_i)
\approx
\widehat h_i.
\]

A practical bridge loss is

\[
\boxed{
\mathcal L_{\mathrm{bridge}}
=
D_{\mathrm{JS}}
\left(
g_\gamma(M_i,Z_i)
\parallel
\widehat h_i
\right).
}
\]

Alternative estimators to test:

- squared moment loss;
- MMD moment matching;
- adversarial conditional moment restriction;
- two-stage regression bridge;
- linear bridge.

---

# 16. Overall Training Objective

The first implementation should use

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{fact}}
+
\lambda_p\mathcal L_{\mathrm{proxy}}
+
\lambda_b\mathcal L_{\mathrm{bridge}}.
}
\]

Recommended strategy:

1. pretrain / warm up semantic verifier with \(\mathcal L_{\mathrm{fact}}\);
2. train proxy model;
3. train bridge model;
4. optionally jointly fine-tune with a small bridge weight.

Do not initially introduce unnecessary adversarial or orthogonality losses.

---

# 17. Stage 6 — Proximal Inference

For test input

\[
X_i=(C_i,E_i),
\]

compute

\[
M_i=f_\theta(X_i).
\]

The target is

\[
\boxed{
P_{\mathrm{prox}}(Y=c\mid X_i)
=
E_W[h_c(M_i,W)].
}
\]

Approximate via Monte Carlo:

\[
\boxed{
\hat P_{\mathrm{prox}}(Y=c\mid X_i)
=
\frac1S
\sum_{s=1}^S
h_c(M_i,W^{(s)}).
}
\]

For the primary estimator:

\[
W^{(s)}
\sim
\widehat P_{\mathrm{train}}(W).
\]

Also evaluate sensitivity to:

- empirical training marginal;
- balanced reference marginal;
- dataset-specific reference marginal.

Finally,

\[
\hat Y_i
=
\arg\max_c
\hat P_{\mathrm{prox}}(Y=c\mid X_i).
\]

---

# 18. Stage 7 — Counterfactual Proxy-Support Enhancement

Construct

\[
X_i^{cf}
\]

such that

\[
M(X_i^{cf})
\approx
M(X_i),
\]

while

\[
Z(X_i^{cf})
\neq
Z(X_i).
\]

Examples:

- negation realization changes that preserve proposition meaning;
- lexical paraphrases;
- syntactic paraphrases;
- style rewrites;
- entity surface-form changes that preserve referent.

Evidence-side counterfactuals may vary \(W\) while approximately preserving the claim-evidence relation.

Goal:

\[
\boxed{
M\text{ fixed},
\qquad
Z/W\text{ varied}.
}
\]

This improves overlap and provides empirical support for the proxy-identification assumptions.

---

# 19. Stage 8 — Semantic Sensitivity Evaluation

A debiased model should not become insensitive to genuine factual changes.

Construct semantic perturbations

\[
X^{sem}
\]

that genuinely change the relation between claim and evidence:

- entity replacement;
- numerical change;
- relation reversal;
- temporal change;
- polarity-changing negation;
- location change;
- causal-role change.

The desired behavior is:

\[
\hat Y(X)
\neq
\hat Y(X^{sem}).
\]

---

# 20. Evaluation Metrics

## 20.1 Standard performance

\[
Accuracy
\]

and

\[
Macro\text{-}F1.
\]

---

## 20.2 Distribution-shift gap

\[
\boxed{
\Delta_{\mathrm{shift}}
=
Perf_{\mathrm{orig}}
-
Perf_{\mathrm{sym}}.
}
\]

Lower is better.

---

## 20.3 Shortcut Counterfactual Consistency

\[
\boxed{
CCR
=
\frac1N
\sum_i
\mathbf 1
[
\hat Y(X_i)
=
\hat Y(X_i^{cf})
].
}
\]

Higher is better.

---

## 20.4 Semantic Sensitivity Rate

\[
\boxed{
SSR
=
\frac1N
\sum_i
\mathbf 1
[
\hat Y(X_i)
\neq
\hat Y(X_i^{sem})
].
}
\]

Higher is better.

Desired joint behavior:

\[
\boxed{
CCR\uparrow,
\qquad
SSR\uparrow.
}
\]

This distinguishes

\[
\text{shortcut invariance}
\]

from

\[
\text{semantic insensitivity}.
\]

---

## 20.5 Bridge residual

Define

\[
\boxed{
R_{\mathrm{bridge}}
=
\frac1N
\sum_i
\left\|
g_\gamma(M_i,Z_i)
-
\widehat E[
h(M_i,W)
\mid M_i,Z_i
]
\right\|_2^2.
}
\]

Lower is better.

---

# 21. Baselines

## 21.1 Standard verifiers

- Claim-only;
- Claim-Evidence;
- RoBERTa verifier;
- DeBERTa-NLI;
- GEAR;
- KGAT.

## 21.2 Causal / debiasing baselines

- CICR;
- CLEVER;
- Causal Walk;
- Dual-Debias;
- EGR-FV.

## 21.3 CABIN-inspired naive adjustment baseline

Define **Naive-CABI**:

\[
\boxed{
P(Y\mid do(X))
\approx
\sum_b
P(Y\mid X,b)P(b).
}
\]

Here \(b\) is an observed shortcut attribute such as negation or lexical style.

The most important comparison is

\[
\boxed{
\text{Naive-CABI}
\quad\text{vs.}\quad
\text{ProxCABI-FV}.
}
\]

This tests the hypothesis:

\[
\boxed{
\text{observed shortcut}
\neq
\text{true confounder}.
}
\]

---

# 22. Ablation Study

| Variant | Purpose |
|---|---|
| Full ProxCABI-FV | Full method |
| w/o \(Z\) | Test treatment-proxy necessity |
| w/o \(W\) | Test outcome-proxy necessity |
| Random \(Z/W\) | Test proxy informativeness |
| Reverse \(Z/W\) | Test role assignment |
| Shuffled \(Z\) | Destroy identifying variation |
| Shuffled \(W\) | Destroy bridge information |
| w/o bridge | Test bridge contribution |
| Naive-CABI | Compare ordinary proxy adjustment |
| w/o CF support | Test overlap enhancement |
| Linear bridge | Test nonlinear bridge necessity |
| Two-stage bridge | Alternative estimator |
| Moment bridge | Alternative estimator |
| w/o semantic bottleneck \(M\) | Test representation role |
| Proxy degradation | Test robustness to proxy quality |

---

# 23. Semi-Synthetic Causal Evaluation

Real datasets do not reveal the true latent confounder \(U\), so observational benchmark improvements alone cannot prove causal identification.

Therefore construct a semi-synthetic benchmark.

Let

\[
U\in\{u_1,\ldots,u_K\}.
\]

Control

\[
P(Z\mid U),
\]

\[
P(W\mid U),
\]

\[
P(X\mid U),
\]

and shortcut strength.

Keep oracle labels determined by the genuine semantic relation

\[
M(X).
\]

Because the data-generation process is controlled, the oracle causal distribution

\[
P^*(Y(x))
\]

is known.

Define causal estimation error

\[
\boxed{
CEE
=
D_{\mathrm{KL}}
\left(
P^*(Y(x))
\parallel
\hat P(Y(x))
\right).
}
\]

Alternative:

\[
CEE_{\mathrm{MSE}}
=
\left\|
P^*(Y(x))
-
\hat P(Y(x))
\right\|_2^2.
\]

Compare:

- Claim-Evidence;
- Dual-Debias;
- Naive-CABI;
- ProxCABI-FV.

Expected trend:

\[
CEE_{\mathrm{ProxCABI}}
<
CEE_{\mathrm{Naive-CABI}}.
\]

---

# 24. LLM Extension

Available hardware:

\[
4\times
\text{NVIDIA A6000 48GB}.
\]

## Main experiments

Use:

- RoBERTa-base;
- DeBERTa-v3-base.

## LLM extension

Use one shared 7B/8B backbone with:

- LoRA / QLoRA;
- gradient checkpointing;
- bf16/fp16 mixed precision;
- lightweight \(Z\), \(W\), \(g\), and \(h\) heads;
- shared semantic encoder.

Recommended first LLM experiments:

1. FEVER + symmetric-FEVER;
2. PolitiHop + symmetric-PolitiHop;

or replace the second group with HOVER.

Avoid training separate 7B/8B models for \(M,Z,W\).

---

# 25. Research Questions

**RQ1.** Does ProxCABI-FV improve fact verification under both in-distribution and symmetric / OOD settings?

**RQ2.** Does proximal adjustment outperform naive backdoor adjustment over observed shortcut attributes?

**RQ3.** Are structured \(Z/W\) proxy assignments necessary for handling latent dataset confounding?

**RQ4.** Can ProxCABI-FV achieve shortcut invariance while preserving genuine semantic sensitivity?

**RQ5.** Does proxy quality control bridge quality and causal robustness?

**RQ6.** Can ProxCABI-FV recover known causal effects under semi-synthetic latent confounding?

**RQ7.** Does counterfactual proxy-support enhancement improve overlap and robustness?

**RQ8.** Does the framework remain effective when scaling from PLMs to 7B/8B LLMs?

---

# 26. Experimental Hypotheses

## H1

\[
Perf_{\mathrm{ProxCABI}}
>
Perf_{\mathrm{Vanilla}}
\]

under symmetric / OOD evaluation.

## H2

\[
\Delta_{\mathrm{shift}}^{\mathrm{ProxCABI}}
<
\Delta_{\mathrm{shift}}^{\mathrm{Vanilla}}.
\]

## H3

\[
CCR_{\mathrm{ProxCABI}}
>
CCR_{\mathrm{Vanilla}}.
\]

## H4

\[
SSR_{\mathrm{ProxCABI}}
\]

should remain high, demonstrating that debiasing does not destroy semantic sensitivity.

## H5

\[
R_{\mathrm{bridge}}
\downarrow
\Longrightarrow
\text{OOD robustness}\uparrow.
\]

## H6

\[
CEE_{\mathrm{ProxCABI}}
<
CEE_{\mathrm{Naive-CABI}}.
\]

---

# 27. Recommended Implementation Order

\[
\boxed{
\text{Bias Audit}
\rightarrow
\text{Proxy Design}
\rightarrow
\text{Proxy Diagnostics}
\rightarrow
\text{Semantic Encoder}
\rightarrow
\text{Proxy Model}
\rightarrow
\text{Bridge Learning}
\rightarrow
\text{Proximal Inference}
\rightarrow
\text{Robustness Evaluation}
\rightarrow
\text{Causal Recovery Evaluation}.
}
\]

The highest-risk stage is proxy design, not bridge-network implementation.

---

# 28. TODO List

## Phase 0 — Theory Finalization

- [ ] Fix notation: \(X=(C,E),U,Z,W,M,Y\).
- [ ] Finalize DAG.
- [ ] State estimand \(P(Y(x))\).
- [ ] Formalize latent exchangeability.
- [ ] Formalize treatment-proxy validity.
- [ ] Formalize outcome-proxy validity.
- [ ] Formalize completeness.
- [ ] Formalize positivity.
- [ ] Formalize bridge existence.
- [ ] Derive bridge equation.
- [ ] Derive proximal \(g\)-formula.
- [ ] Clarify when \(Z=f(X)\) becomes too weak / trivial.
- [ ] Clarify whether test-time \(P(W)\) uses training marginal or reference marginal.

## Phase 1 — Dataset Preparation

- [ ] Prepare FEVER.
- [ ] Prepare symmetric-FEVER.
- [ ] Prepare PolitiHop.
- [ ] Prepare symmetric-PolitiHop.
- [ ] Prepare HOVER.
- [ ] Prepare VitaminC.
- [ ] Standardize label mapping.
- [ ] Fix gold-evidence protocol where applicable.
- [ ] Build unified train/dev/test loaders.

## Phase 2 — Bias Audit

- [ ] Analyze label distributions.
- [ ] Analyze claim lengths.
- [ ] Analyze evidence lengths.
- [ ] Analyze negation frequency.
- [ ] Compute lexical LMI.
- [ ] Analyze syntax-label association.
- [ ] Analyze entity-label association.
- [ ] Analyze topic-label association.
- [ ] Analyze evidence source/domain association.
- [ ] Analyze evidence structure / hop depth.

## Phase 3 — Proxy Design

- [ ] Build candidate \(Z\) for each dataset.
- [ ] Build candidate \(W\) for each dataset.
- [ ] Prefer dataset-generation metadata where available.
- [ ] Detect negation patterns.
- [ ] Extract high-LMI lexical features.
- [ ] Extract syntactic templates.
- [ ] Extract entity statistics.
- [ ] Extract claim style features.
- [ ] Extract source / domain / provenance features.
- [ ] Extract evidence structure features.
- [ ] Build dataset-specific proxy table.
- [ ] Flag proxies that may violate A2/A3.

## Phase 4 — Proxy Diagnostics

- [ ] Compute \(I(Z;Y)\).
- [ ] Compute \(I(W;Y)\).
- [ ] Estimate \(W\sim(X,Z)\).
- [ ] Estimate \(W\sim X\).
- [ ] Compute \(\Delta_{\mathrm{proxy}}\).
- [ ] Compare structured / random / reversed \(Z/W\).
- [ ] Shuffle \(Z\).
- [ ] Shuffle \(W\).
- [ ] Perform proxy-strength degradation.
- [ ] Save all proxy diagnostic plots.

## Phase 5 — Baseline Verifiers

- [ ] Claim-only.
- [ ] Claim-Evidence.
- [ ] RoBERTa baseline.
- [ ] DeBERTa baseline.
- [ ] GEAR.
- [ ] KGAT.
- [ ] CICR.
- [ ] CLEVER.
- [ ] Causal Walk.
- [ ] Dual-Debias.
- [ ] EGR-FV.
- [ ] Naive-CABI.
- [ ] Standardize backbone/training settings.

## Phase 6 — Semantic Encoder

- [ ] Implement \(M=f_\theta(C,E)\).
- [ ] Implement \(\mathcal L_{\mathrm{fact}}\).
- [ ] Validate FEVER performance.
- [ ] Validate PolitiHop performance.
- [ ] Validate HOVER 2/3/4-hop performance.
- [ ] Save \(M\)-representations for analysis.

## Phase 7 — Proxy Conditional Model

- [ ] Decide continuous vs discrete \(W\).
- [ ] Implement \(q_\omega(W\mid M,Z)\).
- [ ] Implement \(\mathcal L_{\mathrm{proxy}}\).
- [ ] Evaluate proxy prediction.
- [ ] Evaluate calibration of \(q_\omega\).
- [ ] Perform proxy noise tests.

## Phase 8 — Bridge Model

- [ ] Implement \(g_\gamma(M,Z)\).
- [ ] Implement \(h_\psi(M,W)\).
- [ ] Implement Monte Carlo \(W^{(s)}\sim q_\omega\).
- [ ] Implement JS bridge loss.
- [ ] Implement squared moment bridge.
- [ ] Implement bridge residual.
- [ ] Compare linear vs nonlinear bridge.
- [ ] Compare two-stage vs moment bridge.
- [ ] Optionally implement adversarial conditional moment bridge.

## Phase 9 — Proximal Inference

- [ ] Implement \(E_W[h(M,W)]\).
- [ ] Compare \(S\in\{8,16,32,64\}\).
- [ ] Evaluate empirical training \(P(W)\).
- [ ] Evaluate balanced/reference \(P(W)\).
- [ ] Record inference overhead.
- [ ] Check probability calibration.

## Phase 10 — Counterfactual Proxy Support

- [ ] Generate meaning-preserving negation rewrites.
- [ ] Generate lexical paraphrases.
- [ ] Generate syntax paraphrases.
- [ ] Generate style rewrites.
- [ ] Generate entity surface-form rewrites.
- [ ] Verify semantic preservation.
- [ ] Vary evidence-side proxy features where possible.
- [ ] Improve \(Z/W\) overlap.
- [ ] Compute CCR.

## Phase 11 — Semantic Sensitivity

- [ ] Entity-change examples.
- [ ] Number-change examples.
- [ ] Relation-change examples.
- [ ] Temporal-change examples.
- [ ] Polarity-changing examples.
- [ ] Location-change examples.
- [ ] Compute SSR.
- [ ] Jointly report CCR + SSR.

## Phase 12 — Main Experiments

- [ ] FEVER.
- [ ] symmetric-FEVER.
- [ ] PolitiHop.
- [ ] symmetric-PolitiHop.
- [ ] HOVER.
- [ ] VitaminC.
- [ ] Run at least 3 random seeds.
- [ ] Report mean ± standard deviation.
- [ ] Report Accuracy / Macro-F1.
- [ ] Report \(\Delta_{\mathrm{shift}}\).
- [ ] Report CCR / SSR.
- [ ] Report bridge residual.

## Phase 13 — Ablation

- [ ] w/o \(Z\).
- [ ] w/o \(W\).
- [ ] random \(Z/W\).
- [ ] reverse \(Z/W\).
- [ ] shuffled \(Z\).
- [ ] shuffled \(W\).
- [ ] w/o bridge.
- [ ] Naive-CABI.
- [ ] w/o counterfactual support.
- [ ] linear bridge.
- [ ] two-stage bridge.
- [ ] moment bridge.
- [ ] proxy degradation.
- [ ] w/o semantic bottleneck \(M\).

## Phase 14 — Semi-Synthetic Causal Evaluation

- [ ] Define latent \(U\).
- [ ] Define causal data-generation process.
- [ ] Control confounding strength.
- [ ] Control \(P(Z\mid U)\).
- [ ] Control \(P(W\mid U)\).
- [ ] Control \(P(X\mid U)\).
- [ ] Preserve oracle semantic labels.
- [ ] Compute true \(P^*(Y(x))\).
- [ ] Compute CEE.
- [ ] Compare vanilla / Dual-Debias / Naive-CABI / ProxCABI-FV.
- [ ] Plot confounding strength vs CEE.
- [ ] Plot proxy strength vs CEE.

## Phase 15 — LLM Extension

- [ ] Select 7B/8B backbone.
- [ ] Implement LoRA / QLoRA.
- [ ] Use one shared backbone.
- [ ] Add lightweight proxy / bridge heads.
- [ ] Run FEVER / symmetric-FEVER.
- [ ] Run PolitiHop / symmetric-PolitiHop or HOVER.
- [ ] Compare PLM vs LLM.
- [ ] Record GPU memory.
- [ ] Record training time.
- [ ] Record inference latency.

## Phase 16 — Analysis for Paper

- [ ] Negation shortcut case study.
- [ ] Entity/topic shortcut case study.
- [ ] Visualize \(Z/W\) representations.
- [ ] Visualize bridge residuals.
- [ ] Proxy strength vs robustness.
- [ ] Proxy strength vs CEE.
- [ ] Hop depth vs performance.
- [ ] Original vs symmetric gap.
- [ ] CCR vs SSR tradeoff.
- [ ] PLM vs LLM scalability.
- [ ] Failure-case analysis.
- [ ] Assumption-violation discussion.

---

# 29. Detailed Identification Derivation

This section provides the theoretical derivation of why two proxies can identify the causal effect of \(X\) on \(Y\) without observing \(U\).

---

## 29.1 Target estimand

The causal estimand is

\[
P(Y(x)=c).
\]

For simplicity, first consider a scalar outcome \(Y\).

Under latent exchangeability,

\[
Y(x)\perp X\mid U.
\]

By consistency,

\[
Y=Y(X).
\]

Therefore,

\[
E[Y(x)]
=
E_U[E[Y(x)\mid U]].
\]

Using exchangeability,

\[
E[Y(x)\mid U]
=
E[Y(x)\mid X=x,U].
\]

By consistency,

\[
E[Y(x)\mid X=x,U]
=
E[Y\mid X=x,U].
\]

Hence

\[
\boxed{
E[Y(x)]
=
E_U[E[Y\mid X=x,U]].
}
\]

This is the ordinary backdoor \(g\)-formula if \(U\) were observed.

The problem is that \(U\) is latent.

---

## 29.2 Bridge assumption at the latent-confounder level

Assume there exists a function \(h(x,w)\) satisfying

\[
\boxed{
E[Y\mid X=x,U]
=
E[h(x,W)\mid U].
}
\]

This equation says that the conditional outcome regression involving the unobserved confounder \(U\) can be represented through the observed proxy \(W\).

Substitute this into the ordinary \(g\)-formula:

\[
E[Y(x)]
=
E_U[E[h(x,W)\mid U]].
\]

By the law of iterated expectations,

\[
E_U[E[h(x,W)\mid U]]
=
E[h(x,W)].
\]

Therefore,

\[
\boxed{
E[Y(x)]
=
E_W[h(x,W)].
}
\]

This is the proximal \(g\)-formula.

Crucially, \(U\) has disappeared.

---

## 29.3 But \(h\) is unknown

The previous result is useful only if the bridge function \(h\) can be identified from observed data.

We cannot directly fit

\[
E[Y\mid X,U]
=
E[h(X,W)\mid U]
\]

because \(U\) is unobserved.

This is where \(Z\) enters.

---

## 29.4 Observable bridge equation

Start from the observable conditional expectation

\[
E[Y\mid X,Z].
\]

Apply the law of iterated expectations:

\[
E[Y\mid X,Z]
=
E[
E[Y\mid X,Z,U]
\mid X,Z
].
\]

By treatment-proxy validity,

\[
Y\perp Z\mid X,U,
\]

so

\[
E[Y\mid X,Z,U]
=
E[Y\mid X,U].
\]

Hence

\[
E[Y\mid X,Z]
=
E[
E[Y\mid X,U]
\mid X,Z
].
\]

Using the latent bridge equation,

\[
E[Y\mid X,U]
=
E[h(X,W)\mid U].
\]

Therefore,

\[
E[Y\mid X,Z]
=
E[
E[h(X,W)\mid U]
\mid X,Z
].
\]

Now use outcome-proxy validity,

\[
W\perp(X,Z)\mid U.
\]

This gives

\[
E[h(X,W)\mid X,Z,U]
=
E[h(X,W)\mid U]
\]

for fixed \(X=x\) inside \(h(x,W)\).

Therefore, by iterated expectation,

\[
E[
E[h(X,W)\mid U]
\mid X,Z
]
=
E[h(X,W)\mid X,Z].
\]

Thus we obtain the fully observable bridge equation:

\[
\boxed{
E[Y\mid X,Z]
=
E[h(X,W)\mid X,Z].
}
\]

This equation contains only observed variables:

\[
X,\quad Z,\quad W,\quad Y.
\]

---

## 29.5 Why \(Z\) identifies \(h\)

For fixed \(X=x\),

\[
E[Y\mid x,z]
=
\int h(x,w)
p(w\mid x,z)
\,dw.
\]

The unknown object is

\[
h(x,w).
\]

The quantities

\[
E[Y\mid x,z]
\]

and

\[
p(w\mid x,z)
\]

are estimable from observed data.

As \(z\) varies,

\[
p(w\mid x,z)
\]

also varies because \(Z\) and \(W\) share the hidden cause \(U\).

Thus different \(z\) values provide a family of integral equations.

In the finite discrete case, assume

\[
W\in\{w_1,\ldots,w_K\}
\]

and

\[
Z\in\{z_1,\ldots,z_J\}.
\]

For fixed \(x\),

\[
m_j(x)
=
E[Y\mid x,z_j].
\]

Let

\[
H_k(x)
=
h(x,w_k).
\]

Then

\[
m_j(x)
=
\sum_{k=1}^K
P(W=w_k\mid x,z_j)
H_k(x).
\]

In matrix form,

\[
\boxed{
\mathbf m(x)
=
\mathbf P_{W\mid x,Z}
\mathbf H(x).
}
\]

If

\[
\mathbf P_{W\mid x,Z}
\]

has sufficient rank, then

\[
\mathbf H(x)
\]

is identifiable.

For square invertible matrices,

\[
\mathbf H(x)
=
\mathbf P_{W\mid x,Z}^{-1}
\mathbf m(x).
\]

In continuous settings, completeness is the functional analogue of this full-rank requirement.

That is why completeness is essential.

---

## 29.6 Why one proxy alone is generally insufficient

Suppose we only observe \(Z\).

A naive strategy is

\[
E[Y(x)]
\overset{?}{=}
E_Z[E[Y\mid X=x,Z]].
\]

This would require \(Z\) itself to form a valid adjustment set:

\[
Y(x)\perp X\mid Z.
\]

But if \(Z\) is merely an imperfect measurement of \(U\),

\[
Z\neq U,
\]

then conditioning on \(Z\) leaves residual variation in \(U\).

Hence the backdoor path

\[
X\leftarrow U\rightarrow Y
\]

may remain open.

Thus

\[
\boxed{
\text{proxy association}
\neq
\text{backdoor sufficiency}.
}
\]

---

## 29.7 Why \(W\) is not a control variable for testing \(Z\to Y\)

The role of \(W\) is sometimes misunderstood.

We are not using \(W\) to test whether the observed association is caused by

\[
Z\rightarrow Y
\]

or

\[
Z\leftarrow U\rightarrow Y.
\]

Instead, the theoretical model assumes

\[
Y\perp Z\mid X,U.
\]

That assumption rules out residual \(Z\to Y\) influence after conditioning on \(X,U\).

The role of \(W\) is different:

\[
\boxed{
W\text{ provides an observed proxy representation through which the latent outcome regression can be bridged.}
}
\]

---

## 29.8 Multiclass classification

For fact verification, define

\[
Y^{(c)}
=
\mathbf 1(Y=c).
\]

For every class \(c\), assume bridge function \(h_c\) satisfying

\[
E[Y^{(c)}\mid X,Z]
=
E[h_c(X,W)\mid X,Z].
\]

Then

\[
P(Y(x)=c)
=
E[Y^{(c)}(x)].
\]

Using the same derivation,

\[
\boxed{
P(Y(x)=c)
=
E_W[h_c(x,W)].
}
\]

Therefore the full causal label distribution is

\[
\boxed{
P(Y(x))
=
\left[
E_W[h_1(x,W)],
\ldots,
E_W[h_{|\mathcal Y|}(x,W)]
\right].
}
\]

Prediction is

\[
\boxed{
\hat Y
=
\arg\max_c
E_W[h_c(x,W)].
}
\]

---

# 30. Neural Approximation of the Bridge

The exact theoretical bridge equation is

\[
E[Y^{(c)}\mid X,Z]
=
E[h_c(X,W)\mid X,Z].
\]

In implementation, replace \(X\) by semantic representation

\[
M=f_\theta(X).
\]

Define

\[
g_\gamma(M,Z)
\approx
P(Y\mid M,Z).
\]

Define

\[
q_\omega(W\mid M,Z).
\]

Then

\[
E[h(M,W)\mid M,Z]
\]

is approximated by

\[
\frac1S
\sum_{s=1}^S
h_\psi(M,W^{(s)}),
\]

where

\[
W^{(s)}
\sim
q_\omega(W\mid M,Z).
\]

Thus the empirical bridge condition is

\[
\boxed{
g_\gamma(M,Z)
\approx
\frac1S
\sum_{s=1}^S
h_\psi(M,W^{(s)}).
}
\]

The JS divergence bridge loss is a practical neural approximation, not by itself a proof of identification.

The theoretical identification comes from:

1. exchangeability;
2. proxy validity;
3. bridge existence;
4. completeness;
5. positivity.

The neural objective is an estimator designed to approximate the bridge implied by these assumptions.

---

# 31. Theoretical Claim to Use in the Paper

A concise theoretical statement is:

> **Proposition.**  
> Suppose there exists an unobserved confounder \(U\) such that \(Y(x)\perp X\mid U\). Let \(Z\) and \(W\) be treatment- and outcome-inducing proxies satisfying \(Y\perp Z\mid X,U\) and \(W\perp(X,Z)\mid U\). Assume positivity, completeness, and existence of an outcome confounding bridge \(h_c\) for every class \(c\). Then the causal veracity distribution is identified from observed data by
>
> \[
> P(Y(x)=c)
> =
> E_W[h_c(x,W)],
> \]
>
> where \(h_c\) satisfies
>
> \[
> E[\mathbf 1(Y=c)\mid X=x,Z=z]
> =
> E[h_c(x,W)\mid X=x,Z=z].
> \]

The key conceptual message is:

\[
\boxed{
\text{ProxCABI-FV does not recover }U.
}
\]

Instead,

\[
\boxed{
\text{two proxy views identify a bridge that removes the need to observe }U.
}
\]

---

# 32. Most Important Risks

The framework has strong proximal-causal motivation, but its practical validity depends on whether the constructed NLP proxies approximately satisfy the theoretical assumptions.

The three main risks are:

1. **Proxy validity:** some lexical features may directly encode factual semantics.
2. **Completeness / strength:** proxies may not contain enough information about the latent dataset mechanism.
3. **Representation substitution:** replacing \(X\) by learned \(M\) in the bridge estimator requires care and should be presented as a modeling approximation unless further formal assumptions are added.

Therefore the strongest paper should combine:

\[
\boxed{
\text{theory}
+
\text{proxy diagnostics}
+
\text{controlled semi-synthetic causal recovery}
+
\text{real benchmark robustness}.
}
\]

---

# 33. Final Experimental Narrative

The complete experimental story is:

\[
\boxed{
\text{Dataset shortcuts exist}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{They are manifestations of latent dataset-generation confounding }U
}
\]

\[
\Downarrow
\]

\[
\boxed{
Z,W\text{ provide two observable proxy views}
}
\]

\[
\Downarrow
\]

\[
\boxed{
E[Y\mid X,Z]
=
E[h(X,W)\mid X,Z]
}
\]

\[
\Downarrow
\]

\[
\boxed{
P(Y(x))
=
E_W[h(x,W)]
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{shortcut robustness}
+
\text{semantic sensitivity}
+
\text{causal recovery}
}
\]

This is the central theoretical and experimental logic of **ProxCABI-FV**.
