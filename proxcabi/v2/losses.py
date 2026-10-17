from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class V2LossWeights:
    std: float = 1.0
    fact: float = 0.25
    g: float = 0.25
    h: float = 0.5
    proxy: float = 0.1
    bridge: float = 0.05
    invariance: float = 0.1
    semantic: float = 0.1

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("loss weights must be non-negative")


def invariant_js(
    p_orig: torch.Tensor,
    p_inv: torch.Tensor,
    inv_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked probability JS divergence; an empty mask returns graph-zero."""

    _validate_probability_pair(p_orig, p_inv, inv_mask)
    if not bool(inv_mask.any()):
        return (p_orig.sum() + p_inv.sum()) * 0.0
    p = _normalize_probs(p_orig[inv_mask].float(), eps)
    q = _normalize_probs(p_inv[inv_mask].float(), eps)
    midpoint = 0.5 * (p + q)
    values = 0.5 * (
        F.kl_div(midpoint.log(), p, reduction="none").sum(dim=-1)
        + F.kl_div(midpoint.log(), q, reduction="none").sum(dim=-1)
    )
    return values.mean()


def semantic_margin_loss(
    logp_sem: torch.Tensor,
    old_label: torch.Tensor,
    new_label: torch.Tensor,
    sem_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require the verified changed label to beat the old label by ``margin``."""

    if margin < 0:
        raise ValueError("semantic margin must be non-negative")
    if logp_sem.ndim != 2 or old_label.shape != new_label.shape or old_label.ndim != 1:
        raise ValueError("semantic inputs need [B,C], [B], and [B] shapes")
    if sem_mask.dtype != torch.bool or sem_mask.shape != old_label.shape:
        raise ValueError("sem_mask must be bool with shape [B]")
    if not bool(sem_mask.any()):
        return logp_sem.sum() * 0.0
    active_old, active_new = old_label[sem_mask], new_label[sem_mask]
    if torch.equal(active_old, active_new) or bool((active_old == active_new).any()):
        raise ValueError("active semantic pairs must change gold label")
    active = logp_sem.float()[sem_mask]
    old_logp = active.gather(1, active_old.unsqueeze(1)).squeeze(1)
    new_logp = active.gather(1, active_new.unsqueeze(1)).squeeze(1)
    return F.relu(float(margin) - new_logp + old_logp).mean()


def bridge_alignment_loss(
    g_probs: torch.Tensor,
    q_probs: torch.Tensor,
    h_all_w_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Predictive JS alignment between g and the q-weighted H mixture."""

    if q_probs.ndim != 2 or h_all_w_probs.ndim != 3:
        raise ValueError("q_probs and h_all_w_probs need [B,K] and [B,K,C]")
    if q_probs.shape != h_all_w_probs.shape[:2]:
        raise ValueError("q and H W dimensions do not match")
    bridge = torch.einsum(
        "bw,bwc->bc",
        _normalize_probs(q_probs.float(), eps),
        _normalize_probs(h_all_w_probs.float(), eps),
    )
    mask = torch.ones(g_probs.size(0), dtype=torch.bool, device=g_probs.device)
    return invariant_js(g_probs, bridge, mask, eps)


def compute_supervised_losses(
    view_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    paired_batch: Mapping[str, object],
    weights: V2LossWeights,
) -> Dict[str, object]:
    """Compute raw supervised/bridge losses over every real (non-dummy) view."""

    del weights  # Kept in the API so callers use one stable loss contract.
    masks = {
        "base": torch.ones_like(paired_batch["old_labels"], dtype=torch.bool),  # type: ignore[arg-type]
        "invariant": paired_batch["inv_mask"],
        "semantic": paired_batch["sem_mask"],
    }
    result: Dict[str, object] = {}
    specifications = {
        "std": "std_logits",
        "fact": "fact_logits",
        "g": "g_logits",
        "h": "h_obs_logits",
        "proxy": "q_logits",
    }
    for loss_name, output_name in specifications.items():
        logits, targets = [], []
        for view_name in ("base", "invariant", "semantic"):
            mask = masks[view_name]
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ValueError(f"{view_name} mask must be a BoolTensor")
            output = view_outputs[view_name]
            if output_name not in output:
                raise KeyError(f"missing {output_name} for {view_name}")
            target_name = "w_ids" if loss_name == "proxy" else "labels"
            target = paired_batch[view_name][target_name]  # type: ignore[index]
            if bool(mask.any()):
                logits.append(output[output_name][mask])
                targets.append(target[mask])
        count = sum(item.size(0) for item in targets)
        anchor = view_outputs["base"][output_name]
        value = F.cross_entropy(torch.cat(logits), torch.cat(targets)) if count else anchor.sum() * 0.0
        result[loss_name] = value
        result[f"{loss_name}_count"] = count
        result[f"{loss_name}_valid"] = count > 0

    bridge_values = []
    bridge_count = 0
    for view_name in ("base", "invariant", "semantic"):
        mask = masks[view_name]
        if bool(mask.any()):
            output = view_outputs[view_name]
            bridge_values.append(bridge_alignment_loss(
                F.softmax(output["g_logits"][mask].float(), dim=-1),
                F.softmax(output["q_logits"][mask].float(), dim=-1),
                F.softmax(output["h_all_w_logits"][mask].float(), dim=-1),
            ) * int(mask.sum()))
            bridge_count += int(mask.sum())
    anchor = view_outputs["base"]["g_logits"]
    result["bridge"] = sum(bridge_values) / bridge_count if bridge_count else anchor.sum() * 0.0
    result["bridge_count"] = bridge_count
    result["bridge_valid"] = bridge_count > 0
    return result


def compute_v2_losses(
    view_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    paired_batch: Mapping[str, object],
    weights: V2LossWeights,
    semantic_margin: float = 0.5,
) -> Dict[str, object]:
    """Compute all eight independently weighted V2 objectives."""

    result = compute_supervised_losses(view_outputs, paired_batch, weights)
    base_probs = view_outputs["base"]["std_logits"].float().exp()
    inv_probs = view_outputs["invariant"]["std_logits"].float().exp()
    inv_mask = paired_batch["inv_mask"]
    sem_mask = paired_batch["sem_mask"]
    result["invariance"] = invariant_js(base_probs, inv_probs, inv_mask)  # type: ignore[arg-type]
    result["invariance_count"] = int(inv_mask.sum())  # type: ignore[union-attr]
    result["invariance_valid"] = bool(inv_mask.any())  # type: ignore[union-attr]
    result["semantic"] = semantic_margin_loss(
        view_outputs["semantic"]["std_logits"],
        paired_batch["old_labels"],  # type: ignore[arg-type]
        paired_batch["new_labels"],  # type: ignore[arg-type]
        sem_mask,  # type: ignore[arg-type]
        semantic_margin,
    )
    result["semantic_count"] = int(sem_mask.sum())  # type: ignore[union-attr]
    result["semantic_valid"] = bool(sem_mask.any())  # type: ignore[union-attr]
    total = (
        weights.std * result["std"]
        + weights.fact * result["fact"]
        + weights.g * result["g"]
        + weights.h * result["h"]
        + weights.proxy * result["proxy"]
        + weights.bridge * result["bridge"]
        + weights.invariance * result["invariance"]
        + weights.semantic * result["semantic"]
    )
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite V2 total loss")
    result["total"] = total
    return result


def _normalize_probs(probs: torch.Tensor, eps: float) -> torch.Tensor:
    result = probs.clamp_min(eps)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(eps)


def _validate_probability_pair(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> None:
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("probability tensors must have identical [B,C] shapes")
    if mask.dtype != torch.bool or mask.ndim != 1 or mask.size(0) != p.size(0):
        raise ValueError("mask must be bool with shape [B]")
