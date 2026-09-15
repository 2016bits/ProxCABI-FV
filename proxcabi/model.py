from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="none").sum(dim=-1)
        + F.kl_div(m.log(), q, reduction="none").sum(dim=-1)
    )


@dataclass
class LossWeights:
    fact: float = 1.0
    proxy: float = 1.0
    bridge: float = 0.2
    g: float = 1.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProxCABIModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        z_buckets: int,
        w_buckets: int,
        proxy_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = int(self.encoder.config.hidden_size)
        self.z_embeddings = nn.Embedding(z_buckets, proxy_dim)
        self.w_embeddings = nn.Embedding(w_buckets, proxy_dim)
        self.fact_head = MLP(hidden, hidden, num_labels, dropout)
        self.proxy_head = MLP(hidden + proxy_dim, hidden, w_buckets, dropout)
        self.g_head = MLP(hidden + proxy_dim, hidden, num_labels, dropout)
        self.h_head = MLP(hidden + proxy_dim, hidden, num_labels, dropout)
        self.num_labels = num_labels
        self.w_buckets = w_buckets

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        summed = (output.last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def h_all_w(self, m: torch.Tensor) -> torch.Tensor:
        batch = m.size(0)
        w_ids = torch.arange(self.w_buckets, device=m.device)
        w_emb = self.w_embeddings(w_ids)
        m_rep = m.unsqueeze(1).expand(batch, self.w_buckets, m.size(-1))
        w_rep = w_emb.unsqueeze(0).expand(batch, self.w_buckets, w_emb.size(-1))
        h_in = torch.cat([m_rep, w_rep], dim=-1)
        return self.h_head(h_in)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_ids: torch.Tensor,
        w_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[LossWeights] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        m = self.encode(input_ids, attention_mask)
        z_emb = self.z_embeddings(z_ids)
        z_in = torch.cat([m, z_emb], dim=-1)
        fact_logits = self.fact_head(m)
        proxy_logits = self.proxy_head(z_in)
        g_logits = self.g_head(z_in)

        out: Dict[str, torch.Tensor] = {
            "m": m,
            "fact_logits": fact_logits,
            "proxy_logits": proxy_logits,
            "g_logits": g_logits,
        }
        if w_ids is not None:
            w_emb = self.w_embeddings(w_ids)
            out["h_logits"] = self.h_head(torch.cat([m, w_emb], dim=-1))

        if labels is not None and w_ids is not None:
            weights = loss_weights or LossWeights()
            fact_loss = F.cross_entropy(fact_logits, labels, weight=label_weights)
            g_loss = F.cross_entropy(g_logits, labels, weight=label_weights)
            proxy_loss = F.cross_entropy(proxy_logits, w_ids)
            q_w = F.softmax(proxy_logits, dim=-1)
            h_probs = F.softmax(self.h_all_w(m), dim=-1)
            bridge_probs = torch.einsum("bw,bwc->bc", q_w, h_probs)
            g_probs = F.softmax(g_logits, dim=-1)
            bridge_loss = js_divergence(g_probs, bridge_probs).mean()
            total = (
                weights.fact * fact_loss
                + weights.g * g_loss
                + weights.proxy * proxy_loss
                + weights.bridge * bridge_loss
            )
            out.update(
                {
                    "loss": total,
                    "fact_loss": fact_loss.detach(),
                    "g_loss": g_loss.detach(),
                    "proxy_loss": proxy_loss.detach(),
                    "bridge_loss": bridge_loss.detach(),
                    "bridge_probs": bridge_probs,
                }
            )
        return out

    def proximal_logits(self, m: torch.Tensor, w_marginal: torch.Tensor) -> torch.Tensor:
        h_probs = F.softmax(self.h_all_w(m), dim=-1)
        weights = w_marginal.to(m.device, h_probs.dtype)
        weights = weights / weights.sum().clamp_min(1e-8)
        probs = torch.einsum("w,bwc->bc", weights, h_probs)
        return torch.log(probs.clamp_min(1e-8))
