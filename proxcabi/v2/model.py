from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F

from proxcabi.model import MLP, _load_auto_model


class ProxCABIV2(nn.Module):
    """Four-head predictive-standardization verifier.

    Shapes use ``B`` for batch size, ``H`` for backbone width, ``K`` for W
    classes, and ``C`` for fact labels. ``forward`` returns ``m [B,H]``,
    ``fact_logits/g_logits/std_logits [B,C]``, ``q_logits [B,K]``,
    ``h_all_w_logits [B,K,C]``, optional ``h_obs_logits [B,C]``, and
    ``bridge_probs [B,C]``.
    """

    def __init__(
        self,
        backbone_name: Optional[str],
        num_labels: int,
        z_dim: int,
        num_w_classes: int,
        pi0: torch.Tensor,
        proxy_dim: int = 64,
        dropout: float = 0.1,
        *,
        encoder: Optional[nn.Module] = None,
        use_z_proxy: bool = True,
        use_w_proxy: bool = True,
    ) -> None:
        super().__init__()
        if num_labels < 2 or z_dim < 1 or num_w_classes < 1:
            raise ValueError("num_labels>=2, z_dim>=1, and num_w_classes>=1 are required")
        if encoder is None:
            if not backbone_name:
                raise ValueError("backbone_name is required when encoder is not supplied")
            encoder = _load_auto_model(backbone_name)
        if not hasattr(encoder, "config") or not hasattr(encoder.config, "hidden_size"):
            raise ValueError("encoder must expose config.hidden_size")
        hidden = int(encoder.config.hidden_size)
        reference = torch.as_tensor(pi0, dtype=torch.float32).flatten()
        if reference.numel() != num_w_classes:
            raise ValueError("pi0 length must equal num_w_classes")
        if not torch.isfinite(reference).all() or (reference < 0).any() or reference.sum() <= 0:
            raise ValueError("pi0 must be finite, non-negative, and have positive mass")

        self.encoder = encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, proxy_dim),
            nn.GELU(),
            nn.LayerNorm(proxy_dim),
        )
        self.w_embeddings = nn.Embedding(num_w_classes, proxy_dim)
        self.fact_head = MLP(hidden, hidden, num_labels, dropout)
        self.proxy_head = MLP(hidden + proxy_dim, hidden, num_w_classes, dropout)
        self.g_head = MLP(hidden + proxy_dim, hidden, num_labels, dropout)
        self.h_head = MLP(hidden + proxy_dim, hidden, num_labels, dropout)
        self.num_labels = num_labels
        self.z_dim = z_dim
        self.num_w_classes = num_w_classes
        self.use_z_proxy = use_z_proxy
        self.use_w_proxy = use_w_proxy
        self.register_buffer("pi0", reference / reference.sum(), persistent=True)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode token pairs into a pooled ``[B,H]`` representation."""

        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler = getattr(output, "pooler_output", None)
        if pooler is not None:
            return pooler
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)):
            hidden = output[0]
        if hidden is None:
            raise ValueError("encoder output needs pooler_output or last_hidden_state")
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def h_all_w_logits(self, m: torch.Tensor) -> torch.Tensor:
        """Enumerate conditional H logits for every W: ``[B,K,C]``."""

        batch_size = m.size(0)
        w_ids = torch.arange(self.num_w_classes, device=m.device)
        w_features = self.w_embeddings(w_ids)
        if not self.use_w_proxy:
            w_features = torch.zeros_like(w_features)
        m_all = m.unsqueeze(1).expand(batch_size, self.num_w_classes, m.size(-1))
        w_all = w_features.unsqueeze(0).expand(batch_size, self.num_w_classes, -1)
        return self.h_head(torch.cat((m_all, w_all), dim=-1))

    def standardized_probs(self, m: torch.Tensor) -> torch.Tensor:
        """Compute ``sum_w pi0(w) h(y|M,w)`` in float32."""

        h_probs = F.softmax(self.h_all_w_logits(m).float(), dim=-1)
        weights = self.pi0.float().to(m.device)
        probs = torch.einsum("w,bwc->bc", weights, h_probs)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def standardized_logits(self, m: torch.Tensor) -> torch.Tensor:
        """Return log standardized probabilities ``[B,C]``."""

        return self.standardized_probs(m).clamp_min(1e-8).log()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_features: torch.Tensor,
        w_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_features.ndim != 2 or z_features.size(-1) != self.z_dim:
            raise ValueError(f"z_features must have shape [B,{self.z_dim}]")
        m = self.encode(input_ids, attention_mask)
        z_encoded = self.z_encoder(z_features.to(m.dtype))
        if not self.use_z_proxy:
            z_encoded = torch.zeros_like(z_encoded)
        mz = torch.cat((m, z_encoded), dim=-1)
        fact_logits = self.fact_head(m)
        q_logits = self.proxy_head(mz)
        g_logits = self.g_head(mz)
        h_all_logits = self.h_all_w_logits(m)
        h_all_probs = F.softmax(h_all_logits.float(), dim=-1)
        q_probs = F.softmax(q_logits.float(), dim=-1)
        bridge_probs = torch.einsum("bw,bwc->bc", q_probs, h_all_probs)
        std_probs = torch.einsum("w,bwc->bc", self.pi0.float(), h_all_probs)
        std_probs = std_probs / std_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = {
            "m": m,
            "fact_logits": fact_logits,
            "q_logits": q_logits,
            "proxy_logits": q_logits,
            "g_logits": g_logits,
            "h_all_w_logits": h_all_logits,
            "std_logits": std_probs.clamp_min(1e-8).log(),
            "bridge_probs": bridge_probs,
        }
        if w_ids is not None:
            if w_ids.ndim != 1 or w_ids.size(0) != m.size(0):
                raise ValueError("w_ids must have shape [B]")
            if ((w_ids < 0) | (w_ids >= self.num_w_classes)).any():
                raise ValueError("w_ids contains an out-of-range category")
            w_features = self.w_embeddings(w_ids)
            if not self.use_w_proxy:
                w_features = torch.zeros_like(w_features)
            result["h_obs_logits"] = self.h_head(torch.cat((m, w_features), dim=-1))
        return result

    def forward_views(self, paired_batch: Mapping[str, object]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run all three aligned views from a ``PairedBatchCollator`` output."""

        result: Dict[str, Dict[str, torch.Tensor]] = {}
        for view_name in ("base", "invariant", "semantic"):
            view = paired_batch[view_name]
            if not isinstance(view, Mapping):
                raise TypeError(f"paired_batch[{view_name!r}] must be a mapping")
            result[view_name] = self(
                input_ids=view["input_ids"],  # type: ignore[arg-type]
                attention_mask=view["attention_mask"],  # type: ignore[arg-type]
                z_features=view["z_features"],  # type: ignore[arg-type]
                w_ids=view.get("w_ids"),  # type: ignore[arg-type]
            )
        return result
