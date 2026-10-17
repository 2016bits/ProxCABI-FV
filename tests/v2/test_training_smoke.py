from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from proxcabi.v2.losses import V2LossWeights, compute_v2_losses
from proxcabi.v2.model import ProxCABIV2


class LearnableEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(16, 12)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids), pooler_output=None)


def _batch() -> dict:
    labels = torch.tensor([0, 1, 2] * 4)
    ids = (labels + 1).unsqueeze(1).repeat(1, 2)
    view = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "z_features": torch.nn.functional.one_hot(labels, num_classes=3).float(),
        "w_ids": labels.remainder(2),
        "labels": labels,
    }
    return {
        "base": view,
        "invariant": view,
        "semantic": view,
        "inv_mask": torch.zeros(len(labels), dtype=torch.bool),
        "sem_mask": torch.zeros(len(labels), dtype=torch.bool),
        "old_labels": labels,
        "new_labels": labels,
    }


def test_tiny_base_only_overfit_and_h_gradient() -> None:
    torch.manual_seed(2)
    model = ProxCABIV2(
        None, 3, 3, 2, torch.tensor([0.5, 0.5]),
        proxy_dim=4, dropout=0.0, encoder=LearnableEncoder(),
    )
    batch = _batch()
    weights = V2LossWeights(invariance=0.0, semantic=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = None
    for _ in range(80):
        outputs = model.forward_views(batch)
        losses = compute_v2_losses(outputs, batch, weights)
        initial = float(losses["total"].detach()) if initial is None else initial
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
    final_outputs = model.forward_views(batch)
    final_loss = compute_v2_losses(final_outputs, batch, weights)["total"]
    predictions = final_outputs["base"]["std_logits"].argmax(dim=-1)

    assert float(final_loss.detach()) < initial * 0.1
    assert torch.equal(predictions, batch["old_labels"])
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.h_head.parameters())
