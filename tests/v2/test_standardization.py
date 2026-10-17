from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from proxcabi.v2.model import ProxCABIV2


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids), pooler_output=None)


def _model(pi0: torch.Tensor | None = None, use_w_proxy: bool = True) -> ProxCABIV2:
    return ProxCABIV2(
        backbone_name=None,
        num_labels=3,
        z_dim=5,
        num_w_classes=2,
        pi0=torch.tensor([0.25, 0.75]) if pi0 is None else pi0,
        proxy_dim=4,
        dropout=0.0,
        encoder=TinyEncoder(),
        use_w_proxy=use_w_proxy,
    )


def test_standardization_shape_normalization_and_exact_weighting() -> None:
    model = _model()
    logits = torch.log(torch.tensor([
        [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]],
        [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]],
    ]))
    model.h_all_w_logits = lambda m: logits.to(m.device)  # type: ignore[method-assign]
    probs = model.standardized_probs(torch.zeros(2, 8))
    expected = 0.25 * logits.exp()[:, 0] + 0.75 * logits.exp()[:, 1]

    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    assert torch.allclose(probs, expected, atol=1e-7)
    assert torch.isfinite(model.standardized_logits(torch.zeros(2, 8))).all()


def test_identical_h_is_invariant_to_pi0() -> None:
    model = _model(torch.tensor([0.99, 0.01]))
    one = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
    model.h_all_w_logits = lambda m: one.expand(m.size(0), -1, -1)  # type: ignore[method-assign]
    expected = torch.softmax(one[:, 0], dim=-1)
    assert torch.allclose(model.standardized_probs(torch.zeros(4, 8)), expected.expand(4, -1))


def test_pi0_is_a_frozen_persistent_buffer() -> None:
    model = _model()
    assert model.pi0.requires_grad is False
    assert "pi0" in dict(model.named_buffers())
    assert all(parameter is not model.pi0 for parameter in model.parameters())
    assert "pi0" in model.state_dict()


def test_checkpoint_round_trip_restores_standardized_prediction(tmp_path) -> None:
    torch.manual_seed(4)
    model = _model().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "z_features": torch.randn(2, 5),
        "w_ids": torch.tensor([0, 1]),
    }
    expected = model(**batch)["std_logits"]
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = _model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual = restored(**batch)["std_logits"]
    assert torch.equal(expected, actual)


def test_no_w_information_keeps_h_supervised_and_trainable() -> None:
    model = _model(use_w_proxy=False)
    output = model(
        input_ids=torch.tensor([[1, 2], [3, 4]]),
        attention_mask=torch.ones(2, 2, dtype=torch.long),
        z_features=torch.zeros(2, 5),
        w_ids=torch.tensor([0, 1]),
    )
    loss = torch.nn.functional.cross_entropy(output["h_obs_logits"], torch.tensor([0, 1]))
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.h_head.parameters())
