from __future__ import annotations

import torch

from proxcabi.v2.losses import invariant_js, semantic_margin_loss


def test_invariant_identical_and_different_predictions() -> None:
    original = torch.tensor([[0.8, 0.1, 0.1], [0.3, 0.3, 0.4]], requires_grad=True)
    identical = original.clone()
    changed = torch.tensor([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]], requires_grad=True)
    mask = torch.tensor([True, False])

    assert torch.allclose(invariant_js(original, identical, mask), torch.tensor(0.0), atol=1e-7)
    loss = invariant_js(original, changed, mask)
    assert loss > 0
    loss.backward()
    assert original.grad is not None and torch.isfinite(original.grad).all()


def test_semantic_margin_direction_and_zero_case() -> None:
    good = torch.log(torch.tensor([[0.05, 0.90, 0.05]], requires_grad=True))
    bad = torch.log(torch.tensor([[0.80, 0.10, 0.10]], requires_grad=True))
    old, new, mask = torch.tensor([0]), torch.tensor([1]), torch.tensor([True])
    assert semantic_margin_loss(good, old, new, mask, margin=0.5).item() == 0.0
    loss = semantic_margin_loss(bad, old, new, mask, margin=0.5)
    assert loss > 0
    loss.backward()
    assert bad.grad_fn is not None


def test_semantic_equal_labels_are_rejected() -> None:
    with torch.no_grad():
        logp = torch.log(torch.tensor([[0.5, 0.5]]))
    try:
        semantic_margin_loss(logp, torch.tensor([0]), torch.tensor([0]), torch.tensor([True]), 0.5)
    except ValueError as error:
        assert "must change" in str(error)
    else:
        raise AssertionError("equal semantic labels were accepted")


def test_all_false_masks_return_graph_connected_zero() -> None:
    original = torch.softmax(torch.randn(3, 3, requires_grad=True), dim=-1)
    changed = torch.softmax(torch.randn(3, 3, requires_grad=True), dim=-1)
    mask = torch.zeros(3, dtype=torch.bool)
    inv = invariant_js(original, changed, mask)
    sem = semantic_margin_loss(changed.log(), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), mask, 0.5)
    total = inv + sem
    assert total.item() == 0.0 and torch.isfinite(total)
    total.backward()
