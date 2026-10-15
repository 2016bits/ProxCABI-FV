import torch

from proxcabi.causal.moment_bridge import OutcomeBridgeNet


def test_outcome_bridge_is_real_valued_and_backpropagates():
    model = OutcomeBridgeNet(2, 2, 1, emb_dim=4, hidden_dim=8)
    value = model(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]))
    assert value.shape == (2,)
    value.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
