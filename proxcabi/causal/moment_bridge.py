from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .schemas import EffectEstimate


class OutcomeBridgeNet(nn.Module):
    """Real-valued h(A,W,L); deliberately has no probability output activation."""

    def __init__(self, n_a: int, n_w: int, n_l: int, emb_dim: int = 16,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        self.a_emb = nn.Embedding(n_a, emb_dim)
        self.w_emb = nn.Embedding(n_w, emb_dim)
        self.l_emb = nn.Embedding(n_l, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, a: torch.Tensor, w: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        value = torch.cat((self.a_emb(a), self.w_emb(w), self.l_emb(l)), dim=-1)
        return self.net(value).squeeze(-1)


class NeuralMomentBridge:
    """Fit h using A×Z×L conditional residual moments and held-out early stopping."""

    def __init__(self, *, epochs: int = 1000, lr: float = 1e-2, weight_decay: float = 1e-5,
                 patience: int = 100, seed: int = 13, emb_dim: int = 8,
                 hidden_dim: int = 32, device: str = "cpu") -> None:
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.patience, self.seed = patience, seed
        self.emb_dim, self.hidden_dim, self.device = emb_dim, hidden_dim, torch.device(device)
        self.model: OutcomeBridgeNet | None = None
        self.maps: dict[str, dict[int, int]] = {}
        self._diagnostics: dict[str, Any] = {}

    def fit(self, a: Any, z: Any, w: Any, l: Any, r: Any, *,
            val: tuple[Any, Any, Any, Any, Any] | None = None) -> "NeuralMomentBridge":
        torch.manual_seed(self.seed)
        arrays = [np.asarray(value) for value in (a, z, w, l, r)]
        if len({len(value) for value in arrays}) != 1 or not len(arrays[0]):
            raise ValueError("bridge arrays must be non-empty and equal length")
        self.maps = {name: {int(value): i for i, value in enumerate(np.unique(array))}
                     for name, array in zip(("a", "z", "w", "l"), arrays[:4])}
        self.model = OutcomeBridgeNet(len(self.maps["a"]), len(self.maps["w"]),
                                      len(self.maps["l"]), self.emb_dim, self.hidden_dim).to(self.device)
        train = self._tensorize(*arrays)
        valid = self._tensorize(*(val if val is not None else arrays))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss, best_state, stale, grad_max = float("inf"), None, 0, 0.0
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            loss, coverage = self._moment_loss(train)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0))
            grad_max = max(grad_max, grad_norm)
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_loss, val_coverage = self._moment_loss(valid)
            score = float(val_loss)
            if score < best_loss - 1e-9:
                best_loss, best_state, stale = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._diagnostics = {
            "epochs_completed": epoch + 1, "heldout_conditional_moment_mse": best_loss,
            "train_cell_coverage": coverage, "heldout_cell_coverage": val_coverage,
            "max_gradient_norm": grad_max, "output_is_real_valued": True,
            "warning": "A small empirical moment residual does not prove identification assumptions.",
        }
        return self

    def estimate_ate(self, target_w: Any, target_l: Any) -> EffectEstimate:
        self._check()
        w, l = np.asarray(target_w), np.asarray(target_l)
        if len(w) != len(l) or not len(w):
            raise ValueError("target arrays must be non-empty and equal length")
        mu = [float(np.mean(self.predict(np.full(len(w), av), w, l))) for av in (0, 1)]
        return EffectEstimate(mu[1] - mu[0], mu[0], mu[1], None, None, None, len(w),
                              "neural_moment_proximal", True)

    def predict(self, a: Any, w: Any, l: Any) -> np.ndarray:
        self._check()
        tensors = [torch.as_tensor(self._encode(name, np.asarray(value)), dtype=torch.long,
                                   device=self.device) for name, value in (("a", a), ("w", w), ("l", l))]
        with torch.no_grad():
            return self.model(*tensors).cpu().numpy()  # type: ignore[misc]

    def diagnose(self) -> dict[str, Any]:
        return dict(self._diagnostics)

    def _tensorize(self, a: Any, z: Any, w: Any, l: Any, r: Any):
        encoded = [torch.as_tensor(self._encode(name, np.asarray(value)), dtype=torch.long,
                                   device=self.device) for name, value in (("a", a), ("z", z), ("w", w), ("l", l))]
        return (*encoded, torch.as_tensor(np.asarray(r), dtype=torch.float32, device=self.device))

    def _encode(self, name: str, values: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self.maps[name][int(value)] for value in values], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"unseen {name} state: {exc.args[0]}") from exc

    def _moment_loss(self, values):
        a, z, w, l, r = values
        residual = r - self.model(a, w, l)  # type: ignore[misc]
        cell_id = (a * len(self.maps["z"]) + z) * len(self.maps["l"]) + l
        moments = []
        possible = len(self.maps["a"]) * len(self.maps["z"]) * len(self.maps["l"])
        for cell in torch.unique(cell_id):
            moments.append(residual[cell_id == cell].mean().square())
        return torch.stack(moments).mean(), len(moments) / possible

    def _check(self) -> None:
        if self.model is None:
            raise RuntimeError("neural bridge has not been fitted")
