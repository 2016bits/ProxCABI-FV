from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import IdentificationError, InsufficientSupportError
from .schemas import EffectEstimate


@dataclass
class _BridgeCell:
    h: np.ndarray
    z_values: np.ndarray
    w_values: np.ndarray
    q: np.ndarray
    m: np.ndarray
    counts: np.ndarray
    rank: int
    min_singular_value: float
    condition_number: float
    train_moment_rmse: float


class DiscreteOutcomeBridge:
    """Finite-state outcome bridge with explicit rank/support gates."""

    def __init__(self, ridge: float = 1e-6, min_cell: int = 20,
                 min_singular_value: float = 1e-3) -> None:
        if ridge < 0 or min_cell < 1 or min_singular_value < 0:
            raise ValueError("invalid bridge hyperparameters")
        self.ridge = float(ridge)
        self.min_cell = int(min_cell)
        self.min_singular_value = float(min_singular_value)
        self.cells: dict[tuple[int, int], _BridgeCell] = {}
        self.w_values: np.ndarray | None = None
        self.l_values: np.ndarray | None = None
        self.n = 0
        self._diagnostics: dict[str, Any] = {}

    def fit(self, a: Any, z: Any, w: Any, l: Any, r: Any) -> "DiscreteOutcomeBridge":
        arrays = [_one_dim(x, name) for x, name in zip((a, z, w, l, r), "azwlr")]
        a_arr, z_arr, w_arr, l_arr, r_arr = arrays
        if len({len(x) for x in arrays}) != 1 or len(a_arr) == 0:
            raise ValueError("a, z, w, l, r must be non-empty arrays of equal length")
        if not np.isfinite(r_arr.astype(float)).all():
            raise ValueError("response contains non-finite values")
        if not set(np.unique(a_arr)).issubset({0, 1}):
            raise ValueError("treatment must be binary")
        self.w_values = np.unique(w_arr)
        self.l_values = np.unique(l_arr)
        self.n = len(a_arr)
        self.cells.clear()
        failures: list[str] = []
        for lv in self.l_values:
            l_mask = l_arr == lv
            for av in (0, 1):
                mask = l_mask & (a_arr == av)
                if not np.any(mask):
                    failures.append(f"missing treatment A={av} in L={lv}")
                    continue
                z_values = np.unique(z_arr[mask])
                counts = np.array([np.sum(mask & (z_arr == zv)) for zv in z_values], dtype=float)
                sparse = [(int(zv), int(n)) for zv, n in zip(z_values, counts) if n < self.min_cell]
                if sparse:
                    failures.append(f"L={lv}, A={av} has sparse Z cells {sparse}; min_cell={self.min_cell}")
                    continue
                q = np.zeros((len(z_values), len(self.w_values)), dtype=float)
                m = np.zeros(len(z_values), dtype=float)
                for j, zv in enumerate(z_values):
                    cell = mask & (z_arr == zv)
                    m[j] = np.mean(r_arr[cell].astype(float))
                    for k, wv in enumerate(self.w_values):
                        q[j, k] = np.mean(w_arr[cell] == wv)
                singular = np.linalg.svd(q, compute_uv=False)
                rank = int(np.linalg.matrix_rank(q))
                min_sv = float(singular[-1]) if singular.size else 0.0
                condition = float(singular[0] / singular[-1]) if min_sv > 0 else float("inf")
                if rank < len(self.w_values):
                    failures.append(
                        f"L={lv}, A={av} bridge rank {rank} < {len(self.w_values)} W states"
                    )
                    continue
                if min_sv < self.min_singular_value:
                    failures.append(
                        f"L={lv}, A={av} min singular value {min_sv:.6g} "
                        f"< threshold {self.min_singular_value:.6g}"
                    )
                    continue
                d = counts / counts.sum()
                lhs = q.T @ (d[:, None] * q) + self.ridge * np.eye(q.shape[1])
                rhs = q.T @ (d * m)
                try:
                    h = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError as exc:
                    failures.append(f"L={lv}, A={av} solve failed: {exc}")
                    continue
                residual = q @ h - m
                self.cells[(av, int(lv))] = _BridgeCell(
                    h=h, z_values=z_values, w_values=self.w_values, q=q, m=m,
                    counts=counts, rank=rank, min_singular_value=min_sv,
                    condition_number=condition,
                    train_moment_rmse=float(np.sqrt(np.average(residual ** 2, weights=d))),
                )
        self._diagnostics = {
            "n": self.n,
            "ridge": self.ridge,
            "min_cell": self.min_cell,
            "min_singular_value_threshold": self.min_singular_value,
            "all_required_matrices_full_column_rank": not failures,
            "passed_identification_gate": not failures,
            "failures": failures,
            "cells": {f"a{a}_l{l}": _cell_diagnostics(cell) for (a, l), cell in self.cells.items()},
        }
        if failures:
            self.cells.clear()
            if any("sparse" in item or "missing treatment" in item for item in failures):
                raise InsufficientSupportError("; ".join(failures))
            raise IdentificationError("; ".join(failures))
        return self

    def predict_h(self, a: int, w: int, l: int) -> float:
        self._check_fitted()
        try:
            cell = self.cells[(int(a), int(l))]
            index = int(np.flatnonzero(cell.w_values == w)[0])
        except (KeyError, IndexError) as exc:
            raise ValueError(f"unsupported (a={a}, w={w}, l={l})") from exc
        return float(cell.h[index])

    def estimate_ate(self, target_w: Any, target_l: Any) -> EffectEstimate:
        self._check_fitted()
        w_arr, l_arr = _one_dim(target_w, "target_w"), _one_dim(target_l, "target_l")
        if len(w_arr) != len(l_arr) or len(w_arr) == 0:
            raise ValueError("target_w and target_l must have equal non-zero length")
        pairs, counts = np.unique(np.column_stack((w_arr, l_arr)), axis=0, return_counts=True)
        weights = counts / counts.sum()
        mu = [
            float(sum(weight * self.predict_h(av, int(wv), int(lv))
                      for (wv, lv), weight in zip(pairs, weights)))
            for av in (0, 1)
        ]
        return EffectEstimate(
            ate=mu[1] - mu[0], mu0=mu[0], mu1=mu[1], se=None,
            ci_low=None, ci_high=None, n=len(w_arr), method="discrete_proximal",
            passed_identification_gate=True,
        )

    def estimate_cate(self, target_w: Any, target_l: Any) -> dict[int, EffectEstimate]:
        """Estimate effects within each supported pre-treatment L stratum."""
        w_arr, l_arr = _one_dim(target_w, "target_w"), _one_dim(target_l, "target_l")
        if len(w_arr) != len(l_arr):
            raise ValueError("target_w and target_l must have equal length")
        estimates = {}
        for lv in np.unique(l_arr):
            mask = l_arr == lv
            estimates[int(lv)] = self.estimate_ate(w_arr[mask], l_arr[mask])
        return estimates

    def heldout_moments(self, a: Any, z: Any, w: Any, l: Any, r: Any) -> dict[str, Any]:
        self._check_fitted()
        arrays = [_one_dim(x, name) for x, name in zip((a, z, w, l, r), "azwlr")]
        if len({len(x) for x in arrays}) != 1:
            raise ValueError("held-out arrays must have equal length")
        a_arr, z_arr, w_arr, l_arr, r_arr = arrays
        rows = []
        weighted_squares, total = 0.0, 0
        for (av, lv), cell in self.cells.items():
            for zv in cell.z_values:
                mask = (a_arr == av) & (l_arr == lv) & (z_arr == zv)
                n = int(mask.sum())
                if n == 0:
                    continue
                residual = np.array([
                    float(rv) - self.predict_h(int(av), int(wv), int(lv))
                    for rv, wv in zip(r_arr[mask], w_arr[mask])
                ])
                mean = float(residual.mean())
                rows.append({"a": int(av), "l": int(lv), "z": int(zv), "n": n, "mean_residual": mean})
                weighted_squares += n * mean ** 2
                total += n
        return {"conditional_moment_rmse": float(np.sqrt(weighted_squares / total)) if total else None,
                "covered_rows": total, "cells": rows}

    def diagnose(self) -> dict[str, Any]:
        return self._diagnostics.copy()

    def _check_fitted(self) -> None:
        if not self.cells:
            raise RuntimeError("bridge has not been successfully fitted")


def _one_dim(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _cell_diagnostics(cell: _BridgeCell) -> dict[str, Any]:
    return {
        "rank": cell.rank,
        "n_w_states": len(cell.w_values),
        "min_singular_value": cell.min_singular_value,
        "condition_number": cell.condition_number,
        "train_moment_rmse": cell.train_moment_rmse,
        "z_cell_counts": {str(int(z)): int(n) for z, n in zip(cell.z_values, cell.counts)},
        "h_range": [float(cell.h.min()), float(cell.h.max())],
    }
