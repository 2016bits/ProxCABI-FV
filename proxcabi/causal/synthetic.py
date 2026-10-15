from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinarySCMConfig:
    n: int = 100_000
    seed: int = 13
    p_u: float = 0.5
    z_accuracy: float = 0.85
    w_accuracy: float = 0.85
    p_a_u0: float = 0.20
    p_a_u1: float = 0.80
    beta: float = 0.30
    delta: float = 0.80
    noise_std: float = 0.10
    direct_z_effect: float = 0.0
    direct_w_to_a: float = 0.0

    def validate(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        for name in ("p_u", "z_accuracy", "w_accuracy", "p_a_u0", "p_a_u1"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


def generate_binary_scm(cfg: BinarySCMConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate observed A/Z/W/L/R and a separately returned oracle record."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    u = rng.binomial(1, cfg.p_u, cfg.n).astype(np.int64)
    z_noise = rng.binomial(1, 1.0 - cfg.z_accuracy, cfg.n)
    w_noise = rng.binomial(1, 1.0 - cfg.w_accuracy, cfg.n)
    z = np.bitwise_xor(u, z_noise).astype(np.int64)
    w = np.bitwise_xor(u, w_noise).astype(np.int64)
    p_a = np.where(u == 1, cfg.p_a_u1, cfg.p_a_u0).astype(float)
    if cfg.direct_w_to_a != 0.0:
        p_a = np.clip(p_a + cfg.direct_w_to_a * (w - 0.5), 0.01, 0.99)
    a = rng.binomial(1, p_a).astype(np.int64)
    eps = rng.normal(0.0, cfg.noise_std, cfg.n)
    r = cfg.beta * a + cfg.delta * u + cfg.direct_z_effect * z + eps
    observed = {
        "a": a,
        "z": z,
        "w": w,
        "l": np.zeros(cfg.n, dtype=np.int64),
        "r": r.astype(np.float64),
    }
    valid = cfg.direct_z_effect == 0.0 and cfg.direct_w_to_a == 0.0
    oracle = {
        "u": u,
        "tau_true": float(cfg.beta),
        "settings": asdict(cfg),
        "assumptions_designed_valid": valid,
        "assumption_status": "designed_valid" if valid else "known_violated",
    }
    return observed, oracle


def naive_difference(observed: dict[str, np.ndarray]) -> float:
    a, r = observed["a"], observed["r"]
    if not np.any(a == 0) or not np.any(a == 1):
        raise ValueError("both treatment arms are required")
    return float(r[a == 1].mean() - r[a == 0].mean())
