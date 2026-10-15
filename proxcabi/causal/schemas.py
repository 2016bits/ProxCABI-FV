from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Mode = Literal["synthetic_strict", "real_strict", "real_exploratory"]


@dataclass(frozen=True)
class CausalObservation:
    row_id: str
    split: str
    treatment: int
    z: int
    w: int
    l: int
    response: float
    sample_id: str = ""
    semantic_group: str = ""
    original_label: int | None = None
    claim: str = ""
    evidence: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.treatment not in (0, 1):
            raise ValueError("treatment must be 0 or 1")


@dataclass(frozen=True)
class EffectEstimate:
    ate: float
    mu0: float
    mu1: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    method: str
    passed_identification_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
