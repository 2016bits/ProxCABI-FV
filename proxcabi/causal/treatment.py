from __future__ import annotations

from typing import Protocol


class TreatmentStrategy(Protocol):
    name: str

    def apply(self, claim: str, evidence: str, a: int, *, seed: int) -> tuple[str, str]: ...

    def verify_label_preservation(self, before: tuple[str, str], after: tuple[str, str],
                                  label: int) -> bool: ...
