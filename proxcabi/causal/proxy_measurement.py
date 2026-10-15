from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ProxyMeasurement(Protocol):
    def measure_z(self, metadata: dict[str, Any]) -> int: ...
    def measure_w(self, metadata: dict[str, Any]) -> int: ...


@dataclass(frozen=True)
class MetadataProxyMeasurement:
    """Read independent measurements only from explicit metadata fields."""
    z_key: str = "z"
    w_key: str = "w"

    def measure_z(self, metadata: dict[str, Any]) -> int:
        return int(metadata[self.z_key])

    def measure_w(self, metadata: dict[str, Any]) -> int:
        return int(metadata[self.w_key])
