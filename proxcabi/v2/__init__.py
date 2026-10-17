"""Semantic-sensitive ProxCABI V2 components.

The V2 package is intentionally separate from the legacy training stack.  Its
objects can therefore evolve without changing existing checkpoints or CLI
commands.
"""

from .schemas import CounterfactualRecord, PairSample
from .structured_proxies import StructuredProxyBuilder

__all__ = ["CounterfactualRecord", "PairSample", "StructuredProxyBuilder"]
