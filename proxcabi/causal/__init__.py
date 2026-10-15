"""Plan A causal-effect estimation tools.

The strict estimators in this package are intentionally separate from the
legacy text-derived proxy model in :mod:`proxcabi.model`.
"""

from .discrete_bridge import DiscreteOutcomeBridge
from .errors import AssumptionValidationError, IdentificationError, InsufficientSupportError
from .schemas import CausalObservation, EffectEstimate
from .synthetic import BinarySCMConfig, generate_binary_scm

__all__ = [
    "AssumptionValidationError",
    "BinarySCMConfig",
    "CausalObservation",
    "DiscreteOutcomeBridge",
    "EffectEstimate",
    "IdentificationError",
    "InsufficientSupportError",
    "generate_binary_scm",
]
