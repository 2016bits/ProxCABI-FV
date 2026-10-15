"""Domain-specific failures for Plan A."""


class CausalError(RuntimeError):
    """Base error for causal pipeline failures."""


class AssumptionValidationError(CausalError):
    """Raised when an input violates a declared identification assumption."""


class IdentificationError(CausalError):
    """Raised when the observable bridge is not identified or is unstable."""


class InsufficientSupportError(IdentificationError):
    """Raised when required treatment/proxy/stratum cells lack support."""
