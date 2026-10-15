from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .errors import AssumptionValidationError


REQUIRED_PROXY_FIELDS = (
    "name", "measured_before_treatment", "derived_from_post_treatment_text",
    "directly_used_by_verifier", "provenance", "exclusion_argument",
)


def synthetic_assumptions(settings: dict[str, Any]) -> dict[str, Any]:
    violated = []
    if float(settings.get("direct_z_effect", 0.0)) != 0.0:
        violated.append("Z outcome-side exclusion restriction")
    if float(settings.get("direct_w_to_a", 0.0)) != 0.0:
        violated.append("W treatment-side exclusion restriction")
    status = "known_violated" if violated else "designed_valid"
    return {
        "mode": "synthetic_strict", "overall_status": status,
        "consistency": {"status": "designed_valid", "source": "binary SCM"},
        "latent_exchangeability": {"status": "designed_valid", "source": "binary SCM"},
        "proxy_exclusion": {"status": status, "known_violations": violated,
                            "source": "SCM structural equations"},
        "completeness": {"status": "untested", "note": "assessed through observable rank gate"},
        "warning": "Diagnostics do not prove unobservable identification assumptions.",
    }


def validate_proxy_protocol(protocol: dict[str, Any], mode: str) -> dict[str, Any]:
    """Validate provenance metadata; never treats metadata as proof of assumptions."""
    if mode == "real_exploratory":
        return {"mode": mode, "overall_status": "untested", "passed_gate": False,
                "estimate_kind": "exploratory"}
    if mode != "real_strict":
        raise AssumptionValidationError(f"unsupported real-data mode: {mode}")
    errors = []
    for proxy_name in ("z_proxy", "w_proxy"):
        proxy = protocol.get(proxy_name)
        if not isinstance(proxy, dict):
            errors.append(f"missing {proxy_name} protocol")
            continue
        for field in REQUIRED_PROXY_FIELDS:
            if field not in proxy or proxy[field] in (None, ""):
                errors.append(f"{proxy_name}.{field} is required")
        if proxy.get("measured_before_treatment") is not True:
            errors.append(f"{proxy_name} was not measured before treatment")
        if proxy.get("derived_from_post_treatment_text") is not False:
            errors.append(f"{proxy_name} is derived from post-treatment text")
        for field in ("provenance", "exclusion_argument"):
            value = proxy.get(field)
            if value and not Path(str(value)).exists():
                errors.append(f"{proxy_name}.{field} does not exist: {value}")
        if proxy.get("known_exclusion_violation") is True:
            errors.append(f"{proxy_name} has a known exclusion violation")
    if errors:
        raise AssumptionValidationError("real_strict gate failed: " + "; ".join(errors))
    return {
        "mode": mode, "overall_status": "argued", "passed_gate": True,
        "estimate_kind": "identified_under_assumptions",
        "warning": "Protocol and empirical diagnostics support, but do not prove, assumptions.",
    }


def validate_no_split_leakage(rows: Iterable[dict[str, Any]]) -> None:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = str(row.get("semantic_group", "")).strip()
        split = str(row.get("split", "")).strip()
        if not group or not split:
            raise AssumptionValidationError("semantic_group and split are required")
        seen[group].add(split)
    leaking = {group: sorted(splits) for group, splits in seen.items() if len(splits) > 1}
    if leaking:
        preview = list(leaking.items())[:5]
        raise AssumptionValidationError(f"semantic groups cross splits: {preview}")
