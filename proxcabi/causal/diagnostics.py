from __future__ import annotations

from typing import Any

import numpy as np


def observable_support_diagnostics(a: Any, z: Any, w: Any, l: Any) -> dict[str, Any]:
    """Describe observable support; this is not a test of latent positivity."""
    arrays = [np.asarray(value) for value in (a, z, w, l)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("diagnostic arrays must have equal length")
    a_arr, z_arr, w_arr, l_arr = arrays
    strata = {}
    warnings = []
    for lv in np.unique(l_arr):
        mask = l_arr == lv
        p = float(np.mean(a_arr[mask] == 1))
        entry = {"n": int(mask.sum()), "p_a1": p,
                 "z_states": int(len(np.unique(z_arr[mask]))),
                 "w_states": int(len(np.unique(w_arr[mask])))}
        strata[str(int(lv))] = entry
        if p < 0.05 or p > 0.95:
            warnings.append(f"L={lv} has scarce observable treatment overlap (P(A=1)={p:.4f})")
    return {"strata": strata, "warnings": warnings,
            "note": "Observable overlap cannot establish positivity conditional on latent U."}
