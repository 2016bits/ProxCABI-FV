import numpy as np
import pytest

from proxcabi.causal.discrete_bridge import DiscreteOutcomeBridge
from proxcabi.causal.errors import IdentificationError
from proxcabi.causal.synthetic import BinarySCMConfig, generate_binary_scm


def test_discrete_bridge_recovers_effect_and_uses_target_population():
    observed, oracle = generate_binary_scm(BinarySCMConfig(n=200_000, seed=13))
    estimator = DiscreteOutcomeBridge(min_cell=20, ridge=1e-6)
    estimator.fit(*(observed[k] for k in ("a", "z", "w", "l", "r")))
    result = estimator.estimate_ate(observed["w"], observed["l"])
    assert result.ate == pytest.approx(oracle["tau_true"], abs=0.06)
    assert estimator.diagnose()["all_required_matrices_full_column_rank"]
    # A treatment-conditioned target distribution would differ, so verify the API result
    # equals manual unconditional target standardization.
    manual = np.mean([
        estimator.predict_h(1, int(w), int(l)) - estimator.predict_h(0, int(w), int(l))
        for w, l in zip(observed["w"], observed["l"])
    ])
    assert result.ate == pytest.approx(manual)


def test_bridge_output_is_real_valued_not_probability():
    observed, _ = generate_binary_scm(BinarySCMConfig(n=100_000, seed=7, delta=-1.5))
    estimator = DiscreteOutcomeBridge().fit(*(observed[k] for k in ("a", "z", "w", "l", "r")))
    values = [estimator.predict_h(a, w, 0) for a in (0, 1) for w in (0, 1)]
    assert min(values) < 0


def test_uninformative_z_fails_rank_gate():
    rng = np.random.default_rng(2)
    n = 20_000
    u = rng.binomial(1, 0.5, n)
    a = rng.binomial(1, np.where(u == 1, 0.8, 0.2))
    z = np.zeros(n, dtype=int)
    w = np.bitwise_xor(u, rng.binomial(1, 0.15, n))
    r = 0.3 * a + 0.8 * u + rng.normal(0, 0.1, n)
    with pytest.raises(IdentificationError, match="rank"):
        DiscreteOutcomeBridge().fit(a, z, w, np.zeros(n, dtype=int), r)
