import numpy as np
import pytest

from proxcabi.causal.synthetic import BinarySCMConfig, generate_binary_scm, naive_difference


def test_scm_is_reproducible_and_separates_oracle():
    cfg = BinarySCMConfig(n=20_000, seed=13)
    observed1, oracle1 = generate_binary_scm(cfg)
    observed2, oracle2 = generate_binary_scm(cfg)
    assert set(observed1) == {"a", "z", "w", "l", "r"}
    assert "u" not in observed1
    assert all(np.array_equal(observed1[k], observed2[k]) for k in observed1)
    assert np.array_equal(oracle1["u"], oracle2["u"])
    assert oracle1["tau_true"] == pytest.approx(0.3)


def test_scm_default_probabilities_and_naive_confounding():
    observed, oracle = generate_binary_scm(BinarySCMConfig(n=100_000, seed=13))
    u, a = oracle["u"], observed["a"]
    assert observed["z"].min() == 0 and observed["z"].max() == 1
    assert np.mean(observed["z"] == u) == pytest.approx(0.85, abs=0.01)
    assert np.mean(observed["w"] == u) == pytest.approx(0.85, abs=0.01)
    assert a[u == 0].mean() == pytest.approx(0.2, abs=0.01)
    assert a[u == 1].mean() == pytest.approx(0.8, abs=0.01)
    assert naive_difference(observed) == pytest.approx(0.78, abs=0.02)


def test_violation_is_explicitly_marked():
    _, oracle = generate_binary_scm(BinarySCMConfig(n=100, direct_z_effect=0.2))
    assert oracle["assumption_status"] == "known_violated"
    assert oracle["assumptions_designed_valid"] is False
