import numpy as np

from proxcabi.causal.effect_guided import effect_guided_group_weights, sample_weights


def test_effect_guided_weights_change_training_weights():
    weights = effect_guided_group_weights({0: 0.1, 1: 0.8}, kappa=1.0, max_weight=4.0)
    samples = sample_weights(np.array([0, 1, 1]), weights)
    assert weights[1] > weights[0] >= 1.0
    assert samples[1] == samples[2] == weights[1]
    assert not np.allclose(samples, 1.0)
