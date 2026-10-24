"""
hmm.py — Gaussian (diagonal-covariance) Hidden Markov Model implemented from scratch in numpy.

VERIFY WITH: python -m pytest tests/ -q
The interface below is required by tests/test_hmm_sanity.py.
"""
import numpy as np


class GaussianHMM:
    """K hidden states, D-dimensional observations, diagonal Gaussian emissions."""

    def __init__(self, n_states: int, n_features: int, var_floor: float = 1e-3, random_state=None):
        raise NotImplementedError

    # log_emission, forward, backward, posteriors, viterbi, score,
    # fit_supervised, fit (Baum–Welch with log-likelihood convergence check), logsumexp
