"""
Correctness tests for src/hmm.py (GaussianHMM, diagonal covariance).

They check the HMM maths against brute-force enumeration on tiny problems, so if these pass
you can state in the report that "forward and Viterbi were verified against exhaustive
enumeration of all state paths". Run from the repo root:

    python -m pytest tests/ -q

Interface expected (see 02_PROMPTS.md, Prompt 4):
    GaussianHMM(n_states, n_features, var_floor=1e-3, random_state=None)
    startprob_ (K,), transmat_ (K,K), means_ (K,D), covars_ (K,D)
    log_emission(X)->(T,K); forward(X)->(log_alpha, loglik); backward(X)->log_beta
    posteriors(X)->(T,K); viterbi(X)->(path, logprob); score(X)->float
    fit_supervised(seqs, label_seqs)->self; fit(seqs, tol, max_iter)->self
    history_, converged_, n_iter_
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hmm import GaussianHMM  # noqa: E402

K, D = 3, 2


def _log_gauss_diag(x, mu, var):
    return -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def _toy_model(seed=0):
    rng = np.random.default_rng(seed)
    m = GaussianHMM(n_states=K, n_features=D, random_state=seed)
    m.startprob_ = np.array([0.5, 0.3, 0.2])
    m.transmat_ = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.05, 0.25, 0.7]])
    m.means_ = np.array([[0.0, 0.0], [3.0, -1.0], [-2.0, 2.5]])
    m.covars_ = np.array([[1.0, 0.5], [0.7, 1.2], [1.5, 0.8]])
    return m, rng


def _sample(m, T, rng):
    z = np.zeros(T, dtype=int)
    X = np.zeros((T, D))
    z[0] = rng.choice(K, p=m.startprob_)
    for t in range(T):
        if t:
            z[t] = rng.choice(K, p=m.transmat_[z[t - 1]])
        X[t] = rng.normal(m.means_[z[t]], np.sqrt(m.covars_[z[t]]))
    return X, z


def _brute_force(m, X):
    """Log P(X) and the best path by enumerating all K^T paths."""
    T = len(X)
    logB = np.array([[_log_gauss_diag(X[t], m.means_[k], m.covars_[k]) for k in range(K)] for t in range(T)])
    best_lp, best_path, all_lp = -np.inf, None, []
    for path in itertools.product(range(K), repeat=T):
        lp = np.log(m.startprob_[path[0]]) + logB[0, path[0]]
        for t in range(1, T):
            lp += np.log(m.transmat_[path[t - 1], path[t]]) + logB[t, path[t]]
        all_lp.append(lp)
        if lp > best_lp:
            best_lp, best_path = lp, np.array(path)
    all_lp = np.array(all_lp)
    mx = all_lp.max()
    return mx + np.log(np.exp(all_lp - mx).sum()), best_path, best_lp, logB


def test_log_emission_matches_formula():
    m, rng = _toy_model()
    X, _ = _sample(m, 6, rng)
    *_, logB = _brute_force(m, X)
    np.testing.assert_allclose(m.log_emission(X), logB, rtol=1e-8, atol=1e-8)


def test_forward_equals_brute_force():
    m, rng = _toy_model(1)
    X, _ = _sample(m, 6, rng)
    ll_true, *_ = _brute_force(m, X)
    _, ll = m.forward(X)
    assert ll == pytest.approx(ll_true, rel=1e-8, abs=1e-8)
    assert m.score(X) == pytest.approx(ll_true, rel=1e-8, abs=1e-8)


def test_backward_consistent_with_forward():
    m, rng = _toy_model(2)
    X, _ = _sample(m, 8, rng)
    _, ll = m.forward(X)
    log_beta = m.backward(X)
    ll_b = np.log(np.sum(m.startprob_ * np.exp(m.log_emission(X)[0] + log_beta[0])))
    assert ll_b == pytest.approx(ll, rel=1e-8, abs=1e-8)


def test_viterbi_equals_brute_force():
    m, rng = _toy_model(3)
    X, _ = _sample(m, 6, rng)
    _, path_true, lp_true, _ = _brute_force(m, X)
    path, lp = m.viterbi(X)
    np.testing.assert_array_equal(np.asarray(path), path_true)
    assert lp == pytest.approx(lp_true, rel=1e-8, abs=1e-8)


def test_posteriors_are_distributions():
    m, rng = _toy_model(4)
    X, _ = _sample(m, 50, rng)
    g = m.posteriors(X)
    assert g.shape == (50, K)
    assert np.all(g >= -1e-12)
    np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-8)


def test_long_sequence_no_underflow():
    m, rng = _toy_model(5)
    X, _ = _sample(m, 5000, rng)
    _, ll = m.forward(X)
    path, lp = m.viterbi(X)
    assert np.isfinite(ll) and np.isfinite(lp)
    assert len(path) == 5000


def test_supervised_fit_recovers_parameters():
    m_true, rng = _toy_model(6)
    seqs, labs = zip(*[_sample(m_true, 300, rng) for _ in range(10)])
    m = GaussianHMM(n_states=K, n_features=D, random_state=0)
    m.fit_supervised(list(seqs), list(labs))
    np.testing.assert_allclose(m.means_, m_true.means_, atol=0.25)
    np.testing.assert_allclose(m.transmat_, m_true.transmat_, atol=0.08)
    np.testing.assert_allclose(m.transmat_.sum(axis=1), 1.0, atol=1e-8)
    assert m.startprob_.sum() == pytest.approx(1.0)


def test_baum_welch_monotonic_and_converges():
    m_true, rng = _toy_model(7)
    seqs, labs = zip(*[_sample(m_true, 200, rng) for _ in range(8)])
    m = GaussianHMM(n_states=K, n_features=D, random_state=0)
    m.fit_supervised(list(seqs), list(labs))
    m.means_ = m.means_ + 0.5  # perturb so EM has work to do
    m.fit(list(seqs), tol=1e-6, max_iter=500)
    h = np.asarray(m.history_)
    assert len(h) >= 2
    assert np.all(np.diff(h) >= -1e-6 * np.abs(h[:-1]).max()), "log-likelihood decreased during Baum–Welch"
    assert m.converged_, "did not meet the convergence criterion before max_iter"
    assert m.n_iter_ < 500
    np.testing.assert_allclose(m.transmat_.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(m.covars_ > 0)


def test_viterbi_decodes_well_separated_data():
    m, rng = _toy_model(8)
    m.means_ = m.means_ * 3  # very separable
    X, z = _sample(m, 400, rng)
    path, _ = m.viterbi(X)
    assert np.mean(np.asarray(path) == z) > 0.97
