"""Gaussian HMM with diagonal covariance, implemented from scratch in numpy (all maths in log space)."""
import numpy as np


class GaussianHMM:
    """K hidden states, D-dimensional observations, diagonal Gaussian emissions."""

    def __init__(self, n_states: int, n_features: int, var_floor: float = 1e-3, random_state=None):
        self.n_states = n_states
        self.n_features = n_features
        self.var_floor = var_floor
        self.rng = np.random.default_rng(random_state)
        self.startprob_ = np.full(n_states, 1.0 / n_states)
        self.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
        self.means_ = np.zeros((n_states, n_features))
        self.covars_ = np.ones((n_states, n_features))
        self.history_ = []
        self.converged_ = False
        self.n_iter_ = 0

    @staticmethod
    def logsumexp(a, axis=None):
        """log(sum(exp(a))) without overflow/underflow."""
        a = np.asarray(a, dtype=float)
        m = np.max(a, axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
        return np.squeeze(out, axis=axis) if axis is not None else out.item()

    def _log_start_trans(self):
        with np.errstate(divide="ignore"):
            return np.log(self.startprob_), np.log(self.transmat_)

    def log_emission(self, X):
        """log N(x_t | mu_k, diag(var_k)) for every t, k -> (T, K)."""
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (np.sum(np.log(2 * np.pi * self.covars_), axis=1)
                       + np.sum(diff ** 2 / self.covars_, axis=2))

    def _forward(self, log_b, log_pi, log_a):
        T = len(log_b)
        log_alpha = np.empty((T, self.n_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            # alpha_t(j) = b_j(x_t) * sum_i alpha_{t-1}(i) a_ij
            log_alpha[t] = log_b[t] + self.logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
        return log_alpha, float(self.logsumexp(log_alpha[-1], axis=0))

    def _backward(self, log_b, log_a):
        T = len(log_b)
        log_beta = np.zeros((T, self.n_states))
        for t in range(T - 2, -1, -1):
            # beta_t(i) = sum_j a_ij b_j(x_{t+1}) beta_{t+1}(j)
            log_beta[t] = self.logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def forward(self, X):
        """Returns (log_alpha (T, K), log P(X))."""
        log_pi, log_a = self._log_start_trans()
        return self._forward(self.log_emission(X), log_pi, log_a)

    def backward(self, X):
        """Returns log_beta (T, K)."""
        _, log_a = self._log_start_trans()
        return self._backward(self.log_emission(X), log_a)

    def posteriors(self, X):
        """gamma_t(k) = P(z_t = k | X) -> (T, K)."""
        log_alpha, ll = self.forward(X)
        return np.exp(log_alpha + self.backward(X) - ll)

    def score(self, X):
        """log P(X)."""
        return self.forward(X)[1]

    def viterbi(self, X):
        """Most probable state path and its log probability."""
        log_pi, log_a = self._log_start_trans()
        log_b = self.log_emission(X)
        T = len(log_b)
        delta = np.empty((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, T):
            # delta_t(j) = max_i [delta_{t-1}(i) + log a_ij] + log b_j(x_t)
            cand = delta[t - 1][:, None] + log_a
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = cand[psi[t], np.arange(self.n_states)] + log_b[t]
        path = np.empty(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1, path[-1]])

    def fit_supervised(self, sequences, label_sequences, alpha: float = 1.0):
        """Estimate pi, A (add-alpha smoothed counts) and per-state mean/variance from labelled sequences."""
        K = self.n_states
        start = np.full(K, alpha)
        trans = np.full((K, K), alpha)
        for z in label_sequences:
            z = np.asarray(z, dtype=int)
            start[z[0]] += 1
            np.add.at(trans, (z[:-1], z[1:]), 1)
        self.startprob_ = start / start.sum()
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)

        X = np.vstack([np.asarray(x, dtype=float) for x in sequences])
        z = np.concatenate([np.asarray(z, dtype=int) for z in label_sequences])
        for k in range(K):
            xk = X[z == k]
            if len(xk):
                self.means_[k] = xk.mean(axis=0)
                self.covars_[k] = np.maximum(xk.var(axis=0), self.var_floor)
        return self

    def fit(self, sequences, tol: float = 1e-4, max_iter: int = 500, verbose: bool = False):
        """Baum-Welch (EM) over multiple sequences, starting from the current parameters.

        Stops when |LL_new - LL_old| < tol * max(1, |LL_old|); max_iter is only a safety cap.
        """
        K, D = self.n_states, self.n_features
        self.history_, self.converged_ = [], False
        for it in range(1, max_iter + 1):
            log_pi, log_a = self._log_start_trans()
            start, trans = np.zeros(K), np.zeros((K, K))
            g_sum, x_sum, x2_sum = np.zeros(K), np.zeros((K, D)), np.zeros((K, D))
            total_ll = 0.0

            # E-step
            for X in sequences:
                X = np.asarray(X, dtype=float)
                log_b = self.log_emission(X)
                log_alpha, ll = self._forward(log_b, log_pi, log_a)
                log_beta = self._backward(log_b, log_a)
                gamma = np.exp(log_alpha + log_beta - ll)
                if len(X) > 1:
                    # xi_t(i, j) = alpha_t(i) a_ij b_j(x_{t+1}) beta_{t+1}(j) / P(X)
                    log_xi = (log_alpha[:-1, :, None] + log_a[None]
                              + (log_b[1:] + log_beta[1:])[:, None, :] - ll)
                    trans += np.exp(log_xi).sum(axis=0)
                start += gamma[0]
                g_sum += gamma.sum(axis=0)
                x_sum += gamma.T @ X
                x2_sum += gamma.T @ X ** 2
                total_ll += ll

            self.history_.append(total_ll)
            self.n_iter_ = it
            if verbose:
                print(f"iter {it:3d}  log-likelihood {total_ll:.4f}")
            if it > 1 and abs(total_ll - self.history_[-2]) < tol * max(1.0, abs(self.history_[-2])):
                self.converged_ = True
                break

            # M-step
            self.startprob_ = start / start.sum()
            row = trans.sum(axis=1, keepdims=True)
            self.transmat_ = np.where(row > 0, trans / np.where(row > 0, row, 1), self.transmat_)
            seen = g_sum > 0
            self.means_[seen] = x_sum[seen] / g_sum[seen, None]
            var = x2_sum[seen] / g_sum[seen, None] - self.means_[seen] ** 2
            self.covars_[seen] = np.maximum(var, self.var_floor)
        return self
