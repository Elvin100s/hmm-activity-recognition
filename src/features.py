"""Windowing, time/frequency-domain features and a train-only Z-score normalizer.

cleaned CSV -> make_windows -> extract_features -> ZScoreNormalizer
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SEC = 2.0   # 100 samples at 50 Hz -> FFT resolution 0.5 Hz, >= 2 gait cycles, ~4 jumps
OVERLAP = 0.5      # 1 s hop = one HMM time step
MIN_LABEL_FRACTION = 0.8   # a window is scored only if >= 80 % of its samples share one label
EPS = 1e-12

ACC = ["acc_x", "acc_y", "acc_z"]
GYRO = ["gyro_x", "gyro_y", "gyro_z"]
GRAV = ["grav_x", "grav_y", "grav_z"]
CHANNELS = ACC + GYRO + GRAV          # 9 channels, this exact order is the feature contract

# frequency bands (Hz)
BAND_LO, BAND_HI = 0.3, 15.0          # everything human-motion-relevant, below Nyquist (25 Hz)
LOCO_LO, LOCO_HI = 0.8, 3.0           # locomotion band: walking cadence and jump rhythm

FEATURE_NAMES = [
    # time domain (intensity)
    "acc_mag_mean", "acc_mag_std", "acc_mag_sma", "acc_mag_peak", "acc_jerk_rms",
    "gyro_mag_rms", "gyro_mag_std", "corr_acc_xy",
    # orientation: yaw-invariant tilt (separates standing from still)
    "grav_tilt_y_deg", "grav_tilt_z_deg",
    # frequency domain: FFT of |a| and |gyro|
    "acc_dom_freq", "acc_spec_energy", "acc_loco_ratio", "acc_spec_entropy", "gyro_dom_freq",
]
N_FEATURES = len(FEATURE_NAMES)


# windowing
def window_samples(fs: float = 50.0, window_sec: float = WINDOW_SEC) -> int:
    """Number of samples in one window (100 at 50 Hz / 2 s)."""
    return int(round(window_sec * fs))


def window_hop(fs: float = 50.0, overlap: float = OVERLAP) -> int:
    """Samples between consecutive window starts (50 at 50 Hz / 50 % overlap = 1 s hop)."""
    return max(1, int(round(window_samples(fs, WINDOW_SEC) * (1.0 - overlap))))


def make_windows(x: np.ndarray, fs: float = 50.0, window_sec: float = WINDOW_SEC,
                 overlap: float = OVERLAP) -> np.ndarray:
    """(T, C) signal -> (n_windows, window_samples, C). A partial last window is dropped, not zero-padded."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, hop = window_samples(fs, window_sec), window_hop(fs, overlap)
    if len(x) < n:
        return np.empty((0, n, x.shape[1]))
    starts = range(0, len(x) - n + 1, hop)
    return np.stack([x[s:s + n] for s in starts])


def window_start_times(n_windows: int, fs: float = 50.0, window_sec: float = WINDOW_SEC,
                       overlap: float = OVERLAP) -> np.ndarray:
    """Start time (s) of each window, i.e. the HMM time step of that observation."""
    hop = window_hop(fs, overlap) / fs
    return np.arange(n_windows) * hop


# features
def _spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of a mean-removed 1-D signal, restricted to [BAND_LO, BAND_HI] Hz."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return np.array([]), np.array([])
    freq = np.fft.rfftfreq(len(x), 1.0 / fs)
    power = (np.abs(np.fft.rfft(x)) / len(x)) ** 2
    keep = (freq >= BAND_LO) & (freq <= BAND_HI)
    return freq[keep], power[keep]


def _dominant_frequency(x: np.ndarray, fs: float) -> float:
    freq, power = _spectrum(x, fs)
    return float(freq[np.argmax(power)]) if len(freq) else 0.0


def _spectral_energy(x: np.ndarray, fs: float) -> float:
    _, power = _spectrum(x, fs)
    return float(np.log10(power.sum() + EPS))


def _loco_ratio(x: np.ndarray, fs: float) -> float:
    freq, power = _spectrum(x, fs)
    if not len(freq):
        return 0.0
    total = power.sum()
    return float(power[(freq >= LOCO_LO) & (freq <= LOCO_HI)].sum() / (total + EPS))


def _spectral_entropy(x: np.ndarray, fs: float) -> float:
    freq, power = _spectrum(x, fs)
    if not len(freq):
        return 0.0
    p = power / (power.sum() + EPS)
    h = -np.sum(p * np.log(p + EPS))
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0.0 when either axis is constant (avoids NaNs in 'still')."""
    if a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _tilt_deg(g_bar: np.ndarray, axis: int) -> float:
    """Unsigned angle (deg) between mean gravity and one phone axis; invariant to yaw."""
    norm = float(np.linalg.norm(g_bar))
    if norm < EPS:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(abs(g_bar[axis]) / norm, 0.0, 1.0))))


def extract_features(window: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """One window (N, 9) in CHANNELS order -> feature vector (N_FEATURES,)."""
    w = np.asarray(window, dtype=float)
    if w.ndim != 2 or w.shape[1] != len(CHANNELS):
        raise ValueError(f"expected (N, {len(CHANNELS)}) window in CHANNELS order, got {w.shape}")
    a, gy, gr = w[:, 0:3], w[:, 3:6], w[:, 6:9]
    a_mag, g_mag = np.linalg.norm(a, axis=1), np.linalg.norm(gy, axis=1)
    g_bar = gr.mean(axis=0)
    return np.array([
        a_mag.mean(),
        a_mag.std(),
        np.abs(a).sum(axis=1).mean(),
        a_mag.max(),
        np.sqrt(np.mean(np.diff(a_mag) ** 2)) if len(a_mag) > 1 else 0.0,
        np.sqrt(np.mean(g_mag ** 2)),
        g_mag.std(),
        _safe_corr(a[:, 0], a[:, 1]),
        _tilt_deg(g_bar, 1),
        _tilt_deg(g_bar, 2),
        _dominant_frequency(a_mag, fs),
        _spectral_energy(a_mag, fs),
        _loco_ratio(a_mag, fs),
        _spectral_entropy(a_mag, fs),
        _dominant_frequency(g_mag, fs),
    ], dtype=float)


def extract_feature_windows(X: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """(n_windows, N, 9) -> (n_windows, N_FEATURES). Empty input -> (0, N_FEATURES)."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0:
        return np.empty((0, N_FEATURES))
    return np.stack([extract_features(w, fs) for w in X])


def read_signal(df: pd.DataFrame) -> np.ndarray:
    """DataFrame -> (T, 9) float array in CHANNELS order."""
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    return df[CHANNELS].to_numpy(dtype=float)


# feature matrices for the two data layouts
def build_feature_matrix(files: list[Path] | list[str], states: list[str], fs: float = 50.0) -> tuple:
    """Labelled clips (activity = folder name) -> (X, y, groups = file index, names)."""
    X_rows, y_rows, groups = [], [], []
    for gi, path in enumerate(files):
        df = pd.read_csv(path)
        state = Path(path).parent.name
        if state not in states:
            raise ValueError(f"{path}: folder '{state}' is not in {states}")
        W = make_windows(read_signal(df), fs)
        if not len(W):
            continue
        X_rows.append(extract_feature_windows(W, fs))
        y_rows.append(np.full(len(W), states.index(state), dtype=int))
        groups.append(np.full(len(W), gi, dtype=int))
    if not X_rows:
        return np.empty((0, N_FEATURES)), np.empty(0, dtype=int), np.empty(0, dtype=int), list(FEATURE_NAMES)
    return (np.vstack(X_rows), np.concatenate(y_rows), np.concatenate(groups), list(FEATURE_NAMES))


def continuous_windows(df: pd.DataFrame, states: list[str], fs: float = 50.0,
                       min_label_fraction: float = MIN_LABEL_FRACTION) -> dict:
    """Continuous test recording -> features of every window (kept contiguous for Viterbi) and
    per-window truth: y = -1 and scored = False unless >= min_label_fraction of samples share a label.
    Returns dict with X, y, scored, frac, starts, raw."""
    raw = read_signal(df)
    W = make_windows(raw, fs)
    n = len(W)
    starts = window_start_times(n, fs)
    n_samp = W.shape[1]

    if "label" not in df.columns:
        raise ValueError("continuous test file needs a per-row 'label' column")
    labels = df["label"].astype(object).to_numpy()

    X = extract_feature_windows(W, fs)
    y = np.full(n, -1, dtype=int)
    frac = np.zeros(n)
    for i in range(n):
        seg = labels[i * window_hop(fs): i * window_hop(fs) + n_samp]
        vals, counts = np.unique(seg[pd.notna(seg)], return_counts=True)
        if not len(vals):
            continue
        j = int(np.argmax(counts))
        frac[i] = counts[j] / n_samp
        if vals[j] in states:
            y[i] = states.index(vals[j])
    scored = (y >= 0) & (frac >= min_label_fraction)
    return {"X": X, "y": y, "scored": scored, "frac": frac, "starts": starts, "raw": raw}


# normalization
class ZScoreNormalizer:
    """Z-score with mean/std from the TRAINING windows only (features have mixed units and the
    diagonal Gaussian is scale-sensitive; min-max would be set by one extreme jump impact)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreNormalizer":
        X = np.asarray(X, dtype=float)
        if not len(X):
            raise ValueError("cannot fit ZScoreNormalizer on zero windows")
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("call fit() before transform()")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_
