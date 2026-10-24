"""
features.py — windowing, time- and frequency-domain features, Z-score normalizer.

Functions/classes to implement: make_windows, extract_features, build_feature_matrix,
ZScoreNormalizer (fit on TRAIN only).
"""
WINDOW_SEC = 2.0  # 100 samples at 50 Hz -> FFT resolution 0.5 Hz, ~3-4 gait/jump cycles
OVERLAP = 0.5     # 1 s hop = one HMM time step
