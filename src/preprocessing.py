"""
preprocessing.py — load raw sensor exports, merge accel + gyro, harmonize sampling rates,
trim, clean, and write labelled CSVs to data/train and data/test.

Functions to implement: load_recording, estimate_fs, resample_uniform, trim_edges, clean,
build_clean_dataset, build_test_file.

Raw layout expected:
    data/raw/train/<activity>_<NN>/Accelerometer.csv, Gyroscope.csv (+ Gravity.csv if recorded)
    data/raw/test/test_<NN>/...                  + test_<NN>_segments.csv (start_s, end_s, label)

Note: Sensor Logger's "Accelerometer" has gravity removed, so it carries no phone-orientation
information. Gravity / TotalAcceleration is what separates standing (held upright) from still
(flat on a table).

build_test_file turns a continuous test recording + its segments file into
data/test/test_<NN>_<name>.csv with a per-row 'label' column.
"""
TARGET_FS = 50.0  # Hz — common grid; human motion < 20 Hz, Nyquist 25 Hz
TRIM_SEC = 1.0    # s removed from each end (button-tap artefacts): 10 s recording -> 8 s clean
