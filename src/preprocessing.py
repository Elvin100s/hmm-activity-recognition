"""Load Sensor Logger exports, merge sensors, harmonize to a uniform rate, trim, clean, write labelled CSVs.

Raw layout:
    data/raw/train/<activity>/<Prefix>_<N>-<timestamp>.zip   (Prefix: Still | Stand | Jump | Walk)
    data/raw/test/Test_<N>-<timestamp>.zip + Test_<N>_segments.csv  (start_s, end_s, label)

Sensor Logger's "Accelerometer" has gravity removed, so the Gravity sensor is kept as its own
columns: it carries the phone orientation that separates standing (held upright) from still (flat).
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_FS = 50.0      # Hz: <= native rate, Nyquist 25 Hz > human motion (< 20 Hz)
CUTOFF_HZ = 20.0      # anti-alias low-pass applied before downsampling
TRIM_SEC = 1.0        # removed from each end (button taps)
MAX_SEC = 10.0        # brief: 5-10 s per training file
ACTIVITIES = ["standing", "walking", "jumping", "still"]
PREFIX_TO_ACTIVITY = {"stand": "standing", "walk": "walking", "jump": "jumping", "still": "still"}
SENSORS = {"Accelerometer": "acc", "Gyroscope": "gyro", "Gravity": "grav"}
COLUMNS = ["timestamp"] + [f"{p}_{a}" for p in SENSORS.values() for a in "xyz"]

# (start, end) seconds to trim instead of TRIM_SEC: the phone was still settling after Start was tapped
TRIM_OVERRIDES = {("still", 11): (3.0, TRIM_SEC), ("still", 13): (2.0, TRIM_SEC)}

RAW_NAME = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<num>\d+)-(?P<stamp>[\d_-]+)\.zip$")


def _read_members(source: Path) -> dict[str, pd.DataFrame | str]:
    """Read the sensor CSVs and Metadata.csv from a zip or an unzipped folder."""
    out = {}
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf:
            names = set(zf.namelist())
            for s in SENSORS:
                out[s] = pd.read_csv(io.BytesIO(zf.read(f"{s}.csv")))
            if "Metadata.csv" in names:
                out["meta"] = zf.read("Metadata.csv").decode()
    else:
        for s in SENSORS:
            out[s] = pd.read_csv(source / f"{s}.csv")
        if (source / "Metadata.csv").exists():
            out["meta"] = (source / "Metadata.csv").read_text()
    return out


def device_name(meta: str | None) -> str:
    """Lowercase alphanumeric device token from Metadata.csv, e.g. 'TECNO BG6' -> 'tecnobg6'."""
    if not meta:
        return "unknown"
    row = pd.read_csv(io.StringIO(meta)).iloc[0]
    return re.sub(r"[^a-z0-9]", "", str(row.get("device name", "unknown")).lower()) or "unknown"


def estimate_fs(timestamps: np.ndarray) -> float:
    """Sampling rate as 1 / median(diff(t)); robust to the odd late sample."""
    dt = np.diff(np.asarray(timestamps, dtype=float))
    return float(1.0 / np.median(dt[dt > 0]))


def load_recording(source: Path) -> tuple[pd.DataFrame, dict]:
    """One recording -> DataFrame on the accelerometer clock with acc/gyro/grav xyz, plus info."""
    m = _read_members(Path(source))
    acc = m["Accelerometer"].sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")
    t = acc["seconds_elapsed"].to_numpy(dtype=float)
    df = pd.DataFrame({"timestamp": t})
    info = {"device": device_name(m.get("meta"))}
    for sensor, p in SENSORS.items():
        s = m[sensor].sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")
        ts = s["seconds_elapsed"].to_numpy(dtype=float)
        info[f"native_fs_{p}"] = estimate_fs(ts)
        for a in "xyz":
            # gyro and gravity interpolated onto the accelerometer timeline
            df[f"{p}_{a}"] = np.interp(t, ts, s[a].to_numpy(dtype=float))
    return df, info


def lowpass(x: np.ndarray, fs: float, cutoff: float = CUTOFF_HZ, numtaps: int = 31) -> np.ndarray:
    """Zero-phase windowed-sinc FIR low-pass (Hamming), applied column-wise with reflect padding."""
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = np.sinc(2 * cutoff / fs * n) * np.hamming(numtaps)
    h /= h.sum()
    pad = numtaps // 2
    x = np.atleast_2d(x.T).T
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    return np.column_stack([np.convolve(xp[:, j], h, mode="valid") for j in range(x.shape[1])])


def resample_uniform(df: pd.DataFrame, target_fs: float = TARGET_FS) -> pd.DataFrame:
    """Low-pass if the native rate is >= 2x target (aliasing), then linear interpolation onto a uniform grid."""
    t = df["timestamp"].to_numpy(dtype=float)
    cols = [c for c in df.columns if c != "timestamp"]
    X = df[cols].to_numpy(dtype=float)
    fs = estimate_fs(t)
    if fs >= 2 * target_fs * 0.95:
        X = lowpass(X, fs, min(CUTOFF_HZ, 0.4 * target_fs))
    grid = np.arange(t[0], t[-1], 1.0 / target_fs)
    out = pd.DataFrame({"timestamp": grid})
    for j, c in enumerate(cols):
        out[c] = np.interp(grid, t, X[:, j])
    return out


def trim_edges(df: pd.DataFrame, start: float = TRIM_SEC, end: float = TRIM_SEC) -> pd.DataFrame:
    """Drop `start` s from the beginning and `end` s from the end."""
    t = df["timestamp"]
    return df[(t >= t.iloc[0] + start) & (t <= t.iloc[-1] - end)].reset_index(drop=True)


def cap_length(df: pd.DataFrame, max_sec: float = MAX_SEC, fs: float = TARGET_FS) -> pd.DataFrame:
    """Keep the central max_sec seconds (exactly max_sec * fs samples) if the clip is longer."""
    n = int(round(max_sec * fs))
    if len(df) <= n:
        return df
    s = (len(df) - n) // 2
    return df.iloc[s:s + n].reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop NaNs and duplicate timestamps, sort, restart time at 0; report gaps > 3 sample periods."""
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]
    dt = np.diff(df["timestamp"].to_numpy())
    problems = []
    if len(dt) and np.any(dt > 3 * np.median(dt)):
        problems.append(f"{int(np.sum(dt > 3 * np.median(dt)))} gaps > 3 sample periods")
    return df, problems


def parse_raw_name(path: Path) -> tuple[str, int]:
    """'Stand_4-2026-10-24_22-16-08.zip' -> ('standing', 4)."""
    m = RAW_NAME.match(path.name)
    if not m or m["prefix"].lower() not in PREFIX_TO_ACTIVITY:
        raise ValueError(f"unexpected raw file name: {path.name}")
    return PREFIX_TO_ACTIVITY[m["prefix"].lower()], int(m["num"])


def process_recording(source: Path, trim: tuple[float, float] = (TRIM_SEC, TRIM_SEC),
                      max_sec: float | None = MAX_SEC) -> tuple[pd.DataFrame, dict]:
    """load -> resample to TARGET_FS -> trim -> cap -> clean."""
    raw, info = load_recording(source)
    info["raw_duration_s"] = float(raw["timestamp"].iloc[-1] - raw["timestamp"].iloc[0])
    df = trim_edges(resample_uniform(raw), *trim)
    if max_sec:
        df = cap_length(df, max_sec)
    df, info["problems"] = clean(df)
    return df, info


def build_clean_dataset(raw_root: Path, out_root: Path, target_fs: float = TARGET_FS) -> pd.DataFrame:
    """data/raw/train/<activity>/*.zip -> data/train/<activity>/<activity>_<device>_<NN>.csv; returns a summary."""
    assert target_fs == TARGET_FS, "change TARGET_FS in this module to use another rate"
    rows = []
    for src in sorted(Path(raw_root, "train").glob("*/*.zip")):
        activity, num = parse_raw_name(src)
        if src.parent.name != activity:
            raise ValueError(f"{src} is in folder '{src.parent.name}' but named as '{activity}'")
        df, info = process_recording(src, TRIM_OVERRIDES.get((activity, num), (TRIM_SEC, TRIM_SEC)))
        df["label"], df["device"] = activity, info["device"]
        dest = Path(out_root, "train", activity, f"{activity}_{info['device']}_{num:02d}.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False, float_format="%.6f")
        rows.append({"file": dest.name, "raw": src.name, "activity": activity, "device": info["device"],
                     "native_fs_acc": info["native_fs_acc"], "native_fs_gyro": info["native_fs_gyro"],
                     "native_fs_grav": info["native_fs_grav"], "raw_duration_s": info["raw_duration_s"],
                     "duration_s": float(df["timestamp"].iloc[-1]), "n_rows": len(df),
                     "problems": "; ".join(info["problems"])})
    return pd.DataFrame(rows)


def build_test_file(source: Path, segments_csv: Path, out_path: Path) -> pd.DataFrame:
    """Continuous test recording + segments (start_s, end_s, label; seconds from recording start)
    -> CSV with a per-row label. Rows outside every segment (the switches) keep an empty label so the
    sequence stays continuous for Viterbi but those rows are never scored. No trimming or capping."""
    df, info = process_recording(source, trim=(0.0, 0.0), max_sec=None)
    seg = pd.read_csv(segments_csv)
    unknown = set(seg["label"]) - set(ACTIVITIES)
    if unknown:
        raise ValueError(f"unknown labels in {segments_csv.name}: {sorted(unknown)}")
    df["label"] = None
    for r in seg.itertuples():
        df.loc[(df["timestamp"] >= r.start_s) & (df["timestamp"] < r.end_s), "label"] = r.label
    df["device"] = info["device"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    return df


def build_test_dataset(raw_root: Path, out_root: Path) -> pd.DataFrame:
    """data/raw/test/Test_<N>-*.zip + Test_<N>_segments.csv -> data/test/test_<NN>_<device>.csv."""
    rows = []
    for src in sorted(Path(raw_root, "test").glob("Test_*.zip")):
        num = int(RAW_NAME.match(src.name)["num"])
        seg = src.parent / f"Test_{num}_segments.csv"
        if not seg.exists():
            print(f"skipping {src.name}: no {seg.name}")
            continue
        device = device_name(_read_members(src).get("meta"))
        df = build_test_file(src, seg, Path(out_root, "test", f"test_{num:02d}_{device}.csv"))
        rows.append({"file": f"test_{num:02d}_{device}.csv", "seconds": len(df) / TARGET_FS,
                     "unlabelled": round(df["label"].isna().sum() / TARGET_FS, 1),
                     **df["label"].value_counts().div(TARGET_FS).round(1).to_dict()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data"
    print(build_test_dataset(root / "raw", root).to_string(index=False))
    summary = build_clean_dataset(root / "raw", root)
    print(summary.groupby("activity").agg(files=("file", "count"), seconds=("duration_s", "sum")).round(1))
    print(summary.groupby("device")[["native_fs_acc", "native_fs_gyro", "native_fs_grav"]].median().round(1))
    bad = summary[summary["problems"] != ""]
    print("problems:", "none" if bad.empty else bad[["file", "problems"]].to_string(index=False))
