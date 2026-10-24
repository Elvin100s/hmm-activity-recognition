#!/usr/bin/env python3
"""
validate_dataset.py — checks the cleaned dataset against the Formative 2 rubric.

Expected layout (relative to the repo root):
    data/train/<activity>/<activity>_<member>_<NN>.csv
    data/test/test_<NN>_<member>.csv          (per-row 'label' column, >= 2 activities)

Required columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
(train files may omit 'label' — it is taken from the folder name).

Usage:
    python scripts/validate_dataset.py                 # uses ./data
    python scripts/validate_dataset.py --data path/to/data
Exit code 0 = rubric targets met, 1 = something to fix.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITIES = ["standing", "walking", "jumping", "still"]
REQUIRED_COLS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TRAIN_NAME = re.compile(r"^(standing|walking|jumping|still)_([a-z0-9]+)_(\d+)\.csv$")
TEST_NAME = re.compile(r"^test_(\d+)_([a-z0-9]+)\.csv$")

TARGET_FILES = 50
TARGET_SEC_PER_ACTIVITY = 90.0
MIN_LEN, MAX_LEN = 5.0, 10.0
TARGET_TEST_FILES = 2

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


def check_timestamps(t: np.ndarray) -> tuple[float, float, list[str]]:
    """Return (duration_s, fs_estimate, problems) for a timestamp array in seconds."""
    problems = []
    if len(t) < 2:
        return 0.0, 0.0, ["fewer than 2 rows"]
    dt = np.diff(t)
    if np.any(dt <= 0):
        problems.append(f"{int(np.sum(dt <= 0))} non-increasing timestamps")
    med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    fs = 1.0 / med if med > 0 else 0.0
    if med > 0 and np.any(dt > 3 * med):
        problems.append(f"{int(np.sum(dt > 3 * med))} gaps > 3 sample periods (max {dt.max():.3f}s)")
    if fs > 0 and t.max() > 1e6:
        problems.append("timestamps look like epoch/ns — convert to seconds from start")
    return float(t[-1] - t[0]), fs, problems


def load(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return None, [f"cannot read: {exc}"]
    problems = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    present = [c for c in REQUIRED_COLS if c in df.columns]
    n_nan = int(df[present].isna().sum().sum())
    if n_nan:
        problems.append(f"{n_nan} NaN values")
    return df, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="data directory (default: ./data)")
    args = ap.parse_args()
    root = Path(args.data)
    all_ok = True

    print(f"\n=== Validating {root.resolve()} ===\n")

    # ---------------- TRAIN ----------------
    train_dir = root / "train"
    files = sorted(train_dir.rglob("*.csv")) if train_dir.exists() else []
    per_act_sec = defaultdict(float)
    per_act_files = defaultdict(int)
    per_member_fs = defaultdict(list)
    rows = []
    for f in files:
        issues = []
        m = TRAIN_NAME.match(f.name)
        if not m:
            issues.append("bad filename (want <activity>_<member>_<NN>.csv, lowercase)")
            act, member = f.parent.name, "?"
        else:
            act, member = m.group(1), m.group(2)
            if f.parent.name != act:
                issues.append(f"in folder '{f.parent.name}' but named '{act}'")
        df, p = load(f)
        issues += p
        dur, fs = 0.0, 0.0
        if df is not None and "timestamp" in df.columns:
            dur, fs, p = check_timestamps(df["timestamp"].to_numpy(dtype=float))
            issues += p
            if not (MIN_LEN - 0.25 <= dur <= MAX_LEN + 0.25):
                issues.append(f"duration {dur:.1f}s outside 5–10 s")
        fatal = ("missing", "cannot", "non-increasing", "bad filename", "but named")
        status = OK if not issues else (BAD if any(k in i for i in issues for k in fatal) else WARN)
        if status != BAD and act in ACTIVITIES:
            per_act_sec[act] += dur
            per_act_files[act] += 1
            per_member_fs[member].append(fs)
        rows.append((status, f.relative_to(root), dur, fs, "; ".join(issues)))

    print("Per-file report (problems only):")
    shown = 0
    for status, rel, dur, fs, msg in rows:
        if status != OK:
            print(f"  [{status}] {rel}  ({dur:.1f}s, {fs:.0f} Hz)  {msg}")
            shown += 1
    if not shown:
        print("  none — every file clean")

    n_good = sum(per_act_files.values())
    print(f"\nTRAIN files usable: {n_good} / {len(files)}   (target ≥ {TARGET_FILES})  "
          f"[{OK if n_good >= TARGET_FILES else BAD}]")
    all_ok &= n_good >= TARGET_FILES

    print(f"\n{'Activity':<10} {'files':>5} {'seconds':>8}  status (target ≥ {TARGET_SEC_PER_ACTIVITY:.0f}s)")
    for a in ACTIVITIES:
        s = per_act_sec.get(a, 0.0)
        good = s >= TARGET_SEC_PER_ACTIVITY
        all_ok &= good
        print(f"{a:<10} {per_act_files.get(a, 0):>5} {s:>8.1f}  [{OK if good else BAD}]")

    print("\nMeasured sampling rate per member (put this table in your report):")
    print(f"{'member':<12} {'files':>5} {'median Hz':>10} {'min Hz':>8} {'max Hz':>8}")
    for mem, fss in sorted(per_member_fs.items()):
        a = np.array(fss)
        print(f"{mem:<12} {len(a):>5} {np.median(a):>10.1f} {a.min():>8.1f} {a.max():>8.1f}")
    all_fs = np.concatenate([np.array(v) for v in per_member_fs.values()]) if per_member_fs else np.array([])
    if all_fs.size and all_fs.max() / max(all_fs.min(), 1e-9) > 1.2:
        print(f"  [{WARN}] rates differ ({all_fs.min():.0f}–{all_fs.max():.0f} Hz): did you resample to a common rate? "
              "(fine if these are RAW files; cleaned files should all be ≈ TARGET_FS)")

    # ---------------- TEST ----------------
    test_dir = root / "test"
    tfiles = sorted(p for p in test_dir.glob("*.csv") if TEST_NAME.match(p.name)) if test_dir.exists() else []
    print(f"\nTEST files: {len(tfiles)}   (target ≥ {TARGET_TEST_FILES})  "
          f"[{OK if len(tfiles) >= TARGET_TEST_FILES else BAD}]")
    all_ok &= len(tfiles) >= TARGET_TEST_FILES
    for f in tfiles:
        df, issues = load(f)
        if df is None:
            print(f"  [{BAD}] {f.name}: {issues}")
            all_ok = False
            continue
        dur, fs, p = check_timestamps(df["timestamp"].to_numpy(dtype=float)) if "timestamp" in df.columns else (0, 0, [])
        issues += p
        if "label" not in df.columns:
            issues.append("no per-row 'label' column")
            labels = []
        else:
            labels = sorted(set(df["label"].dropna().astype(str)))
            unknown = [l for l in labels if l not in ACTIVITIES]
            if unknown:
                issues.append(f"unknown labels {unknown}")
            if len(labels) < 2:
                issues.append("only one activity — continuous multi-activity test files test transitions/Viterbi far better")
        status = OK if not issues else WARN
        if any("label" in i and "no per-row" in i for i in issues):
            status, all_ok = BAD, False
        print(f"  [{status}] {f.name}: {dur:.1f}s @ {fs:.0f} Hz, labels={labels} {'; '.join(issues)}")

    print("\nReminder: test recordings must come from a NEW session (different day/room/person) and never be")
    print("used for training, normalization statistics, or tuning. Say how you obtained them in the report.")

    print("\n" + ("ALL RUBRIC DATA TARGETS MET ✅" if all_ok else "SOMETHING TO FIX ❌ — see FAIL lines above"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
