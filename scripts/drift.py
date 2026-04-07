"""
Statistical drift detection between a reference (training-era) dataset and incoming/current data.

Uses normalized mean shift on Global_active_power: |mean_cur - mean_ref| / std_ref.
If that exceeds --threshold, drift_detected is True.

Writes model/drift_status.json for the API/dashboard; can exit with non-zero code for CI gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from new_data_io import read_new_data_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = PROJECT_ROOT / "data" / "cleaned_data.csv"
DEFAULT_CURRENT = PROJECT_ROOT / "data" / "new_data.csv"
DRIFT_STATUS_PATH = PROJECT_ROOT / "model" / "drift_status.json"

# Default: require ~0.5 sigma shift in mean power before flagging drift
DEFAULT_THRESHOLD = 0.5


def load_power_series(path: Path) -> pd.Series:
    """
    Load CSV and return a numeric target series for drift checks.

    Prefers Global_active_power (ground truth). Falls back to predicted_power from the API log
    when labels are unavailable (operational proxy for consumption level).
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = read_new_data_csv(path)
    if "global_active_power" in df.columns and "Global_active_power" not in df.columns:
        df = df.rename(columns={"global_active_power": "Global_active_power"})
    gap = pd.to_numeric(df["Global_active_power"], errors="coerce") if "Global_active_power" in df.columns else None
    pred = pd.to_numeric(df["predicted_power"], errors="coerce") if "predicted_power" in df.columns else None
    if gap is not None and pred is not None:
        s = gap.fillna(pred).dropna()
    elif gap is not None:
        s = gap.dropna()
    elif pred is not None:
        s = pred.dropna()
    else:
        raise ValueError(f"No usable power column in {path} (expected Global_active_power or predicted_power)")
    if len(s) == 0:
        raise ValueError(f"No usable power column in {path} (expected Global_active_power or predicted_power)")
    return s


def mean_shift_score(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Return (mean_ref, mean_cur, std_ref, normalized_shift).

    normalized_shift = |mean_cur - mean_ref| / std_ref, with epsilon if std_ref is ~0.
    """
    mean_ref = float(np.mean(reference))
    mean_cur = float(np.mean(current))
    std_ref = float(np.std(reference))
    denom = std_ref if std_ref > 1e-9 else 1e-9
    shift = abs(mean_cur - mean_ref) / denom
    return mean_ref, mean_cur, std_ref, shift


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    threshold: float,
) -> tuple[bool, dict]:
    """Compute drift flag and diagnostic stats."""
    mean_ref, mean_cur, std_ref, shift = mean_shift_score(reference, current)
    drift_detected = bool(shift > threshold)
    details = {
        "drift_detected": drift_detected,
        "threshold": threshold,
        "reference_mean": mean_ref,
        "current_mean": mean_cur,
        "reference_std": std_ref,
        "normalized_mean_shift": shift,
        "reference_n": int(len(reference)),
        "current_n": int(len(current)),
    }
    return drift_detected, details


def write_drift_status(details: dict, path: Path = DRIFT_STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(details, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect data drift on Global_active_power.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Normalized mean-shift threshold (|Δmean|/std_ref).",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 when drift_detected is True (optional CI gate).",
    )
    parser.add_argument(
        "--max-reference-rows",
        type=int,
        default=None,
        help="Optional cap on reference rows sampled from the start (for large files).",
    )
    parser.add_argument(
        "--max-current-rows",
        type=int,
        default=None,
        help="Optional cap on current rows.",
    )
    args = parser.parse_args()

    ref_series = load_power_series(args.reference)
    try:
        cur_series = load_power_series(args.current)
    except (FileNotFoundError, ValueError):
        cur_series = pd.Series(dtype=float)

    if len(cur_series) == 0:
        details = {
            "drift_detected": False,
            "threshold": args.threshold,
            "error": "current dataset empty; cannot assess drift",
            "reference_n": int(len(ref_series)),
            "current_n": 0,
        }
        write_drift_status(details)
        print(json.dumps(details, indent=2))
        return

    ref_vals = ref_series.to_numpy(dtype=float)
    cur_vals = cur_series.to_numpy(dtype=float)

    if args.max_reference_rows:
        ref_vals = ref_vals[: args.max_reference_rows]
    if args.max_current_rows:
        cur_vals = cur_vals[: args.max_current_rows]

    drift_detected, details = detect_drift(ref_vals, cur_vals, threshold=args.threshold)
    write_drift_status(details)

    print(json.dumps(details, indent=2))
    print(f"drift_detected={drift_detected}")

    if args.fail_on_drift and drift_detected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
