"""
Retraining pipeline: compare reference vs incoming data; if drift is detected, refit the model.

Saves a versioned artifact (model/model_v{N}.pkl) and promotes it to model/model.pkl.
Updates model_metadata.json with new version and timestamp.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

# Allow `python scripts/retrain.py` from repo root
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from drift import DEFAULT_REFERENCE, detect_drift, load_power_series, write_drift_status
from new_data_io import read_new_data_csv
from train import METADATA_PATH, METRICS_PATH, MODEL_PATH, PROJECT_ROOT, train_and_evaluate

VERSIONED_DIR = PROJECT_ROOT / "model"


def next_version(metadata_path: Path = METADATA_PATH) -> int:
    if not metadata_path.is_file():
        return 2
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(meta.get("model_version", 1)) + 1


def load_training_frame(reference_path: Path, new_path: Path | None) -> pd.DataFrame:
    """Build training dataframe from cleaned reference plus logged rows (actual or predicted target)."""
    ref = pd.read_csv(reference_path, parse_dates=["datetime"])[["datetime", "Global_active_power"]]
    if new_path and new_path.is_file():
        try:
            extra = pd.read_csv(new_path, parse_dates=["datetime"])
        except pd.errors.ParserError:
            extra = read_new_data_csv(new_path)
            extra["datetime"] = pd.to_datetime(extra["datetime"], errors="coerce")
        if "Global_active_power" in extra.columns and "predicted_power" in extra.columns:
            g = pd.to_numeric(extra["Global_active_power"], errors="coerce")
            extra = extra.copy()
            extra["Global_active_power"] = g.fillna(pd.to_numeric(extra["predicted_power"], errors="coerce"))
        elif "Global_active_power" not in extra.columns and "predicted_power" in extra.columns:
            extra["Global_active_power"] = extra["predicted_power"]
        if "Global_active_power" in extra.columns:
            extra = extra[["datetime", "Global_active_power"]].copy()
            extra["Global_active_power"] = pd.to_numeric(extra["Global_active_power"], errors="coerce")
            extra = extra.dropna(subset=["Global_active_power"])
            ref = pd.concat([ref, extra], ignore_index=True)
    ref = ref.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    return ref


def safe_load_current(path: Path) -> pd.Series:
    """Load drift signal from new_data; return empty series if file is missing or unusable."""
    if not path.is_file():
        return pd.Series(dtype=float)
    try:
        return load_power_series(path)
    except ValueError:
        return pd.Series(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain model when drift is detected.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, default=PROJECT_ROOT / "data" / "new_data.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report drift; do not retrain even if drift is detected.",
    )
    args = parser.parse_args()

    ref_series = load_power_series(args.reference)
    cur_series = safe_load_current(args.current)

    if len(cur_series) == 0:
        details = {
            "drift_detected": False,
            "threshold": args.threshold,
            "message": "No current/new data yet; skip retraining.",
            "reference_n": int(len(ref_series)),
            "current_n": 0,
        }
        write_drift_status(details)
        print(json.dumps(details, indent=2))
        return

    drift_detected, details = detect_drift(
        ref_series.to_numpy(dtype=float),
        cur_series.to_numpy(dtype=float),
        threshold=args.threshold,
    )
    write_drift_status(details)
    print(json.dumps(details, indent=2))

    if args.dry_run:
        print("dry-run: not retraining.")
        return

    if not drift_detected:
        print("No drift; model unchanged.")
        return

    version = next_version()
    df = load_training_frame(args.reference, args.current)
    model, metrics = train_and_evaluate(df)

    versioned_path = VERSIONED_DIR / f"model_v{version}.pkl"
    VERSIONED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, versioned_path)
    shutil.copy2(versioned_path, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "model_version": version,
        "last_trained_at": now,
        "last_retrained_at": now,
        "model_file": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "versioned_file": str(versioned_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "retrained_due_to_drift": True,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Retrained: saved {versioned_path} and promoted to {MODEL_PATH}")


if __name__ == "__main__":
    main()
