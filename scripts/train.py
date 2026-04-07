"""
Train a baseline regression model on cleaned household power data.

Features: hour, day, month from datetime. Target: Global_active_power.
Persists model.joblib to model/model.pkl and writes training metadata / metrics for the dashboard.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "cleaned_data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day-of-month, and month from the datetime column."""
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["day"] = out["datetime"].dt.day
    out["month"] = out["datetime"].dt.month
    return out


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[LinearRegression, dict]:
    """Fit LinearRegression and return model + dict of metrics."""
    data = add_time_features(df)
    feature_cols = ["hour", "day", "month"]
    X = data[feature_cols].to_numpy(dtype=float)
    y = data["Global_active_power"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": feature_cols,
    }
    return model, metrics


def save_artifacts(
    model: LinearRegression,
    metrics: dict,
    *,
    version: int,
    model_path: Path = MODEL_PATH,
) -> None:
    """Save model, metrics JSON, and metadata (version, timestamp)."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "model_version": version,
        "last_trained_at": now,
        "last_retrained_at": now,
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train energy consumption regression model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Cleaned CSV path (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=MODEL_PATH,
        help=f"Where to save the trained model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Model version label to record in metadata.",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Cleaned data not found: {args.data}. Run scripts/preprocess.py first.")

    df = pd.read_csv(args.data, parse_dates=["datetime"])

    model, metrics = train_and_evaluate(df)
    save_artifacts(model, metrics, version=args.version, model_path=args.output_model)

    print(f"Saved model to {args.output_model}")
    print(f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
