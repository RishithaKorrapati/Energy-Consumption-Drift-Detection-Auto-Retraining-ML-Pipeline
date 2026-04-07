"""
FastAPI service for energy consumption prediction.

POST /predict accepts either a timestamp or explicit hour/day/month features.
Each request is appended to data/new_data.csv for drift monitoring and retraining.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
LOG_PATH = PROJECT_ROOT / "data" / "new_data.csv"

app = FastAPI(title="Energy consumption predictor", version="1.0.0")
_model = None
_log_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run scripts/train.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    """Incoming payload: provide `datetime` OR all of `hour`, `day`, `month`."""

    datetime: Optional[str] = Field(default=None, description="ISO 8601 or DD/MM/YYYY HH:MM:SS")
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    day: Optional[int] = Field(default=None, ge=1, le=31)
    month: Optional[int] = Field(default=None, ge=1, le=12)

    @model_validator(mode="after")
    def validate_features(self):
        has_dt = self.datetime is not None
        has_parts = all(x is not None for x in (self.hour, self.day, self.month))
        if not has_dt and not has_parts:
            raise ValueError("Provide `datetime` or all of `hour`, `day`, `month`.")
        if has_dt and has_parts:
            raise ValueError("Provide either `datetime` or explicit features, not both.")
        return self


class PredictResponse(BaseModel):
    predicted_power: float
    hour: int
    day: int
    month: int


def _parse_datetime(value: str) -> datetime:
    """Try ISO first, then European day-first format used in the raw UCI file."""
    value = value.strip()
    parsers = (
        lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
        lambda v: datetime.strptime(v, "%d/%m/%Y %H:%M:%S"),
    )
    for fn in parsers:
        try:
            return fn(value)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime format: {value}")


def _features_from_request(body: PredictRequest) -> tuple[int, int, int, datetime]:
    if body.datetime is not None:
        dt = _parse_datetime(body.datetime)
    else:
        # Use a fixed year for consistency when only calendar features are provided
        dt = datetime(2006, body.month, body.day, body.hour)
    return dt.hour, dt.day, dt.month, dt


def append_prediction_log(dt: datetime, hour: int, day: int, month: int, predicted: float) -> None:
    """Persist request features and prediction for downstream drift checks."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "datetime": dt.isoformat(sep=" "),
        "hour": hour,
        "day": day,
        "month": month,
        "predicted_power": predicted,
    }
    df = pd.DataFrame([row])
    # Serialize concurrent /predict appends so CSV rows are never interleaved (Windows + parallel load).
    with _log_lock:
        write_header = not LOG_PATH.exists()
        df.to_csv(LOG_PATH, mode="a", index=False, header=write_header)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.is_file()}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    try:
        hour, day, month, dt = _features_from_request(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model = get_model()
    x = np.array([[hour, day, month]], dtype=float)
    pred = float(model.predict(x)[0])

    append_prediction_log(dt, hour, day, month, pred)
    return PredictResponse(predicted_power=pred, hour=hour, day=day, month=month)


def main():
    # Run from repo root: `python api/app.py` or `uvicorn api.app:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
