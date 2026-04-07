"""
Read data/new_data.csv even when the file mixes schemas (legacy 2-col vs API 5-col rows).

Avoids pandas ParserError when the header implies 2 columns but later rows have 5 fields.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def read_new_data_csv(path: Path) -> pd.DataFrame:
    """Parse prediction log CSV row-by-row; skip headers and malformed rows."""
    records: list[dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if not row:
                continue
            n = len(row)
            key = row[0].strip().lower()
            if key == "datetime" and n in (2, 5):
                continue
            if n == 2:
                records.append(
                    {
                        "datetime": row[0],
                        "Global_active_power": row[1],
                        "hour": None,
                        "day": None,
                        "month": None,
                        "predicted_power": None,
                    }
                )
            elif n == 5:
                records.append(
                    {
                        "datetime": row[0],
                        "hour": row[1],
                        "day": row[2],
                        "month": row[3],
                        "predicted_power": row[4],
                        "Global_active_power": None,
                    }
                )
            # else: skip corrupted / interleaved partial lines from concurrent writes

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    for c in ("hour", "day", "month", "predicted_power", "Global_active_power"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
