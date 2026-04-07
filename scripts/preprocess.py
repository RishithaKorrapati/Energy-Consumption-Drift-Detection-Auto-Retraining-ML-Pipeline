from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Project root is parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "household_power_consumption.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "cleaned_data.csv"


def preprocess(
    input_path: Path,
    output_path: Path,
    *,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """
    Load, clean, and return the preprocessed dataframe. Optionally limit rows (for CI/smoke tests).

    Steps:
    1. Read CSV with ';' separator and treat '?' as NA.
    2. Build a single datetime from Date + Time (European day-first format).
    3. Keep only datetime and Global_active_power.
    4. Coerce power to numeric and drop rows with missing target.
    5. Sort by time ascending.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Large file: explicit dtypes where possible reduces memory; power is float after coercion
    df = pd.read_csv(
        input_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # Combine date and time into one timestamp column
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df[["datetime", "Global_active_power"]].copy()
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # Drop any row missing datetime or target
    df = df.dropna(subset=["datetime", "Global_active_power"])

    df = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)

    if sample_rows is not None and sample_rows > 0:
        df = df.head(sample_rows).copy()

    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess household power consumption data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to raw dataset (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for cleaned CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="If set, keep only the first N rows after cleaning (useful for quick CI runs).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = preprocess(args.input, args.output, sample_rows=args.sample_rows)
    print(f"Wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
