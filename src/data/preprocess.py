"""
preprocess.py — Clean and preprocess the raw Kaggle DataFrame.

Responsibility: datetime parsing, sensor normalisation (min-max),
categorical encoding, missing-value handling, and saving processed output.

Missing-value strategy
----------------------
- Sensor columns (numeric): filled with the column **median**.
  Sensors occasionally drop out mid-shift; median imputation preserves
  the distribution without being pulled by extreme failure spikes.
- Categorical columns: filled with the string ``'unknown'`` before
  encoding, so the category code is deterministic and the category
  is not silently dropped.
- ``transaction_date``: rows where the date cannot be parsed are
  dropped (they carry no temporal signal).
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

np.random.seed(42)

PROCESSED_DATA_DIR: Path = Path("data/processed")

SENSOR_COLUMNS: list[str] = [
    "temp_bearing_degC",
    "temp_motor_degC",
    "vibration_h_mms",
    "vibration_v_mms",
    "oil_pressure_bar",
    "load_pct",
    "power_consumption_kw",
    "shaft_rpm",
]

CATEGORICAL_COLUMNS: list[str] = [
    "machine_type",
    "part_family",
    "criticality",
    "uom",
    "wo_type",
    "plant_code",
]

DATE_COLUMN: str = "transaction_date"


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the transaction_date column to datetime and drop unparseable rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame containing a ``transaction_date`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``transaction_date`` as ``datetime64[ns]``.
        Rows where parsing failed are dropped.
    """
    df = df.copy()
    before: int = len(df)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    dropped: int = df[DATE_COLUMN].isnull().sum()
    if dropped:
        print(f"[preprocess] Dropped {dropped}/{before} rows — unparseable dates.")
    df = df.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)
    return df


def normalise_sensors(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Min-max normalise sensor columns to [0, 1] and record the scalers.

    Missing values in sensor columns are imputed with the column median
    **before** scaling so the scaler parameters are always finite.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the sensor columns listed in
        ``SENSOR_COLUMNS``.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with sensor columns replaced by their normalised values.
    scalers : dict[str, dict[str, float]]
        Mapping ``{column: {"min": ..., "max": ...}}`` for inverse transform.
        Use ``original = scaled * (max - min) + min``.
    """
    df = df.copy()
    scalers: Dict[str, Dict[str, float]] = {}

    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            print(f"[preprocess] WARNING: sensor column '{col}' not found — skipped.")
            continue

        # Impute missing values with median before computing scale parameters
        median_val: float = float(df[col].median())
        null_count: int = int(df[col].isnull().sum())
        if null_count:
            print(f"[preprocess] '{col}': imputed {null_count} nulls with median ({median_val:.4f}).")
        df[col] = df[col].fillna(median_val)

        col_min: float = float(df[col].min())
        col_max: float = float(df[col].max())
        scalers[col] = {"min": col_min, "max": col_max}

        col_range: float = col_max - col_min
        if col_range == 0.0:
            # Constant column — map everything to 0.0
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / col_range

    return df, scalers


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as integer category codes.

    Missing values are replaced with ``'unknown'`` before encoding so
    that every row receives a valid integer code.

    The original string column is replaced by an integer code column
    with the same name. The category mapping is printed to stdout for
    reference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the categorical columns listed in
        ``CATEGORICAL_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns encoded as ``int16`` codes.
    """
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            print(f"[preprocess] WARNING: categorical column '{col}' not found — skipped.")
            continue

        null_count: int = int(df[col].isnull().sum())
        if null_count:
            print(f"[preprocess] '{col}': filled {null_count} nulls with 'unknown'.")
        df[col] = df[col].fillna("unknown").astype("category")

        mapping: Dict[int, str] = dict(enumerate(df[col].cat.categories))
        print(f"[preprocess] '{col}' encoding: {mapping}")

        df[col] = df[col].cat.codes.astype("int16")

    return df


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save the processed DataFrame to data/processed/ as a CSV.

    Creates the output directory if it does not exist.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame to persist.
    filename : str
        Output filename (e.g. ``'updated_data_processed.csv'``).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Saved processed data → {out_path}  (shape {df.shape})")
    return out_path


def preprocess(
    df: pd.DataFrame,
    output_filename: str = "processed.csv",
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Run the full preprocessing pipeline on a raw DataFrame.

    Steps (in order):
    1. Parse ``transaction_date`` → datetime; drop unparseable rows.
    2. Impute & min-max normalise sensor columns → [0, 1].
    3. Impute & encode categorical columns → integer codes.
    4. Save result to ``data/processed/<output_filename>``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by ``loader.load_csv()``.
    output_filename : str, optional
        Name for the saved CSV file (default ``'processed.csv'``).

    Returns
    -------
    df_clean : pd.DataFrame
        Fully preprocessed DataFrame.
    scalers : dict[str, dict[str, float]]
        Scaler parameters for each sensor column (min / max).
        Required for inverse-transforming predictions back to raw units.
    """
    print(f"[preprocess] Starting pipeline — input shape {df.shape}")

    df_clean: pd.DataFrame = parse_dates(df)
    df_clean, scalers = normalise_sensors(df_clean)
    df_clean = encode_categoricals(df_clean)

    save_processed(df_clean, output_filename)

    print(f"[preprocess] Pipeline complete — output shape {df_clean.shape}")
    return df_clean, scalers


def main() -> None:
    """Entry point for standalone preprocessing.

    Usage
    -----
    .. code-block:: bash

        python -m src.data.preprocess <input.csv> [output.csv]

    Example
    -------
    .. code-block:: bash

        python -m src.data.preprocess updated_data.csv updated_data_processed.csv
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.preprocess <input.csv> [output.csv]")
        sys.exit(1)

    input_filename: str = sys.argv[1]
    output_filename: str = sys.argv[2] if len(sys.argv) >= 3 else "processed.csv"

    # Import here to avoid circular dependency at module level
    from src.data.loader import load_csv

    print(f"[preprocess] Loading '{input_filename}'...")
    raw_df: pd.DataFrame = load_csv(input_filename)

    df_clean, scalers = preprocess(raw_df, output_filename)

    print("\n[preprocess] Scaler parameters (for inverse transform):")
    for col, params in scalers.items():
        print(f"  {col:<30} min={params['min']:.4f}  max={params['max']:.4f}")


if __name__ == "__main__":
    main()
