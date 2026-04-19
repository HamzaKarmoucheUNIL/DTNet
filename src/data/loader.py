"""
loader.py — Load and inspect the raw Kaggle CSV dataset.

Responsibility: single-file CSV ingestion with basic diagnostics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

# Resolve from __file__ so path is correct regardless of CWD.
# loader.py lives at <root>/src/data/loader.py  →  3 parents up = project root.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR: Path = _PROJECT_ROOT / "data" / "raw"


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/raw/ and return it as a DataFrame.

    Parameters
    ----------
    filename : str
        Name of the CSV file (e.g. ``'updated_data.csv'``).

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``data/raw/<filename>``.
    """
    filepath: Path = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"[loader] File not found: {filepath}")
    df: pd.DataFrame = pd.read_csv(filepath)
    return df


def print_basic_info(df: pd.DataFrame) -> None:
    """Print a structured diagnostic summary of a DataFrame.

    Reports shape, column names, dtypes, null counts, and unique
    value counts for every column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    """
    n_rows, n_cols = df.shape
    print(f"\n{'='*50}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Shape          : {n_rows} rows × {n_cols} columns")
    print(f"\n  Columns ({n_cols}):")
    for col in df.columns:
        print(f"    - {col}")

    print(f"\n  Column details:")
    header: str = f"  {'Column':<35} {'Dtype':<15} {'Nulls':>8} {'Unique':>8}"
    print(header)
    print(f"  {'-'*35} {'-'*15} {'-'*8} {'-'*8}")
    for col in df.columns:
        dtype: str = str(df[col].dtype)
        nulls: int = int(df[col].isnull().sum())
        unique: int = int(df[col].nunique())
        print(f"  {col:<35} {dtype:<15} {nulls:>8} {unique:>8}")

    print(f"{'='*50}\n")


def main() -> None:
    """Entry point for standalone inspection of a raw CSV file.

    Usage
    -----
    .. code-block:: bash

        python -m src.data.loader <filename.csv>

    Example
    -------
    .. code-block:: bash

        python -m src.data.loader updated_data.csv
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.loader <filename.csv>")
        sys.exit(1)

    filename: str = sys.argv[1]
    print(f"[loader] Loading '{filename}' from {RAW_DATA_DIR}/...")
    df: pd.DataFrame = load_csv(filename)
    print_basic_info(df)
    print(f"[loader] Done. DataFrame ready with shape {df.shape}.")


if __name__ == "__main__":
    main()
