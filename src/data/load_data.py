from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Class"
ROW_ID_COLUMN = "row_id"


def load_data(
    csv_path: str | Path,
    add_row_id: bool = True,
) -> pd.DataFrame:
    """
    Load the credit card fraud dataset from CSV.

    Args:
        csv_path:
            Path to the dataset CSV.
        add_row_id:
            If True, add a stable row identifier column preserving the
            original row index from the raw dataset.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError:
            If the CSV file does not exist.
        ValueError:
            If the target column is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if add_row_id:
        df = df.copy()
        if ROW_ID_COLUMN not in df.columns:
            df[ROW_ID_COLUMN] = df.index

    return df