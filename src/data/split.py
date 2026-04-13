from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import ROW_ID_COLUMN, TARGET_COLUMN


RANDOM_STATE = 42


@dataclass
class SplitFrames:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class SplitArrays:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> SplitFrames:
    """
    Split a dataframe into train / validation / test using stratification
    on the target column.

    Notes:
    - val_size is interpreted as a fraction of the full dataset,
      consistent with the rest of the project.
    - The resulting split proportions are approximately:
        train = 0.6
        val = 0.2
        test = 0.2
      when using the defaults.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataframe.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )

    adjusted_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[TARGET_COLUMN],
        random_state=random_state,
    )

    return SplitFrames(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def split_features_target(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    drop_row_id_from_X: bool = False,
) -> SplitArrays:
    """
    Split into X/y train, validation, and test sets.

    Args:
        df:
            Full dataframe including target column.
        test_size:
            Fraction for test split.
        val_size:
            Fraction for validation split relative to the full dataset.
        random_state:
            Random seed for reproducibility.
        drop_row_id_from_X:
            If True, remove row_id from X splits.
            If False, preserve row_id in X so downstream analyses can keep it.

    Returns:
        SplitArrays containing X_train, X_val, X_test, y_train, y_val, y_test.
    """
    splits = split_dataframe(
        df=df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    def _split_xy(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = split_df.drop(columns=[TARGET_COLUMN]).copy()
        y = split_df[TARGET_COLUMN].astype(int).copy()

        if drop_row_id_from_X and ROW_ID_COLUMN in X.columns:
            X = X.drop(columns=[ROW_ID_COLUMN])

        return X, y

    X_train, y_train = _split_xy(splits.train_df)
    X_val, y_val = _split_xy(splits.val_df)
    X_test, y_test = _split_xy(splits.test_df)

    return SplitArrays(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def sample_fraud_focused_subset(
    df: pd.DataFrame,
    nonfraud_multiplier: int = 3,
    max_nonfraud: int | None = 1500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Build a fraud-focused subset from a dataframe.

    Keeps:
    - all fraud cases
    - a random sample of non-fraud cases

    Useful for GPLVM or exploratory latent analysis.

    Args:
        df:
            Input dataframe containing Class and optionally row_id.
        nonfraud_multiplier:
            Target ratio of non-fraud samples per fraud sample.
        max_nonfraud:
            Optional cap on the number of non-fraud samples.
        random_state:
            Random seed.

    Returns:
        Shuffled subset dataframe.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataframe.")

    fraud_df = df[df[TARGET_COLUMN] == 1].copy()
    nonfraud_df = df[df[TARGET_COLUMN] == 0].copy()

    n_fraud = len(fraud_df)
    n_nonfraud_target = n_fraud * nonfraud_multiplier

    if max_nonfraud is not None:
        n_nonfraud_target = min(n_nonfraud_target, max_nonfraud)

    n_nonfraud_target = min(n_nonfraud_target, len(nonfraud_df))

    sampled_nonfraud = nonfraud_df.sample(
        n=n_nonfraud_target,
        random_state=random_state,
        replace=False,
    )

    subset = pd.concat([fraud_df, sampled_nonfraud], axis=0)
    subset = subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return subset