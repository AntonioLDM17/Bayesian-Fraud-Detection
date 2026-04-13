from __future__ import annotations

from typing import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import ROW_ID_COLUMN, TARGET_COLUMN


def get_feature_columns(
    columns: Iterable[str],
    exclude_target: bool = True,
    exclude_row_id: bool = True,
    extra_excluded: set[str] | None = None,
) -> list[str]:
    """
    Build a clean list of feature columns.

    Args:
        columns:
            Column names from a dataframe.
        exclude_target:
            Whether to exclude the target column.
        exclude_row_id:
            Whether to exclude the row_id column.
        extra_excluded:
            Any additional columns to exclude.

    Returns:
        List of feature columns.
    """
    excluded = set(extra_excluded or set())

    if exclude_target:
        excluded.add(TARGET_COLUMN)

    if exclude_row_id:
        excluded.add(ROW_ID_COLUMN)

    return [col for col in columns if col not in excluded]


def build_numeric_preprocessor(
    feature_names: list[str],
    scale: bool = True,
) -> ColumnTransformer:
    """
    Build a numeric preprocessing pipeline.

    Args:
        feature_names:
            Names of numeric feature columns.
        scale:
            Whether to apply StandardScaler after imputation.

    Returns:
        A sklearn ColumnTransformer.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_names),
        ]
    )

    return preprocessor


def build_standard_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """
    Standard preprocessor used by logistic regression, BNN, and GPLVM.

    Includes:
    - median imputation
    - standard scaling
    """
    return build_numeric_preprocessor(feature_names=feature_names, scale=True)


def build_tree_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """
    Preprocessor for tree-based models.

    Includes:
    - median imputation
    - no scaling
    """
    return build_numeric_preprocessor(feature_names=feature_names, scale=False)