from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    LOGISTIC_MAX_ITER,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
)


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_names),
        ]
    )

    return preprocessor


def build_baseline_models(feature_names: list[str]) -> dict[str, Pipeline]:
    """Create baseline sklearn pipelines."""
    preprocessor = build_preprocessor(feature_names)

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    min_samples_split=RF_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_pipeline,
        "random_forest": rf_pipeline,
    }
