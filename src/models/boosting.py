from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    For tree-based boosting models, scaling is not strictly necessary,
    but we keep a simple and robust numeric preprocessing pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Trees do not need scaling, so we omit StandardScaler here.
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_names),
        ]
    )

    return preprocessor


def build_xgboost_pipeline(feature_names: list[str], scale_pos_weight: float) -> Pipeline:
    """
    Build XGBoost pipeline for imbalanced fraud detection.
    """
    preprocessor = build_preprocessor(feature_names)

    xgb_model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_model),
        ]
    )

    return pipeline



def build_boosting_models(
    feature_names: list[str],
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    """
    Return all boosting pipelines.
    """
    return {
        "xgboost": build_xgboost_pipeline(
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
        ),
    }
