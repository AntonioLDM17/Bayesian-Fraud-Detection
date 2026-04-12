from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42


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
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
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


def build_lightgbm_pipeline(feature_names: list[str], scale_pos_weight: float) -> Pipeline:
    """
    Build LightGBM pipeline for imbalanced fraud detection.
    """
    preprocessor = build_preprocessor(feature_names)

    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", lgbm_model),
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
        "lightgbm": build_lightgbm_pipeline(
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
        ),
    }