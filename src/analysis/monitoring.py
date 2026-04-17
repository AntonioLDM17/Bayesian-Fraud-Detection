from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH,
    DECISION_CSV_PATH,
    EXPERIMENTS_DIR,
    UNCERTAINTY_CSV_PATH,
)
from src.data.load_data import load_data
from src.data.split import split_features_target


MONITORING_RESULTS_DIR = EXPERIMENTS_DIR / "monitoring_results"
MONITORING_SUMMARY_PATH = MONITORING_RESULTS_DIR / "monitoring_summary.json"
MONITORING_REPORT_PATH = MONITORING_RESULTS_DIR / "monitoring_report.csv"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def compute_feature_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute summary statistics for numeric features.
    """
    summary: dict[str, dict[str, float]] = {}

    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) == 0:
            continue

        summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }

    return summary


def compute_prediction_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize predictive probabilities and uncertainties.
    """
    summary: dict[str, Any] = {"num_rows": int(len(df))}

    if "predicted_probability" in df.columns:
        probs = df["predicted_probability"].dropna()
        if len(probs) > 0:
            summary["predicted_probability"] = {
                "mean": float(probs.mean()),
                "std": float(probs.std(ddof=0)),
                "median": float(probs.median()),
                "p90": float(probs.quantile(0.90)),
                "p95": float(probs.quantile(0.95)),
                "share_above_0_5": float((probs > 0.5).mean()),
                "share_above_0_8": float((probs > 0.8).mean()),
                "share_above_0_9": float((probs > 0.9).mean()),
            }

    if "uncertainty_std" in df.columns:
        unc = df["uncertainty_std"].dropna()
        if len(unc) > 0:
            summary["uncertainty_std"] = {
                "mean": float(unc.mean()),
                "std": float(unc.std(ddof=0)),
                "median": float(unc.median()),
                "p90": float(unc.quantile(0.90)),
                "p95": float(unc.quantile(0.95)),
                "share_above_0_1": float((unc > 0.1).mean()),
                "share_above_0_2": float((unc > 0.2).mean()),
                "share_above_0_25": float((unc > 0.25).mean()),
            }

    return summary


def compute_decision_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize decision distribution if decisions are available.
    """
    summary: dict[str, Any] = {}

    if "decision" not in df.columns:
        summary["available"] = False
        return summary

    counts = df["decision"].value_counts(dropna=False).to_dict()
    shares = (df["decision"].value_counts(dropna=False) / len(df)).to_dict()

    summary["available"] = True
    summary["counts"] = {str(k): int(v) for k, v in counts.items()}
    summary["shares"] = {str(k): float(v) for k, v in shares.items()}

    return summary


def compare_numeric_summaries(
    reference_summary: dict[str, dict[str, float]],
    current_summary: dict[str, dict[str, float]],
) -> list[dict[str, float | str | None]]:
    """
    Compare numeric summaries feature by feature.
    """
    rows: list[dict[str, float | str | None]] = []

    common_keys = sorted(set(reference_summary) & set(current_summary))
    for key in common_keys:
        ref = reference_summary[key]
        cur = current_summary[key]

        ref_mean = safe_float(ref.get("mean"))
        cur_mean = safe_float(cur.get("mean"))
        ref_std = safe_float(ref.get("std"))
        cur_std = safe_float(cur.get("std"))

        mean_diff = None if ref_mean is None or cur_mean is None else cur_mean - ref_mean
        abs_mean_diff = None if mean_diff is None else abs(mean_diff)

        std_ratio = None
        if ref_std is not None and ref_std > 1e-12 and cur_std is not None:
            std_ratio = cur_std / ref_std

        rows.append(
            {
                "metric_or_feature": key,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_diff": mean_diff,
                "abs_mean_diff": abs_mean_diff,
                "reference_std": ref_std,
                "current_std": cur_std,
                "std_ratio": std_ratio,
            }
        )

    return rows


def compare_prediction_summaries(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> list[dict[str, float | str | None]]:
    """
    Compare probability/uncertainty summaries.
    """
    rows: list[dict[str, float | str | None]] = []

    for group_name in ["predicted_probability", "uncertainty_std"]:
        ref_group = reference.get(group_name, {})
        cur_group = current.get(group_name, {})
        metric_names = sorted(set(ref_group.keys()) | set(cur_group.keys()))

        for metric_name in metric_names:
            ref_value = safe_float(ref_group.get(metric_name))
            cur_value = safe_float(cur_group.get(metric_name))

            diff = None
            if ref_value is not None and cur_value is not None:
                diff = cur_value - ref_value

            rows.append(
                {
                    "metric_or_feature": f"{group_name}.{metric_name}",
                    "reference_mean": ref_value,
                    "current_mean": cur_value,
                    "mean_diff": diff,
                    "abs_mean_diff": None if diff is None else abs(diff),
                    "reference_std": None,
                    "current_std": None,
                    "std_ratio": None,
                }
            )

    return rows


def compare_decision_summaries(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> list[dict[str, float | str | None]]:
    """
    Compare decision distributions.
    """
    rows: list[dict[str, float | str | None]] = []

    if not reference.get("available", False) or not current.get("available", False):
        return rows

    ref_shares = reference.get("shares", {})
    cur_shares = current.get("shares", {})
    labels = sorted(set(ref_shares.keys()) | set(cur_shares.keys()))

    for label in labels:
        ref_value = safe_float(ref_shares.get(label))
        cur_value = safe_float(cur_shares.get(label))
        diff = None
        if ref_value is not None and cur_value is not None:
            diff = cur_value - ref_value

        rows.append(
            {
                "metric_or_feature": f"decision_share.{label}",
                "reference_mean": ref_value,
                "current_mean": cur_value,
                "mean_diff": diff,
                "abs_mean_diff": None if diff is None else abs(diff),
                "reference_std": None,
                "current_std": None,
                "std_ratio": None,
            }
        )

    return rows


def compute_monitoring_alerts(
    prediction_report_rows: list[dict[str, float | str | None]],
    decision_report_rows: list[dict[str, float | str | None]],
) -> list[dict[str, str | float]]:
    """
    Build simple monitoring alerts from summary shifts.
    """
    alerts: list[dict[str, str | float]] = []

    def find_metric(rows: list[dict[str, float | str | None]], name: str) -> dict[str, float | str | None] | None:
        for row in rows:
            if row["metric_or_feature"] == name:
                return row
        return None

    mean_unc = find_metric(prediction_report_rows, "uncertainty_std.mean")
    share_unc_high = find_metric(prediction_report_rows, "uncertainty_std.share_above_0_2")
    share_prob_high = find_metric(prediction_report_rows, "predicted_probability.share_above_0_9")
    review_share = find_metric(decision_report_rows, "decision_share.REVIEW")
    block_share = find_metric(decision_report_rows, "decision_share.BLOCK")

    if mean_unc is not None and mean_unc["mean_diff"] is not None and mean_unc["mean_diff"] > 0.03:
        alerts.append(
            {
                "level": "warning",
                "message": "Average uncertainty increased materially versus reference.",
                "metric": "uncertainty_std.mean",
                "delta": float(mean_unc["mean_diff"]),
            }
        )

    if share_unc_high is not None and share_unc_high["mean_diff"] is not None and share_unc_high["mean_diff"] > 0.05:
        alerts.append(
            {
                "level": "warning",
                "message": "Share of high-uncertainty cases increased materially.",
                "metric": "uncertainty_std.share_above_0_2",
                "delta": float(share_unc_high["mean_diff"]),
            }
        )

    if share_prob_high is not None and share_prob_high["mean_diff"] is not None and share_prob_high["mean_diff"] > 0.02:
        alerts.append(
            {
                "level": "info",
                "message": "Share of very high fraud scores increased.",
                "metric": "predicted_probability.share_above_0_9",
                "delta": float(share_prob_high["mean_diff"]),
            }
        )

    if review_share is not None and review_share["mean_diff"] is not None and review_share["mean_diff"] > 0.05:
        alerts.append(
            {
                "level": "warning",
                "message": "Review rate increased materially.",
                "metric": "decision_share.REVIEW",
                "delta": float(review_share["mean_diff"]),
            }
        )

    if block_share is not None and block_share["mean_diff"] is not None and block_share["mean_diff"] > 0.02:
        alerts.append(
            {
                "level": "info",
                "message": "Automatic block rate increased.",
                "metric": "decision_share.BLOCK",
                "delta": float(block_share["mean_diff"]),
            }
        )

    if not alerts:
        alerts.append(
            {
                "level": "info",
                "message": "No major monitoring alerts detected.",
                "metric": "none",
                "delta": 0.0,
            }
        )

    return alerts


def load_reference_feature_data() -> pd.DataFrame:
    """
    Use the saved project test split as reference population for feature summaries.
    """
    df = load_data(DATA_PATH, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=True)
    return splits.X_test.copy()


def load_prediction_monitoring_data(
    uncertainty_csv_path: str | Path = UNCERTAINTY_CSV_PATH,
    decision_csv_path: str | Path = DECISION_CSV_PATH,
) -> pd.DataFrame:
    """
    Load saved prediction outputs and merge decisions if available.
    """
    uncertainty_csv_path = Path(uncertainty_csv_path)
    if not uncertainty_csv_path.exists():
        raise FileNotFoundError(f"Could not find uncertainty CSV: {uncertainty_csv_path}")

    pred_df = pd.read_csv(uncertainty_csv_path)

    decision_csv_path = Path(decision_csv_path)
    if decision_csv_path.exists():
        decision_df = pd.read_csv(decision_csv_path)
        if {"row_id", "decision"}.issubset(decision_df.columns):
            pred_df = pred_df.merge(
                decision_df[["row_id", "decision"]],
                on="row_id",
                how="left",
            )

    return pred_df


def run_monitoring_report(
    reference_feature_df: pd.DataFrame | None = None,
    current_feature_df: pd.DataFrame | None = None,
    uncertainty_csv_path: str | Path = UNCERTAINTY_CSV_PATH,
    decision_csv_path: str | Path = DECISION_CSV_PATH,
    output_dir: str | Path = MONITORING_RESULTS_DIR,
) -> dict[str, Any]:
    """
    Build a simple offline monitoring report.

    Default behavior:
    - reference features: test split from the main dataset
    - current predictions: saved uncertainty/decision outputs
    - current features: if not provided, same as reference features

    This is meant as a lightweight production-style monitoring snapshot.
    """
    output_dir = ensure_dir(output_dir)

    if reference_feature_df is None:
        reference_feature_df = load_reference_feature_data()

    if current_feature_df is None:
        current_feature_df = reference_feature_df.copy()

    prediction_df = load_prediction_monitoring_data(
        uncertainty_csv_path=uncertainty_csv_path,
        decision_csv_path=decision_csv_path,
    )

    reference_feature_summary = compute_feature_summary(reference_feature_df)
    current_feature_summary = compute_feature_summary(current_feature_df)

    reference_prediction_summary = compute_prediction_summary(prediction_df)
    current_prediction_summary = compute_prediction_summary(prediction_df)

    reference_decision_summary = compute_decision_summary(prediction_df)
    current_decision_summary = compute_decision_summary(prediction_df)

    feature_report = compare_numeric_summaries(
        reference_feature_summary,
        current_feature_summary,
    )
    prediction_report = compare_prediction_summaries(
        reference_prediction_summary,
        current_prediction_summary,
    )
    decision_report = compare_decision_summaries(
        reference_decision_summary,
        current_decision_summary,
    )

    alerts = compute_monitoring_alerts(
        prediction_report_rows=prediction_report,
        decision_report_rows=decision_report,
    )

    report_rows = feature_report + prediction_report + decision_report
    report_df = pd.DataFrame(report_rows)

    summary: dict[str, Any] = {
        "metadata": {
            "uncertainty_csv_path": str(uncertainty_csv_path),
            "decision_csv_path": str(decision_csv_path),
            "num_reference_feature_rows": int(len(reference_feature_df)),
            "num_current_feature_rows": int(len(current_feature_df)),
            "num_prediction_rows": int(len(prediction_df)),
        },
        "reference_feature_summary": reference_feature_summary,
        "current_feature_summary": current_feature_summary,
        "prediction_summary": current_prediction_summary,
        "decision_summary": current_decision_summary,
        "alerts": alerts,
    }

    report_df.to_csv(MONITORING_REPORT_PATH, index=False)
    with MONITORING_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved monitoring report to: {MONITORING_REPORT_PATH}")
    print(f"Saved monitoring summary to: {MONITORING_SUMMARY_PATH}")

    return summary


if __name__ == "__main__":
    run_monitoring_report()