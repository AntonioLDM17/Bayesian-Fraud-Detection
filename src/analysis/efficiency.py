from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BNN_CHECKPOINT_PATH,
    BNN_PREPROCESSOR_PATH,
    DATA_PATH,
    DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
    DEFAULT_MODEL_DIRS,
    DEVICE,
    EXPERIMENTS_DIR,
)
from src.data.load_data import load_data
from src.data.split import split_features_target
from src.evaluation.evaluate_models import collect_model_artifacts
from src.inference.bnn_inference import (
    load_bnn_artifacts_for_inference,
    predict_proba_and_uncertainty,
    transform_features,
)


EFFICIENCY_RESULTS_DIR = EXPERIMENTS_DIR / "efficiency_results"
EFFICIENCY_SUMMARY_PATH = EFFICIENCY_RESULTS_DIR / "efficiency_summary.json"
EFFICIENCY_TABLE_PATH = EFFICIENCY_RESULTS_DIR / "efficiency_table.csv"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifact_size_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size / (1024 * 1024))


def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    df = load_data(DATA_PATH, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=True)
    return splits.X_test.copy(), splits.y_test.copy()


def try_extract_training_time_seconds(artifact: dict[str, Any]) -> float | None:
    """
    Best-effort lookup for training time from a sibling metrics.json file.
    Returns None if not available.
    """
    model_path = Path(artifact["model_path"])
    metrics_path = model_path.parent / "metrics.json"
    if not metrics_path.exists():
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        return None

    candidate_keys = [
        "training_time_seconds",
        "train_time_seconds",
        "fit_time_seconds",
    ]
    for key in candidate_keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])

    return None


def benchmark_sklearn_model(
    model_path: str | Path,
    X_test: pd.DataFrame,
    num_repeats: int = 3,
) -> dict[str, float]:
    """
    Measure load and inference timings for a sklearn model.
    """
    model_path = Path(model_path)

    load_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        _ = joblib.load(model_path)
        t1 = time.perf_counter()
        load_times.append(t1 - t0)

    model = joblib.load(model_path)

    inference_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        _ = model.predict_proba(X_test)[:, 1]
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

    inference_time_total = float(np.mean(inference_times))
    num_samples = int(len(X_test))

    return {
        "load_time_seconds": float(np.mean(load_times)),
        "inference_time_total_seconds": inference_time_total,
        "latency_ms_per_sample": float((inference_time_total / num_samples) * 1000.0),
        "num_samples": num_samples,
    }


def benchmark_bnn_model(
    checkpoint_path: str | Path,
    preprocessor_path: str | Path,
    X_test: pd.DataFrame,
    num_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
    num_repeats: int = 3,
) -> dict[str, float | int]:
    """
    Measure load and inference timings for the BNN.
    """
    checkpoint_path = Path(checkpoint_path)
    preprocessor_path = Path(preprocessor_path)

    load_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        _ = load_bnn_artifacts_for_inference(
            checkpoint_path=checkpoint_path,
            preprocessor_path=preprocessor_path,
        )
        t1 = time.perf_counter()
        load_times.append(t1 - t0)

    artifacts = load_bnn_artifacts_for_inference(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
    )

    X_transformed = transform_features(
        X=X_test,
        preprocessor=artifacts["preprocessor"],
        feature_names=artifacts["feature_names"],
    )

    inference_times = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        _ = predict_proba_and_uncertainty(
            model=artifacts["model"],
            guide=artifacts["guide"],
            X_transformed=X_transformed,
            num_mc_samples=num_mc_samples,
        )
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

    inference_time_total = float(np.mean(inference_times))
    num_samples = int(len(X_test))

    return {
        "load_time_seconds": float(np.mean(load_times)),
        "inference_time_total_seconds": inference_time_total,
        "latency_ms_per_sample": float((inference_time_total / num_samples) * 1000.0),
        "num_samples": num_samples,
        "mc_samples_used": int(num_mc_samples),
    }


def benchmark_artifact(
    artifact: dict[str, Any],
    X_test: pd.DataFrame,
    bnn_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
) -> dict[str, Any]:
    artifact_type = artifact["artifact_type"]
    model_path = Path(artifact["model_path"])

    record: dict[str, Any] = {
        "model": artifact["qualified_name"],
        "artifact_type": artifact_type,
        "model_path": str(model_path),
        "artifact_size_mb": get_artifact_size_mb(model_path),
        "training_time_seconds": try_extract_training_time_seconds(artifact),
        "device": DEVICE,
    }

    if artifact_type == "sklearn":
        timings = benchmark_sklearn_model(
            model_path=model_path,
            X_test=X_test,
        )
        record.update(timings)
        record["mc_samples_used"] = None
        return record

    if artifact_type == "bnn":
        preprocessor_path = Path(artifact["preprocessor_path"])
        timings = benchmark_bnn_model(
            checkpoint_path=model_path,
            preprocessor_path=preprocessor_path,
            X_test=X_test,
            num_mc_samples=bnn_mc_samples,
        )
        record.update(timings)
        record["preprocessor_path"] = str(preprocessor_path)
        record["preprocessor_size_mb"] = get_artifact_size_mb(preprocessor_path)
        record["artifact_size_mb_total"] = (
            record["artifact_size_mb"] + record["preprocessor_size_mb"]
        )
        return record

    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def summarize_efficiency(results_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_models": int(len(results_df)),
    }

    if results_df.empty:
        return summary

    fastest_load = results_df.sort_values("load_time_seconds", ascending=True).iloc[0]
    fastest_inference = results_df.sort_values(
        "latency_ms_per_sample", ascending=True
    ).iloc[0]
    smallest_artifact = results_df.sort_values("artifact_size_mb", ascending=True).iloc[0]

    summary["fastest_load"] = {
        "model": fastest_load["model"],
        "load_time_seconds": float(fastest_load["load_time_seconds"]),
    }
    summary["fastest_inference"] = {
        "model": fastest_inference["model"],
        "latency_ms_per_sample": float(fastest_inference["latency_ms_per_sample"]),
    }
    summary["smallest_artifact"] = {
        "model": smallest_artifact["model"],
        "artifact_size_mb": float(smallest_artifact["artifact_size_mb"]),
    }

    if "artifact_type" in results_df.columns:
        grouped = (
            results_df.groupby("artifact_type", dropna=False)[
                ["load_time_seconds", "latency_ms_per_sample", "artifact_size_mb"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary["mean_by_artifact_type"] = grouped.to_dict(orient="records")

    return summary


def run_efficiency_benchmark(
    model_dirs: list[str | Path] | None = None,
    output_dir: str | Path = EFFICIENCY_RESULTS_DIR,
    bnn_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
) -> dict[str, Any]:
    """
    Benchmark load and inference efficiency of all saved model artifacts.
    """
    if model_dirs is None:
        model_dirs = DEFAULT_MODEL_DIRS

    output_dir = ensure_dir(output_dir)

    X_test, _ = load_test_split()
    artifacts = collect_model_artifacts(model_dirs)

    if not artifacts:
        raise FileNotFoundError("No supported model artifacts found to benchmark.")

    records = []
    for artifact in artifacts:
        print(f"Benchmarking {artifact['qualified_name']} ({artifact['artifact_type']})...")
        record = benchmark_artifact(
            artifact=artifact,
            X_test=X_test,
            bnn_mc_samples=bnn_mc_samples,
        )
        records.append(record)

    results_df = pd.DataFrame(records).sort_values(
        "latency_ms_per_sample", ascending=True
    ).reset_index(drop=True)

    summary = summarize_efficiency(results_df)
    summary["metadata"] = {
        "device": DEVICE,
        "num_test_samples": int(len(X_test)),
        "bnn_mc_samples": int(bnn_mc_samples),
        "model_dirs": [str(p) for p in model_dirs],
    }

    results_df.to_csv(EFFICIENCY_TABLE_PATH, index=False)
    with EFFICIENCY_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved efficiency table to: {EFFICIENCY_TABLE_PATH}")
    print(f"Saved efficiency summary to: {EFFICIENCY_SUMMARY_PATH}")

    return summary


if __name__ == "__main__":
    run_efficiency_benchmark()