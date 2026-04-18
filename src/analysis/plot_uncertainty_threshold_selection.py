from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PLOT_DPI


CURVE_CSV_PATH = Path(
    "experiments/bnn_results/uncertainty_analysis/uncertainty_threshold_curve.csv"
)
CURVE_SUMMARY_PATH = Path(
    "experiments/bnn_results/uncertainty_analysis/uncertainty_threshold_curve_summary.json"
)
OUTPUT_FIG_PATH = Path(
    "experiments/bnn_results/uncertainty_analysis/uncertainty_threshold_selection.png"
)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run() -> None:
    if not CURVE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find curve CSV at: {CURVE_CSV_PATH}\n"
            "Run `python -m src.analysis.uncertainty_threshold_curve` first."
        )

    df = pd.read_csv(CURVE_CSV_PATH)

    selected_tau = None
    if CURVE_SUMMARY_PATH.exists():
        with CURVE_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        suggested = summary.get("suggested_threshold")
        if suggested is not None:
            selected_tau = float(suggested["uncertainty_threshold"])

    output_path = ensure_parent(OUTPUT_FIG_PATH)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left axis: accept rate
    line1 = ax1.plot(
        df["uncertainty_threshold"],
        df["accept_rate"],
        linewidth=2,
        label="Accept rate",
    )
    ax1.set_xlabel(r"Uncertainty threshold $\tau_u$")
    ax1.set_ylabel("Accept rate")
    ax1.set_ylim(0.0, 1.0)

    # Right axis: fraud in accept
    ax2 = ax1.twinx()
    fraud_accept = df["fraud_in_accept_rate"].astype(float)
    line2 = ax2.plot(
        df["uncertainty_threshold"],
        fraud_accept,
        linewidth=2,
        linestyle="--",
        label="Fraud rate in ACCEPT",
    )
    ax2.set_ylabel("Fraud rate inside ACCEPT")

    if selected_tau is not None:
        ax1.axvline(
            selected_tau,
            linestyle=":",
            linewidth=2,
        )

        closest_idx = (df["uncertainty_threshold"] - selected_tau).abs().idxmin()
        row = df.loc[closest_idx]

        ax1.scatter(
            [row["uncertainty_threshold"]],
            [row["accept_rate"]],
            s=50,
            zorder=5,
        )
        ax2.scatter(
            [row["uncertainty_threshold"]],
            [row["fraud_in_accept_rate"]],
            s=50,
            zorder=5,
        )

        ax1.text(
            row["uncertainty_threshold"],
            min(row["accept_rate"] + 0.06, 0.98),
            rf"$\tau_u \approx {row['uncertainty_threshold']:.3f}$",
        )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title(r"Choosing the uncertainty threshold $\tau_u$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    run()