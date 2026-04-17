from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import (
    BNN_CHECKPOINT_PATH,
    BNN_PREPROCESSOR_PATH,
    DECISION_CSV_PATH,
    FULL_EVALUATION_PATH,
    UNCERTAINTY_CSV_PATH,
    UNCERTAINTY_SUMMARY_PATH,
)
from src.inference.bnn_inference import load_bnn_artifacts_for_inference
from src.inference.decision_inference import (
    load_decision_context,
    predict_decision_for_batch,
)


st.set_page_config(
    page_title="Bayesian Fraud Detection",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_app_resources() -> dict:
    artifacts = load_bnn_artifacts_for_inference(
        checkpoint_path=BNN_CHECKPOINT_PATH,
        preprocessor_path=BNN_PREPROCESSOR_PATH,
    )
    decision_context = load_decision_context(
        evaluation_json_path=FULL_EVALUATION_PATH,
    )
    return {
        "artifacts": artifacts,
        "decision_context": decision_context,
    }


@st.cache_data(show_spinner=False)
def load_uncertainty_dataset() -> pd.DataFrame:
    path = Path(UNCERTAINTY_CSV_PATH)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_decision_dataset() -> pd.DataFrame:
    path = Path(DECISION_CSV_PATH)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uncertainty_summary() -> dict:
    path = Path(UNCERTAINTY_SUMMARY_PATH)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_full_evaluation() -> dict:
    path = Path(FULL_EVALUATION_PATH)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model_comparison_table(eval_data: dict) -> pd.DataFrame:
    if not eval_data or "models" not in eval_data:
        return pd.DataFrame()

    rows = []
    for qualified_name, model_info in eval_data["models"].items():
        test = model_info["test"]
        cls = test["classification_metrics"]
        prob = test["proper_scoring_rules"]
        cal = test["calibration_metrics"]

        rows.append(
            {
                "model": qualified_name,
                "artifact_type": model_info.get("artifact_type", ""),
                "pr_auc_test": cls["pr_auc"],
                "roc_auc_test": cls["roc_auc"],
                "f1_test": cls["f1"],
                "precision_test": cls["precision"],
                "recall_test": cls["recall"],
                "nll_test": prob["nll"],
                "brier_test": prob["brier_score"],
                "ece_test": cal["ece"],
                "optimal_threshold": model_info.get(
                    "selected_threshold_from_validation",
                    test.get("optimal_threshold"),
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pr_auc_test", ascending=False).reset_index(drop=True)

    return df


def get_feature_names() -> list[str]:
    resources = load_app_resources()
    return resources["artifacts"]["feature_names"]


def render_sidebar_info() -> None:
    resources = load_app_resources()
    decision_context = resources["decision_context"]

    st.sidebar.header("Model info")
    st.sidebar.write(f"Checkpoint: `{Path(BNN_CHECKPOINT_PATH).name}`")
    st.sidebar.write(f"Preprocessor: `{Path(BNN_PREPROCESSOR_PATH).name}`")
    st.sidebar.write(f"Evaluation file: `{Path(FULL_EVALUATION_PATH).name}`")

    st.sidebar.header("Decision thresholds")
    st.sidebar.write(
        {
            "optimal_threshold": round(decision_context["optimal_threshold"], 6),
            "block_probability_threshold": round(
                decision_context["block_probability_threshold"], 6
            ),
            "uncertainty_threshold": round(
                decision_context["uncertainty_threshold"], 6
            ),
        }
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Note: input features V1–V28 are anonymized PCA components from the dataset, "
        "so this interface is designed for case inspection, batch scoring, and monitoring."
    )


def validate_uploaded_dataframe(df: pd.DataFrame, feature_names: list[str]) -> None:
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required feature columns: "
            f"{missing}"
        )


def classify_risk_level(probability: float, optimal_threshold: float) -> str:
    if probability >= optimal_threshold:
        return "High"
    if probability >= 0.5 * optimal_threshold:
        return "Medium"
    return "Low"


def classify_confidence_level(uncertainty: float, uncertainty_threshold: float) -> str:
    if uncertainty <= 0.5 * uncertainty_threshold:
        return "High"
    if uncertainty <= uncertainty_threshold:
        return "Medium"
    return "Low"


def add_context_columns(
    df: pd.DataFrame,
    decision_context: dict,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()

    df["risk_level"] = df["predicted_probability"].apply(
        lambda p: classify_risk_level(
            probability=float(p),
            optimal_threshold=decision_context["optimal_threshold"],
        )
    )
    df["confidence_level"] = df["uncertainty_std"].apply(
        lambda u: classify_confidence_level(
            uncertainty=float(u),
            uncertainty_threshold=decision_context["uncertainty_threshold"],
        )
    )

    if reference_df is not None and not reference_df.empty:
        ref_probs = reference_df["predicted_probability"].to_numpy()
        ref_unc = reference_df["uncertainty_std"].to_numpy()

        df["probability_percentile"] = df["predicted_probability"].apply(
            lambda x: float((ref_probs <= x).mean() * 100.0)
        )
        df["uncertainty_percentile"] = df["uncertainty_std"].apply(
            lambda x: float((ref_unc <= x).mean() * 100.0)
        )

    return df


def explain_decision(row: pd.Series, decision_context: dict) -> str:
    prob = float(row["predicted_probability"])
    unc = float(row["uncertainty_std"])
    decision = row["decision"]

    if decision == "BLOCK":
        return (
            f"Blocked because fraud probability ({prob:.4f}) is above the optimal threshold "
            f"({decision_context['optimal_threshold']:.4f}) and uncertainty ({unc:.4f}) "
            f"is below the uncertainty threshold ({decision_context['uncertainty_threshold']:.4f})."
        )

    if decision == "REVIEW":
        if unc > decision_context["uncertainty_threshold"]:
            return (
                f"Sent to review because uncertainty ({unc:.4f}) is above the uncertainty "
                f"threshold ({decision_context['uncertainty_threshold']:.4f})."
            )
        return (
            "Sent to review because the case is suspicious but does not meet the "
            "strict blocking rule with sufficient confidence."
        )

    return (
        f"Accepted because fraud probability ({prob:.4f}) is below the optimal threshold "
        f"({decision_context['optimal_threshold']:.4f})."
    )


def compute_system_status(dashboard_df: pd.DataFrame) -> tuple[str, str]:
    error_rate = (dashboard_df["y_pred"] != dashboard_df["y_true"]).mean()
    mean_unc = dashboard_df["uncertainty_std"].mean()

    review_rate = 0.0
    if "decision" in dashboard_df.columns:
        review_rate = dashboard_df["decision"].eq("REVIEW").mean()

    if error_rate > 0.02 or mean_unc > 0.25:
        return (
            "🔴 Risk",
            "Model is making too many errors or is highly uncertain.",
        )

    if mean_unc > 0.15 or review_rate > 0.3:
        return (
            "🟡 Caution",
            "Model uncertainty is significant. Monitor predictions carefully.",
        )

    return (
        "🟢 Healthy",
        "Model is performing reliably with low uncertainty.",
    )


def compute_system_alerts(dashboard_df: pd.DataFrame) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []

    error_rate = (dashboard_df["y_pred"] != dashboard_df["y_true"]).mean()
    mean_unc = dashboard_df["uncertainty_std"].mean()

    review_rate = 0.0
    block_rate = 0.0
    if "decision" in dashboard_df.columns:
        review_rate = dashboard_df["decision"].eq("REVIEW").mean()
        block_rate = dashboard_df["decision"].eq("BLOCK").mean()

    high_unc_rate = (dashboard_df["uncertainty_std"] > 0.25).mean()

    fn_rate = 0.0
    if {"y_true", "y_pred"}.issubset(dashboard_df.columns):
        positives = (dashboard_df["y_true"] == 1).sum()
        if positives > 0:
            false_negatives = (
                (dashboard_df["y_true"] == 1) & (dashboard_df["y_pred"] == 0)
            ).sum()
            fn_rate = false_negatives / positives

    if error_rate > 0.02:
        alerts.append(("error", f"High observed error rate: {error_rate:.2%}"))

    if fn_rate > 0.15:
        alerts.append(("error", f"False negative rate is elevated: {fn_rate:.2%}"))

    if mean_unc > 0.20:
        alerts.append(("warning", f"Average uncertainty is high: {mean_unc:.4f}"))

    if review_rate > 0.40:
        alerts.append(("warning", f"Review rate is high: {review_rate:.2%}"))

    if high_unc_rate > 0.20:
        alerts.append(
            ("warning", f"Large fraction of cases have high uncertainty: {high_unc_rate:.2%}")
        )

    if block_rate == 0:
        alerts.append(("info", "No transactions are being automatically blocked."))

    if not alerts:
        alerts.append(("info", "No major alerts detected."))

    return alerts


def render_system_status(dashboard_df: pd.DataFrame) -> None:
    status, message = compute_system_status(dashboard_df)

    st.markdown("## 🧭 System Status")

    if "🟢" in status:
        st.success(f"{status} — {message}")
    elif "🟡" in status:
        st.warning(f"{status} — {message}")
    else:
        st.error(f"{status} — {message}")


def render_system_alerts(dashboard_df: pd.DataFrame) -> None:
    alerts = compute_system_alerts(dashboard_df)

    st.markdown("## 🚨 Alerts")
    for level, message in alerts:
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)


def render_batch_interpretation(results: pd.DataFrame) -> None:
    if len(results) == 0:
        return

    review_rate = results["decision"].eq("REVIEW").mean()
    block_rate = results["decision"].eq("BLOCK").mean()
    mean_unc = results["uncertainty_std"].mean()

    messages: list[str] = []

    if review_rate > 0.4:
        messages.append("High uncertainty: many transactions require manual review.")

    if block_rate > 0.05:
        messages.append("Non-trivial fraud detected: automatic blocking is active.")

    if mean_unc > 0.2:
        messages.append("Batch is harder than usual (high average uncertainty).")

    if messages:
        st.markdown("### Key insights")
        for msg in messages:
            st.warning(msg)


def render_case_interpretation(case: pd.Series) -> None:
    messages: list[str] = []

    risk = case.get("risk_level")
    confidence = case.get("confidence_level")
    confusion = case.get("confusion_type")

    if risk == "High" and confidence == "High":
        messages.append("High-risk transaction with high confidence.")
    elif confidence == "Low":
        messages.append("Model is uncertain: decision should be treated carefully.")

    if confusion == "FN":
        messages.append("False negative: fraud not detected (critical error).")
    elif confusion == "FP":
        messages.append("False positive: legitimate transaction flagged.")

    if messages:
        st.markdown("### Key interpretation")
        for msg in messages:
            st.warning(msg)


def render_model_comparison_insights(comparison_df: pd.DataFrame) -> None:
    if comparison_df.empty:
        return

    best_pr = comparison_df.sort_values("pr_auc_test", ascending=False).iloc[0]
    best_cal = comparison_df.sort_values("ece_test", ascending=True).iloc[0]

    st.markdown("### Key conclusions")
    st.info(
        f"Best model (ranking): {best_pr['model']} (PR-AUC {best_pr['pr_auc_test']:.4f})"
    )
    st.info(
        f"Best calibrated model: {best_cal['model']} (ECE {best_cal['ece_test']:.4f})"
    )


def render_dashboard_interpretation(
    dashboard_df: pd.DataFrame,
    uncertainty_summary: dict,
) -> None:
    messages: list[str] = []

    error_rate = (dashboard_df["y_pred"] != dashboard_df["y_true"]).mean()
    mean_unc = dashboard_df["uncertainty_std"].mean()

    if error_rate < 0.001:
        messages.append("Very low error rate on test data.")

    if mean_unc > 0.18:
        messages.append("Uncertainty is significant → REVIEW step is justified.")

    if messages:
        st.markdown("### Key system insights")
        for msg in messages:
            st.info(msg)


def render_batch_scoring(feature_names: list[str]) -> None:
    st.subheader("Batch scoring")
    st.write(
        "Upload a CSV file containing the same feature columns used during training. "
        "This mode is the most realistic for this dataset because the input variables "
        "are anonymized PCA components."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_csv_uploader",
    )

    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(uploaded_file)
        validate_uploaded_dataframe(input_df, feature_names)

        st.write("Preview of uploaded data")
        st.dataframe(input_df.head(), use_container_width=True)

        if st.button("Run batch scoring", type="primary"):
            with st.spinner("Running inference..."):
                decision_context = load_app_resources()["decision_context"]
                reference_df = load_uncertainty_dataset()

                results = predict_decision_for_batch(
                    X=input_df,
                    evaluation_json_path=FULL_EVALUATION_PATH,
                    num_mc_samples=200,
                    include_input_columns=True,
                )
                results = add_context_columns(
                    results,
                    decision_context=decision_context,
                    reference_df=reference_df if not reference_df.empty else None,
                )
                results["decision_explanation"] = results.apply(
                    lambda row: explain_decision(row, decision_context),
                    axis=1,
                )

            st.success("Batch scoring completed.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(results)}")
            c2.metric("Mean fraud probability", f"{results['predicted_probability'].mean():.4f}")
            c3.metric("Mean uncertainty", f"{results['uncertainty_std'].mean():.4f}")
            c4.metric("Review rate", f"{(results['decision'].eq('REVIEW').mean() * 100):.2f}%")

            render_batch_interpretation(results)

            st.write("Decision counts")
            decision_counts = (
                results["decision"]
                .value_counts(dropna=False)
                .rename_axis("decision")
                .reset_index(name="count")
            )
            st.dataframe(decision_counts, use_container_width=True)

            st.write("Filters")
            f1, f2 = st.columns(2)
            selected_decisions = f1.multiselect(
                "Decision",
                options=sorted(results["decision"].unique().tolist()),
                default=sorted(results["decision"].unique().tolist()),
            )
            min_probability = f2.slider(
                "Minimum fraud probability",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )

            filtered = results[
                results["decision"].isin(selected_decisions)
                & (results["predicted_probability"] >= min_probability)
            ].copy()

            st.write("Scored transactions")
            st.dataframe(filtered, use_container_width=True)

            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download filtered results as CSV",
                data=csv_bytes,
                file_name="fraud_predictions_with_uncertainty.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Batch scoring failed: {exc}")


def render_case_explorer() -> None:
    st.subheader("Case explorer")
    st.write(
        "Inspect saved predictions on the test set. This is more interpretable than manual "
        "input because it shows real scored cases from the project pipeline."
    )

    uncertainty_df = load_uncertainty_dataset()
    decision_df = load_decision_dataset()
    decision_context = load_app_resources()["decision_context"]

    if uncertainty_df.empty:
        st.info(
            "No saved uncertainty dataset found. Run `python -m src.analysis.uncertainty` first."
        )
        return

    df = uncertainty_df.copy()
    if not decision_df.empty and "row_id" in decision_df.columns:
        if "decision" in decision_df.columns:
            df = df.merge(
                decision_df[["row_id", "decision"]],
                on="row_id",
                how="left",
            )

    df = add_context_columns(df, decision_context=decision_context, reference_df=uncertainty_df)
    if "decision" in df.columns:
        df["decision_explanation"] = df.apply(
            lambda row: explain_decision(row, decision_context),
            axis=1,
        )

    c1, c2, c3 = st.columns(3)
    selected_confusion = c1.selectbox(
        "Filter by confusion type",
        options=["All"] + sorted(df["confusion_type"].dropna().unique().tolist()),
    )
    selected_decision = c2.selectbox(
        "Filter by decision",
        options=["All"] + sorted(df["decision"].dropna().unique().tolist()) if "decision" in df.columns else ["All"],
    )
    sort_by = c3.selectbox(
        "Sort by",
        options=["predicted_probability", "uncertainty_std", "row_id"],
        index=1,
    )

    filtered = df.copy()
    if selected_confusion != "All":
        filtered = filtered[filtered["confusion_type"] == selected_confusion]
    if "decision" in filtered.columns and selected_decision != "All":
        filtered = filtered[filtered["decision"] == selected_decision]

    filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)

    st.write(f"Matching cases: {len(filtered)}")
    st.dataframe(filtered.head(200), use_container_width=True)

    if len(filtered) == 0:
        return

    selected_index = st.number_input(
        "Select row index to inspect",
        min_value=0,
        max_value=max(len(filtered) - 1, 0),
        value=0,
        step=1,
    )

    case = filtered.iloc[int(selected_index)]

    st.markdown("### Selected case")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Fraud probability", f"{float(case['predicted_probability']):.4f}")
    cc2.metric("Uncertainty", f"{float(case['uncertainty_std']):.4f}")
    cc3.metric("True label", str(case["y_true"]))
    cc4.metric("Predicted label", str(case["y_pred"]))

    dd1, dd2, dd3 = st.columns(3)
    dd1.metric("Confusion type", str(case["confusion_type"]))
    dd2.metric("Risk level", str(case["risk_level"]))
    dd3.metric("Confidence level", str(case["confidence_level"]))

    if "decision" in case.index:
        st.metric("Decision", str(case["decision"]))

    if "probability_percentile" in case.index and "uncertainty_percentile" in case.index:
        st.write(
            {
                "probability_percentile": round(float(case["probability_percentile"]), 2),
                "uncertainty_percentile": round(float(case["uncertainty_percentile"]), 2),
            }
        )

    if "decision_explanation" in case.index:
        st.info(str(case["decision_explanation"]))

    render_case_interpretation(case)

    with st.expander("Raw case data"):
        st.write(case.to_dict())


def render_model_comparison(eval_data: dict) -> None:
    st.markdown("### Model comparison")

    comparison_df = build_model_comparison_table(eval_data)
    if comparison_df.empty:
        st.info("No model comparison data available.")
        return

    st.dataframe(comparison_df, use_container_width=True)

    metric = st.selectbox(
        "Select comparison metric",
        options=[
            "pr_auc_test",
            "roc_auc_test",
            "f1_test",
            "recall_test",
            "nll_test",
            "brier_test",
            "ece_test",
        ],
        index=0,
    )

    plot_df = comparison_df.set_index("model")[[metric]]
    st.bar_chart(plot_df)

    best_row = comparison_df.sort_values(
        metric,
        ascending=(metric in {"nll_test", "brier_test", "ece_test"}),
    ).iloc[0]

    st.write(
        {
            "best_model_for_selected_metric": best_row["model"],
            "metric": metric,
            "value": round(float(best_row[metric]), 6),
        }
    )

    render_model_comparison_insights(comparison_df)


def render_system_dashboard() -> None:
    st.subheader("System dashboard")
    st.write(
        "This dashboard summarizes the current behavior of the fraud detection system "
        "on the saved test-set outputs."
    )

    uncertainty_df = load_uncertainty_dataset()
    decision_df = load_decision_dataset()
    uncertainty_summary = load_uncertainty_summary()
    full_evaluation = load_full_evaluation()

    if uncertainty_df.empty:
        st.info(
            "No saved uncertainty dataset found. Run `python -m src.analysis.uncertainty` first."
        )
        return

    dashboard_df = uncertainty_df.copy()
    if (
        not decision_df.empty
        and "row_id" in decision_df.columns
        and "decision" in decision_df.columns
    ):
        dashboard_df = dashboard_df.merge(
            decision_df[["row_id", "decision"]],
            on="row_id",
            how="left",
        )

    render_system_status(dashboard_df)
    render_system_alerts(dashboard_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(dashboard_df)}")
    c2.metric("Mean fraud probability", f"{dashboard_df['predicted_probability'].mean():.4f}")
    c3.metric("Mean uncertainty", f"{dashboard_df['uncertainty_std'].mean():.4f}")
    c4.metric("Observed error rate", f"{(dashboard_df['y_pred'] != dashboard_df['y_true']).mean():.4%}")

    render_dashboard_interpretation(dashboard_df, uncertainty_summary)
    render_model_comparison(full_evaluation)

    st.markdown("### Decision distribution")
    if "decision" in dashboard_df.columns:
        decision_counts = dashboard_df["decision"].value_counts(dropna=False)
        st.bar_chart(decision_counts)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Fraud probability distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(dashboard_df["predicted_probability"], bins=40)
        ax1.set_xlabel("Predicted fraud probability")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of predicted probabilities")
        st.pyplot(fig1)

    with col_right:
        st.markdown("### Uncertainty distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(dashboard_df["uncertainty_std"], bins=40)
        ax2.set_xlabel("Predictive uncertainty (std)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of predictive uncertainty")
        st.pyplot(fig2)

    st.markdown("### Fraud probability vs uncertainty")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(
        dashboard_df["predicted_probability"],
        dashboard_df["uncertainty_std"],
        s=8,
        alpha=0.35,
    )
    ax3.set_xlabel("Predicted fraud probability")
    ax3.set_ylabel("Predictive uncertainty")
    ax3.set_title("Probability vs uncertainty")
    st.pyplot(fig3)

    if uncertainty_summary:
        coverage_risk = uncertainty_summary.get("coverage_risk", [])
        selective_prediction = uncertainty_summary.get("selective_prediction", {})

        if coverage_risk:
            st.markdown("### Coverage vs risk")
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            coverage = [point["coverage"] for point in coverage_risk]
            risk = [point["risk"] for point in coverage_risk]
            ax4.plot(coverage, risk)
            ax4.set_xlabel("Coverage")
            ax4.set_ylabel("Risk (error rate)")
            ax4.set_title("Coverage vs risk")
            st.pyplot(fig4)

        if selective_prediction:
            st.markdown("### Coverage vs accuracy")
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            selective_items = sorted(
                selective_prediction.items(),
                key=lambda kv: float(kv[1]["coverage"]),
            )
            valid_items = [
                item for item in selective_items if item[1]["accuracy"] is not None
            ]
            coverage_vals = [float(item[1]["coverage"]) for item in valid_items]
            accuracy_vals = [float(item[1]["accuracy"]) for item in valid_items]

            ax5.plot(coverage_vals, accuracy_vals)
            ax5.set_xlabel("Coverage")
            ax5.set_ylabel("Accuracy")
            ax5.set_title("Coverage vs accuracy")
            st.pyplot(fig5)

        st.markdown("### Saved uncertainty summary")
        st.json(
            uncertainty_summary["overall"]
            if "overall" in uncertainty_summary
            else uncertainty_summary
        )


def main() -> None:
    st.title("🧠 Bayesian Fraud Detection")
    st.markdown(
        """
        This app focuses on **usable inspection and monitoring** rather than manual entry of
        anonymized PCA features.

        It provides three views:

        - **Batch scoring** for real transaction files
        - **Case explorer** for inspecting saved model outputs on test cases
        - **System dashboard** for monitoring probability, uncertainty, decisions, and model comparison
        """
    )

    render_sidebar_info()

    try:
        feature_names = get_feature_names()
    except Exception as exc:
        st.error(f"Failed to load model resources: {exc}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["Batch scoring", "Case explorer", "System dashboard"]
    )

    with tab1:
        render_batch_scoring(feature_names)

    with tab2:
        render_case_explorer()

    with tab3:
        render_system_dashboard()

    with st.expander("Required model input columns"):
        st.write(feature_names)


if __name__ == "__main__":
    main()