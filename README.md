# 🧠 Uncertainty-Aware Fraud Detection

### *Bayesian Classification, Proper Scoring Rules, Risk-Aware Decisions, and Latent Structure Analysis*

---

## 📌 Overview

Fraud detection in financial transactions is a challenging probabilistic AI problem because it combines:

- **Extreme class imbalance** (fraud is very rare)
- **Asymmetric costs** between false positives and false negatives
- The need for **probability estimates**, not just hard labels
- The need for **uncertainty-aware decisions** in ambiguous cases

Traditional machine learning models often focus on point predictions only. In real-world fraud detection, however, institutions need to know:

- how confident the model is,
- when a case should be escalated,
- and whether fraud exhibits hidden latent structure.

This project builds a **probabilistic fraud detection pipeline** that combines strong predictive baselines with Bayesian modeling, uncertainty analysis, and latent variable modeling.

---

## 🎯 Project Goal

The goal of this project is to study whether probabilistic AI methods add practical value to fraud detection beyond standard classification accuracy.

In particular, we focus on four questions:

1. How do Bayesian and non-Bayesian models compare in fraud detection performance?
2. Are predictive probabilities well calibrated?
3. Is model uncertainty informative for identifying difficult cases?
4. Can latent variable models reveal hidden structure in fraudulent transactions?

---

## 🧩 Methodology

The project is organized into four main components.

---

### 1. Model Benchmarking

We compare deterministic and probabilistic approaches on the same fraud dataset.

#### Deterministic models
- Logistic Regression
- Random Forest
- Boosting model: XGBoost

#### Probabilistic model
- Bayesian Neural Network (BNN) implemented with **Pyro + PyTorch**

The goal is not only to compare predictive accuracy, but also to understand the trade-off between predictive performance and uncertainty awareness.

---

### 2. Probabilistic Evaluation

Models are evaluated both as classifiers and as probabilistic predictors.

#### Classification metrics
These are especially important under extreme imbalance:
- **PR-AUC**
- **ROC-AUC**
- **F1-score**
- **Precision**
- **Recall**
- **Specificity**
- **Accuracy**

#### Proper scoring rules
To evaluate probability quality:
- **Negative Log-Likelihood (NLL)**
- **Brier Score**
- **Log Score**

#### Calibration analysis
To assess whether predicted probabilities reflect real frequencies:
- **Expected Calibration Error (ECE)**
- **Maximum Calibration Error (MCE)**
- calibration tables by confidence bin

This allows us to measure whether a model is only good at ranking fraud cases, or whether it also outputs meaningful probabilities.

---

### 3. Uncertainty-Aware Decision-Making

Using the Bayesian Neural Network, we estimate predictive uncertainty through **Monte Carlo sampling**.

We then analyze:
- uncertainty distribution across fraud and non-fraud,
- uncertainty in correct vs incorrect predictions,
- relationship between uncertainty and ambiguous cases.

#### Decision policy

Rather than using the BNN as a simple binary classifier, we turn it into a **risk-aware decision system**:

- **ACCEPT** → low predicted fraud risk
- **REVIEW** → medium risk or high uncertainty
- **BLOCK** → high fraud probability with low uncertainty

This is one of the key contributions of the project: uncertainty is used not only for interpretation, but also for **actionable decision-making**.

---

### 4. Latent Structure Analysis with GP-LVM

To explore hidden fraud patterns, we train a **Gaussian Process Latent Variable Model (GP-LVM)** on a manageable subset of the test split.

The GP-LVM learns a low-dimensional latent representation of transactions, which we use to study:

- whether fraud forms one or several patterns,
- whether fraud overlaps with normal behavior,
- whether uncertainty is concentrated in ambiguous regions,
- whether review/block decisions align with latent structure.

This part adds interpretability and helps explain **why fraud detection is difficult**.

---

## 📊 Dataset

We use the **Credit Card Fraud Detection dataset**.

### Main properties
- 284,807 transactions
- 28 anonymized PCA-based variables (`V1` to `V28`)
- `Time`
- `Amount`
- target column: `Class`
- extreme imbalance: roughly **1 fraud per 578 transactions**

---

## 🧠 Main Findings So Far

The current implementation supports the following conclusions:

### Predictive performance
- **XGBoost** is currently the strongest predictive model in terms of PR-AUC and F1.
- **Random Forest** is also very competitive.
- The **Bayesian Neural Network does not outperform boosting methods** in raw classification performance.

### Probability quality
- Proper scoring rules and calibration metrics allow us to compare models beyond accuracy alone.
- In this dataset, strong boosting models also perform very well probabilistically.

### Uncertainty
- The BNN produces meaningful predictive uncertainty.
- Incorrect predictions tend to be associated with **higher uncertainty** than correct predictions.
- This makes the BNN useful even when it is not the best pure classifier.

### Decision policy
- A simple uncertainty-aware policy can separate:
  - obviously safe transactions,
  - highly suspicious transactions,
  - ambiguous cases requiring manual review.

### Latent structure
- Fraud does **not** behave as a single coherent cluster.
- GP-LVM suggests that fraud is **heterogeneous**, with multiple latent patterns and partial overlap with normal behavior.
- This helps explain why some fraud cases are easy while others are much harder.

---

## ⚙️ Tech Stack

- **Python**
- **PyTorch**
- **Pyro**
- **scikit-learn**
- **XGBoost**
- **NumPy / pandas**
- **Matplotlib**

---

## 📁 Project Structure

```bash
.
├── data/
│   └── raw/
│       └── creditcard.csv
├── experiments/
│   ├── baseline_results/
│   ├── boosting_results/
│   ├── bnn_results/
│   ├── gplvm_results/
│   └── full_evaluation.json
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── boosting.py
│   │   ├── bnn.py
│   │   └── gplvm.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_boosting.py
│   │   ├── train_bnn.py
│   │   └── train_gplvm.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── proper_scoring.py
│   │   └── evaluate_models.py
│   └── analysis/
│       ├── uncertainty.py
│       ├── decision_rules.py
│       └── latent_analysis.py
├── notebooks/
├── reports/
├── requirements.txt
└── README.md
````

---

## 🚀 How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Place the dataset

Put the CSV file at:

```bash
data/raw/creditcard.csv
```

---

## 3. Train deterministic baselines

```bash
python -m src.training.train_baseline
python -m src.training.train_boosting
```

This generates:

* saved models in `experiments/baseline_results/`
* saved models in `experiments/boosting_results/`

---

## 4. Train the Bayesian Neural Network

```bash
python -m src.training.train_bnn
```

This generates:

* BNN checkpoint
* fitted preprocessor
* training metrics

inside:

```bash
experiments/bnn_results/
```

---

## 5. Run full model evaluation

```bash
python -m src.evaluation.evaluate_models
```

This creates a consolidated evaluation file:

```bash
experiments/full_evaluation.json
```

---

## 6. Run uncertainty analysis

```bash
python -m src.analysis.uncertainty
```

This generates:

* per-sample uncertainty CSV
* uncertainty summary JSON
* plots of uncertainty by class / error type

inside:

```bash
experiments/bnn_results/uncertainty_analysis/
```

---

## 7. Build uncertainty-aware decisions

```bash
python -m src.analysis.decision_rules
```

This generates:

* per-sample decision CSV
* decision summary JSON

inside:

```bash
experiments/bnn_results/decision_analysis/
```

---

## 8. Train the GP-LVM

```bash
python -m src.training.train_gplvm
```

This generates:

* GP-LVM checkpoint
* latent embeddings CSV
* latent plots

inside:

```bash
experiments/gplvm_results/
```

---

## 9. Run latent analysis

```bash
python -m src.analysis.latent_analysis
```

This merges:

* latent embeddings,
* BNN uncertainty,
* and decision outputs

using `row_id`, and produces:

* merged CSV
* latent-space plots by class, uncertainty, confusion type, and decision

inside:

```bash
experiments/gplvm_results/latent_analysis/
```

---

## 📈 Recommended Execution Order

For full reproducibility, run:

```bash
python -m src.training.train_baseline
python -m src.training.train_boosting
python -m src.training.train_bnn
python -m src.evaluation.evaluate_models
python -m src.analysis.uncertainty
python -m src.analysis.decision_rules
python -m src.training.train_gplvm
python -m src.analysis.latent_analysis
```

---

## 🔬 Research Questions

This project is centered around the following questions:

1. Do Bayesian models improve uncertainty awareness, even if they do not win in pure predictive performance?
2. Are proper scoring rules necessary to evaluate fraud models meaningfully?
3. Can uncertainty be used to design better operational decisions?
4. Does latent structure help explain fraud heterogeneity and model errors?

---

## 🛠 Current Status

At the current stage, the repository already supports:

* baseline model benchmarking,
* probabilistic evaluation,
* BNN uncertainty estimation,
* uncertainty-aware decision rules,
* GP-LVM latent analysis,
* row-level alignment between latent space and uncertainty analysis.

The next natural improvements would be:

* tuning decision thresholds on validation data,
* strengthening the final report and visual presentation,
* and optionally extending the project with hierarchical or deeper probabilistic models.

---

## 👤 Authors

Antonio Lorenzo Díaz-Meco
Javier Ríos Montes

MSc in Artificial Intelligence — ICAI

```

