# 🧠 Uncertainty-Aware Fraud Detection

### *Bayesian Classification, Probabilistic Evaluation, Uncertainty-Aware Decisions, Selective Prediction, GP-LVM Analysis, and Streamlit Deployment*

---

## 📌 Overview

Fraud detection is a natural probabilistic AI problem because it combines:

- **extreme class imbalance**
- **asymmetric costs** between false positives and false negatives
- the need for **probability estimates**, not only hard labels
- the need to know **when the model should not be trusted**
- the need to convert predictions into **operational actions**

This repository implements an end-to-end fraud detection pipeline that combines:

- strong deterministic baselines
- a **Bayesian Neural Network (BNN)** for predictive uncertainty
- **probabilistic evaluation** beyond accuracy
- **uncertainty-aware decision rules**
- **selective prediction / coverage-risk analysis**
- **latent structure analysis** with **GP-LVM**
- a **Streamlit app** for scoring, inspection, and monitoring
- simple **efficiency** and **monitoring** utilities
- a **Dockerized deployment path** for the app

---

## 🎯 Project Goal

The goal of the project is to study whether probabilistic AI methods add practical value to fraud detection beyond standard predictive performance.

More specifically, the project asks:

1. How do Bayesian and non-Bayesian models compare in fraud detection?
2. Are predicted probabilities meaningful and well calibrated?
3. Is model uncertainty useful for identifying difficult cases?
4. Can uncertainty be translated into real actions such as **ACCEPT / REVIEW / BLOCK**?
5. Can latent variable models reveal hidden structure in fraudulent transactions?

---

## 🧩 Current Pipeline

The project is organized as a full pipeline:

1. **Train deterministic baselines**
   - Logistic Regression
   - Random Forest
   - XGBoost

2. **Train a Bayesian Neural Network**
   - Pyro + PyTorch
   - Monte Carlo predictive inference
   - uncertainty estimation from Bayesian weight posterior samples

3. **Evaluate all models probabilistically**
   - classification metrics
   - proper scoring rules
   - calibration metrics
   - validation-based threshold selection

4. **Run uncertainty analysis**
   - predictive probability
   - predictive uncertainty
   - uncertainty by class and error type

5. **Build uncertainty-aware decision rules**
   - `ACCEPT`
   - `REVIEW`
   - `BLOCK`

6. **Run selective prediction analysis**
   - coverage vs risk
   - coverage vs accuracy

7. **Train a GP-LVM**
   - latent 2D representation of a test subset
   - latent interpretation of class structure, uncertainty, errors, and decisions

8. **Serve the system through a Streamlit app**
   - batch scoring
   - case explorer
   - system dashboard
   - model comparison
   - monitoring-style status and alerts

---

## 🤖 Models Included

### Deterministic models
- Logistic Regression
- Random Forest
- XGBoost

### Probabilistic model
- Bayesian Neural Network implemented with **Pyro + PyTorch**

### Latent variable model
- Gaussian Process Latent Variable Model (**GP-LVM**)

---

## 📊 Evaluation

The repository evaluates models both as classifiers and as probabilistic predictors.

### Classification metrics
- PR-AUC
- ROC-AUC
- F1-score
- Precision
- Recall
- Specificity
- Accuracy

### Proper scoring rules
- Negative Log-Likelihood (NLL)
- Brier Score
- Log Score

### Calibration metrics
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- calibration tables by confidence bin

### Uncertainty-aware analysis
- uncertainty by true class
- uncertainty by confusion type
- probability vs uncertainty
- coverage-risk curves
- selective prediction summaries

---

## 🧠 Uncertainty-Aware Decision Policy

The BNN is not used only as a binary classifier. It is turned into a decision system:

- **ACCEPT** → low predicted fraud risk
- **REVIEW** → uncertain or ambiguous case
- **BLOCK** → high fraud probability with sufficiently low uncertainty

This is one of the main practical contributions of the repository:  
uncertainty is used not only for interpretation, but also for **actionable decision support**.

---

## 🔍 Selective Prediction

The repository includes a selective prediction analysis based on uncertainty:

- if we only act on low-uncertainty cases, coverage decreases
- but risk also decreases
- this quantifies the trade-off between **automation** and **safety**

Generated outputs include:

- `coverage vs risk`
- `coverage vs accuracy`
- JSON summaries with operating points

---

## 🧬 Latent Structure Analysis with GP-LVM

A GP-LVM is trained on a manageable subset of the test split in order to study:

- latent structure of fraud vs non-fraud
- concentration of uncertainty
- regions associated with false positives / false negatives
- relationship between latent geometry and `ACCEPT / REVIEW / BLOCK`

This adds an interpretability layer beyond standard predictive metrics.

---

## 📦 Dataset

The repository uses the **Credit Card Fraud Detection dataset**.

### Main properties
- 284,807 transactions
- 28 anonymized PCA-based variables: `V1` to `V28`
- `Time`
- `Amount`
- target column: `Class`
- extreme imbalance: roughly **1 fraud per 578 transactions**

Because `V1`–`V28` are anonymized PCA components, the app is designed for:

- **batch scoring**
- **case inspection**
- **system monitoring**

rather than fully semantic manual data entry.

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Pyro
- scikit-learn
- XGBoost
- NumPy
- pandas
- Matplotlib
- Streamlit
- Docker

---

## 📁 Project Structure

```bash
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── README.md
├── experiments/
│   ├── baseline_results/
│   ├── boosting_results/
│   ├── bnn_results/
│   │   ├── decision_analysis/
│   │   └── uncertainty_analysis/
│   ├── gplvm_results/
│   │   └── latent_analysis/
│   ├── efficiency_results/
│   ├── monitoring_results/
│   ├── full_evaluation.json
│   └── model_comparison_table.csv
├── notebooks/
│   └── comparative_results_summary.ipynb
├── scripts/
│   ├── run_baseline.sh
│   ├── run_bnn.sh
│   ├── run_full_pipeline.ps1
│   └── run_full_pipeline.sh
├── src/
│   ├── analysis/
│   │   ├── coverage_risk.py
│   │   ├── decision_rules.py
│   │   ├── efficiency.py
│   │   ├── latent_analysis.py
│   │   ├── model_comparison.py
│   │   ├── monitoring.py
│   │   └── uncertainty.py
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── evaluation/
│   │   ├── calibration.py
│   │   ├── evaluate_models.py
│   │   ├── metrics.py
│   │   ├── proper_scoring.py
│   │   └── thresholds.py
│   ├── inference/
│   │   ├── bnn_inference.py
│   │   └── decision_inference.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── bnn.py
│   │   ├── boosting.py
│   │   └── gplvm.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_bnn.py
│   │   ├── train_boosting.py
│   │   └── train_gplvm.py
│   ├── utils/
│   │   └── seed.py
│   └── visualization/
│       └── calibration_plots.py
├── Dockerfile
├── requirements.txt
└── README.md
````

---

## 🚀 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```bash
data/raw/creditcard.csv
```

---

## ▶️ Recommended Execution Order

### Option A — full pipeline in one command

#### PowerShell

```powershell
.\scripts\run_full_pipeline.ps1
```

#### Bash

```bash
bash scripts/run_full_pipeline.sh
```

This runs:

1. baseline training
2. boosting training
3. BNN training
4. evaluation
5. uncertainty analysis
6. decision rules
7. GP-LVM training
8. latent analysis
9. coverage-risk analysis

---

### Option B — run each step manually

#### 1. Train deterministic baselines

```bash
python -m src.training.train_baseline
python -m src.training.train_boosting
```

#### 2. Train the Bayesian Neural Network

```bash
python -m src.training.train_bnn
```

#### 3. Evaluate all models

```bash
python -m src.evaluation.evaluate_models
```

#### 4. Run uncertainty analysis

```bash
python -m src.analysis.uncertainty
```

#### 5. Build decision rules

```bash
python -m src.analysis.decision_rules
```

#### 6. Train GP-LVM

```bash
python -m src.training.train_gplvm
```

#### 7. Run latent analysis

```bash
python -m src.analysis.latent_analysis
```

#### 8. Run coverage-risk / selective prediction analysis

```bash
python -m src.analysis.coverage_risk
```

#### 9. Benchmark efficiency

```bash
python -m src.analysis.efficiency
```

#### 10. Build final model comparison table

```bash
python -m src.analysis.model_comparison
```

#### 11. Build monitoring summary

```bash
python -m src.analysis.monitoring
```

---

## 📂 Main Outputs

### Core evaluation

* `experiments/full_evaluation.json`

### BNN uncertainty analysis

* `experiments/bnn_results/uncertainty_analysis/`

  * per-sample CSV
  * summary JSON
  * uncertainty plots
  * coverage-risk outputs

### Decision analysis

* `experiments/bnn_results/decision_analysis/`

  * per-sample decisions
  * decision summary JSON

### GP-LVM analysis

* `experiments/gplvm_results/`
* `experiments/gplvm_results/latent_analysis/`

### Efficiency benchmarking

* `experiments/efficiency_results/`

### Monitoring snapshot

* `experiments/monitoring_results/`

### Final comparison table

* `experiments/model_comparison_table.csv`
* `experiments/model_comparison_table.md`

---

## 🖥 Streamlit App

The repository includes a Streamlit app for:

* **batch scoring**
* **case explorer**
* **system dashboard**
* **model comparison**
* **status and alerts**

Run it locally with:

```bash
streamlit run app/streamlit_app.py
```

The app expects the trained artifacts and generated experiment outputs to exist, especially:

* `experiments/full_evaluation.json`
* `experiments/bnn_results/bayesian_neural_network.pt`
* `experiments/bnn_results/preprocessor.joblib`
* `experiments/bnn_results/uncertainty_analysis/test_uncertainty_per_sample.csv`
* `experiments/bnn_results/decision_analysis/test_decisions_per_sample.csv`

---

## 🐳 Docker Deployment

A `Dockerfile` is included for containerized deployment of the Streamlit app.

### Build the image

```bash
docker build -t bayesian-fraud-detection .
```

### Run the container

```bash
docker run -p 8501:8501 bayesian-fraud-detection
```

Then open:

```text
http://localhost:8501
```

### Note

The Docker image copies the repository contents as-is, so the required experiment artifacts must already be present when building the image.

---

## 📓 Notebook

The repository includes one exploratory notebook:

* `notebooks/comparative_results_summary.ipynb`

It is intended as a lightweight analysis notebook for inspecting saved evaluation outputs and comparison tables.

---

## 🛠 Current Status

The repository currently supports:

* end-to-end model training
* probabilistic evaluation
* validation-based threshold reuse on test
* BNN uncertainty estimation
* uncertainty-aware decision rules
* selective prediction / coverage-risk analysis
* GP-LVM latent analysis
* reusable BNN inference utilities
* Streamlit app for scoring and monitoring
* efficiency benchmarking
* monitoring snapshot generation
* Docker-based app deployment

### Still incomplete / lightweight
* monitoring is currently an **offline snapshot-style** report, not a live production service
* Docker deployment currently targets the **app only**, not the full training pipeline

---

## 🔬 Research Questions

This project is centered around the following questions:

1. Do Bayesian models improve uncertainty awareness, even if they do not win in pure predictive performance?
2. Are proper scoring rules necessary to evaluate fraud models meaningfully?
3. Can uncertainty be used to design better operational decisions?
4. Does selective prediction reduce risk in practice?
5. Does latent structure help explain fraud heterogeneity and model errors?

---

## 👤 Authors

Antonio Lorenzo Díaz-Meco
Javier Ríos Montes

MSc in Artificial Intelligence — ICAI
