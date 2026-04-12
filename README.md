# 🧠 Uncertainty-Aware Fraud Detection

### *Bayesian Classification, Proper Scoring Rules & Latent Structure via GP-LVM*

---

## 📌 Overview

Fraud detection in financial transactions is a highly challenging problem due to:

* **Extreme class imbalance** (~1:578 fraud ratio)
* **High cost asymmetry** between false positives and false negatives
* **Uncertainty in model predictions**

Traditional machine learning models focus on **point predictions**, which are often insufficient for real-world decision-making.

---

## 🎯 Project Goal

This project develops a **probabilistic AI framework for fraud detection**, going beyond classification accuracy by incorporating:

* **Bayesian modeling and uncertainty estimation**
* **Proper probabilistic evaluation (proper scoring rules)**
* **Bayesian model comparison and calibration analysis**
* **Latent variable modeling (GP-LVM)** to uncover hidden fraud patterns

---

## 🧩 Methodology

The project is structured into four main components:

---

### 🔹 1. Model Benchmarking

We compare deterministic and probabilistic models:

**Deterministic models**

* Logistic Regression
* Random Forest
* Gradient Boosting (XGBoost / LightGBM)

**Probabilistic model**

* Bayesian Neural Network (BNN) implemented with **Pyro + PyTorch**

---

### 🔹 2. Probabilistic Evaluation

We evaluate models not only on classification performance but also on **probability quality**.

#### Classification metrics (imbalanced setting)

* Precision-Recall AUC (PR-AUC)
* F1-Score
* Recall

#### Proper scoring rules

* Negative Log-Likelihood (NLL)
* Brier Score
* Expected Calibration Error (ECE)

👉 This allows us to evaluate how well models estimate **true probabilities**, not just labels.

---

### 🔹 3. Uncertainty & Decision-Making

Using the Bayesian Neural Network, we quantify predictive uncertainty via Monte Carlo sampling.

We analyze:

* Relationship between uncertainty and classification errors
* Distribution of uncertainty across fraud vs non-fraud
* Detection of ambiguous cases

#### Decision framework

We propose a **risk-aware decision system**:

| Risk Level                             | Decision |
| -------------------------------------- | -------- |
| Low probability + low uncertainty      | ✅ Accept |
| Medium probability or high uncertainty | ⚠ Review |
| High probability + low uncertainty     | ❌ Block  |

👉 This connects probabilistic modeling with **real-world operational decisions**.

---

### 🔹 4. Latent Structure Analysis (GP-LVM)

To explore hidden patterns in fraudulent behavior, we apply a **Gaussian Process Latent Variable Model (GP-LVM)**.

#### Objective

Identify whether fraudulent transactions exhibit **latent structure** not observable in the original feature space.

#### Approach

* Apply GP-LVM on:

  * all fraud cases, or
  * fraud + stratified sample of non-fraud
* Learn a low-dimensional latent space (e.g., 2D)

#### Analysis

* Cluster structure of fraud cases
* Relationship between latent regions and:

  * transaction amount
  * time patterns
  * model uncertainty
  * classification errors

👉 This provides a **probabilistic interpretation of fraud heterogeneity**.

---

## 📊 Dataset

We use the **Credit Card Fraud Detection dataset**:

* ~284,807 transactions
* 28 anonymized PCA features + Amount + Time
* Highly imbalanced fraud distribution

---

## 🧠 Key Research Questions

1. Do Bayesian models improve probability calibration compared to deterministic models?
2. Is uncertainty informative for identifying difficult or ambiguous fraud cases?
3. Can latent variable models reveal hidden structure in fraudulent behavior?
4. How does probabilistic modeling impact decision-making under risk?

---

## ⚙️ Tech Stack

* **PyTorch** — Deep learning
* **Pyro** — Probabilistic programming
* **Scikit-learn** — Baselines
* **XGBoost / LightGBM** — Boosting models
* **properscoring** — Probabilistic metrics
* **GPyTorch / GPflow** — Gaussian Processes / GP-LVM
* **MLflow** — Experiment tracking
* **Streamlit** — Interactive demo

---

## 📁 Project Structure

```bash
.
├── data/                  # Dataset (or download instructions)
├── src/
│   ├── data.py           # Preprocessing & loading
│   ├── models/
│   │   ├── baseline.py
│   │   ├── bnn.py
│   │   ├── gplvm.py
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_bnn.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   ├── analysis/
│   │   ├── uncertainty.py
│   │   ├── latent_space.py
│   ├── utils.py
├── notebooks/            # Exploratory analysis
├── reports/              # Proposal & final report
├── app/                  # Streamlit demo
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train baseline models

```bash
python src/train/train_baseline.py
```

### Train Bayesian Neural Network

```bash
python src/train/train_bnn.py
```

### Evaluate models

```bash
python src/evaluation/run_evaluation.py
```

### Run GP-LVM analysis

```bash
python src/analysis/latent_space.py
```

### Launch demo

```bash
streamlit run app/app.py
```

---

## 📈 Expected Contributions

This project aims to demonstrate that:

* Deterministic models may excel in **classification performance**
* Bayesian models provide better **uncertainty estimation and calibration**
* Proper scoring rules are essential for evaluating probabilistic models
* Latent variable models can uncover **hidden fraud patterns**
* Combining prediction + uncertainty leads to **better decision-making systems**

---

## 🔬 Future Work

* Hierarchical Bayesian models (if group structure available)
* Fully Bayesian hyperparameter marginalization
* Integration with real-time fraud detection pipelines
* Advanced latent models (Deep GP, VAE)

---

## 👤 Author

Antonio Lorenzo Díaz-Meco
Javier Ríos Montes
MSc Artificial Intelligence — ICAI

---
