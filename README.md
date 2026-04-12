# 🧠 Uncertainty-Aware Fraud Detection

*A probabilistic machine learning approach to fraud detection using Bayesian Neural Networks and uncertainty-aware decision making.*

---

## 📌 Overview

Fraud detection in financial transactions is a highly challenging binary classification problem characterized by:

* **Extreme class imbalance** (~1:578 fraud ratio)
* **Noisy and anonymized features**
* **High cost asymmetry** between false positives and false negatives

Traditional machine learning models provide **point predictions**, which limits their usefulness in real-world decision-making scenarios.

This project explores a **probabilistic AI approach**, combining deterministic and Bayesian models to:

* Predict fraud probability
* Quantify predictive uncertainty
* Improve decision-making under risk

---

## 🎯 Objectives

The goal of this project is not only to maximize classification performance, but to:

* Compare **deterministic vs probabilistic models**
* Evaluate the **quality of predicted probabilities**
* Analyze **uncertainty as a decision-making signal**
* Bridge the gap between **ML performance and business impact**

---

## 🧩 Models

We implement and compare the following approaches:

### 🔹 Baseline Models

* Logistic Regression
* Random Forest / Gradient Boosting (e.g., XGBoost, LightGBM)

### 🔹 Probabilistic Models

* Bayesian Neural Networks (BNNs) using Variational Inference
* (Optional extensions: hierarchical models or latent variable models)

---

## 📊 Evaluation Metrics

Given the extreme class imbalance, we use robust evaluation metrics:

### Classification Metrics

* **Precision-Recall AUC (PR-AUC)**
* **F1-Score**
* **Recall (Fraud Detection Rate)**

### Probabilistic Metrics (Proper Scoring Rules)

* **Negative Log-Likelihood (NLL)**
* **Brier Score**
* **Expected Calibration Error (ECE)**

👉 These metrics allow us to evaluate not only *what the model predicts*, but *how well it predicts probabilities*.

---

## 🧠 Uncertainty Modeling

A key contribution of this project is the analysis of predictive uncertainty:

* **Aleatoric uncertainty** (data noise)
* **Epistemic uncertainty** (model uncertainty)

We estimate uncertainty using **Monte Carlo sampling** over the posterior in Bayesian models.

---

## 💼 Decision-Making Framework

We extend the model into a **risk-aware decision system**:

* ✅ Low risk → Accept transaction
* ⚠ Medium risk + high uncertainty → Flag for human review
* ❌ High risk → Block transaction

This demonstrates how uncertainty can be translated into **real business value**.

---

## ⚙️ Tech Stack

* **PyTorch** (core deep learning framework)
* **Pyro** (probabilistic programming for Bayesian models)
* **Scikit-learn / XGBoost / LightGBM** (baselines)
* **MLflow** (experiment tracking)
* **Streamlit** (interactive demo)

---

## 📁 Project Structure

```
.
├── data/                # Dataset (or download instructions)
├── src/
│   ├── data.py         # Data loading and preprocessing
│   ├── models/
│   │   ├── baseline.py
│   │   ├── bnn.py
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_bnn.py
│   ├── evaluate.py
│   ├── utils.py
├── notebooks/          # Exploratory analysis
├── reports/            # Proposal and final report
├── app/                # Streamlit demo
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train baseline models

```bash
python src/train/train_baseline.py
```

### 3. Train Bayesian Neural Network

```bash
python src/train/train_bnn.py
```

### 4. Evaluate models

```bash
python src/evaluate.py
```

### 5. Launch demo

```bash
streamlit run app/app.py
```

---

## 📈 Expected Contributions

This project aims to show that:

* Deterministic models may achieve strong classification performance
* Bayesian models provide **better-calibrated probabilities**
* Uncertainty enables **more robust decision-making**

👉 The key insight:

> *The best model is not only the most accurate, but the one that knows when it is uncertain.*

---

## 🔬 Future Work

* Hierarchical Bayesian models (user-level structure)
* Latent variable models (e.g., GP-LVM)
* Fully Bayesian hyperparameter marginalization
* Deployment in real-time decision systems

---

## 👤 Authors

Antonio Lorenzo Díaz-Meco
Javier Ríos Montes
MSc Artificial Intelligence — ICAI

---
