#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/raw/creditcard.csv"

echo "========================================"
echo "Running full fraud detection pipeline"
echo "========================================"

# 🔍 Check dataset
if [ ! -f "$DATA_PATH" ]; then
  echo "❌ ERROR: Dataset not found at $DATA_PATH"
  echo "👉 Please place the file at: data/raw/creditcard.csv"
  exit 1
fi

echo "✅ Dataset found"

echo ""
echo "[1/9] Training baseline models..."
python.exe -m src.training.train_baseline

echo ""
echo "[2/9] Training boosting models..."
python.exe -m src.training.train_boosting

echo ""
echo "[3/9] Training Bayesian Neural Network..."
python.exe -m src.training.train_bnn

echo ""
echo "[4/9] Evaluating all models..."
python.exe -m src.evaluation.evaluate_models

echo ""
echo "[5/9] Running uncertainty analysis..."
python.exe -m src.analysis.uncertainty

echo ""
echo "[6/9] Building uncertainty-aware decision rules..."
python.exe -m src.analysis.decision_rules

echo ""
echo "[7/9] Training GP-LVM..."
python.exe -m src.training.train_gplvm

echo ""
echo "[8/9] Running latent analysis..."
python.exe -m src.analysis.latent_analysis

echo ""
echo "[9/9] Running coverage analysis..."
python.exe -m src.analysis.coverage_analysis
echo ""
echo "========================================"
echo "Full pipeline finished successfully"
echo "========================================"