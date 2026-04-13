#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/raw/creditcard.csv"

echo "========================================"
echo "Running Bayesian Neural Network pipeline"
echo "========================================"

# 🔍 Check dataset
if [ ! -f "$DATA_PATH" ]; then
  echo "❌ ERROR: Dataset not found at $DATA_PATH"
  echo "👉 Please place the file at: data/raw/creditcard.csv"
  exit 1
fi

echo "✅ Dataset found"

echo ""
echo "[1/3] Training BNN..."
python.exe -m src.training.train_bnn

echo ""
echo "[2/3] Running uncertainty analysis..."
python.exe -m src.analysis.uncertainty

echo ""
echo "[3/3] Building decision rules..."
python.exe -m src.analysis.decision_rules

echo ""
echo "========================================"
echo "BNN pipeline finished successfully"
echo "========================================"