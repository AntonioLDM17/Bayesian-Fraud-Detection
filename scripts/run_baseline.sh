#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/raw/creditcard.csv"

echo "========================================"
echo "Running baseline training"
echo "========================================"

# 🔍 Check dataset
if [ ! -f "$DATA_PATH" ]; then
  echo "❌ ERROR: Dataset not found at $DATA_PATH"
  echo "👉 Please place the file at: data/raw/creditcard.csv"
  exit 1
fi

echo "✅ Dataset found"

echo ""
echo "[1/2] Training baseline models..."
python.exe -m src.training.train_baseline

echo ""
echo "[2/2] Training boosting models..."
python.exe -m src.training.train_boosting

echo ""
echo "========================================"
echo "Baseline pipeline finished successfully"
echo "========================================"