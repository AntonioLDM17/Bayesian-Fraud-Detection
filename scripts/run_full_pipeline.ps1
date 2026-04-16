$ErrorActionPreference = "Stop"

$DATA_PATH = "data/raw/creditcard.csv"

Write-Host "========================================"
Write-Host "Running full fraud detection pipeline"
Write-Host "========================================"

# 🔍 Check dataset
if (!(Test-Path $DATA_PATH)) {
    Write-Host "❌ ERROR: Dataset not found at $DATA_PATH"
    Write-Host "👉 Please place the file at: data/raw/creditcard.csv"
    exit 1
}

Write-Host "✅ Dataset found"

Write-Host ""
Write-Host "[1/9] Training baseline models..."
python -m src.training.train_baseline

Write-Host ""
Write-Host "[2/9] Training boosting models..."
python -m src.training.train_boosting

Write-Host ""
Write-Host "[3/9] Training Bayesian Neural Network..."
python -m src.training.train_bnn

Write-Host ""
Write-Host "[4/9] Evaluating all models..."
python -m src.evaluation.evaluate_models

Write-Host ""
Write-Host "[5/9] Running uncertainty analysis..."
python -m src.analysis.uncertainty

Write-Host ""
Write-Host "[6/9] Building uncertainty-aware decision rules..."
python -m src.analysis.decision_rules

Write-Host ""
Write-Host "[7/9] Training GP-LVM..."
python -m src.training.train_gplvm

Write-Host ""
Write-Host "[8/9] Running latent analysis..."
python -m src.analysis.latent_analysis

Write-Host ""
Write-Host "[9/9] Running coverage analysis..."
python -m src.analysis.coverage_risk

Write-Host ""
Write-Host "========================================"
Write-Host "Full pipeline finished successfully"
Write-Host "========================================"