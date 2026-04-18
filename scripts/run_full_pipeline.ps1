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
Write-Host "[1/11] Training baseline models..."
python -m src.training.train_baseline

Write-Host ""
Write-Host "[2/11] Training boosting models..."
python -m src.training.train_boosting

Write-Host ""
Write-Host "[3/11] Training Bayesian Neural Network..."
python -m src.training.train_bnn

Write-Host ""
Write-Host "[4/11] Evaluating all models..."
python -m src.evaluation.evaluate_models

Write-Host ""
Write-Host "[5/11] Running uncertainty analysis..."
python -m src.analysis.uncertainty

Write-Host ""
Write-Host "[6/11] Building uncertainty-aware decision rules..."
python -m src.analysis.decision_rules

Write-Host ""
Write-Host "[7/11] Training GP-LVM..."
python -m src.training.train_gplvm

Write-Host ""
Write-Host "[8/11] Running latent analysis..."
python -m src.analysis.latent_analysis

Write-Host ""
Write-Host "[9/11] Running coverage analysis..."
python -m src.analysis.coverage_risk

Write-Host ""
Write-Host "[10/11] Running decision rules by model..."
python -m src.analysis.decision_rules_by_model

Write-Host ""
Write-Host "[11/11] Running uncertainty threshold curve analysis..."
python -m src.analysis.uncertainty_threshold_curve

Write-Host ""
Write-Host "========================================"
Write-Host "Full pipeline finished successfully"
Write-Host "========================================"