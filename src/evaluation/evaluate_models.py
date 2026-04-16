from __future__ import annotations

from pathlib import Path

import torch


# ============================================================
# Core project constants
# ============================================================

RANDOM_STATE: int = 42
TARGET_COLUMN: str = "Class"
ROW_ID_COLUMN: str = "row_id"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

BASELINE_RESULTS_DIR = EXPERIMENTS_DIR / "baseline_results"
BOOSTING_RESULTS_DIR = EXPERIMENTS_DIR / "boosting_results"
BNN_RESULTS_DIR = EXPERIMENTS_DIR / "bnn_results"
GPLVM_RESULTS_DIR = EXPERIMENTS_DIR / "gplvm_results"

BNN_UNCERTAINTY_DIR = BNN_RESULTS_DIR / "uncertainty_analysis"
BNN_DECISION_DIR = BNN_RESULTS_DIR / "decision_analysis"
GPLVM_LATENT_ANALYSIS_DIR = GPLVM_RESULTS_DIR / "latent_analysis"

FULL_EVALUATION_PATH = EXPERIMENTS_DIR / "full_evaluation.json"

BNN_CHECKPOINT_PATH = BNN_RESULTS_DIR / "bayesian_neural_network.pt"
BNN_PREPROCESSOR_PATH = BNN_RESULTS_DIR / "preprocessor.joblib"
BNN_METRICS_PATH = BNN_RESULTS_DIR / "metrics.json"

GPLVM_CHECKPOINT_PATH = GPLVM_RESULTS_DIR / "gplvm.pt"
GPLVM_PREPROCESSOR_PATH = GPLVM_RESULTS_DIR / "preprocessor.joblib"
GPLVM_LATENT_CSV_PATH = GPLVM_RESULTS_DIR / "latent_embeddings.csv"
GPLVM_TRAINING_SUMMARY_PATH = GPLVM_RESULTS_DIR / "training_summary.json"

UNCERTAINTY_CSV_PATH = BNN_UNCERTAINTY_DIR / "test_uncertainty_per_sample.csv"
UNCERTAINTY_SUMMARY_PATH = BNN_UNCERTAINTY_DIR / "test_uncertainty_summary.json"

DECISION_CSV_PATH = BNN_DECISION_DIR / "test_decisions_per_sample.csv"
DECISION_SUMMARY_PATH = BNN_DECISION_DIR / "test_decision_summary.json"


# ============================================================
# Data split defaults
# ============================================================

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_VAL_SIZE: float = 0.2


# ============================================================
# Evaluation defaults
# ============================================================

DEFAULT_THRESHOLD: float = 0.5
DEFAULT_N_BINS: int = 10

DEFAULT_BNN_MC_SAMPLES: int = 100
DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES: int = 200


# ============================================================
# Active model families
# ============================================================

BASELINE_MODEL_NAMES: list[str] = [
    "logistic_regression",
    "random_forest",
]

BOOSTING_MODEL_NAMES: list[str] = [
    "xgboost",
]

USE_LIGHTGBM: bool = False


# ============================================================
# Baseline model hyperparameters
# ============================================================

LOGISTIC_MAX_ITER: int = 2000
RF_N_ESTIMATORS: int = 300
RF_MAX_DEPTH: int | None = None
RF_MIN_SAMPLES_SPLIT: int = 2
RF_MIN_SAMPLES_LEAF: int = 1


# ============================================================
# Boosting model hyperparameters
# ============================================================

XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 5
XGB_LEARNING_RATE: float = 0.05
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE_BYTREE: float = 0.8


# ============================================================
# Bayesian Neural Network configuration
# ============================================================

BNN_HIDDEN_DIM_1: int = 256
BNN_HIDDEN_DIM_2: int = 128
BNN_PRIOR_SCALE: float = 0.9
BNN_DROPOUT_RATE: float = 0.0  # disabled for cleaner uncertainty interpretation

BNN_LEARNING_RATE: float = 2e-4
BNN_NUM_EPOCHS: int = 150
BNN_BATCH_SIZE: int = 512

BNN_TRAIN_MC_SAMPLES: int = 100
BNN_EVAL_MC_SAMPLES: int = 200

BNN_EARLY_STOPPING_PATIENCE: int = 30
BNN_MIN_DELTA: float = 1e-4


# ============================================================
# GP-LVM configuration
# ============================================================

GPLVM_LATENT_DIM: int = 2
GPLVM_NUM_EPOCHS: int = 1500
GPLVM_LEARNING_RATE: float = 0.03
GPLVM_LATENT_REG_WEIGHT: float = 1e-3
GPLVM_PRINT_EVERY: int = 50

GPLVM_NONFRAUD_MULTIPLIER: int = 3
GPLVM_MAX_NONFRAUD: int = 1500
GPLVM_SOURCE_SPLIT: str = "test"


# ============================================================
# Uncertainty-aware decision policy
# ============================================================

BLOCK_PROBABILITY_THRESHOLD: float = 0.80
REVIEW_PROBABILITY_THRESHOLD: float = 0.30
UNCERTAINTY_THRESHOLD: float = 0.10


# ============================================================
# Convenience collections
# ============================================================

DEFAULT_MODEL_DIRS: list[Path] = [
    BASELINE_RESULTS_DIR,
    BOOSTING_RESULTS_DIR,
    BNN_RESULTS_DIR,
    GPLVM_RESULTS_DIR
]

PLOT_DPI: int = 200
LATENT_PLOT_DPI: int = 220