"""
Configuration module for Urban Mobility Demand Prediction.
Contains directory paths, feature schemas, hyperparameters, and global constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_PLOTS_DIR = OUTPUTS_DIR / "eda_plots"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, EDA_PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data File Paths
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Output File Paths
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
FEATURE_IMPORTANCE_CSV = OUTPUTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = OUTPUTS_DIR / "feature_importance.png"

# Target and ID Columns
TARGET_COL = "demand"
ID_COL = "id"

# Raw Column Types
CATEGORICAL_COLS = [
    "season_code",
    "category_weather",
]

BINARY_COLS = [
    "flag_holiday",
    "flag_working",
]

NUMERICAL_COLS = [
    "feature_temp",
    "feature_feel_temp",
    "feature_humidity",
    "feature_wind",
]

TEMPORAL_COLS = [
    "Year",
    "Month",
    "Day_of_Week",
    "Hour",
]

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Cross Validation Configuration
N_SPLITS = 5
SHUFFLE_CV = True

# Primary Metric
PRIMARY_METRIC = "RMSE"
