"""
Inference Pipeline for Urban Mobility Demand Prediction.
Loads trained models and preprocessor, predicts on test.csv,
strictly validates submission format, and exports submission.csv.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    ID_COL,
    OUTPUTS_DIR,
    PREPROCESSOR_PATH,
    PROJECT_ROOT,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
)
from src.data_preprocessing import UrbanMobilityPreprocessor


def generate_submission(
    test_path: Union[str, Path] = TEST_PATH,
    sample_sub_path: Union[str, Path] = SAMPLE_SUBMISSION_PATH,
    model_path: Union[str, Path] = BEST_MODEL_PATH,
    preprocessor_path: Union[str, Path] = PREPROCESSOR_PATH,
    output_path: Union[str, Path] = SUBMISSION_PATH,
) -> pd.DataFrame:
    """
    Generate, validate, and save competition submission.
    """
    print("=" * 70)
    print("🚀 GENERATING PREDICTIONS FOR KAGGLE SUBMISSION")
    print("=" * 70)

    # 1. Load test data and sample submission
    print(f"[Inference] Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    # 2. Load model and preprocessor
    print(f"[Inference] Loading preprocessor from {preprocessor_path}...")
    preprocessor = UrbanMobilityPreprocessor.load(preprocessor_path)
    
    print(f"[Inference] Loading model from {model_path}...")
    model = joblib.load(model_path)

    # 3. Transform test features
    X_test = preprocessor.transform(test_df)
    print(f"[Inference] Preprocessed test shape: {X_test.shape}")

    # 4. Generate predictions
    raw_predictions = model.predict(X_test)

    # 5. Post-process predictions
    # Demand cannot be negative; clip at 0 (or lower bound of 0)
    predictions = np.clip(raw_predictions, 0, None)

    # 6. Create submission dataframe
    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: predictions,
    })

    # 7. Strict Validation Checks
    print("\n--- RUNNING SUBMISSION VALIDATION CHECKS ---")
    
    # Check 1: Row count
    assert len(submission) == len(sample_sub), (
        f"Row count mismatch! Expected {len(sample_sub)}, got {len(submission)}"
    )
    print(f"✅ Row count check passed: {len(submission)} rows")

    # Check 2: Column names and order
    assert list(submission.columns) == list(sample_sub.columns), (
        f"Column mismatch! Expected {list(sample_sub.columns)}, got {list(submission.columns)}"
    )
    print(f"✅ Column names & order check passed: {list(submission.columns)}")

    # Check 3: ID alignment
    assert (submission[ID_COL].values == sample_sub[ID_COL].values).all(), (
        "ID alignment mismatch between submission and sample_submission!"
    )
    print("✅ ID alignment check passed")

    # Check 4: No null / NaN / Inf values
    assert not submission.isna().any().any(), "Submission contains NaN values!"
    assert not np.isinf(submission[TARGET_COL]).any(), "Submission contains infinite values!"
    print("✅ Null and Inf check passed: 0 missing values")

    # Check 5: Non-negative values
    assert (submission[TARGET_COL] >= 0).all(), "Submission contains negative demand values!"
    print("✅ Non-negative check passed: all demand values >= 0")

    # 8. Save submission
    submission.to_csv(output_path, index=False)
    # Also save to root directory for user convenience
    root_sub_path = PROJECT_ROOT / "submission.csv"
    submission.to_csv(root_sub_path, index=False)

    print(f"\n[Inference] Saved submission to:")
    print(f"  1. {output_path}")
    print(f"  2. {root_sub_path}")

    # Display prediction statistics
    print("\n--- PREDICTION SUMMARY STATISTICS ---")
    print(submission[TARGET_COL].describe().to_string())

    print("\n--- FIRST 10 SUBMISSION ROWS ---")
    print(submission.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ SUBMISSION FILE READY FOR KAGGLE UPLOAD!")
    print("=" * 70)

    return submission


if __name__ == "__main__":
    generate_submission()
