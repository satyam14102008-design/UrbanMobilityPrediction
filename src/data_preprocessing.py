"""
Data Preprocessing module for Urban Mobility Demand Prediction.
Handles data loading, validation, missing value imputation, and pipeline transformations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    DATA_DIR,
    ID_COL,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
)
from src.feature_engineering import engineer_all_features


def load_raw_data(
    train_path: Union[str, Path] = TRAIN_PATH,
    test_path: Union[str, Path] = TEST_PATH,
    sample_sub_path: Union[str, Path] = SAMPLE_SUBMISSION_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and sample submission datasets from CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub_df = pd.read_csv(sample_sub_path)
    
    print(f"[Data Loader] Loaded Train: {train_df.shape}, Test: {test_df.shape}, Sample Sub: {sample_sub_df.shape}")
    return train_df, test_df, sample_sub_df


class UrbanMobilityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leak-free Preprocessing Pipeline for Urban Mobility dataset.
    Learns imputation medians, scaling, and encodings strictly from training data.
    """

    def __init__(self, one_hot_encode: bool = False, scale_numerical: bool = False):
        self.one_hot_encode = one_hot_encode
        self.scale_numerical = scale_numerical
        self.imputers: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.ohe: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit imputers and encoders on training data.
        """
        df = X.copy()
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        # Step 1: Feature engineering
        df_feat = engineer_all_features(df)

        # Step 2: Learn median imputation for all numeric columns containing NaN
        for col in df_feat.columns:
            if df_feat[col].isna().sum() > 0:
                self.imputers[col] = float(df_feat[col].median())

        df_imputed = df_feat.copy()
        for col, val in self.imputers.items():
            df_imputed[col] = df_imputed[col].fillna(val)

        # Step 3: Fit Scaler / OHE if requested
        if self.scale_numerical:
            self.scaler = StandardScaler()
            self.scaler.fit(df_imputed.select_dtypes(include=[np.number]))

        if self.one_hot_encode:
            cat_cols = [c for c in CATEGORICAL_COLS if c in df_imputed.columns]
            if cat_cols:
                self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                self.ohe.fit(df_imputed[cat_cols])

        self.feature_names_ = list(df_imputed.columns)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset using fitted parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")

        df = X.copy()
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        # Step 1: Feature engineering
        df_feat = engineer_all_features(df)

        # Step 2: Apply learned imputation
        for col, val in self.imputers.items():
            if col in df_feat.columns:
                df_feat[col] = df_feat[col].fillna(val)

        # Fallback for any unexpected new NaNs
        df_feat = df_feat.fillna(0.0)

        # Step 3: Apply Scaling if configured
        if self.scale_numerical and self.scaler is not None:
            num_cols = df_feat.select_dtypes(include=[np.number]).columns
            df_feat[num_cols] = self.scaler.transform(df_feat[num_cols])

        return df_feat

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, file_path: Union[str, Path]):
        """Save preprocessor state to disk."""
        joblib.dump(self, file_path)
        print(f"[Preprocessor] Saved to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "UrbanMobilityPreprocessor":
        """Load preprocessor state from disk."""
        obj = joblib.load(file_path)
        print(f"[Preprocessor] Loaded from {file_path}")
        return obj


def prepare_datasets(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    preprocessor: Optional[UrbanMobilityPreprocessor] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, UrbanMobilityPreprocessor]:
    """
    Convenience function to prepare X_train, y_train, X_test, and test_ids.
    """
    y_train = train_df[TARGET_COL].copy()
    test_ids = test_df[ID_COL].copy()

    if preprocessor is None:
        preprocessor = UrbanMobilityPreprocessor()
        X_train = preprocessor.fit_transform(train_df)
    else:
        X_train = preprocessor.transform(train_df)

    X_test = preprocessor.transform(test_df)

    return X_train, y_train, X_test, test_ids, preprocessor
