"""
Evaluation module for Urban Mobility Demand Prediction.
Contains calculation of competition metrics (RMSE, RMSLE, MAE, R2),
cross-validation routines, and model benchmarking tables.
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.config import N_SPLITS, RANDOM_STATE, SHUFFLE_CV


def calculate_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate comprehensive regression evaluation metrics.
    Ensures negative predictions are clipped at 0 for valid score calculation.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_clipped = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_clipped))
    
    # MAE
    mae = mean_absolute_error(y_true_arr, y_pred_clipped)
    
    # R2
    r2 = r2_score(y_true_arr, y_pred_clipped)

    # RMSLE (using safe log1p calculation)
    log_true = np.log1p(y_true_arr)
    log_pred = np.log1p(y_pred_clipped)
    rmsle = np.sqrt(mean_squared_error(log_true, log_pred))

    return {
        "RMSE": float(rmse),
        "RMSLE": float(rmsle),
        "MAE": float(mae),
        "R2": float(r2),
    }


def cross_validate_model(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    use_log_target: bool = False,
    n_splits: int = N_SPLITS,
    shuffle: bool = SHUFFLE_CV,
    random_state: int = RANDOM_STATE,
    is_time_series_split: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, List[Any]]:
    """
    Perform cross-validation for a given model.
    
    Parameters:
        model_factory: function returning a fresh model instance
        X: Feature matrix
        y: Target series
        use_log_target: Whether to train on log1p(y) and invert predictions
        n_splits: Number of folds
        shuffle: Whether to shuffle folds (for standard KFold)
        random_state: Random seed
        is_time_series_split: If True, uses TimeSeriesSplit; else KFold
        
    Returns:
        mean_metrics: Dict with average CV metrics across folds
        oof_predictions: Out-of-fold prediction array
        trained_models: List of models fitted per fold
    """
    if is_time_series_split:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    oof_preds = np.zeros(len(X), dtype=np.float64)
    fold_metrics: List[Dict[str, float]] = []
    trained_models: List[Any] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = model_factory()

        if use_log_target:
            y_tr_fit = np.log1p(y_tr)
            model.fit(X_tr, y_tr_fit)
            raw_val_pred = model.predict(X_va)
            val_pred = np.expm1(raw_val_pred)
        else:
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_va)

        val_pred = np.clip(val_pred, 0, None)
        oof_preds[val_idx] = val_pred

        metrics = calculate_metrics(y_va, val_pred)
        fold_metrics.append(metrics)
        trained_models.append(model)

    # Compute aggregate metrics across out-of-fold predictions
    overall_metrics = calculate_metrics(y, oof_preds)
    
    return overall_metrics, oof_preds, trained_models


def print_metrics_summary(model_name: str, metrics: Dict[str, float], use_log: bool):
    """Print formatted metrics summary."""
    transform_str = "log1p(y)" if use_log else "raw target"
    print(f"[{model_name} | {transform_str}] "
          f"RMSE: {metrics['RMSE']:.4f} | "
          f"RMSLE: {metrics['RMSLE']:.4f} | "
          f"MAE: {metrics['MAE']:.4f} | "
          f"R2: {metrics['R2']:.4f}")
