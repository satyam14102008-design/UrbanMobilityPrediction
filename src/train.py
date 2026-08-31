"""
Training Pipeline for Urban Mobility Demand Prediction.
Runs baseline models, target transformation comparisons, Optuna hyperparameter tuning,
blending ensemble, model evaluation, feature importance analysis, and model serialization.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_IMPORTANCE_CSV,
    FEATURE_IMPORTANCE_PNG,
    FEATURE_NAMES_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    N_SPLITS,
    OUTPUTS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TRAIN_PATH,
)
from src.data_preprocessing import UrbanMobilityPreprocessor, load_raw_data
from src.evaluate import calculate_metrics, cross_validate_model, print_metrics_summary

# Set Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_baseline_model_factories() -> Dict[str, Callable[[], Any]]:
    """
    Return dictionary of baseline model factory callables.
    """
    return {
        "Ridge_Regression": lambda: Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Random_Forest": lambda: RandomForestRegressor(
            n_estimators=150, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Hist_Gradient_Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=200, max_depth=10, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "CatBoost": lambda: cb.CatBoostRegressor(
            iterations=350,
            learning_rate=0.05,
            depth=7,
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        ),
    }


def tune_lightgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 25) -> Dict[str, Any]:
    """
    Tune LightGBM hyperparameters with Optuna on log1p(y).
    """
    print(f"\n[Hyperparameter Tuning] Running Optuna for LightGBM ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": -1,
        }
        factory = lambda: lgb.LGBMRegressor(**params)
        metrics, _, _ = cross_validate_model(factory, X, y, use_log_target=True, n_splits=5)
        return metrics["RMSE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    print(f"[Optuna LightGBM] Best RMSE: {study.best_value:.4f}")
    return study.best_params


def tune_catboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
    """
    Tune CatBoost hyperparameters with Optuna on log1p(y).
    """
    print(f"\n[Hyperparameter Tuning] Running Optuna for CatBoost ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            "depth": trial.suggest_int("depth", 6, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "thread_count": -1,
        }
        factory = lambda: cb.CatBoostRegressor(**params)
        metrics, _, _ = cross_validate_model(factory, X, y, use_log_target=True, n_splits=5)
        return metrics["RMSE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    print(f"[Optuna CatBoost] Best RMSE: {study.best_value:.4f}")
    return study.best_params


def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters with Optuna on log1p(y).
    """
    print(f"\n[Hyperparameter Tuning] Running Optuna for XGBoost ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        factory = lambda: xgb.XGBRegressor(**params)
        metrics, _, _ = cross_validate_model(factory, X, y, use_log_target=True, n_splits=5)
        return metrics["RMSE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    print(f"[Optuna XGBoost] Best RMSE: {study.best_value:.4f}")
    return study.best_params


class EnsembleModel:
    """
    Weighted Blend Ensemble of multiple regression models trained on log1p(y).
    """

    def __init__(self, models: List[Tuple[str, Any]], weights: List[float], use_log_target: bool = True):
        self.models = models  # list of (name, fitted_model)
        self.weights = np.array(weights) / np.sum(weights)
        self.use_log_target = use_log_target

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for name, model in self.models:
            raw_p = model.predict(X)
            if self.use_log_target:
                p = np.expm1(raw_p)
            else:
                p = raw_p
            preds.append(p)

        ensemble_pred = np.zeros(len(X))
        for w, p in zip(self.weights, preds):
            ensemble_pred += w * p

        return np.clip(ensemble_pred, 0, None)


def optimize_ensemble_weights(oof_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """
    Find non-negative weights that minimize ensemble RMSE on out-of-fold predictions.
    """
    model_names = list(oof_dict.keys())
    matrix = np.column_stack([oof_dict[name] for name in model_names])

    def loss(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = matrix @ w
        return np.sqrt(np.mean((y_true - pred) ** 2))

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in model_names]
    res = minimize(loss, init_weights, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1})
    
    weights = res.x / np.sum(res.x)
    return weights


def plot_and_save_feature_importance(model: Any, feature_names: List[str]):
    """
    Extract, print, save feature importance table and chart.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
    else:
        print("[Feature Importance] Model does not expose feature_importances_.")
        return

    # Normalize importances
    importances = 100.0 * (importances / np.sum(importances))
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    
    # Save CSV
    feat_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    print(f"\n[Feature Importance] Saved top features to {FEATURE_IMPORTANCE_CSV}")
    print("\n--- TOP 15 MOST IMPORTANT FEATURES ---")
    print(feat_df.head(15).to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 8), dpi=200)
    top_feats = feat_df.head(20).sort_values("Importance", ascending=True)
    colors = sns.color_palette("mako", len(top_feats))
    bars = plt.barh(top_feats["Feature"], top_feats["Importance"], color=colors, edgecolor="black", alpha=0.85)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontsize=9)

    plt.title("Top 20 Features Driving Mobility Demand Prediction", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance (%)", fontsize=12)
    plt.ylabel("Engineered & Raw Features", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG)
    plt.close()
    print(f"[Feature Importance] Chart saved to {FEATURE_IMPORTANCE_PNG}")


def run_training_pipeline() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Execute full end-to-end training, benchmarking, tuning, and artifact generation.
    """
    print("=" * 70)
    print("🚀 STARTING URBAN MOBILITY DEMAND PREDICTION TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load data
    train_df, test_df, sample_sub = load_raw_data()

    # 2. Fit preprocessor
    print("\n[Data Preparation] Fitting preprocessor and engineering features...")
    preprocessor = UrbanMobilityPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["demand"].copy()
    feature_names = preprocessor.feature_names_
    
    print(f"[Data Preparation] Preprocessed Features: {X_train.shape[1]} columns")
    preprocessor.save(PREPROCESSOR_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    # 3. Model benchmarking (Phase 6 & 7: Raw vs Log1p target)
    print("\n" + "=" * 70)
    print("PHASE 6 & 7: BENCHMARKING MODELS (RAW TARGET VS LOG1P TARGET)")
    print("=" * 70)

    baseline_factories = get_baseline_model_factories()
    comparison_records: List[Dict[str, Any]] = []
    oof_predictions_dict: Dict[str, np.ndarray] = {}

    for name, factory in baseline_factories.items():
        # Evaluate without target transform
        metrics_raw, oof_raw, _ = cross_validate_model(factory, X_train, y_train, use_log_target=False)
        print_metrics_summary(name, metrics_raw, use_log=False)
        comparison_records.append({
            "Model": name,
            "Target_Transform": "None (Raw)",
            "RMSE": metrics_raw["RMSE"],
            "RMSLE": metrics_raw["RMSLE"],
            "MAE": metrics_raw["MAE"],
            "R2": metrics_raw["R2"],
        })

        # Evaluate with log1p target transform
        metrics_log, oof_log, _ = cross_validate_model(factory, X_train, y_train, use_log_target=True)
        print_metrics_summary(name, metrics_log, use_log=True)
        comparison_records.append({
            "Model": name,
            "Target_Transform": "log1p(demand)",
            "RMSE": metrics_log["RMSE"],
            "RMSLE": metrics_log["RMSLE"],
            "MAE": metrics_log["MAE"],
            "R2": metrics_log["R2"],
        })

        oof_predictions_dict[f"{name}_log1p"] = oof_log

    # 4. Phase 8: Hyperparameter Tuning
    print("\n" + "=" * 70)
    print("PHASE 8: OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)

    best_lgb_params = tune_lightgbm(X_train, y_train, n_trials=25)
    best_cb_params = tune_catboost(X_train, y_train, n_trials=20)
    best_xgb_params = tune_xgboost(X_train, y_train, n_trials=20)

    # Evaluate Tuned Models
    tuned_factories = {
        "LightGBM_Tuned": lambda: lgb.LGBMRegressor(**best_lgb_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        "CatBoost_Tuned": lambda: cb.CatBoostRegressor(**best_cb_params, random_seed=RANDOM_STATE, verbose=False, thread_count=-1),
        "XGBoost_Tuned": lambda: xgb.XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, n_jobs=-1),
    }

    tuned_oof_dict: Dict[str, np.ndarray] = {}
    for name, factory in tuned_factories.items():
        metrics, oof, _ = cross_validate_model(factory, X_train, y_train, use_log_target=True)
        print_metrics_summary(name, metrics, use_log=True)
        comparison_records.append({
            "Model": name,
            "Target_Transform": "log1p(demand)",
            "RMSE": metrics["RMSE"],
            "RMSLE": metrics["RMSLE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
        })
        tuned_oof_dict[name] = oof

    # 5. Phase 9: Ensembling
    print("\n" + "=" * 70)
    print("PHASE 9: BLENDED ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    opt_weights = optimize_ensemble_weights(tuned_oof_dict, y_train.values)
    model_names = list(tuned_oof_dict.keys())
    print("[Ensemble] Optimal Model Weights:")
    for name, weight in zip(model_names, opt_weights):
        print(f"  - {name}: {weight * 100:.2f}%")

    ensemble_matrix = np.column_stack([tuned_oof_dict[name] for name in model_names])
    ensemble_oof_preds = np.clip(ensemble_matrix @ opt_weights, 0, None)
    ensemble_metrics = calculate_metrics(y_train.values, ensemble_oof_preds)
    print_metrics_summary("Ensemble (LightGBM + CatBoost + XGBoost)", ensemble_metrics, use_log=True)

    comparison_records.append({
        "Model": "Ensemble (LGBM+CatBoost+XGB)",
        "Target_Transform": "log1p(demand)",
        "RMSE": ensemble_metrics["RMSE"],
        "RMSLE": ensemble_metrics["RMSLE"],
        "MAE": ensemble_metrics["MAE"],
        "R2": ensemble_metrics["R2"],
    })

    # Save Model Comparison Table
    comp_df = pd.DataFrame(comparison_records).sort_values("RMSE", ascending=True).reset_index(drop=True)
    comp_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\n[Comparison Table] Saved results to {MODEL_COMPARISON_PATH}")
    print("\n" + comp_df.to_string(index=False))

    # 6. Phase 10: Final Model Fitting & Serialization
    print("\n" + "=" * 70)
    print("PHASE 10: TRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 70)

    y_train_log = np.log1p(y_train)
    
    # Fit each individual winning model on full training set
    fitted_lgb = lgb.LGBMRegressor(**best_lgb_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    fitted_lgb.fit(X_train, y_train_log)

    fitted_cb = cb.CatBoostRegressor(**best_cb_params, random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
    fitted_cb.fit(X_train, y_train_log)

    fitted_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    fitted_xgb.fit(X_train, y_train_log)

    final_ensemble = EnsembleModel(
        models=[
            ("LightGBM", fitted_lgb),
            ("CatBoost", fitted_cb),
            ("XGBoost", fitted_xgb),
        ],
        weights=opt_weights,
        use_log_target=True,
    )

    # Save Ensemble Model
    joblib.dump(final_ensemble, BEST_MODEL_PATH)
    print(f"[Model Registry] Serialized final ensemble model to {BEST_MODEL_PATH}")

    # Feature Importance from top tree model
    plot_and_save_feature_importance(fitted_lgb, feature_names)

    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "best_model": final_ensemble,
        "preprocessor": preprocessor,
        "metrics": ensemble_metrics,
        "weights": dict(zip(model_names, opt_weights)),
    }, comp_df


if __name__ == "__main__":
    run_training_pipeline()
