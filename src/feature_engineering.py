"""
Feature Engineering module for Urban Mobility Demand Prediction.
Generates domain-specific temporal, cyclical, peak, and weather interaction features.
"""

import numpy as np
import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time, rush hour, peak, and trend features.
    """
    df = df.copy()

    # Peak hour indicators
    df["is_morning_peak"] = ((df["Hour"] >= 7) & (df["Hour"] <= 9)).astype(int)
    df["is_evening_peak"] = ((df["Hour"] >= 16) & (df["Hour"] <= 19)).astype(int)
    
    # Rush hour: peak hours on working days
    df["is_rush_hour"] = (
        ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)) 
        & (df["flag_working"] == 1)
    ).astype(int)

    # Weekend / Off-day indicator
    df["is_weekend"] = (
        (df["Day_of_Week"] >= 5) | ((df["flag_working"] == 0) & (df["flag_holiday"] == 0))
    ).astype(int)

    # Daytime vs Nighttime
    df["is_night"] = ((df["Hour"] >= 0) & (df["Hour"] <= 5)).astype(int)
    df["is_daytime"] = ((df["Hour"] >= 6) & (df["Hour"] <= 21)).astype(int)

    # Time Trend: continuous monthly index from 2011 to 2012
    df["time_trend"] = (df["Year"] - 2011) * 12 + df["Month"]
    
    # Quarter of year
    df["quarter"] = ((df["Month"] - 1) // 3 + 1).astype(int)

    # Interactions between time components
    df["hour_working_interaction"] = df["Hour"] * df["flag_working"]
    df["hour_day_interaction"] = df["Day_of_Week"] * 24 + df["Hour"]
    df["season_month_ratio"] = df["Month"] / df["season_code"]

    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sine and cosine transformations for cyclical time variables.
    """
    df = df.copy()

    # Hour (24-hour cycle: 0 to 23)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

    # Month (12-month cycle: 1 to 12)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    # Day of Week (7-day cycle: 0 to 6)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7.0)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7.0)

    # Season (4 seasons: 1 to 4)
    df["Season_sin"] = np.sin(2 * np.pi * df["season_code"] / 4.0)
    df["Season_cos"] = np.cos(2 * np.pi * df["season_code"] / 4.0)

    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate weather interaction, difference, and severity indices.
    """
    df = df.copy()

    # Missing indicators before imputation
    df["humidity_is_na"] = df["feature_humidity"].isna().astype(int)
    df["wind_is_na"] = df["feature_wind"].isna().astype(int)

    # Temperature difference (feels-like vs actual temperature)
    df["temp_feel_diff"] = df["feature_feel_temp"] - df["feature_temp"]

    # Humidity and Temperature interactions
    df["temp_humidity_interaction"] = df["feature_temp"] * df["feature_humidity"]
    df["temp_humidity_ratio"] = df["feature_temp"] / (df["feature_humidity"] + 0.01)

    # Wind and Temperature interaction
    df["wind_temp_interaction"] = df["feature_wind"] * df["feature_temp"]
    df["wind_feel_interaction"] = df["feature_wind"] * df["feature_feel_temp"]

    # Discomfort Index proxy: DI = Temp - 0.55 * (1 - Humidity) * (Temp - 0.58)
    df["discomfort_index"] = (
        df["feature_temp"] - 0.55 * (1.0 - df["feature_humidity"]) * (df["feature_temp"] - 0.58)
    )

    # Weather severity indicator
    df["is_severe_weather"] = (df["category_weather"] >= 3).astype(int)
    df["is_good_weather"] = (df["category_weather"] == 1).astype(int)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.
    """
    df = create_time_features(df)
    df = create_cyclical_features(df)
    df = create_weather_features(df)
    return df
