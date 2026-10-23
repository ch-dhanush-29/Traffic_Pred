"""
M3: Feature Engineering Module
Builds temporal encodings, strictly causal lags, rolling statistics,
capacity indicators, weather flags, and calendar interaction terms.
Guarantees NO future-window lookahead leakage.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from src.utils import logger, load_config
from src.clean import clean_and_merge

def engineer_features(df_input: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Generates all engineered feature sets from the merged dataframe."""
    df = df_input.copy()
    logger.info("Starting feature engineering...")
    
    # Ensure correct datetime parsing and chronological ordering per road
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['road_id', 'timestamp']).reset_index(drop=True)
    
    # 1. Temporal & Cyclical Encodings
    dt = df['timestamp'].dt
    df['day_of_week'] = dt.dayofweek  # 0=Monday, 6=Sunday
    df['hour_int'] = dt.hour
    df['minute_int'] = dt.minute
    df['hour_float'] = df['hour_int'] + df['minute_int'] / 60.0
    
    df['sin_hour'] = np.sin(2.0 * np.pi * df['hour_float'] / 24.0)
    df['cos_hour'] = np.cos(2.0 * np.pi * df['hour_float'] / 24.0)
    df['sin_dow'] = np.sin(2.0 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2.0 * np.pi * df['day_of_week'] / 7.0)
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Peak periods: Morning peak (07:30 - 10:00) and Evening peak (17:00 - 20:00)
    morning_peak = (df['hour_float'] >= 7.5) & (df['hour_float'] <= 10.0)
    evening_peak = (df['hour_float'] >= 17.0) & (df['hour_float'] <= 20.0)
    df['is_peak'] = (morning_peak | evening_peak).astype(int)
    
    # 2. Lag Features (grouped strictly by road_id)
    # Lags: t-1 (30m), t-2 (60m), t-48 (24h prior)
    logger.info("Computing causal lag features (t-1, t-2, t-48)...")
    for col in ['traffic_volume', 'avg_speed', 'occupancy']:
        df[f'{col}_lag_1'] = df.groupby('road_id')[col].shift(1)
        df[f'{col}_lag_2'] = df.groupby('road_id')[col].shift(2)
        df[f'{col}_lag_48'] = df.groupby('road_id')[col].shift(48)
        
        # Fill edge nulls for initial windows using lag_1 or median to retain sample size
        df[f'{col}_lag_2'] = df[f'{col}_lag_2'].fillna(df[f'{col}_lag_1'])
        df[f'{col}_lag_48'] = df[f'{col}_lag_48'].fillna(df[f'{col}_lag_1'])
        df[f'{col}_lag_1'] = df[f'{col}_lag_1'].fillna(df.groupby(['road_id', 'time'])[col].transform('median'))
        df[f'{col}_lag_2'] = df[f'{col}_lag_2'].fillna(df[f'{col}_lag_1'])
        df[f'{col}_lag_48'] = df[f'{col}_lag_48'].fillna(df[f'{col}_lag_1'])

    # 3. Rolling Statistics (Strictly causal using shift(1) to prevent leakage)
    logger.info("Computing causal rolling window statistics...")
    for col in ['traffic_volume', 'avg_speed']:
        # Shifted series to exclude current window
        shifted = df.groupby('road_id')[col].shift(1)
        
        # 4 windows = 2 hours
        roll_4 = shifted.groupby(df['road_id']).rolling(window=4, min_periods=1)
        df[f'{col}_roll_mean_4'] = roll_4.mean().reset_index(level=0, drop=True).fillna(df[f'{col}_lag_1'])
        df[f'{col}_roll_std_4'] = roll_4.std().reset_index(level=0, drop=True).fillna(0.0)
        
        # 8 windows = 4 hours
        roll_8 = shifted.groupby(df['road_id']).rolling(window=8, min_periods=1)
        df[f'{col}_roll_mean_8'] = roll_8.mean().reset_index(level=0, drop=True).fillna(df[f'{col}_lag_1'])
        df[f'{col}_roll_std_8'] = roll_8.std().reset_index(level=0, drop=True).fillna(0.0)

    # 4. Capacity & Interaction Features
    half_capacity = df['road_capacity'] / 2.0
    df['vc_ratio_lag1'] = df['traffic_volume_lag_1'] / half_capacity
    df['capacity_headroom_lag1'] = half_capacity - df['traffic_volume_lag_1']
    
    # 5. Weather Features
    df['is_rain'] = ((df['rainfall'] > 0.0) | (df['weather_condition'] == 'Rain')).astype(int)
    df['low_visibility'] = (df['visibility'] < 2000.0).astype(int)
    
    # Temperature bands: Cold (<15C), Moderate (15-30C), Hot (>30C)
    df['temp_is_cold'] = (df['temperature'] < 15.0).astype(int)
    df['temp_is_hot'] = (df['temperature'] > 30.0).astype(int)
    
    # 6. Calendar & Event Interactions
    df['holiday_x_peak'] = df['public_holiday'] * df['is_peak']
    df['event_x_peak'] = df['event_flag'] * df['is_peak']
    
    # One-hot encoding for weather condition and road ID for linear models
    weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather', drop_first=True, dtype=float)
    df = pd.concat([df, weather_dummies], axis=1)
    
    # Feature columns specification
    feature_cols = [
        # Temporal
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow', 'is_weekend', 'is_peak',
        # Lags
        'traffic_volume_lag_1', 'traffic_volume_lag_2', 'traffic_volume_lag_48',
        'avg_speed_lag_1', 'avg_speed_lag_2', 'avg_speed_lag_48',
        'occupancy_lag_1',
        # Rolling stats
        'traffic_volume_roll_mean_4', 'traffic_volume_roll_std_4',
        'traffic_volume_roll_mean_8', 'traffic_volume_roll_std_8',
        'avg_speed_roll_mean_4', 'avg_speed_roll_std_4',
        'avg_speed_roll_mean_8', 'avg_speed_roll_std_8',
        # Capacity
        'road_capacity', 'signal_timing', 'vc_ratio_lag1', 'capacity_headroom_lag1',
        # Vehicle distributions
        'share_2w', 'share_car', 'share_lcv', 'share_hcv',
        # Weather
        'temperature', 'rainfall', 'visibility', 'is_rain', 'low_visibility',
        'temp_is_cold', 'temp_is_hot',
        # Calendar
        'public_holiday', 'event_flag', 'roadwork_flag', 'holiday_x_peak', 'event_x_peak'
    ] + [c for c in weather_dummies.columns]
    
    metadata = {
        "total_records": len(df),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "date_range": [str(df['date'].min()), str(df['date'].max())],
        "roads_covered": df['road_id'].nunique()
    }
    logger.info(f"Feature engineering complete: {len(feature_cols)} features engineered across {len(df)} records.")
    return df, feature_cols, metadata

def build_and_save_dataset(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, List[str]]:
    """Runs clean and merge, generates features, and persists the analysis-ready dataset."""
    cfg = load_config(config_path)
    df_merged, clean_stats = clean_and_merge(config_path)
    df_features, feature_cols, meta = engineer_features(df_merged, cfg)
    
    # Save processed parquet and csv
    os.makedirs(os.path.dirname(cfg['paths']['processed_data']), exist_ok=True)
    logger.info(f"Saving processed dataset to {cfg['paths']['processed_data']}...")
    df_features.to_parquet(cfg['paths']['processed_data'], index=False)
    
    logger.info("Saving sample CSV for accessibility...")
    df_features.head(1000).to_csv(cfg['paths']['processed_csv'], index=False)
    
    logger.info("Processed dataset ready for modelling.")
    return df_features, feature_cols

if __name__ == "__main__":
    df, feats = build_and_save_dataset()
    print("Features count:", len(feats))
    print("Dataset shape:", df.shape)
