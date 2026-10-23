"""
M2: Data Cleaning & Wrangling Module
Performs deduplication, outlier clipping/imputation, categorical harmonization,
temporal alignment, and source merging according to PRD Section 9.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.utils import logger, load_config
from src.ingest import ingest_all

WEATHER_MAP = {
    'clear': 'Clear',
    'clear ': 'Clear',
    'cloudy': 'Cloudy',
    'overcast': 'Cloudy',
    'rain': 'Rain',
    'rainy': 'Rain',
    'rain ': 'Rain',
    'fog': 'Fog',
    'foggy': 'Fog',
}

def clean_weather(df_weather: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans and standardizes weather observations."""
    df = df_weather.copy()
    
    # Harmonize date format from DD/MM/YYYY to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df['hour'] = df['time'].str.slice(0, 2)
    
    # Harmonize categorical weather conditions
    df['weather_condition_clean'] = df['weather_condition'].str.strip().str.lower().map(WEATHER_MAP).fillna('Clear')
    
    # Sort chronologically by station and time
    df = df.sort_values(by=['station_id', 'date', 'time']).reset_index(drop=True)
    
    # Impute missing temperature and visibility via interpolation & median per station
    initial_null_temp = df['temperature'].isnull().sum()
    initial_null_vis = df['visibility'].isnull().sum()
    
    df['temperature'] = df.groupby('station_id')['temperature'].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both')
    )
    df['visibility'] = df.groupby('station_id')['visibility'].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both')
    )
    
    df['temperature'] = df['temperature'].fillna(df['temperature'].median())
    df['visibility'] = df['visibility'].fillna(df['visibility'].median())
    
    stats = {
        "weather_initial_rows": len(df_weather),
        "weather_imputed_temp": int(initial_null_temp),
        "weather_imputed_visibility": int(initial_null_vis),
        "weather_conditions_mapped": df['weather_condition_clean'].value_counts().to_dict()
    }
    logger.info(f"Weather cleaned: {len(df)} rows, imputed {initial_null_temp} temp, {initial_null_vis} vis.")
    return df, stats

def derive_congestion_level(volume: pd.Series, capacity: pd.Series) -> pd.Series:
    """
    Derives congestion level based on Volume/Capacity ratio for a 30-min window:
    V/C = volume / (road_capacity / 2)
    <0.50: Free-flow
    0.50-0.79: Moderate
    0.80-0.99: Heavy
    >=1.00: Severe
    """
    vc = volume / (capacity / 2.0)
    conditions = [
        vc < 0.50,
        (vc >= 0.50) & (vc < 0.80),
        (vc >= 0.80) & (vc < 1.00),
        vc >= 1.00
    ]
    choices = ['Free-flow', 'Moderate', 'Heavy', 'Severe']
    return pd.Series(np.select(conditions, choices, default='Free-flow'), index=volume.index)

def parse_vehicle_distribution(series: pd.Series) -> pd.DataFrame:
    """Parses JSON vehicle_type_dist string into separate float columns."""
    records = []
    for item in series:
        if isinstance(item, str):
            try:
                records.append(json.loads(item))
            except Exception:
                records.append({"2W": 0.35, "Car": 0.45, "LCV": 0.12, "HCV": 0.08})
        elif isinstance(item, dict):
            records.append(item)
        else:
            records.append({"2W": 0.35, "Car": 0.45, "LCV": 0.12, "HCV": 0.08})
    
    df_dist = pd.DataFrame(records)
    df_dist = df_dist.rename(columns={
        "2W": "share_2w",
        "Car": "share_car",
        "LCV": "share_lcv",
        "HCV": "share_hcv"
    })
    return df_dist.fillna(0.0)

def clean_traffic(df_traffic: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans traffic sensor telemetry: deduplication, outlier replacement, interpolation, derivation."""
    df = df_traffic.copy()
    initial_rows = len(df)
    
    # 1. Deduplication on road_id + date + time
    df = df.drop_duplicates(subset=['road_id', 'date', 'time'], keep='first').reset_index(drop=True)
    dedup_rows = len(df)
    duplicates_dropped = initial_rows - dedup_rows
    
    # 2. Outlier detection and replacement with NaN
    neg_vol_count = (df['traffic_volume'] < 0).sum()
    df.loc[df['traffic_volume'] < 0, 'traffic_volume'] = np.nan
    
    overspeed_count = (df['avg_speed'] > config['cleaning']['max_speed']).sum()
    df.loc[df['avg_speed'] > config['cleaning']['max_speed'], 'avg_speed'] = np.nan
    df.loc[df['avg_speed'] < 0, 'avg_speed'] = np.nan
    
    overocc_count = (df['occupancy'] > config['cleaning']['max_occupancy']).sum()
    df.loc[df['occupancy'] > config['cleaning']['max_occupancy'], 'occupancy'] = np.nan
    df.loc[df['occupancy'] < 0, 'occupancy'] = np.nan
    
    # 3. Create datetime timestamp and sort per segment
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values(by=['road_id', 'timestamp']).reset_index(drop=True)
    
    # 4. Missing value imputation via forward/backward interpolation within each segment
    # and fallback to segment-window median
    for col in ['traffic_volume', 'avg_speed', 'occupancy']:
        # Step A: Short-gap interpolation per road
        df[col] = df.groupby('road_id')[col].transform(
            lambda s: s.interpolate(method='linear', limit=4, limit_direction='both')
        )
        # Step B: Segment-by-time median fallback
        median_map = df.groupby(['road_id', 'time'])[col].transform('median')
        df[col] = df[col].fillna(median_map)
        # Step C: Road overall median fallback
        road_median = df.groupby('road_id')[col].transform('median')
        df[col] = df[col].fillna(road_median)
    
    # Ensure vehicle_count equals traffic_volume if not aligned
    df['traffic_volume'] = df['traffic_volume'].round()
    df['vehicle_count'] = df['traffic_volume'].astype(int)
    
    # 5. Derive blank congestion_level rows
    derived_congestion = derive_congestion_level(df['traffic_volume'], df['road_capacity'])
    blank_cong_count = df['congestion_level'].isnull().sum()
    df['congestion_level'] = df['congestion_level'].fillna(derived_congestion)
    
    # 6. Parse vehicle distribution JSON
    logger.info("Parsing vehicle distribution JSON...")
    df_dist = parse_vehicle_distribution(df['vehicle_type_dist'])
    for col in df_dist.columns:
        df[col] = df_dist[col].values
        
    # 7. Derive accident risk binary target
    df['accident_risk'] = (df['accident_count'] > 0).astype(int)
    
    stats = {
        "traffic_initial_rows": initial_rows,
        "traffic_dedup_rows": dedup_rows,
        "duplicates_removed": duplicates_dropped,
        "negative_volume_remediated": int(neg_vol_count),
        "overspeed_remediated": int(overspeed_count),
        "overoccupancy_remediated": int(overocc_count),
        "blank_congestion_derived": int(blank_cong_count),
        "accident_positive_count": int(df['accident_risk'].sum())
    }
    logger.info(f"Traffic cleaned: {dedup_rows} rows retained, {blank_cong_count} congestion levels derived.")
    return df, stats

def clean_and_merge(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Performs the complete M2 cleaning and merging pipeline."""
    cfg = load_config(config_path)
    df_traffic, df_weather, df_cal, ingest_report = ingest_all(config_path)
    
    # Clean components
    df_weather_clean, weather_stats = clean_weather(df_weather)
    df_traffic_clean, traffic_stats = clean_traffic(df_traffic, cfg)
    
    # Join hourly weather onto half-hourly traffic
    # Hour key for join: '00', '01', ...
    df_traffic_clean['hour'] = df_traffic_clean['time'].str.slice(0, 2)
    
    weather_join_cols = [
        'station_id', 'date', 'hour',
        'weather_condition_clean', 'temperature', 'rainfall', 'visibility'
    ]
    df_weather_sub = df_weather_clean[weather_join_cols].rename(columns={
        'station_id': 'weather_station_id',
        'weather_condition_clean': 'weather_condition'
    })
    
    logger.info("Merging traffic and weather...")
    merged = pd.merge(
        df_traffic_clean,
        df_weather_sub,
        on=['weather_station_id', 'date', 'hour'],
        how='left'
    )
    
    # Fill any remaining unjoined weather records with corridor defaults
    merged['weather_condition'] = merged['weather_condition'].fillna('Clear')
    merged['temperature'] = merged['temperature'].fillna(merged['temperature'].median())
    merged['rainfall'] = merged['rainfall'].fillna(0.0)
    merged['visibility'] = merged['visibility'].fillna(10000.0)
    
    # Join calendar events
    logger.info("Merging calendar events...")
    cal_join_cols = ['date', 'public_holiday', 'holiday_name', 'event_flag', 'event_name', 'roadwork_flag']
    merged = pd.merge(merged, df_cal[cal_join_cols], on='date', how='left')
    
    merged['public_holiday'] = merged['public_holiday'].fillna(0).astype(int)
    merged['event_flag'] = merged['event_flag'].fillna(0).astype(int)
    merged['roadwork_flag'] = merged['roadwork_flag'].fillna(0).astype(int)
    merged['holiday_name'] = merged['holiday_name'].fillna('None')
    merged['event_name'] = merged['event_name'].fillna('None')
    
    # Sort chronologically
    merged = merged.sort_values(by=['road_id', 'timestamp']).reset_index(drop=True)
    
    # Persist interim merged dataset
    interim_path = os.path.join(cfg['paths']['interim_dir'], "merged_cleaned_interim.parquet")
    merged.to_parquet(interim_path, index=False)
    
    report = {
        "ingestion": ingest_report,
        "traffic_cleaning": traffic_stats,
        "weather_cleaning": weather_stats,
        "final_merged_rows": len(merged),
        "final_columns": merged.columns.tolist()
    }
    logger.info(f"Cleaning & merge complete: {len(merged)} analysis-ready records.")
    return merged, report

if __name__ == "__main__":
    df_merged, rep = clean_and_merge()
    print("Clean and merge summary:")
    print("Final rows:", rep["final_merged_rows"])
    print("Columns:", rep["final_columns"])
