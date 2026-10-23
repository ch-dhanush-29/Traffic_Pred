"""
M1: Data Ingestion & Validation Module
Loads raw sensor, weather, and calendar tables, validates schema and types,
and logs/quarantines invalid or anomalous records.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from src.utils import logger, load_config

def validate_calendar(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validates calendar_events schema and integrity."""
    required_cols = ['date', 'public_holiday', 'holiday_name', 'event_flag', 'event_name', 'roadwork_flag']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"calendar_events missing required columns: {missing_cols}")
    
    report = {
        "table": "calendar_events",
        "total_rows": len(df),
        "unique_dates": df['date'].nunique(),
        "holidays": int(df['public_holiday'].sum()),
        "events": int(df['event_flag'].sum()),
        "roadworks": int(df['roadwork_flag'].sum())
    }
    logger.info(f"Calendar validated: {report['total_rows']} rows, {report['unique_dates']} unique dates.")
    return df, report

def validate_weather(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Validates weather_observations schema, ranges, and tracks missing/quarantined records."""
    required_cols = ['station_id', 'date', 'time', 'weather_condition', 'temperature', 'rainfall', 'visibility']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"weather_observations missing required columns: {missing_cols}")
    
    total_raw = len(df)
    # Range check masks
    invalid_temp = (df['temperature'] < -30) | (df['temperature'] > 60)
    invalid_rain = df['rainfall'] < 0
    invalid_vis = (df['visibility'] < 0) | (df['visibility'] > 50000)
    
    quarantine_mask = invalid_temp | invalid_rain | invalid_vis
    quarantine_df = df[quarantine_mask].copy()
    valid_df = df.copy()
    
    report = {
        "table": "weather_observations",
        "total_rows": total_raw,
        "stations": df['station_id'].unique().tolist(),
        "missing_temp": int(df['temperature'].isnull().sum()),
        "missing_visibility": int(df['visibility'].isnull().sum()),
        "invalid_range_rows": int(quarantine_mask.sum()),
        "unique_weather_conditions": df['weather_condition'].nunique()
    }
    logger.info(f"Weather validated: {total_raw} rows, {report['invalid_range_rows']} out-of-range.")
    return valid_df, quarantine_df, report

def validate_traffic(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Validates traffic_sensor_log schema, duplicate keys, and physical boundaries."""
    required_cols = [
        'road_id', 'road_name', 'latitude', 'longitude', 'weather_station_id',
        'date', 'time', 'traffic_volume', 'vehicle_count', 'vehicle_type_dist',
        'avg_speed', 'occupancy', 'congestion_level', 'travel_time',
        'accident_count', 'signal_timing', 'road_capacity'
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"traffic_sensor_log missing required columns: {missing_cols}")
    
    total_raw = len(df)
    
    # Check duplicate keys
    dup_mask = df.duplicated(subset=['road_id', 'date', 'time'], keep='first')
    
    # Check physical range anomalies
    neg_vol = df['traffic_volume'] < 0
    high_speed = df['avg_speed'] > config['cleaning']['max_speed']
    high_occ = df['occupancy'] > config['cleaning']['max_occupancy']
    
    quarantine_mask = dup_mask | neg_vol | high_speed | high_occ
    quarantine_df = df[quarantine_mask].copy()
    
    report = {
        "table": "traffic_sensor_log",
        "total_rows": total_raw,
        "unique_roads": df['road_id'].nunique(),
        "duplicate_rows": int(dup_mask.sum()),
        "negative_volume_rows": int(neg_vol.fillna(False).sum()),
        "overspeed_rows": int(high_speed.fillna(False).sum()),
        "overoccupancy_rows": int(high_occ.fillna(False).sum()),
        "missing_volume": int(df['traffic_volume'].isnull().sum()),
        "missing_speed": int(df['avg_speed'].isnull().sum()),
        "missing_occupancy": int(df['occupancy'].isnull().sum()),
        "blank_congestion": int(df['congestion_level'].isnull().sum()),
        "accident_records": int((df['accident_count'] > 0).sum())
    }
    logger.info(
        f"Traffic validated: {total_raw} rows, {report['duplicate_rows']} duplicates, "
        f"{report['negative_volume_rows']} neg-volume, {report['overspeed_rows']} overspeed."
    )
    return df, quarantine_df, report

def ingest_all(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Loads and validates all three source tables."""
    cfg = load_config(config_path)
    logger.info("Starting raw data ingestion...")
    
    df_cal = pd.read_csv(cfg['paths']['raw_calendar'])
    df_cal, cal_report = validate_calendar(df_cal)
    
    df_weather = pd.read_csv(cfg['paths']['raw_weather'])
    df_weather, weather_quarantine, weather_report = validate_weather(df_weather)
    
    df_traffic = pd.read_csv(cfg['paths']['raw_traffic'])
    df_traffic, traffic_quarantine, traffic_report = validate_traffic(df_traffic, cfg)
    
    combined_report = {
        "calendar": cal_report,
        "weather": weather_report,
        "traffic": traffic_report,
        "traffic_quarantined_count": len(traffic_quarantine),
        "weather_quarantined_count": len(weather_quarantine)
    }
    
    # Save quarantine logs to interim
    os.makedirs(cfg['paths']['interim_dir'], exist_ok=True)
    traffic_quarantine.to_csv(os.path.join(cfg['paths']['interim_dir'], "quarantine_traffic.csv"), index=False)
    weather_quarantine.to_csv(os.path.join(cfg['paths']['interim_dir'], "quarantine_weather.csv"), index=False)
    
    logger.info("Ingestion & validation complete. Quarantined records saved to interim.")
    return df_traffic, df_weather, df_cal, combined_report

if __name__ == "__main__":
    t, w, c, rep = ingest_all()
    print("Ingestion Report Summary:")
    print(json.dumps(rep, indent=2))
