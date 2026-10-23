"""
FlowCast: Master End-to-End Execution Pipeline
Runs data ingestion, cleaning, feature engineering, exploratory data analysis,
classical model training, deep learning sequence modeling, and evaluation benchmarking
with a single reproducible command.
"""

import sys
import time
from src.utils import logger, load_config, set_seed
from src.ingest import ingest_all
from src.clean import clean_and_merge
from src.features import build_and_save_dataset
from src.eda import generate_eda_figures, write_data_quality_report
from src.ml_models import run_classical_pipeline
from src.dl_model import train_dl_pipeline
from src.evaluate import run_evaluation_benchmark

def main():
    start_time = time.time()
    logger.info("==================================================================")
    logger.info("   FLOWCAST: SMART TRAFFIC FLOW PREDICTION SYSTEM - PIPELINE")
    logger.info("   Meridian Mobility Systems · Northline Corridor AI Programme")
    logger.info("==================================================================")
    
    cfg = load_config("config.yaml")
    set_seed(cfg['project']['random_seed'])
    
    # Stage 1: Data Cleaning, Merging & Feature Engineering (M1, M2, M3)
    logger.info("\n>>> STAGE 1: Data Engineering & Feature Engineering (M1-M3)...")
    df_features, feature_cols = build_and_save_dataset("config.yaml")
    
    # Stage 2: Exploratory Data Analysis & Reporting (M4)
    logger.info("\n>>> STAGE 2: EDA & Data Quality Audit (M4)...")
    generate_eda_figures("config.yaml")
    write_data_quality_report("config.yaml")
    
    # Stage 3: Classical Machine Learning Models (M5)
    logger.info("\n>>> STAGE 3: Classical Machine Learning Suite (M5)...")
    model_cards = run_classical_pipeline("config.yaml")
    
    # Stage 4: Deep Learning Sequence Engine (M6)
    logger.info("\n>>> STAGE 4: Recurrent Sequence Neural Network (M6)...")
    dl_metrics = train_dl_pipeline("config.yaml")
    
    # Stage 5: Evaluation, Benchmark Scoreboard & Figures
    logger.info("\n>>> STAGE 5: Evaluation & Benchmarking...")
    scoreboard, success_audit = run_evaluation_benchmark("config.yaml")
    
    elapsed = time.time() - start_time
    logger.info("==================================================================")
    logger.info(f"   PIPELINE EXECUTION COMPLETED IN {elapsed:.1f} SECONDS")
    logger.info("   All artifacts, models, figures, and reports generated.")
    logger.info("==================================================================")

if __name__ == "__main__":
    main()
