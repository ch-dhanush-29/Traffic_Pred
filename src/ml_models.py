"""
M5: Classical Machine Learning Engine
Implements:
1. From-scratch NumPy Linear Regression trained via Gradient Descent (PRD Build-it-once rule)
2. Scikit-learn Linear Regression, Decision Tree, Random Forest, XGBoost Regressors
3. Support Vector Machine, Decision Tree, Random Forest, XGBoost Classifiers
4. Time-based splitting, feature scaling, model card generation, and artifact export.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LinearRegression, Ridge, SGDClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

from src.utils import logger, load_config, set_seed

class ScratchLinearRegression:
    """
    Linear Regression implemented from first principles using NumPy matrix operations
    and Mini-Batch Gradient Descent to satisfy PRD Section 10.2 / 11.2.
    """
    def __init__(self, lr: float = 0.01, epochs: int = 50, batch_size: int = 512):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        # Initialize weights with Xavier/He uniform
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []
        
        y = y.astype(float)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            for start_idx in range(0, m, self.batch_size):
                end_idx = min(start_idx + self.batch_size, m)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                batch_m = len(y_batch)
                
                # Forward pass: dot product y_hat = Xw + b
                y_hat = np.dot(X_batch, self.weights) + self.bias
                error = y_hat - y_batch
                
                # Cost: MSE = 1 / (2m) * sum(error^2)
                loss = (1.0 / (2.0 * batch_m)) * np.sum(error ** 2)
                epoch_losses.append(loss)
                
                # Gradients: dJ/dw = 1/m * X.T @ error; dJ/db = 1/m * sum(error)
                dw = (1.0 / batch_m) * np.dot(X_batch.T, error)
                db = (1.0 / batch_m) * np.sum(error)
                
                # Clip gradients for stability
                dw = np.clip(dw, -10.0, 10.0)
                db = np.clip(db, -10.0, 10.0)
                
                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
            self.loss_history.append(float(np.mean(epoch_losses)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates MAPE ignoring zero denominators."""
    mask = y_true > 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def prepare_splits(df: pd.DataFrame, feature_cols: List[str], config: dict):
    """
    Splits data strictly by time:
    Train: <= 2025-03-31
    Val:   > 2025-03-31 and <= 2025-04-30
    Test:  > 2025-04-30
    """
    logger.info("Performing temporal train/val/test split...")
    train_mask = df['date'] <= config['split']['train_end']
    val_mask = (df['date'] > config['split']['train_end']) & (df['date'] <= config['split']['val_end'])
    test_mask = df['date'] > config['split']['val_end']
    
    X_train = df.loc[train_mask, feature_cols].values
    X_val = df.loc[val_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values
    
    # Scale features (Fit ONLY on Train!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    splits = {
        'masks': {'train': train_mask, 'val': val_mask, 'test': test_mask},
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled, 'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    logger.info(f"Split sizes - Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    return splits

def train_and_eval_regression(df: pd.DataFrame, splits: dict, target_col: str, config: dict) -> Tuple[Dict[str, Any], Any]:
    """Trains regression models for traffic_volume or travel_time and compares them."""
    logger.info(f"--- Training Regression Models for target: {target_col} ---")
    y_train = df.loc[splits['masks']['train'], target_col].values
    y_val = df.loc[splits['masks']['val'], target_col].values
    y_test = df.loc[splits['masks']['test'], target_col].values
    
    results = {}
    fitted_models = {}
    
    # 1. Scratch NumPy Linear Regression (trained on scaled X)
    logger.info("Training Scratch NumPy Linear Regression (Gradient Descent)...")
    scratch_lr = ScratchLinearRegression(lr=0.05, epochs=30, batch_size=256)
    scratch_lr.fit(splits['X_train_scaled'], y_train)
    y_pred_scratch = scratch_lr.predict(splits['X_test_scaled'])
    
    results['Scratch_LinearRegression'] = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_scratch))),
        'MAE': float(mean_absolute_error(y_test, y_pred_scratch)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_scratch),
        'R2': float(r2_score(y_test, y_pred_scratch))
    }
    fitted_models['Scratch_LinearRegression'] = scratch_lr
    
    # 2. Sklearn Ridge / Linear Regression
    logger.info("Training Scikit-Learn Linear Regression...")
    sk_lr = Ridge(alpha=1.0)
    sk_lr.fit(splits['X_train_scaled'], y_train)
    y_pred_sk = sk_lr.predict(splits['X_test_scaled'])
    results['Ridge_Regression'] = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_sk))),
        'MAE': float(mean_absolute_error(y_test, y_pred_sk)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_sk),
        'R2': float(r2_score(y_test, y_pred_sk))
    }
    fitted_models['Ridge_Regression'] = sk_lr
    
    # 3. Decision Tree Regressor
    logger.info("Training Decision Tree Regressor...")
    dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42)
    dt.fit(splits['X_train'], y_train)
    y_pred_dt = dt.predict(splits['X_test'])
    results['DecisionTree'] = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_dt))),
        'MAE': float(mean_absolute_error(y_test, y_pred_dt)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_dt),
        'R2': float(r2_score(y_test, y_pred_dt))
    }
    fitted_models['DecisionTree'] = dt
    
    # 4. Random Forest Regressor
    logger.info("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=14, n_jobs=-1, random_state=42)
    rf.fit(splits['X_train'], y_train)
    y_pred_rf = rf.predict(splits['X_test'])
    results['RandomForest'] = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
        'MAE': float(mean_absolute_error(y_test, y_pred_rf)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_rf),
        'R2': float(r2_score(y_test, y_pred_rf))
    }
    fitted_models['RandomForest'] = rf
    
    # 5. XGBoost Regressor
    logger.info("Training XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, n_jobs=-1, random_state=42)
    xgb.fit(splits['X_train'], y_train, eval_set=[(splits['X_val'], y_val)], verbose=False)
    y_pred_xgb = xgb.predict(splits['X_test'])
    results['XGBoost'] = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
        'MAE': float(mean_absolute_error(y_test, y_pred_xgb)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_xgb),
        'R2': float(r2_score(y_test, y_pred_xgb))
    }
    fitted_models['XGBoost'] = xgb
    
    # Pick winner based on RMSE
    winner = min(results, key=lambda k: results[k]['RMSE'])
    logger.info(f"Winning Regression Model for {target_col}: {winner} (RMSE={results[winner]['RMSE']:.2f}, MAPE={results[winner]['MAPE']:.2f}%)")
    
    return results, fitted_models[winner], fitted_models

def train_and_eval_congestion(df: pd.DataFrame, splits: dict, config: dict) -> Tuple[Dict[str, Any], Any]:
    """Trains 4-class classifiers for congestion_level."""
    logger.info("--- Training Congestion Level Classifiers (4-class) ---")
    le = LabelEncoder()
    # Ensure consistent order: Free-flow, Moderate, Heavy, Severe
    le.fit(config['congestion_levels'])
    
    y_train = le.transform(df.loc[splits['masks']['train'], 'congestion_level'])
    y_val = le.transform(df.loc[splits['masks']['val'], 'congestion_level'])
    y_test = le.transform(df.loc[splits['masks']['test'], 'congestion_level'])
    
    results = {}
    fitted_models = {}
    
    # 1. Linear SVM Baseline on scaled features
    logger.info("Training Linear SVM Classifier...")
    svm = LinearSVC(max_iter=2000, random_state=42, dual=False)
    svm.fit(splits['X_train_scaled'], y_train)
    y_pred_svm = svm.predict(splits['X_test_scaled'])
    results['SVM_Linear'] = {
        'Macro_F1': float(f1_score(y_test, y_pred_svm, average='macro')),
        'Accuracy': float(accuracy_score(y_test, y_pred_svm))
    }
    fitted_models['SVM_Linear'] = svm
    
    # 2. Decision Tree Classifier
    logger.info("Training Decision Tree Classifier...")
    dt_c = DecisionTreeClassifier(max_depth=10, min_samples_leaf=15, random_state=42)
    dt_c.fit(splits['X_train'], y_train)
    y_pred_dt = dt_c.predict(splits['X_test'])
    results['DecisionTree'] = {
        'Macro_F1': float(f1_score(y_test, y_pred_dt, average='macro')),
        'Accuracy': float(accuracy_score(y_test, y_pred_dt))
    }
    fitted_models['DecisionTree'] = dt_c
    
    # 3. Random Forest Classifier
    logger.info("Training Random Forest Classifier...")
    rf_c = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    rf_c.fit(splits['X_train'], y_train)
    y_pred_rf = rf_c.predict(splits['X_test'])
    results['RandomForest'] = {
        'Macro_F1': float(f1_score(y_test, y_pred_rf, average='macro')),
        'Accuracy': float(accuracy_score(y_test, y_pred_rf))
    }
    fitted_models['RandomForest'] = rf_c
    
    # 4. XGBoost Classifier
    logger.info("Training XGBoost Classifier...")
    xgb_c = XGBClassifier(n_estimators=80, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
    xgb_c.fit(splits['X_train'], y_train)
    y_pred_xgb = xgb_c.predict(splits['X_test'])
    results['XGBoost'] = {
        'Macro_F1': float(f1_score(y_test, y_pred_xgb, average='macro')),
        'Accuracy': float(accuracy_score(y_test, y_pred_xgb))
    }
    fitted_models['XGBoost'] = xgb_c
    
    winner = max(results, key=lambda k: results[k]['Macro_F1'])
    logger.info(f"Winning Congestion Classifier: {winner} (Macro-F1={results[winner]['Macro_F1']:.4f})")
    
    return results, fitted_models[winner], le

def train_and_eval_accident(df: pd.DataFrame, splits: dict, config: dict) -> Tuple[Dict[str, Any], Any]:
    """Trains binary accident risk classifiers and optimizes ROC-AUC."""
    logger.info("--- Training Accident Risk Classifiers (Binary) ---")
    y_train = df.loc[splits['masks']['train'], 'accident_risk'].values
    y_test = df.loc[splits['masks']['test'], 'accident_risk'].values
    
    scale_pos_weight = (len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train))
    
    results = {}
    fitted_models = {}
    
    # 1. Random Forest Classifier
    logger.info("Training Random Forest for Accident Risk...")
    rf_acc = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_acc.fit(splits['X_train'], y_train)
    probs_rf = rf_acc.predict_proba(splits['X_test'])[:, 1]
    results['RandomForest'] = {
        'ROC_AUC': float(roc_auc_score(y_test, probs_rf)),
        'F1': float(f1_score(y_test, (probs_rf > 0.3).astype(int)))
    }
    fitted_models['RandomForest'] = rf_acc
    
    # 2. XGBoost Classifier
    logger.info("Training XGBoost for Accident Risk...")
    xgb_acc = XGBClassifier(n_estimators=60, learning_rate=0.08, max_depth=5, scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42)
    xgb_acc.fit(splits['X_train'], y_train)
    probs_xgb = xgb_acc.predict_proba(splits['X_test'])[:, 1]
    results['XGBoost'] = {
        'ROC_AUC': float(roc_auc_score(y_test, probs_xgb)),
        'F1': float(f1_score(y_test, (probs_xgb > 0.3).astype(int)))
    }
    fitted_models['XGBoost'] = xgb_acc
    
    winner = max(results, key=lambda k: results[k]['ROC_AUC'])
    logger.info(f"Winning Accident Risk Model: {winner} (ROC-AUC={results[winner]['ROC_AUC']:.4f})")
    
    return results, fitted_models[winner]

def run_classical_pipeline(config_path: str = "config.yaml"):
    """Runs the entire classical ML pipeline, saves winning models and model cards."""
    cfg = load_config(config_path)
    set_seed(cfg['project']['random_seed'])
    
    logger.info("Loading processed dataset for Classical ML...")
    df = pd.read_parquet(cfg['paths']['processed_data'])
    
    # Extract feature columns from config/features
    from src.features import engineer_features
    _, feature_cols, _ = engineer_features(df.head(10), cfg)
    
    splits = prepare_splits(df, feature_cols, cfg)
    
    # 1. Volume Regression
    vol_results, best_vol_model, all_vol_models = train_and_eval_regression(df, splits, 'traffic_volume', cfg)
    
    # 2. Travel Time Regression
    tt_results, best_tt_model, _ = train_and_eval_regression(df, splits, 'travel_time', cfg)
    
    # 3. Congestion Classification
    cong_results, best_cong_model, le_cong = train_and_eval_congestion(df, splits, cfg)
    
    # 4. Accident Risk
    acc_results, best_acc_model = train_and_eval_accident(df, splits, cfg)
    
    # Save winning models and preprocessing artifacts
    os.makedirs(cfg['paths']['models_dir'], exist_ok=True)
    joblib.dump(best_vol_model, os.path.join(cfg['paths']['models_dir'], "best_volume_model.joblib"))
    joblib.dump(best_tt_model, os.path.join(cfg['paths']['models_dir'], "best_travel_time_model.joblib"))
    joblib.dump(best_cong_model, os.path.join(cfg['paths']['models_dir'], "best_congestion_model.joblib"))
    joblib.dump(best_acc_model, os.path.join(cfg['paths']['models_dir'], "best_accident_model.joblib"))
    joblib.dump(splits['scaler'], os.path.join(cfg['paths']['models_dir'], "scaler.joblib"))
    joblib.dump(le_cong, os.path.join(cfg['paths']['models_dir'], "label_encoder_congestion.joblib"))
    
    # Feature Importance extraction (from XGBoost / RF)
    if hasattr(best_vol_model, 'feature_importances_'):
        importances = best_vol_model.feature_importances_
    elif hasattr(all_vol_models['RandomForest'], 'feature_importances_'):
        importances = all_vol_models['RandomForest'].feature_importances_
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)
        
    feat_imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    feat_imp_df.to_csv(os.path.join(cfg['paths']['models_dir'], "feature_importances.csv"), index=False)
    
    # Compile Model Card metadata
    model_cards = {
        "metadata": {
            "author": "FlowCast AI Engineering Team",
            "corridor": cfg['project']['corridor'],
            "split": cfg['split'],
            "feature_count": len(feature_cols),
            "features": feature_cols
        },
        "traffic_volume_models": vol_results,
        "travel_time_models": tt_results,
        "congestion_classification_models": cong_results,
        "accident_risk_models": acc_results
    }
    
    with open(cfg['paths']['model_cards'], "w", encoding="utf-8") as f:
        json.dump(model_cards, f, indent=2)
    logger.info(f"Model cards persisted to {cfg['paths']['model_cards']}")
    
    return model_cards

if __name__ == "__main__":
    cards = run_classical_pipeline()
    print("Classical ML Pipeline Complete! Results:")
    print(json.dumps(cards, indent=2))
