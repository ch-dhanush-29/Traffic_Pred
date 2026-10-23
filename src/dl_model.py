"""
M6: Deep Learning Engine (PyTorch)
Implements:
1. Recurrent LSTM sequence neural network trained from scratch (PRD Section 12)
2. Sequence builder respecting segment temporal boundaries
3. Multi-head output (Traffic volume regression + Congestion classification)
4. Early stopping, learning rate scheduler, dropout regularisation
5. GPU/CUDA acceleration and training curve logging
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score

from src.utils import logger, load_config, set_seed
from src.features import engineer_features
from src.ml_models import mean_absolute_percentage_error

class TrafficSequenceDataset(Dataset):
    """PyTorch Dataset for sliding sequence windows per segment."""
    def __init__(self, sequences: np.ndarray, targets_vol: np.ndarray, targets_cong: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets_vol = torch.tensor(targets_vol, dtype=torch.float32)
        self.targets_cong = torch.tensor(targets_cong, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_vol[idx], self.targets_cong[idx]

class FlowCastLSTM(nn.Module):
    """
    Recurrent sequence architecture:
    Input: [Batch, Seq_Len, Feature_Dim]
    Core: 2-layer LSTM with Dropout
    Heads:
      - Volume Head: Fully Connected -> scalar volume forecast
      - Congestion Head: Fully Connected -> 4-class logits
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, num_classes: int = 4):
        super(FlowCastLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.volume_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.congestion_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the final time step hidden representation
        final_repr = lstm_out[:, -1, :]
        final_repr = self.dropout(final_repr)
        
        vol_pred = self.volume_head(final_repr).squeeze(-1)
        cong_logits = self.congestion_head(final_repr)
        return vol_pred, cong_logits

def build_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """
    Builds sliding sequence windows per road segment to prevent cross-segment leakage.
    Returns:
      X_seq: [N_samples, seq_len, num_features]
      y_vol: [N_samples]
      y_cong: [N_samples]
      dates: Series of dates for the predicted window
    """
    X_list, y_vol_list, y_cong_list, date_list = [], [], [], []
    
    # Label encode congestion level
    le = LabelEncoder()
    le.fit(['Free-flow', 'Moderate', 'Heavy', 'Severe'])
    cong_encoded = le.transform(df['congestion_level'])
    
    # Group by road_id
    road_groups = df.groupby('road_id')
    
    for _, group in road_groups:
        group_features = group[feature_cols].values
        group_vols = group['traffic_volume'].values
        group_congs = cong_encoded[group.index]
        group_dates = group['date'].values
        
        n_rows = len(group)
        for i in range(seq_len, n_rows):
            X_list.append(group_features[i - seq_len : i])
            y_vol_list.append(group_vols[i])
            y_cong_list.append(group_congs[i])
            date_list.append(group_dates[i])
            
    return np.array(X_list), np.array(y_vol_list), np.array(y_cong_list), pd.Series(date_list), le

def train_dl_pipeline(config_path: str = "config.yaml"):
    """Trains the PyTorch sequence model from scratch with validation early stopping."""
    cfg = load_config(config_path)
    set_seed(cfg['project']['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device for Deep Learning: {device}")
    
    # 1. Load data
    df = pd.read_parquet(cfg['paths']['processed_data'])
    _, feature_cols, _ = engineer_features(df.head(10), cfg)
    
    # 2. Build sequences
    seq_len = cfg['deep_learning']['sequence_length']
    logger.info(f"Constructing sliding sequence windows (seq_len={seq_len} = {seq_len*0.5} hours)...")
    X_seq, y_vol, y_cong, dates_series, le = build_sequences(df, feature_cols, seq_len=seq_len)
    
    # 3. Time-based splits
    train_mask = (dates_series <= cfg['split']['train_end']).values
    val_mask = ((dates_series > cfg['split']['train_end']) & (dates_series <= cfg['split']['val_end'])).values
    test_mask = (dates_series > cfg['split']['val_end']).values
    
    # 4. Standardize features across the 3D sequence array (Fit on train only!)
    N, S, F = X_seq.shape
    num_features = F
    scaler = StandardScaler()
    X_train_flat = X_seq[train_mask].reshape(-1, F)
    scaler.fit(X_train_flat)
    
    # Vectorized instantaneous transform of all sequences
    X_seq_scaled = scaler.transform(X_seq.reshape(-1, F)).reshape(N, S, F).astype(np.float32)
        
    # Create Datasets and DataLoaders
    train_ds = TrafficSequenceDataset(X_seq_scaled[train_mask], y_vol[train_mask], y_cong[train_mask])
    val_ds = TrafficSequenceDataset(X_seq_scaled[val_mask], y_vol[val_mask], y_cong[val_mask])
    test_ds = TrafficSequenceDataset(X_seq_scaled[test_mask], y_vol[test_mask], y_cong[test_mask])
    
    batch_size = cfg['deep_learning']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 5. Initialize Model
    model = FlowCastLSTM(
        input_dim=num_features,
        hidden_dim=cfg['deep_learning']['hidden_size'],
        num_layers=cfg['deep_learning']['num_layers'],
        dropout=cfg['deep_learning']['dropout'],
        num_classes=4
    ).to(device)
    
    criterion_vol = nn.MSELoss()
    criterion_cong = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['deep_learning']['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 6. Training Loop with Early Stopping
    epochs = cfg['deep_learning']['epochs']
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_vol_rmse': [], 'val_vol_rmse': []}
    best_weights_path = os.path.join(cfg['paths']['models_dir'], "flowcast_lstm_best.pt")
    
    logger.info(f"Starting LSTM training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_vol_sq_err = 0.0
        train_samples = 0
        
        for batch_X, batch_y_vol, batch_y_cong in train_loader:
            batch_X = batch_X.to(device)
            batch_y_vol = batch_y_vol.to(device)
            batch_y_cong = batch_y_cong.to(device)
            
            optimizer.zero_grad()
            pred_vol, pred_cong = model(batch_X)
            
            loss_v = criterion_vol(pred_vol, batch_y_vol)
            loss_c = criterion_cong(pred_cong, batch_y_cong)
            # Combined multi-task loss (balanced weighting)
            loss = loss_v + 100.0 * loss_c
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_accum += loss.item() * len(batch_X)
            train_vol_sq_err += torch.sum((pred_vol - batch_y_vol)**2).item()
            train_samples += len(batch_X)
            
        epoch_train_loss = train_loss_accum / train_samples
        epoch_train_vol_rmse = np.sqrt(train_vol_sq_err / train_samples)
        
        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_vol_sq_err = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y_vol, batch_y_cong in val_loader:
                batch_X = batch_X.to(device)
                batch_y_vol = batch_y_vol.to(device)
                batch_y_cong = batch_y_cong.to(device)
                
                pred_vol, pred_cong = model(batch_X)
                loss_v = criterion_vol(pred_vol, batch_y_vol)
                loss_c = criterion_cong(pred_cong, batch_y_cong)
                loss = loss_v + 100.0 * loss_c
                
                val_loss_accum += loss.item() * len(batch_X)
                val_vol_sq_err += torch.sum((pred_vol - batch_y_vol)**2).item()
                val_samples += len(batch_X)
                
        epoch_val_loss = val_loss_accum / val_samples
        epoch_val_vol_rmse = np.sqrt(val_vol_sq_err / val_samples)
        
        scheduler.step(epoch_val_loss)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_vol_rmse'].append(epoch_train_vol_rmse)
        history['val_vol_rmse'].append(epoch_val_vol_rmse)
        
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_train_loss:.2f} (Vol RMSE: {epoch_train_vol_rmse:.2f}) | "
            f"Val Loss: {epoch_val_loss:.2f} (Vol RMSE: {epoch_val_vol_rmse:.2f})"
        )
        
        # Save best weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_weights_path)
            
    # 7. Evaluate on held-out test set
    logger.info("Evaluating best LSTM model on held-out test window (May 2025)...")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()
    
    test_preds_vol, test_actuals_vol = [], []
    test_preds_cong, test_actuals_cong = [], []
    
    with torch.no_grad():
        for batch_X, batch_y_vol, batch_y_cong in test_loader:
            batch_X = batch_X.to(device)
            pred_vol, pred_cong = model(batch_X)
            
            test_preds_vol.extend(pred_vol.cpu().numpy())
            test_actuals_vol.extend(batch_y_vol.numpy())
            test_preds_cong.extend(torch.argmax(pred_cong, dim=1).cpu().numpy())
            test_actuals_cong.extend(batch_y_cong.numpy())
            
    test_preds_vol = np.array(test_preds_vol)
    test_actuals_vol = np.array(test_actuals_vol)
    test_preds_cong = np.array(test_preds_cong)
    test_actuals_cong = np.array(test_actuals_cong)
    
    dl_metrics = {
        'volume': {
            'RMSE': float(np.sqrt(mean_squared_error(test_actuals_vol, test_preds_vol))),
            'MAE': float(mean_absolute_error(test_actuals_vol, test_preds_vol)),
            'MAPE': mean_absolute_percentage_error(test_actuals_vol, test_preds_vol),
            'R2': float(r2_score(test_actuals_vol, test_preds_vol))
        },
        'congestion': {
            'Macro_F1': float(f1_score(test_actuals_cong, test_preds_cong, average='macro')),
            'Accuracy': float(accuracy_score(test_actuals_cong, test_preds_cong))
        },
        'history': history
    }
    
    logger.info(
        f"Deep Learning Test Results - Volume RMSE: {dl_metrics['volume']['RMSE']:.2f}, "
        f"MAPE: {dl_metrics['volume']['MAPE']:.2f}%, Congestion Macro-F1: {dl_metrics['congestion']['Macro_F1']:.4f}"
    )
    
    # Save metrics and training history
    with open(os.path.join(cfg['paths']['models_dir'], "dl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dl_metrics, f, indent=2)
        
    return dl_metrics

if __name__ == "__main__":
    metrics = train_dl_pipeline()
    print("DL Sequence Model Training Complete:")
    print("Volume RMSE:", metrics['volume']['RMSE'])
    print("Volume MAPE:", metrics['volume']['MAPE'], "%")
    print("Congestion Macro-F1:", metrics['congestion']['Macro_F1'])
