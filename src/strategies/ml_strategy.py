"""Machine Learning Strategy Implementation."""
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import optuna
import lightgbm as lgb
from numpy.typing import NDArray
import traceback
import json
from scipy import stats
import time
from datetime import datetime, timedelta
import torch.nn.functional as F
import ta  # Import the ta library
import warnings
from src.config.config import ModelType  # Import ModelType from config

# Suppress warnings
warnings.filterwarnings('ignore')

# Type aliases
FloatArray = NDArray[np.float64]
MLModel = Union[nn.Module, lgb.LGBMRegressor]
ValidationType = Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
EvalSetType = List[Tuple[np.ndarray, np.ndarray]]

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    position_type: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data."""
    def __init__(self, X, y):
        """Initialize dataset."""
        self.X = X  # Already a torch tensor on GPU
        self.y = y  # Already a torch tensor on GPU

    def __len__(self):
        """Get dataset length."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get item by index."""
        return self.X[idx], self.y[idx]

class BaseModel(nn.Module):
    """Base model class with common functionality."""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = None  # Initialize logger after attributes are set

    def get_name(self):
        """Get model name with parameters."""
        return self.__class__.__name__

    def _init_logger(self):
        """Initialize logger after attributes are set."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

class SimpleLSTM(BaseModel):
    """Simple LSTM model."""
    def __init__(self, input_size: int = 21, hidden_size: int = 64, dropout: float = 0.2):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # Define layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_logger()

    def forward(self, x):
        """Forward pass."""
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Get last output
        last_out = lstm_out[:, -1, :]
        
        # Dropout and final layer
        out = self.dropout_layer(last_out)
        out = self.fc(out)
        
        return out

class DeepLSTM(BaseModel):
    """Deep LSTM model with multiple layers."""
    def __init__(self, input_size: int = 21, hidden_sizes: List[int] = [64, 32], dropout: float = 0.2):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        curr_size = input_size
        for hidden_size in hidden_sizes:
            self.lstm_layers.append(nn.LSTM(curr_size, hidden_size, batch_first=True))
            curr_size = hidden_size
        
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden_sizes))])
        self.fc = nn.Linear(hidden_sizes[-1], 1)
        
        self._init_logger()

    def forward(self, x):
        """Forward pass through multiple LSTM layers."""
        # x shape: [batch_size, seq_len, input_size]
        
        # Pass through LSTM layers
        current_out = x
        for lstm, dropout in zip(self.lstm_layers, self.dropouts):
            current_out, _ = lstm(current_out)
            current_out = dropout(current_out)
        
        # Take the output from the last time step
        last_out = current_out[:, -1, :]
        
        # Pass through final linear layer
        out = self.fc(last_out)
        return out

class ConvLSTM(BaseModel):
    """CNN-LSTM hybrid model."""
    def __init__(self, input_size: int = 21, hidden_size: int = 64, dropout: float = 0.2):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # CNN layers
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_logger()

    def forward(self, x):
        """Forward pass combining CNN and LSTM."""
        # Reshape for CNN: [batch, sequence, features] -> [batch, features, sequence]
        x = x.permute(0, 2, 1)
        
        # CNN layers
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        conv_out = F.relu(conv_out)
        
        # Reshape back for LSTM: [batch, features, sequence] -> [batch, sequence, features]
        conv_out = conv_out.permute(0, 2, 1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(conv_out)
        
        # Get last output
        last_out = lstm_out[:, -1, :]
        
        # Dropout and final layer
        out = self.dropout_layer(last_out)
        out = self.fc(out)
        
        return out

class AttentionLSTM(BaseModel):
    """LSTM with attention mechanism."""
    def __init__(self, input_size: int = 21, hidden_size: int = 64, dropout: float = 0.2):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_logger()

    def attention_net(self, lstm_output):
        """Compute attention weights."""
        attn_weights = self.attention(lstm_output)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        return context.squeeze(2)

    def forward(self, x):
        """Forward pass with attention mechanism."""
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out = self.attention_net(lstm_out)
        
        # Dropout and final layer
        out = self.dropout_layer(attn_out)
        out = self.fc(out)
        
        return out

class HybridTransformerLSTM(BaseModel):
    """Hybrid model combining Transformer and LSTM."""
    def __init__(self, input_size: int = 21, hidden_size: int = 64, dropout: float = 0.2):
        super(HybridTransformerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,  # Number of attention heads
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_logger()

    def forward(self, x):
        """Forward pass combining Transformer and LSTM."""
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # LSTM layer
        lstm_out, _ = self.lstm(transformer_out)
        
        # Get last output
        last_out = lstm_out[:, -1, :]
        
        # Dropout and final layer
        out = self.dropout_layer(last_out)
        out = self.fc(out)
        
        return out

class CustomLoss(nn.Module):
    """Custom loss function that combines MSE, MAE, and R² score."""
    
    def __init__(self, mse_weight=0.4, mae_weight=0.3, r2_weight=0.3):
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.r2_weight = r2_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, y_pred, y_true):
        # Calculate MSE
        mse_loss = self.mse(y_pred, y_true)
        
        # Calculate MAE
        mae_loss = self.mae(y_pred, y_true)
        
        # Calculate R² score
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)  # Add epsilon to prevent division by zero
        r2_loss = 1 - r2  # Convert to loss (0 is best)
        
        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.mae_weight * mae_loss +
            self.r2_weight * r2_loss
        )
        
        return total_loss, mse_loss, mae_loss, r2

class MetricsTracker:
    """Track and log model metrics during training."""
    
    def __init__(self, logger):
        self.logger = logger
        self.reset()
    
    def reset(self):
        self.current_metrics = {
            'train_loss': [],
            'val_loss': [],
            'mse': [],
            'mae': [],
            'r2': [],
            'best_val_loss': float('inf'),
            'best_r2': -float('inf'),
            'best_epoch': 0,
            'epochs_without_improvement': 0
        }
    
    def update(self, metrics, epoch):
        """Update metrics and check for improvements."""
        for key, value in metrics.items():
            if key in self.current_metrics:
                self.current_metrics[key].append(value)
        
        # Check for improvement
        if metrics['val_loss'] < self.current_metrics['best_val_loss']:
            self.current_metrics['best_val_loss'] = metrics['val_loss']
            self.current_metrics['best_r2'] = metrics['r2']
            self.current_metrics['best_epoch'] = epoch
            self.current_metrics['epochs_without_improvement'] = 0
            return True
        else:
            self.current_metrics['epochs_without_improvement'] += 1
            return False
    
    def should_stop_early(self, patience):
        """Check if early stopping should be triggered."""
        return self.current_metrics['epochs_without_improvement'] >= patience
    
    def log_epoch(self, epoch):
        """Log metrics for current epoch."""
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {self.current_metrics['train_loss'][-1]:.6f}, "
            f"Val Loss: {self.current_metrics['val_loss'][-1]:.6f}, "
            f"MSE: {self.current_metrics['mse'][-1]:.6f}, "
            f"MAE: {self.current_metrics['mae'][-1]:.6f}, "
            f"R²: {self.current_metrics['r2'][-1]:.6f}"
        )
    
    def log_best_metrics(self):
        """Log best metrics achieved during training."""
        self.logger.info("\nBest Model Metrics:")
        self.logger.info(f"Best Validation Loss: {self.current_metrics['best_val_loss']:.6f}")
        self.logger.info(f"Best R² Score: {self.current_metrics['best_r2']:.6f}")
        self.logger.info(f"Best Epoch: {self.current_metrics['best_epoch']}")

class ModelOptimizer:
    """Optimize model hyperparameters using Optuna."""
    
    def __init__(self, model_class, train_data, val_data, device, logger):
        self.model_class = model_class
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.logger = logger
    
    def objective(self, trial: optuna.trial.Trial):
        """Optuna objective function for hyperparameter optimization."""
        # Hyperparameter search space
        params = {
            'input_size': self.train_data[0].shape[-1],
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }
        
        # Initialize model with trial parameters
        model = self.model_class(**params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = CustomLoss()
        
        # Create data loaders with trial batch size
        train_loader = DataLoader(self.train_data, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=params['batch_size'])
        
        # Training loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        patience = 10
        
        for epoch in range(50):  # Maximum 50 epochs for trial
            model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss, mse_loss, mae_loss, r2 = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss, mse_loss, mae_loss, r2 = criterion(output, target)
                    val_loss += val_loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    break
        
        return best_val_loss
    
    def optimize(self, n_trials=100):
        """Run Optuna optimization."""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.logger.info("Best trial:")
        trial = study.best_trial
        self.logger.info(f"  Value: {trial.value:.6f}")
        self.logger.info("  Params: ")
        for key, value in trial.params.items():
            self.logger.info(f"    {key}: {value}")
        
        return study.best_params

class MLStrategy:
    """Machine Learning trading strategy."""
    
    def __init__(self, model_config, logger=None):
        """Initialize ML strategy."""
        self.model_config = model_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Configure logging
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # Ensure GPU is available
        if not torch.cuda.is_available():
            self.logger.warning("GPU not available, using CPU")
            self.device = torch.device('cpu')
        else:
            # Configure GPU
            torch.cuda.empty_cache()
            # Set memory growth limit to 70% of available memory
            torch.cuda.set_per_process_memory_fraction(0.7)
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        self.models = {}
        self.scaler = StandardScaler()
        self.model_metrics = {}
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
        self.logger.info("ML Strategy initialized successfully")
        
    def _log_gpu_memory(self, step: str):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()/1024**2
            memory_cached = torch.cuda.memory_reserved()/1024**2
            self.logger.info(f"GPU Memory at {step}: Allocated={memory_allocated:.2f}MB, Cached={memory_cached:.2f}MB")
            
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cleared GPU memory cache")
            
    def _init_models(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize all model architectures."""
        try:
            self.logger.info("Initializing models...")
            n_features = input_shape[1]  # Number of features
            
            # Initialize models with GPU
            models = [
                SimpleLSTM(input_size=n_features, hidden_size=64).to(self.device),
                DeepLSTM(input_size=n_features, hidden_sizes=[64, 32]).to(self.device),
                ConvLSTM(input_size=n_features, hidden_size=64).to(self.device),
                AttentionLSTM(input_size=n_features, hidden_size=64).to(self.device),
                HybridTransformerLSTM(input_size=n_features, hidden_size=64).to(self.device)
            ]
            
            self.models = models
            self.logger.info(f"Initialized {len(models)} models")
            
            # Log model architectures
            for model in self.models:
                self.logger.info(f"\nModel: {model.__class__.__name__}")
                self.logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def train_models(self, stock_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]):
        """Train all models with the prepared data."""
        try:
            self.logger.info("Starting model training process...")
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, test_prices = self._prepare_data(stock_data, market_data)
            
            # Create data loaders
            train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
            
            # Optimize hyperparameters for each model
            for model_name, model_class in self.models.items():
                self.logger.info(f"\nTraining {model_name}...")
                
                # Optimize hyperparameters
                optimizer = ModelOptimizer(
                    model_class,
                    train_data,
                    val_data,
                    self.device,
                    self.logger
                )
                best_params = optimizer.optimize()
                
                # Train model with best parameters
                model = model_class(**best_params).to(self.device)
                self._train_model(model, train_data, val_data)
                
                # Evaluate trading performance
                self.logger.info(f"\nEvaluating {model_name} trading performance...")
                performance = self.evaluate_model_performance(model, X_test, test_prices)
                
                if performance:
                    # Store model and performance metrics
                    self.models[model_name] = model
                    self.model_metrics[model_name].update({
                        'trading_metrics': performance
                    })
            
            # Print final performance summary
            self._print_performance_summary()
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _print_performance_summary(self):
        """Print a summary of model performance and trading metrics."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Final Model Performance Summary")
        self.logger.info("=" * 50)
        
        # Sort models by CAGR
        sorted_models = sorted(
            self.model_metrics.items(),
            key=lambda x: x[1]['trading_metrics']['CAGR']['value'],
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            self.logger.info(f"\nModel: {model_name}")
            self.logger.info("-" * 40)
            
            # ML Metrics
            self.logger.info("ML Metrics:")
            self.logger.info(f"Best R² Score: {metrics['best_r2']:.6f}")
            self.logger.info(f"MSE: {metrics['mse'][-1]:.6f}")
            self.logger.info(f"MAE: {metrics['mae'][-1]:.6f}")
            
            # Trading Metrics
            self.logger.info("\nTrading Metrics:")
            trading_metrics = metrics['trading_metrics']
            for metric, data in trading_metrics.items():
                status = '✅' if data['meets'] else '❌'
                self.logger.info(f"{metric}: {data['value']:.2f}% (Target: {data['target']}%) {status}")
            
            # Model Stability
            r2_std = np.std(metrics['r2'])
            self.logger.info(f"\nModel Stability:")
            self.logger.info(f"R² Score Std Dev: {r2_std:.6f}")
            
    def train_model(self, model, train_loader, val_loader, optimizer, criterion):
        """Train a single model."""
        try:
            model.train()  # Set model to training mode
            model.to(self.device)  # Ensure model is on correct device
            
            best_val_loss = float('inf')
            patience = self.model_config.patience
            epochs_without_improvement = 0
            
            metrics_tracker = MetricsTracker(self.logger)
            
            for epoch in range(self.model_config.num_epochs):
                # Training phase
                total_train_loss = 0
                model.train()
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # data shape: [batch_size, seq_len, features]
                    output = model(data)
                    loss, mse_loss, mae_loss, r2 = criterion(output, target)
                    
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f'Epoch {epoch}: [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')
                        self._log_gpu_memory(f"Batch {batch_idx}")
                
                avg_train_loss = total_train_loss / len(train_loader)
                
                # Validation phase
                model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        val_loss, mse_loss, mae_loss, r2 = criterion(output, target)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                
                # Update metrics
                metrics = {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'mse': mse_loss.item(),
                    'mae': mae_loss.item(),
                    'r2': r2.item()
                }
                improved = metrics_tracker.update(metrics, epoch)
                
                # Log epoch metrics
                metrics_tracker.log_epoch(epoch)
                
                # Check for early stopping
                if metrics_tracker.should_stop_early(self.model_config.patience):
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Clear memory after each epoch
                self._clear_gpu_memory()
            
            # Log best metrics
            metrics_tracker.log_best_metrics()
            
            return metrics_tracker.current_metrics
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def evaluate_model(self, model, val_loader, criterion):
        """Evaluate model performance with detailed metrics."""
        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    # No need to move to GPU as data is already there
                    outputs = model(data)
                    loss, mse_loss, mae_loss, r2 = criterion(outputs, target)
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    # Store predictions and targets for metric calculation
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                except RuntimeError as e:
                    self.logger.error(f"Error in validation: {str(e)}")
                    continue
                
                del outputs, loss
                torch.cuda.empty_cache()
        
        # Calculate metrics
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'val_loss': avg_val_loss,
            'mse': mean_squared_error(all_targets, all_predictions),
            'mae': mean_absolute_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions)
        }
        
        return metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction using all models."""
        try:
            self._log_gpu_memory("Before prediction")
            
            # Add stock symbol prefix to features
            features_with_prefix = features.copy()
            prefix = list(self.models.keys())[0].lower() + '_'  # Use first model's stock symbol
            features_with_prefix.columns = [prefix + col for col in features.columns]
            
            # Scale features
            scaled_features = self.scaler.transform(features_with_prefix)
            
            # Convert to tensor and reshape for LSTM input [batch_size, seq_len, features]
            # For single prediction, batch_size=1, seq_len=1
            X = torch.FloatTensor(scaled_features).unsqueeze(0)
            if len(X.shape) == 2:  # If input is [batch, features]
                X = X.unsqueeze(1)  # Add sequence dimension [batch, seq_len=1, features]
            X = X.to(self.device)
            
            self.logger.debug(f"Input tensor shape: {X.shape}")
            
            predictions = []
            for model_name, model in self.models.items():
                try:
                    model.eval()
                    with torch.no_grad():
                        pred = model(X)
                        predictions.append(pred.cpu().numpy())
                    
                    # Clear memory after each model
                    del pred
                    self._clear_gpu_memory()
                    
                except Exception as e:
                    self.logger.error(f"Error in model {model_name} prediction: {str(e)}")
                    continue
            
            # Clear input tensor
            del X
            self._clear_gpu_memory()
            
            self._log_gpu_memory("After prediction")
            
            if not predictions:
                raise ValueError("No valid predictions from any model")
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred.squeeze()
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def evaluate_model_performance(self, model, test_data, test_prices):
        """Evaluate model performance against trading targets."""
        try:
            model.eval()
            predictions = []
            actual_returns = []
            positions = []
            
            with torch.no_grad():
                for i in range(len(test_data)):
                    features = torch.FloatTensor(test_data[i:i+1]).to(self.device)
                    pred = model(features).cpu().numpy()[0][0]
                    predictions.append(pred)
                    
                    # Calculate actual returns
                    if i < len(test_data) - 1:
                        actual_return = (test_prices[i+1] - test_prices[i]) / test_prices[i]
                        actual_returns.append(actual_return)
                        
                        # Determine position (1 for long, -1 for short, 0 for no position)
                        position = 1 if pred > 0 else -1 if pred < 0 else 0
                        positions.append(position)
            
            # Calculate strategy returns
            strategy_returns = pd.Series(actual_returns) * pd.Series(positions[:-1])
            
            # Calculate performance metrics
            metrics = PerformanceMetrics(strategy_returns, positions)
            performance = metrics.evaluate_strategy()
            
            self.logger.info("\nTrading Performance Metrics:")
            self.logger.info("=" * 40)
            self.logger.info(metrics.get_summary())
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _prepare_data(self, stock_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]):
        """Prepare data for training."""
        try:
            # Process stock data
            features_list = []
            
            # Process each stock separately
            for stock_symbol, stock_df in stock_data.items():
                # Calculate technical indicators for the stock
                stock_features = self._calculate_technical_indicators(stock_df)
                if stock_features is None:
                    raise ValueError(f"Failed to calculate technical indicators for {stock_symbol}")
                
                # Add stock symbol prefix to columns
                stock_features = stock_features.add_prefix(f'{stock_symbol.lower()}_')
                features_list.append(stock_features)
            
            # Merge all stock features
            features = pd.concat(features_list, axis=1)
            
            # Process market data
            for market_symbol, market_df in market_data.items():
                # Calculate technical indicators for market data
                market_features = self._calculate_technical_indicators(market_df)
                if market_features is not None:
                    # Add market symbol prefix to columns
                    market_features = market_features.add_prefix(f'{market_symbol.lower()}_')
                    features = pd.merge(features, market_features, left_index=True, right_index=True, how='left')
            
            # Create target variable (next day's returns for each stock)
            target = pd.DataFrame(index=features.index)
            for stock_symbol in stock_data.keys():
                close_col = f'{stock_symbol.lower()}_close'
                if close_col in features.columns:
                    # Calculate returns
                    target[stock_symbol] = features[close_col].pct_change().shift(-1)
            
            # Drop rows with NaN values
            valid_idx = ~(features.isna().any(axis=1) | target.isna().any(axis=1))
            features = features[valid_idx]
            target = target[valid_idx]
            
            # Scale features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
            scaled_features = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)
            
            # Create sequences for LSTM
            sequence_length = self.model_config.sequence_length
            X_sequences = []
            y_sequences = []
            
            for i in range(len(scaled_features) - sequence_length):
                X_sequences.append(scaled_features.iloc[i:i+sequence_length].values)
                y_sequences.append(target.iloc[i+sequence_length].values)
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Split data into training and validation sets
            split_idx = int(len(X_sequences) * 0.8)
            
            # Convert to PyTorch tensors with proper shapes for LSTM
            # Shape: [batch_size, sequence_length, n_features]
            X_train = torch.tensor(X_sequences[:split_idx], dtype=torch.float32).to(self.device)
            y_train = torch.tensor(y_sequences[:split_idx], dtype=torch.float32).to(self.device)
            X_val = torch.tensor(X_sequences[split_idx:], dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_sequences[split_idx:], dtype=torch.float32).to(self.device)
            
            # Split validation set into validation and test sets
            test_split_idx = int(len(X_val) * 0.5)
            X_val = X_val[:test_split_idx]
            y_val = y_val[:test_split_idx]
            X_test = X_val[test_split_idx:]
            y_test = y_val[test_split_idx:]
            
            test_prices = features['close'].iloc[-len(X_test):]
            
            self.logger.info(f"Training data shape: {X_train.shape}")
            self.logger.info(f"Training target shape: {y_train.shape}")
            self.logger.info(f"Validation data shape: {X_val.shape}")
            self.logger.info(f"Validation target shape: {y_val.shape}")
            self.logger.info(f"Test data shape: {X_test.shape}")
            self.logger.info(f"Test target shape: {y_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test, test_prices
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        try:
            # Convert data to DataFrame if it's a Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Create a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Initialize indicators dictionary
            indicators = pd.DataFrame(index=df.index)
            
            # Price-based indicators
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            indicators['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
            indicators['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            indicators['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            indicators['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
            
            # Momentum indicators
            indicators['rsi'] = ta.momentum.rsi(df['close'], window=14)
            indicators['macd'] = ta.trend.macd_diff(df['close'])
            indicators['macd_signal'] = ta.trend.macd_signal(df['close'])
            indicators['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            indicators['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            indicators['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            indicators['roc'] = ta.momentum.roc(df['close'])
            
            # Volatility indicators
            indicators['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            indicators['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            indicators['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            indicators['natr'] = indicators['atr'] / df['close']  # Normalized ATR
            
            # Volume indicators
            indicators['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            indicators['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
            indicators['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            indicators['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Trend indicators
            indicators['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            indicators['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            indicators['dpo'] = ta.trend.dpo(df['close'])
            indicators['kst'] = ta.trend.kst(df['close'])
            indicators['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
            indicators['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
            
            # Price transformations
            indicators['log_return'] = np.log(df['close'] / df['close'].shift(1))
            indicators['return_volatility'] = indicators['log_return'].rolling(window=20).std()
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                indicators[f'close_lag_{lag}'] = df['close'].shift(lag)
                indicators[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                indicators[f'return_lag_{lag}'] = indicators['log_return'].shift(lag)
            
            # Add price and volume data
            indicators['open'] = df['open']
            indicators['high'] = df['high']
            indicators['low'] = df['low']
            indicators['close'] = df['close']
            indicators['volume'] = df['volume']
            
            # Forward fill any missing values
            indicators = indicators.fillna(method='ffill')
            
            # Handle remaining NaN values
            indicators = indicators.fillna(0)  # Replace remaining NaNs with 0
            
            # Remove outliers using IQR method
            for column in indicators.columns:
                Q1 = indicators[column].quantile(0.25)
                Q3 = indicators[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                indicators[column] = indicators[column].clip(lower=lower_bound, upper=upper_bound)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

class PerformanceMetrics:
    """Calculate trading strategy performance metrics."""
    
    def __init__(self, returns, positions, risk_free_rate=0.05):
        self.returns = returns
        self.positions = positions
        self.risk_free_rate = risk_free_rate
    
    def calculate_cagr(self):
        """Calculate Compound Annual Growth Rate."""
        total_days = len(self.returns)
        total_return = (1 + self.returns).prod()
        cagr = (total_return ** (252/total_days)) - 1
        return cagr * 100
    
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe Ratio."""
        excess_returns = self.returns - (self.risk_free_rate/252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        return sharpe_ratio
    
    def calculate_max_drawdown(self):
        """Calculate Maximum Drawdown percentage."""
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return drawdowns.min() * 100
    
    def calculate_avg_profit(self):
        """Calculate average profit percentage per trade."""
        profitable_trades = self.returns[self.returns > 0]
        avg_profit = profitable_trades.mean() if len(profitable_trades) > 0 else 0
        return avg_profit * 100
    
    def evaluate_strategy(self):
        """Evaluate if strategy meets target metrics."""
        cagr = self.calculate_cagr()
        sharpe = self.calculate_sharpe_ratio()
        max_dd = self.calculate_max_drawdown()
        avg_profit = self.calculate_avg_profit()
        
        meets_targets = {
            'CAGR': {'value': cagr, 'target': 25, 'meets': cagr >= 25},
            'Sharpe': {'value': sharpe, 'target': 1, 'meets': sharpe >= 1},
            'Max Drawdown': {'value': max_dd, 'target': -20, 'meets': max_dd >= -20},
            'Avg Profit': {'value': avg_profit, 'target': 0.75, 'meets': avg_profit >= 0.75}
        }
        
        return meets_targets
    
    def get_summary(self):
        """Get summary of all performance metrics."""
        metrics = self.evaluate_strategy()
        summary = []
        for name, data in metrics.items():
            status = '✅' if data['meets'] else '❌'
            summary.append(f"{name}: {data['value']:.2f}% (Target: {data['target']}%) {status}")
        return '\n'.join(summary)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
