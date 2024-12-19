"""Configuration settings for the algorithmic trading system."""
from datetime import datetime, timedelta
from typing import Dict, Any, List
from enum import Enum

class ModelType(Enum):
    """Model types supported by the system."""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"

class Config:
    """Trading system configuration."""
    
    def __init__(self):
        # Stock symbols to trade
        self.stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Market symbols to track
        self.market_symbols = ['SPY', 'QQQ', 'VIX']
        
        # Date range
        self.start_date = '2023-01-01'
        self.end_date = '2023-12-31'
        
        # Account settings
        self.account_size = 100000.0
        self.max_position_size = 0.1  # 10% of account
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.04  # 4%
        
        # Model configuration
        self.model = ModelConfig()
        
        # Model parameters
        self.model_params = {
            'max_depth': 7,
            'learning_rate': 0.005,
            'n_estimators': 1000,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.1,
            'scale_pos_weight': 2,
            'max_delta_step': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'error'],
            'use_label_encoder': False,
            'random_state': 42
        }
        
        # Feature engineering parameters
        self.feature_params = {
            'volatility_windows': [5, 10, 20],
            'volume_windows': [5, 20],
            'rsi_periods': [7, 14, 21],
            'bb_windows': [10, 20, 30],
            'momentum_windows': [5, 10, 20]
        }
        
        # Trading parameters
        self.trading_params = {
            'position_threshold_upper': 0.75,
            'position_threshold_lower': 0.25,
            'max_leverage': 1.5,
            'min_profit_target': 0.0075,  # 0.75%
            'max_drawdown_limit': 0.20,   # 20%
            'risk_free_rate': 0.05        # 5% annual
        }
        
        # Backtest parameters
        self.backtest_params = {
            'initial_capital': 100000,
            'transaction_cost': 0.001,    # 0.1% per trade
            'slippage': 0.0005           # 0.05% slippage assumption
        }

class ModelConfig:
    """Model configuration."""
    
    def __init__(self):
        self.model_type = ModelType.LSTM
        self.input_size = 126  # Number of features
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 1
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 10
        self.device = 'cuda'  # or 'cpu'
