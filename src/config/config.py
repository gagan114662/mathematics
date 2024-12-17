"""Configuration settings for the algorithmic trading system."""
from datetime import datetime, timedelta
from typing import Dict, Any

class TradingConfig:
    """Trading system configuration."""
    
    # Data fetching settings
    DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
    MAX_WORKERS = 4  # Number of parallel workers for data fetching
    
    # Model parameters
    MODEL_PARAMS = {
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
    FEATURE_PARAMS = {
        'volatility_windows': [5, 10, 20],
        'volume_windows': [5, 20],
        'rsi_periods': [7, 14, 21],
        'bb_windows': [10, 20, 30],
        'momentum_windows': [5, 10, 20]
    }
    
    # Trading parameters
    TRADING_PARAMS = {
        'position_threshold_upper': 0.75,
        'position_threshold_lower': 0.25,
        'max_leverage': 1.5,
        'min_profit_target': 0.0075,  # 0.75%
        'max_drawdown_limit': 0.20,   # 20%
        'risk_free_rate': 0.05        # 5% annual
    }
    
    # Backtest parameters
    BACKTEST_PARAMS = {
        'initial_capital': 100000,
        'transaction_cost': 0.001,    # 0.1% per trade
        'slippage': 0.0005           # 0.05% slippage assumption
    }
    
    @classmethod
    def get_model_params(cls) -> Dict[str, Any]:
        """Get model parameters."""
        return cls.MODEL_PARAMS.copy()
    
    @classmethod
    def get_feature_params(cls) -> Dict[str, Any]:
        """Get feature engineering parameters."""
        return cls.FEATURE_PARAMS.copy()
    
    @classmethod
    def get_trading_params(cls) -> Dict[str, Any]:
        """Get trading parameters."""
        return cls.TRADING_PARAMS.copy()
    
    @classmethod
    def get_backtest_params(cls) -> Dict[str, Any]:
        """Get backtest parameters."""
        return cls.BACKTEST_PARAMS.copy()

class FeatureConfig:
    """Feature engineering configuration."""
    def __init__(self):
        self.use_ta = True
        self.ta_params = {
            'price': True,
            'volume': True,
            'momentum': True,
            'volatility': True
        }

class MLStrategyConfig:
    """ML Strategy configuration."""
    def __init__(self):
        self.model = None  # ModelConfig instance
        self.data = None   # DataConfig instance
        self.risk = None   # RiskConfig instance
        self.feature_params = FeatureConfig()
