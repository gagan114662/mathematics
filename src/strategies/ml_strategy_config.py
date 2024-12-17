"""Configuration for ML-based trading strategy."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union

class ScalingMethod(Enum):
    """Available scaling methods for feature preprocessing."""
    STANDARD = auto()
    ROBUST = auto()
    MINMAX = auto()
    NONE = auto()

class ModelType(Enum):
    """Available model types for ML strategy."""
    LSTM = auto()
    XGB = auto()
    LIGHTGBM = auto()
    RANDOM_FOREST = auto()
    XGBOOST = auto()

@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    window_size: int = 20
    prediction_horizon: int = 1
    train_split: float = 0.7
    validation_split: float = 0.15
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    feature_selection_threshold: float = 0.05
    min_samples: int = 1000
    target_type: str = "binary"  # "binary" or "regression"
    target_threshold: float = 0.0  # For binary classification
    sequence_length: int = 10  # For LSTM models
    n_features: int = 20  # Number of features to use
    use_market_features: bool = True
    feature_selection_method: str = "mutual_info"
    max_features: int = 50
    scaling_params: Dict = field(default_factory=lambda: {
        "with_mean": True,
        "with_std": True
    })

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_position_size: float = 1.0
    base_position_size: float = 0.5
    stop_loss: float = 0.02
    take_profit: float = 0.04
    trailing_stop: bool = True
    trailing_stop_distance: float = 0.01
    max_drawdown: float = 0.2
    position_sizing_method: str = "dynamic"  # "fixed", "dynamic", "kelly"
    volatility_lookback: int = 21
    volatility_scaling: bool = True
    risk_free_rate: float = 0.02  # Annual risk-free rate
    daily_risk_free_rate: float = 0.02 / 252  # Daily risk-free rate
    max_portfolio_volatility: float = 0.15
    annual_risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    max_correlation: float = 0.7

@dataclass
class MLStrategyConfig:
    """Main configuration for ML strategy."""
    model_type: ModelType = ModelType.LSTM
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    random_seed: int = 42
    verbose: bool = True
    save_models: bool = True
    model_path: str = "models"
    log_path: str = "logs"
    model: ModelType = ModelType.LSTM
    
    # Model-specific parameters
    lstm_params: Dict = field(default_factory=lambda: {
        "units": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10
    })
    
    xgb_params: Dict = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    
    lightgbm_params: Dict = field(default_factory=lambda: {
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "binary",
        "metric": "auc",
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    
    # Feature engineering parameters
    feature_params: Dict = field(default_factory=lambda: {
        "use_ta": True,
        "use_sentiment": False,
        "use_fundamentals": False,
        "custom_features": [],
        "ta_params": {
            "volume": True,
            "volatility": True,
            "trend": True,
            "momentum": True,
            "others": False
        }
    })
    
    # Optimization parameters
    optimization_params: Dict = field(default_factory=lambda: {
        "use_optuna": False,
        "n_trials": 100,
        "timeout": 3600,
        "optimization_metric": "sharpe_ratio"
    })
