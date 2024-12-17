"""ML Strategy Configuration."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple

class ModelType(str, Enum):
    """Model types."""
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"

class ScalingMethod(str, Enum):
    """Scaling method for features."""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"

@dataclass
class DataConfig:
    """Data configuration class."""
    symbols: List[str]  # List of stock symbols
    market_symbols: List[str]  # Market indicators like VIX, SPY
    sector_etfs: List[str]  # Sector ETFs for market regime
    start_date: str
    end_date: str
    sequence_length: int
    n_features: Optional[int]
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    cache_dir: str = 'data/cache'
    min_samples: int = 1000
    max_samples: int = 10000
    
    def __post_init__(self):
        """Validate configuration."""
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.n_features is not None and self.n_features <= 0:
            raise ValueError("n_features must be positive")
        if not 0 <= self.validation_split <= 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 <= self.test_split <= 1:
            raise ValueError("test_split must be between 0 and 1")
        if self.validation_split + self.test_split >= 1:
            raise ValueError("validation_split + test_split must be less than 1")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.max_samples <= self.min_samples:
            raise ValueError("max_samples must be greater than min_samples")

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    price_features: bool = True
    volume_features: bool = True
    technical_features: bool = True
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    trend_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    use_yeo_johnson: bool = True
    missing_threshold: float = 0.05
    feature_list: List[str] = field(default_factory=list)
    ma_periods: List[int] = field(default_factory=list)
    rsi_period: int = 14
    macd_periods: Tuple[int, int, int] = (12, 26, 9)
    bb_period: int = 20
    bb_std: float = 2.0
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    window_sizes: Dict[str, int] = field(default_factory=dict)

@dataclass
class RiskConfig:
    """Risk configuration."""
    account_size: float  # Account size in base currency
    max_position_size: float  # Maximum position size as percentage of account
    stop_loss: float  # Stop loss percentage
    take_profit: float  # Take profit percentage
    max_trades_per_day: int = 10
    max_portfolio_volatility: float = 0.15
    max_correlation: float = 0.7
    max_drawdown: float = 0.2
    max_leverage: float = 1.0
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    risk_per_trade: float = 0.02  # Risk 2% of account per trade
    min_position_size: float = 0.01  # Minimum position size as percentage of account
    stop_loss_atr_multiplier: float = 2.0  # Stop loss distance in ATR units
    take_profit_atr_multiplier: float = 3.0  # Take profit distance in ATR units
    min_volatility: float = 0.01  # Minimum volatility required to enter trade
    max_volatility: float = 0.05  # Maximum volatility allowed for trade entry

class ModelConfig:
    """Model configuration."""
    def __init__(
        self,
        model_type: ModelType = ModelType.LSTM,
        input_dim: Optional[int] = None,
        sequence_length: int = 60,  # Number of time steps for LSTM
        lstm_units: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        optimize_hyperparameters: bool = False,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        l2_reg: float = 1e-6
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimize_hyperparameters = optimize_hyperparameters
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.l2_reg = l2_reg
        
        # Validate configuration
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if input_dim is not None and input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    max_depth: int = 5
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    validation_batch_size: Optional[int] = None

@dataclass
class MLStrategyConfig:
    """Configuration for ML Strategy."""
    
    data: DataConfig
    model: ModelConfig
    risk: RiskConfig
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig(**self.data)
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**self.model)
        if not isinstance(self.risk, RiskConfig):
            self.risk = RiskConfig(**self.risk)
        if not isinstance(self.feature_config, FeatureConfig):
            self.feature_config = FeatureConfig(**self.feature_config)
        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig(**self.training)
