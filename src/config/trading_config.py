"""Trading configuration."""
from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    lookback_days: int = 365  # Days of historical data to use
    max_symbols: int = 40  # Maximum number of symbols to trade
    min_data_points: int = 252  # Minimum number of data points required
    backtest_days: int = 30  # Number of days to use for backtesting
