"""Risk configuration for trading strategies."""

class RiskConfig:
    """Configuration class for risk management parameters."""
    
    def __init__(self):
        # Position sizing parameters
        self.risk_per_trade = 0.02  # Risk 2% of account per trade
        self.max_position_size = 0.25  # Maximum position size as percentage of account
        self.min_position_size = 0.01  # Minimum position size as percentage of account
        
        # Stop loss and take profit parameters
        self.stop_loss_atr_multiplier = 2.0  # Stop loss distance in ATR units
        self.take_profit_atr_multiplier = 3.0  # Take profit distance in ATR units
        
        # Risk management parameters
        self.max_open_positions = 5  # Maximum number of concurrent open positions
        self.max_daily_loss = 0.05  # Maximum daily loss as percentage of account
        self.max_drawdown = 0.20  # Maximum drawdown before stopping trading
        
        # Volatility parameters
        self.min_volatility = 0.01  # Minimum volatility required to enter trade
        self.max_volatility = 0.05  # Maximum volatility allowed for trade entry
        
        # Portfolio risk parameters
        self.max_sector_exposure = 0.30  # Maximum exposure to any single sector
        self.max_correlation = 0.75  # Maximum correlation between positions
        
        # Time-based parameters
        self.min_holding_period = 1  # Minimum holding period in days
        self.max_holding_period = 30  # Maximum holding period in days
