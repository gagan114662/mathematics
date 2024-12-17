"""Risk management module for trading strategies."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import logging

class RiskManager:
    """Risk management for trading strategies."""
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager.
        
        Args:
            config: Dictionary containing risk management parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_drawdown = 0.0
        self.peak_value = 1.0
        self.trailing_stop_price = None
        
    def calculate_position_size(self, 
                              confidence: float, 
                              volatility: float,
                              current_price: float,
                              portfolio_value: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            confidence: Model prediction confidence (0-1)
            volatility: Current market volatility
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            float: Position size as fraction of portfolio
            
        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        # Validate required config parameters
        required_params = ['base_position_size', 'position_sizing_method', 'max_position_size', 'volatility_scaling']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter in risk config: {param}")
            
        # Start with base position size
        position_size = self.config["base_position_size"]
        
        # Adjust based on position sizing method
        if self.config["position_sizing_method"] == "fixed":
            position_size = self.config["base_position_size"]
            
        elif self.config["position_sizing_method"] == "dynamic":
            # Scale position size by confidence and inverse volatility
            vol_factor = 1.0 / (1.0 + volatility)
            position_size *= confidence * vol_factor
            
        elif self.config["position_sizing_method"] == "kelly":
            # Simplified Kelly Criterion
            win_rate = confidence
            win_loss_ratio = 1.5  # Assumed ratio
            kelly_size = win_rate - ((1 - win_rate) / win_loss_ratio)
            position_size = max(0.0, kelly_size * self.config["base_position_size"])
            
        # Apply volatility scaling if enabled
        if self.config["volatility_scaling"]:
            target_vol = 0.15  # Annual target volatility
            current_vol = volatility * np.sqrt(252)  # Annualized volatility
            vol_scalar = target_vol / current_vol if current_vol > 0 else 1.0
            position_size *= vol_scalar
            
        # Ensure position size is within limits
        position_size = np.clip(position_size, 0.0, self.config["max_position_size"])
        
        # Calculate actual position size in units
        max_position_value = portfolio_value * position_size
        position_units = max_position_value / current_price
        
        return float(position_units)
            
    def update_stops(self, 
                    current_price: float, 
                    position_side: str,
                    entry_price: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Update stop loss and take profit levels.
        
        Args:
            current_price: Current asset price
            position_side: Position side ('long' or 'short')
            entry_price: Position entry price
            
        Returns:
            Tuple[Optional[float], Optional[float]]: Stop loss and take profit prices
        """
        try:
            stop_loss = None
            take_profit = None
            
            if position_side == "long":
                # Stop loss
                stop_loss = entry_price * (1 - self.config["stop_loss"])
                
                # Take profit
                take_profit = entry_price * (1 + self.config["take_profit"])
                
                # Trailing stop
                if self.config["trailing_stop"]:
                    if self.trailing_stop_price is None:
                        self.trailing_stop_price = stop_loss
                    else:
                        trailing_stop = current_price * (1 - self.config["trailing_stop_distance"])
                        self.trailing_stop_price = max(self.trailing_stop_price, trailing_stop)
                        stop_loss = max(stop_loss, self.trailing_stop_price)
                        
            elif position_side == "short":
                # Stop loss
                stop_loss = entry_price * (1 + self.config["stop_loss"])
                
                # Take profit
                take_profit = entry_price * (1 - self.config["take_profit"])
                
                # Trailing stop
                if self.config["trailing_stop"]:
                    if self.trailing_stop_price is None:
                        self.trailing_stop_price = stop_loss
                    else:
                        trailing_stop = current_price * (1 + self.config["trailing_stop_distance"])
                        self.trailing_stop_price = min(self.trailing_stop_price, trailing_stop)
                        stop_loss = min(stop_loss, self.trailing_stop_price)
                        
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error updating stops: {str(e)}")
            return None, None
            
    def check_drawdown(self, portfolio_value: float) -> bool:
        """
        Check if maximum drawdown has been exceeded.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            bool: True if max drawdown exceeded, False otherwise
        """
        try:
            # Update peak value
            self.peak_value = max(self.peak_value, portfolio_value)
            
            # Calculate current drawdown
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            
            # Check if max drawdown exceeded
            return self.current_drawdown > self.config["max_drawdown"]
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {str(e)}")
            return False
            
    def reset(self):
        """Reset risk manager state."""
        self.current_drawdown = 0.0
        self.peak_value = 1.0
        self.trailing_stop_price = None
