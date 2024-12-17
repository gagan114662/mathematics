"""Advanced risk management module with dynamic position sizing."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    volatility: float
    var: float
    cvar: float
    beta: float
    correlation: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float

class RiskManager:
    """Advanced risk management with dynamic position sizing."""
    
    def __init__(self, 
                 max_position_size: float = 1.0,
                 max_portfolio_volatility: float = 0.15,
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95,
                 max_correlation: float = 0.7):
        """Initialize risk manager."""
        self.max_position_size = max_position_size
        self.max_portfolio_volatility = max_portfolio_volatility
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.max_correlation = max_correlation
        
    def calculate_position_size(self,
                              predicted_return: float,
                              confidence: float,
                              volatility: float,
                              current_portfolio: Dict[str, float],
                              risk_metrics: RiskMetrics) -> float:
        """Calculate position size using Kelly Criterion with adjustments."""
        try:
            # Basic Kelly Criterion
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            # Estimate win/loss ratio based on predicted return and volatility
            win_ratio = abs(predicted_return) / volatility
            
            # Kelly position size
            kelly_fraction = (win_prob * win_ratio - loss_prob) / win_ratio
            
            # Apply fractional Kelly (more conservative)
            kelly_fraction *= 0.5
            
            # Adjust for portfolio volatility
            portfolio_adjustment = (
                self.max_portfolio_volatility / 
                max(risk_metrics.volatility, self.max_portfolio_volatility)
            )
            
            # Adjust for correlation
            correlation_penalty = max(0, 1 - risk_metrics.correlation / self.max_correlation)
            
            # Final position size with constraints
            position_size = (
                kelly_fraction * 
                portfolio_adjustment * 
                correlation_penalty * 
                self.max_position_size
            )
            
            return max(min(position_size, self.max_position_size), -self.max_position_size)
            
        except Exception as e:
            raise ValueError(f"Error in position size calculation: {str(e)}")
            
    def calculate_risk_metrics(self, 
                             returns: pd.Series,
                             market_returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # VaR and CVaR
            var = self._calculate_var(returns)
            cvar = self._calculate_cvar(returns)
            
            # Market-related metrics
            beta = self._calculate_beta(returns, market_returns)
            correlation = returns.corr(market_returns)
            
            # Performance metrics
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe = self._calculate_sharpe_ratio(returns)
            sortino = self._calculate_sortino_ratio(returns)
            
            return RiskMetrics(
                volatility=volatility,
                var=var,
                cvar=cvar,
                beta=beta,
                correlation=correlation,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino
            )
            
        except Exception as e:
            raise ValueError(f"Error in risk metrics calculation: {str(e)}")
            
    def _calculate_var(self, returns: pd.Series) -> float:
        """Calculate Value at Risk."""
        return abs(np.percentile(returns, (1 - self.confidence_level) * 100))
        
    def _calculate_cvar(self, returns: pd.Series) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(returns)
        return abs(returns[returns <= -var].mean())
        
    def _calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market."""
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1.0
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return abs(drawdowns.min())
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        if len(excess_returns) > 0:
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        return 0.0
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return 0.0
        
    def calculate_dynamic_stops(self,
                              entry_price: float,
                              position_size: float,
                              volatility: float,
                              atr: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels."""
        try:
            # Use ATR for more adaptive stops
            stop_multiple = 2.0 * (1 + volatility)  # Increase stops in high vol
            profit_multiple = 3.0 * (1 + volatility)  # Increase targets in high vol
            
            if position_size > 0:
                stop_loss = entry_price - atr * stop_multiple
                take_profit = entry_price + atr * profit_multiple
            else:
                stop_loss = entry_price + atr * stop_multiple
                take_profit = entry_price - atr * profit_multiple
                
            return stop_loss, take_profit
            
        except Exception as e:
            raise ValueError(f"Error in stop calculation: {str(e)}")
            
    def adjust_for_market_impact(self,
                               price: float,
                               volume: float,
                               position_size: float,
                               avg_volume: float) -> Tuple[float, float]:
        """Adjust execution price for market impact."""
        try:
            # Calculate market impact based on position size relative to volume
            volume_ratio = (position_size * price) / (avg_volume * price)
            impact_factor = 0.1  # Base impact factor
            
            # Non-linear impact model
            market_impact = impact_factor * (volume_ratio ** 0.5)
            
            # Calculate effective price after impact
            effective_price = price * (1 + market_impact * np.sign(position_size))
            
            # Calculate estimated slippage
            slippage = abs(effective_price - price) / price
            
            return effective_price, slippage
            
        except Exception as e:
            raise ValueError(f"Error in market impact calculation: {str(e)}")
