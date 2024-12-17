"""Performance metrics calculation utilities."""
from typing import Dict, Union
import numpy as np
import pandas as pd

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate various performance metrics from a series of returns.
        
        Args:
            returns: pandas Series of returns
            
        Returns:
            Dictionary containing performance metrics:
                - cagr: Compound Annual Growth Rate (%)
                - sharpe_ratio: Risk-adjusted return measure
                - max_drawdown: Maximum peak to trough decline (%)
                - win_rate: Percentage of winning trades (%)
                - avg_profit_per_trade: Average profit per trade (%)
                - profit_factor: Ratio of gross profits to gross losses
                
        Raises:
            ValueError: If returns is not a pandas Series
            TypeError: If returns contains non-numeric values
        """
        if not isinstance(returns, pd.Series):
            raise ValueError("Input must be a pandas Series")
            
        if not pd.api.types.is_numeric_dtype(returns):
            raise TypeError("Returns must contain numeric values")
            
        if returns.empty:
            return {
                'cagr': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'profit_factor': 0.0
            }
            
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # CAGR
            n_years = len(returns) / 252  # Assuming 252 trading days per year
            total_return = cum_returns.iloc[-1] - 1
            cagr = (((1 + total_return) ** (1/n_years)) - 1) * 100 if n_years > 0 else 0.0
            
            # Sharpe Ratio (with 5% annual risk-free rate)
            annual_rf = 0.05
            daily_rf = (1 + annual_rf) ** (1/252) - 1  # Convert annual to daily
            excess_returns = returns - daily_rf
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0.0
            
            # Maximum Drawdown
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = float(drawdowns.min() * 100)
            
            # Win Rate
            wins = (returns > 0).sum()
            total_trades = (~returns.isna()).sum()
            win_rate = float((wins / total_trades * 100) if total_trades > 0 else 0.0)
            
            # Average Profit per Trade
            avg_profit = float(returns.mean() * 100) if not returns.empty else 0.0
            
            # Profit Factor
            gross_profits = returns[returns > 0].sum()
            gross_losses = abs(returns[returns < 0].sum())
            profit_factor = float(gross_profits / gross_losses if gross_losses != 0 else float('inf'))
            
            return {
                'cagr': cagr,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            raise RuntimeError(f"Error calculating performance metrics: {str(e)}")
