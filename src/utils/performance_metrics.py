import numpy as np
import pandas as pd
from typing import Dict
import logging

class PerformanceMetrics:
    """Class to calculate trading strategy performance metrics."""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            # Drop any NaN values
            returns = returns.dropna()
            
            if len(returns) == 0:
                return {
                    'cagr': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0
                }
            
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            trading_days = len(returns)
            years = trading_days / 252
            
            # CAGR
            cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility and Sharpe Ratio
            annualized_vol = returns.std() * np.sqrt(252)
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Maximum Drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Win Rate and Average Profit
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_rate = len(wins) / len(returns) * 100 if len(returns) > 0 else 0
            avg_profit = returns.mean() * 100 if len(returns) > 0 else 0
            
            # Profit Factor
            gross_profits = wins.sum() if len(wins) > 0 else 0
            gross_losses = abs(losses.sum()) if len(losses) > 0 else 0
            profit_factor = abs(gross_profits / gross_losses) if gross_losses != 0 else float('inf')
            
            return {
                'cagr': cagr * 100,  # Convert to percentage
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,  # Convert to percentage
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'profit_factor': profit_factor,
                'total_trades': len(returns)
            }
            
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'cagr': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
