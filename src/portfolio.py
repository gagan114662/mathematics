import pandas as pd
import numpy as np
from typing import Dict
import logging

class Portfolio:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []  # list of trade dictionaries
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, signals: Dict[str, pd.Series], price_data: Dict[str, pd.DataFrame]):
        """Run backtest with signals and price data"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Combine all signals into a single DataFrame
        all_signals = pd.DataFrame(signals)
        dates = sorted(list(set([date for signal_series in signals.values() for date in signal_series.index])))
        
        for date in dates:
            daily_pnl = 0
            
            # Update positions with current prices
            portfolio_value = self.current_capital
            for symbol, quantity in self.positions.items():
                if date in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date, 'Close']
                    position_value = quantity * current_price
                    portfolio_value += position_value
            
            # Process signals for each symbol
            for symbol in signals.keys():
                if date in signals[symbol].index and date in price_data[symbol].index:
                    signal = signals[symbol][date]
                    price = price_data[symbol].loc[date]
                    
                    # Check for exit signals
                    if symbol in self.positions:
                        quantity = self.positions[symbol]
                        entry_price = next(t['entry_price'] for t in reversed(self.trades) 
                                        if t['symbol'] == symbol and t['status'] == 'open')
                        
                        # Check stop loss and take profit
                        pnl_pct = (price['Close'] - entry_price) / entry_price
                        if (quantity > 0 and (pnl_pct <= -0.025 or pnl_pct >= 0.05)) or \
                           (quantity < 0 and (-pnl_pct <= -0.025 or -pnl_pct >= 0.05)) or \
                           (signal != np.sign(quantity)):  # Exit on opposite signal
                            
                            # Close position
                            trade_value = quantity * price['Close']
                            self.current_capital += trade_value
                            daily_pnl += trade_value - (quantity * entry_price)
                            
                            # Record trade
                            for trade in reversed(self.trades):
                                if trade['symbol'] == symbol and trade['status'] == 'open':
                                    trade['status'] = 'closed'
                                    trade['exit_price'] = price['Close']
                                    trade['exit_date'] = date
                                    trade['pnl'] = daily_pnl
                                    trade['pnl_pct'] = pnl_pct
                                    break
                            
                            del self.positions[symbol]
                    
                    # Check for entry signals
                    elif signal != 0 and len(self.positions) < 5:  # Max 5 concurrent positions
                        # Calculate position size (15% of portfolio per position)
                        position_value = portfolio_value * 0.15
                        quantity = int(position_value / price['Close'])
                        if signal < 0:  # Short position
                            quantity = -quantity
                        
                        # Open position
                        self.positions[symbol] = quantity
                        position_cost = quantity * price['Close']
                        self.current_capital -= position_cost
                        
                        # Record trade
                        self.trades.append({
                            'symbol': symbol,
                            'entry_date': date,
                            'entry_price': price['Close'],
                            'quantity': quantity,
                            'status': 'open'
                        })
            
            # Record daily portfolio value
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl
            })
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {
                'cagr': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_profit': 0
            }
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        
        # CAGR calculation
        days = (equity_df.index[-1] - equity_df.index[0]).days
        total_return = (equity_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        cagr = ((1 + total_return) ** (365 / days) - 1) * 100
        
        # Sharpe Ratio (using 5% risk-free rate)
        risk_free_daily = 0.05 / 252  # Daily risk-free rate
        excess_returns = equity_df['daily_return'] - risk_free_daily
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Maximum Drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdowns = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # Average Profit per Trade
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        if closed_trades:
            profits = [t['pnl_pct'] * 100 for t in closed_trades]
            avg_profit = np.mean(profits)
        else:
            avg_profit = 0
        
        # Additional metrics
        win_rate = len([t for t in closed_trades if t['pnl'] > 0]) / len(closed_trades) if closed_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)) / \
                       abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0)) if closed_trades else 0
        
        return {
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_profit': avg_profit,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': len(closed_trades),
            'final_capital': equity_df['portfolio_value'].iloc[-1]
        }
