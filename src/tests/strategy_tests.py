import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.strategies.ml_strategy import MLStrategy
from src.portfolio import Portfolio
import yfinance as yf
from pandas_datareader import data as pdr

class TestMLStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Test configuration
        cls.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'AMD', 'TSLA', 'META', 'JPM', 'GS']
        cls.end_date = datetime.now()
        cls.start_date = cls.end_date - timedelta(days=730)  # 2 years
        cls.test_start = cls.end_date - timedelta(days=180)  # 6 months test period
        
        # Performance targets
        cls.target_cagr = 25.0
        cls.target_sharpe = 1.0
        cls.target_max_dd = -20.0
        cls.target_avg_profit = 0.75
        cls.risk_free_rate = 5.0
        
    def setUp(self):
        self.strategy = MLStrategy(self.symbols, self.start_date, self.end_date)
        
    def test_performance_metrics(self):
        """Test if strategy meets all performance targets"""
        # Get data
        yf.pdr_override()
        training_data = {}
        for symbol in self.symbols:
            data = pdr.get_data_yahoo(symbol, start=self.start_date, end=self.end_date)
            if len(data) > 0:
                training_data[symbol] = data
        
        # Train strategy
        self.strategy.train_all(training_data)
        
        # Run backtest
        backtest_results = {}
        for symbol, data in training_data.items():
            test_data = data[data.index >= self.test_start]
            if len(test_data) > 0:
                signals = self.strategy.backtest(test_data)
                backtest_results[symbol] = signals
        
        # Calculate metrics
        portfolio = Portfolio(initial_capital=100000)
        portfolio.run_backtest(backtest_results, training_data)
        metrics = portfolio.calculate_metrics()
        
        # Log performance
        self.logger.info("\nStrategy Performance:")
        self.logger.info(f"CAGR: {metrics['cagr']:.2f}% (Target: >{self.target_cagr}%)")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (Target: >{self.target_sharpe})")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}% (Target: >{self.target_max_dd}%)")
        self.logger.info(f"Average Profit: {metrics['avg_profit']:.2f}% (Target: >{self.target_avg_profit}%)")
        
        # Assert performance targets
        self.assertGreater(metrics['cagr'], self.target_cagr, 
                          f"CAGR {metrics['cagr']:.2f}% below target of {self.target_cagr}%")
        
        self.assertGreater(metrics['sharpe_ratio'], self.target_sharpe,
                          f"Sharpe {metrics['sharpe_ratio']:.2f} below target of {self.target_sharpe}")
        
        self.assertGreater(metrics['max_drawdown'], self.target_max_dd,
                          f"Max Drawdown {metrics['max_drawdown']:.2f}% worse than target of {self.target_max_dd}%")
        
        self.assertGreater(metrics['avg_profit'], self.target_avg_profit,
                          f"Avg Profit {metrics['avg_profit']:.2f}% below target of {self.target_avg_profit}%")
        
    def test_feature_importance(self):
        """Test if feature importance is properly calculated"""
        self.assertIsNotNone(self.strategy.feature_importance,
                            "Feature importance not calculated")
        
    def test_signal_distribution(self):
        """Test if signals are reasonably distributed"""
        # Get sample data
        yf.pdr_override()
        symbol = self.symbols[0]  # Test with one symbol
        data = pdr.get_data_yahoo(symbol, start=self.start_date, end=self.end_date)
        
        # Train and get signals
        self.strategy.train_all({symbol: data})
        signals = self.strategy.backtest(data)
        
        # Check signal distribution
        unique_signals = np.unique(signals)
        signal_counts = pd.Series(signals).value_counts()
        
        # Ensure we have both long and short signals
        self.assertIn(1, unique_signals, "No long signals generated")
        self.assertIn(-1, unique_signals, "No short signals generated")
        
        # Check if signals are reasonably distributed
        total_signals = len(signals)
        long_ratio = signal_counts.get(1, 0) / total_signals
        short_ratio = signal_counts.get(-1, 0) / total_signals
        
        self.assertLess(long_ratio, 0.4, "Too many long signals")
        self.assertLess(short_ratio, 0.4, "Too many short signals")
        
    def test_risk_management(self):
        """Test risk management parameters"""
        self.assertLessEqual(self.strategy.position_size, 0.2,
                           "Position size too large")
        self.assertGreaterEqual(self.strategy.stop_loss, 0.02,
                              "Stop loss too tight")
        self.assertLessEqual(self.strategy.max_positions, 5,
                           "Too many concurrent positions allowed")

if __name__ == '__main__':
    unittest.main()
