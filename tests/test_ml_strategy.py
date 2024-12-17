"""Test ML Strategy."""
import logging
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.config.ml_strategy_config import (
    MLStrategyConfig,
    DataConfig,
    RiskConfig,
    FeatureConfig,
    ModelConfig,
    ModelType,
    ScalingMethod,
    TrainingConfig
)
from src.strategies.ml_strategy import MLStrategy
from typing import cast

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestMLStrategy(unittest.TestCase):
    """Test ML Strategy."""
    
    @patch('src.data.data_fetcher.DataFetcher')
    def test_backtest_with_metrics(self, mock_data_fetcher):
        """Test backtesting with performance metrics calculation."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
        data = pd.DataFrame(
            index=dates,
            data={
                'open': close_prices * (1 + np.random.randn(len(dates)) * 0.001),
                'high': close_prices * (1 + abs(np.random.randn(len(dates))) * 0.002),
                'low': close_prices * (1 - abs(np.random.randn(len(dates))) * 0.002),
                'close': close_prices,
                'volume': np.random.lognormal(mean=np.log(1000000), sigma=0.5, size=len(dates)),
                'target': np.random.choice([-1, 1], size=len(dates))  # Binary targets
            }
        )
        
        # Mock DataFetcher
        mock_instance = Mock()
        mock_instance.get_data.return_value = data
        mock_data_fetcher.return_value = mock_instance
        
        # Create feature config
        feature_config = FeatureConfig(
            feature_list=['open', 'high', 'low', 'close', 'volume'],
            ma_periods=[5, 10, 20],
            rsi_period=14,
            macd_periods=(12, 26, 9),
            bb_period=20,
            bb_std=2.0,
            technical_indicators={'rsi': True, 'macd': True},
            window_sizes={'open': 5, 'high': 5, 'low': 5, 'close': 5, 'volume': 5}
        )
        
        # Initialize strategy with LightGBM model
        config = MLStrategyConfig(
            data=DataConfig(
                symbol="TEST",
                start_date="2023-01-01",
                end_date="2023-12-31",
                sequence_length=10,
                n_features=5,
                target_type='classification',
                scaling_method=ScalingMethod.STANDARD,
                validation_split=0.2,
                target_threshold=0.0,
                prediction_horizon=5
            ),
            model=ModelConfig(
                model_type=ModelType.LIGHTGBM,
                optimize_hyperparameters=False,
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 3
                }
            ),
            risk=RiskConfig(
                initial_capital=100000.0,
                max_position_size=0.1,
                stop_loss=0.02,
                take_profit=0.05,
                max_trades_per_day=10,
                max_portfolio_volatility=0.15,
                max_correlation=0.7,
                max_drawdown=0.2,
                max_leverage=1.0,
                risk_free_rate=0.02,
                confidence_level=0.95
            ),
            features=feature_config,
            training=TrainingConfig()
        )
        strategy = MLStrategy(config)
        
        # Train the model
        X_train = cast(pd.DataFrame, data[['open', 'high', 'low', 'close', 'volume']])
        y_train = cast(pd.Series, data['target'])
        strategy.train(X_train, y_train)
        
        # Run backtest
        backtest_returns = strategy.backtest(cast(pd.DataFrame, data))
        
        # Verify backtest returns
        self.assertIsNotNone(backtest_returns)
        self.assertIsInstance(backtest_returns, pd.Series)
        self.assertEqual(len(backtest_returns), len(data))
        
        # Get metrics
        metrics = strategy._calculate_metrics(backtest_returns)
        
        # Verify metrics structure
        expected_metrics = {
            'total_returns',
            'annualized_returns',
            'annualized_volatility',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'total_trades'
        }
        self.assertEqual(set(metrics.keys()), expected_metrics)
        
        # Verify metric values
        self.assertIsInstance(metrics['total_returns'], float)
        self.assertIsInstance(metrics['annualized_returns'], float)
        self.assertIsInstance(metrics['annualized_volatility'], float)
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertIsInstance(metrics['max_drawdown'], float)
        self.assertIsInstance(metrics['win_rate'], float)
        self.assertIsInstance(metrics['total_trades'], int)
        
        # Verify metric ranges
        self.assertGreaterEqual(metrics['win_rate'], 0.0)
        self.assertLessEqual(metrics['win_rate'], 1.0)
        self.assertLessEqual(metrics['max_drawdown'], 0.0)
        self.assertGreaterEqual(metrics['total_trades'], 0)
        
        # Test with empty returns
        empty_metrics = strategy._calculate_metrics(pd.Series([]))
        self.assertEqual(empty_metrics['total_returns'], 0.0)
        self.assertEqual(empty_metrics['total_trades'], 0)
        
        # Test with None returns
        none_metrics = strategy._calculate_metrics(None)
        self.assertEqual(none_metrics['total_returns'], 0.0)
        self.assertEqual(none_metrics['total_trades'], 0)

    @patch('src.data.data_fetcher.DataFetcher')
    def test_backtest_realistic_scenario(self, mock_data_fetcher):
        """Test backtesting with realistic market data."""
        # Create feature config
        feature_config = FeatureConfig(
            feature_list=['open', 'high', 'low', 'close', 'volume'],
            ma_periods=[5, 10, 20],
            rsi_period=14,
            macd_periods=(12, 26, 9),
            bb_period=20,
            bb_std=2.0,
            technical_indicators={'rsi': True, 'macd': True, 'bollinger': True},
            window_sizes={'close': 5, 'volume': 5}
        )
        
        # Initialize strategy with LightGBM model
        config = MLStrategyConfig(
            data=DataConfig(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-12-31",
                sequence_length=10,
                n_features=5,
                target_type='classification',
                scaling_method=ScalingMethod.STANDARD,
                validation_split=0.2,
                target_threshold=0.0,
                prediction_horizon=5
            ),
            model=ModelConfig(
                model_type=ModelType.LIGHTGBM,
                optimize_hyperparameters=False,
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1
                }
            ),
            risk=RiskConfig(
                initial_capital=100000.0,
                max_position_size=0.1,
                stop_loss=0.02,
                take_profit=0.05,
                max_trades_per_day=10,
                max_portfolio_volatility=0.15,
                max_correlation=0.7,
                max_drawdown=0.2,
                max_leverage=1.0,
                risk_free_rate=0.02,
                confidence_level=0.95
            ),
            features=feature_config,
            training=TrainingConfig()
        )

        # Create realistic test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Generate synthetic price data that follows a random walk
        close_prices = 100 * (1 + np.random.randn(n_samples).cumsum() * 0.02)
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
        
        # Calculate target before creating DataFrame
        target = np.where(np.roll(close_prices, -1) > close_prices, 1, -1)[:-1]
        dates = dates[:-1]  # Adjust dates to match target length
        
        test_data = pd.DataFrame(
            index=dates,
            data={
                'open': close_prices[:-1] * (1 + np.random.randn(len(dates)) * 0.001),
                'high': close_prices[:-1] * (1 + abs(np.random.randn(len(dates))) * 0.002),
                'low': close_prices[:-1] * (1 - abs(np.random.randn(len(dates))) * 0.002),
                'close': close_prices[:-1],
                'volume': np.random.lognormal(mean=np.log(1000000), sigma=0.5, size=len(dates)),
                'target': target
            }
        )

        # Mock DataFetcher
        mock_instance = Mock()
        mock_instance.get_data.return_value = test_data
        mock_data_fetcher.return_value = mock_instance

        strategy = MLStrategy(config)
        
        # Split data into train and test
        train_data = test_data[:'2023-06-30']
        test_data = test_data['2023-07-01':]
        
        # Train the model
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        X_train = cast(pd.DataFrame, train_data[feature_cols])
        y_train = cast(pd.Series, train_data['target'])
        strategy.train(X_train, y_train)
        
        # Run backtest on test data
        returns = strategy.backtest(cast(pd.DataFrame, test_data))
        metrics = strategy._calculate_metrics(returns)
        
        # Verify the strategy performs better than random
        win_rate = metrics.get('win_rate')
        sharpe_ratio = metrics.get('sharpe_ratio')
        
        # Ensure metrics are not None before comparison
        self.assertIsNotNone(win_rate, "Win rate should not be None")
        self.assertIsNotNone(sharpe_ratio, "Sharpe ratio should not be None")
        
        if win_rate is not None and sharpe_ratio is not None:
            self.assertGreater(float(win_rate), 0.45, "Win rate should be better than random guessing")
            self.assertGreater(float(sharpe_ratio), 0.0, "Sharpe ratio should be positive")
        
        # Additional metrics validation
        self.assertIsNotNone(metrics.get('total_returns'), "Total returns should not be None")
        self.assertIsNotNone(metrics.get('max_drawdown'), "Max drawdown should not be None")
        self.assertIsNotNone(metrics.get('total_trades'), "Total trades should not be None")

        # Verify returns are valid
        self.assertIsInstance(returns, pd.Series, "Backtest should return a pandas Series")
        self.assertGreater(len(returns), 0, "Backtest returns should not be empty")
        
def main():
    """Run ML strategy test."""
    # Create configuration
    data_config = DataConfig(
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2023-12-31",
        sequence_length=10,
        n_features=31,  # Match the actual number of features created
        validation_split=0.2,
        target_type='regression',
        target_threshold=0.0,
        prediction_horizon=1,
        scaling_method=ScalingMethod.STANDARD
    )

    risk_config = RiskConfig(
        initial_capital=100000.0,
        max_position_size=10000.0,
        stop_loss=0.02,
        take_profit=0.04,
        max_trades_per_day=10,
        max_portfolio_volatility=0.15,
        max_correlation=0.7,
        max_drawdown=0.2,
        max_leverage=1.0,
        risk_free_rate=0.02,
        confidence_level=0.95
    )

    feature_config = FeatureConfig(
        feature_list=[
            'ma_20', 'ma_50', 'ma_200',
            'rsi',
            'macd',
            'bollinger_bands'
        ],
        ma_periods=[20, 50, 200],
        rsi_period=14,
        macd_periods=(12, 26, 9),
        bb_period=20,
        bb_std=2.0,
        technical_indicators={},
        window_sizes={'volatility': 20, 'momentum': 14}
    )

    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,  # Using LightGBM for faster training
        optimize_hyperparameters=False,  # Set to False for testing
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
    )

    training_config = TrainingConfig(
        epochs=10,  # Reduced for testing
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=3,
        reduce_lr_patience=2
    )

    config = MLStrategyConfig(
        data=data_config,
        risk=risk_config,
        features=feature_config,
        model=model_config,
        training=training_config
    )

    # Initialize strategy
    strategy = MLStrategy(config)

    try:
        # Fetch and prepare data
        logging.info("Fetching training data...")
        train_data = strategy.data_fetcher.fetch_data(
            symbol=data_config.symbol,
            start_date=data_config.start_date,
            end_date=data_config.end_date
        )
        if train_data is None or train_data.empty:
            raise ValueError(f"Failed to fetch training data for {data_config.symbol} or received empty DataFrame")

        logging.info("Preparing features...")
        train_features = strategy.prepare_features(train_data)
        if train_features is None or train_features.empty:
            raise ValueError("Feature preparation failed or resulted in empty DataFrame")
            
        if 'target' not in train_features.columns:
            raise ValueError("Target column not found in prepared features")
            
        train_labels = train_features.pop('target')  # Separate target from features

        # Initialize and train model
        logging.info("Training model...")
        strategy._init_model()  # Initialize model first
        strategy.train(x_train=train_features, y_train=train_labels)

        # Run backtest
        logging.info("Running backtest...")
        backtest_data = strategy.data_fetcher.fetch_data(
            symbol=data_config.symbol,
            start_date=data_config.start_date,
            end_date=data_config.end_date
        )
        if backtest_data is None or backtest_data.empty:
            raise ValueError(f"Failed to fetch backtest data for {data_config.symbol} or received empty DataFrame")
            
        results = strategy.backtest(backtest_data)
        if results is None:
            raise ValueError("Backtest returned no results")

        # Print results
        logging.info("Backtest Results:")
        logging.info(f"Total Returns: {results.get('total_returns', 'N/A')}")
        logging.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 'N/A')}")
        logging.info(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")
        logging.info(f"Win Rate: {results.get('win_rate', 'N/A')}")
        logging.info(f"Total Trades: {results.get('total_trades', 'N/A')}")

    except Exception as e:
        logging.error(f"Error running strategy: {str(e)}")
        raise

if __name__ == "__main__":
    unittest.main(exit=False)
    main()
