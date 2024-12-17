"""Test script for ML trading strategy."""
import pandas as pd
import numpy as np
from datetime import datetime
from src.strategies.ml_strategy import MLStrategy
from src.config.ml_strategy_config import MLStrategyConfig, ModelType, DataConfig, RiskConfig, ModelConfig, FeatureConfig, ScalingMethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_strategy():
    """Test MLStrategy class."""
    try:
        logging.info("Initializing strategy...")
        
        # Initialize configuration
        model_config = ModelConfig(
            model_type=ModelType.LIGHTGBM,  # Using LightGBM for faster testing
            optimize_hyperparameters=True,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        )
        
        data_config = DataConfig(
            symbol='AAPL',  # Required
            start_date='2023-01-01',  # Required
            end_date='2024-01-01',  # Required
            sequence_length=20,  # Increased sequence length for better prediction
            n_features=35,
            validation_split=0.2,
            target_type='binary',
            target_threshold=0.0,
            prediction_horizon=5,  # Predict 5 days ahead
            scaling_method=ScalingMethod.STANDARD
        )
        
        risk_config = RiskConfig(
            initial_capital=100000.0,  # Required
            max_position_size=1000,
            stop_loss=0.02,
            take_profit=0.05,
            max_trades_per_day=5,
            max_portfolio_volatility=0.15,
            max_correlation=0.7,
            max_drawdown=0.2,
            max_leverage=1.0,
            risk_free_rate=0.02,
            confidence_level=0.95
        )

        feature_config = FeatureConfig(
            feature_list=['close', 'volume', 'rsi', 'macd', 'bollinger'],
            ma_periods=[5, 10, 20, 50],
            rsi_period=14,
            macd_periods=(12, 26, 9),
            bb_period=20,
            bb_std=2.0,
            technical_indicators={
                'momentum': True,
                'volatility': True,
                'trend': True
            },
            window_sizes={
                'volatility': 20,
                'momentum': 10,
                'trend': 50
            }
        )

        # Create MLStrategyConfig with risk_config
        config = MLStrategyConfig(
            data=data_config,
            risk=risk_config,  # Keep original RiskConfig for MLStrategyConfig
            features=feature_config,
            model=model_config
        )
        
        # Initialize strategy with config
        strategy = MLStrategy(config)

        try:
            # Prepare training data
            logging.info("Preparing training data...")
            train_data = strategy.data_fetcher.fetch_data(
                symbol=data_config.symbol,
                start_date=data_config.start_date,
                end_date=data_config.end_date
            )
            
            if train_data is None:
                raise ValueError("Failed to fetch training data")

            # Prepare features and fit scaler on training data
            train_features = strategy.prepare_features(train_data, fit_scaler=True)
            
            # Create labels with proper normalization
            train_labels = pd.Series(
                train_data['close'].pct_change().shift(-1),
                index=train_data.index
            )
            
            # Apply Winsorization to handle outliers
            lower_bound = train_labels.quantile(0.01)
            upper_bound = train_labels.quantile(0.99)
            train_labels = train_labels.clip(lower=lower_bound, upper=upper_bound)
            
            # Forward fill missing values and drop any remaining NaN values
            train_labels = train_labels.ffill()
            valid_idx = ~train_labels.isna()
            train_features = train_features.loc[valid_idx].to_frame() if isinstance(train_features.loc[valid_idx], pd.Series) else train_features.loc[valid_idx]
            train_labels = pd.Series(train_labels[valid_idx], name='target')

            # Train the model
            logging.info("Training model...")
            strategy.train(train_features, train_labels, x_val=None, y_val=None)

            # Define test symbols and dates
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Using fewer symbols for faster testing
            test_start = "2024-01-01"
            test_end = "2024-01-10"
            
            # Fetch test data for backtesting (using a different time period)
            logging.info("Fetching test data...")
            test_data = []
            for symbol in symbols:
                data = strategy.data_fetcher.fetch_data(symbol, test_start, test_end)
                if data is not None and isinstance(data, pd.DataFrame):
                    test_data.append(data)
            
            if not test_data:
                raise ValueError("No test data fetched")
                
            # Use first symbol's data for testing
            backtest_data = test_data[0]
            if not isinstance(backtest_data, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for backtest, got {type(backtest_data)}")
            
            # Prepare features for prediction (don't fit scaler on test data)
            test_features = strategy.prepare_features(backtest_data, fit_scaler=False)
            
            # Make predictions
            logging.info("Making predictions...")
            predictions = strategy.predict(test_features)
            
            # Run backtest
            logging.info("Running backtest...")
            results = strategy.backtest(backtest_data)
            
            # Get and print performance metrics
            metrics = strategy.get_performance_metrics()
            
            logger.info("\nStrategy Performance:")
            for metric, value in metrics.items():
                if 'ratio' in metric.lower():
                    logger.info(f"{metric}: {value:.2f}")
                else:
                    logger.info(f"{metric}: {value:.2%}")
        
        except Exception as e:
            logger.error(f"Error in testing: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_ml_strategy()
