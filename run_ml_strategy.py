"""Run ML trading strategy."""
import logging
import logging.handlers
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Import from local modules
from src.strategies.ml_strategy import MLStrategy
from src.config.ml_strategy_config import (
    MLStrategyConfig, ModelConfig, DataConfig, 
    RiskConfig, FeatureConfig, ModelType, ScalingMethod
)
from src.data.data_fetcher import DataFetcher

# Configure logging with both file and console handlers
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.handlers.RotatingFileHandler(
    'ml_strategy.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Setup logger
logger = logging.getLogger('MLStrategy')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def evaluate_predictions(strategy: MLStrategy, test_data: pd.DataFrame) -> dict:
    """
    Evaluate model predictions against actual results.
    
    Args:
        strategy: Trained ML strategy instance
        test_data: Test dataset with actual values
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    try:
        # Prepare features and make predictions
        features = strategy.prepare_features(test_data, fit_scaler=False)
        predictions = strategy.predict(features)
        
        # Get actual values (next day returns) and align with predictions
        actual_returns = test_data['close'].pct_change().shift(-1).iloc[:-1]
        predicted_returns = pd.Series(predictions, index=features.index)[:-1]  # Remove last prediction as we don't have actual value for it
        
        # Ensure both arrays are aligned and have same length
        common_index = actual_returns.index.intersection(predicted_returns.index)
        actual_returns = actual_returns[common_index]
        predicted_returns = predicted_returns[common_index]
        
        # Convert to numpy arrays for metric calculations
        actual_returns = actual_returns.values
        predicted_returns = predicted_returns.values
        
        # Calculate regression metrics
        mse = mean_squared_error(actual_returns, predicted_returns)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_returns, predicted_returns)
        r2 = r2_score(actual_returns, predicted_returns)
        
        # Calculate directional accuracy
        actual_direction = np.sign(actual_returns)
        predicted_direction = np.sign(predicted_returns)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Calculate classification metrics for direction prediction
        precision = precision_score(actual_direction > 0, predicted_direction > 0, zero_division=0)
        recall = recall_score(actual_direction > 0, predicted_direction > 0, zero_division=0)
        f1 = f1_score(actual_direction > 0, predicted_direction > 0, zero_division=0)
        
        # Calculate portfolio metrics
        returns_series = pd.Series(actual_returns)
        predicted_returns_series = pd.Series(predicted_returns)
        
        # Sharpe Ratio (assuming daily returns)
        risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
        excess_returns = returns_series - risk_free_rate
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Information Ratio
        tracking_error = (returns_series - predicted_returns_series).std()
        information_ratio = np.sqrt(252) * ((returns_series - predicted_returns_series).mean() / tracking_error)
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Hit Ratio (% of times prediction direction matches actual direction)
        hit_ratio = np.mean((returns_series > 0) == (predicted_returns_series > 0))
        
        # Profit Factor
        winning_trades = returns_series[returns_series > 0].sum()
        losing_trades = abs(returns_series[returns_series < 0].sum())
        profit_factor = winning_trades / losing_trades if losing_trades != 0 else float('inf')
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_predictions: {str(e)}")
        raise

def main():
    """Run ML trading strategy."""
    try:
        logger.info("Starting ML trading strategy execution...")
        
        # Configure feature engineering
        logger.info("Configuring feature engineering...")
        feature_config = FeatureConfig(
            feature_list=[
                'returns', 'log_returns', 'volume_change',  # Basic features
                'sma_5', 'sma_10', 'sma_20', 'sma_50',     # SMAs
                'ema_5', 'ema_10', 'ema_20', 'ema_50',     # EMAs
                'macd', 'macd_signal', 'macd_hist',        # MACD
                'bb_upper', 'bb_lower', 'bb_width',        # Bollinger Bands
                'rsi',                                     # RSI
                'volatility', 'volatility_ratio',          # Volatility
                'volume_ma', 'volume_ratio',               # Volume
                'high_low_ratio', 'close_position'         # Price patterns
            ],
            ma_periods=[5, 10, 20, 50],
            rsi_period=14,
            macd_periods=(12, 26, 9),
            bb_period=20,
            bb_std=2.0
        )
        logger.info("Feature configuration completed")

        # Configure model
        logger.info("Configuring model...")
        model_config = {
            'lstm_units': [64, 32],  # Increased units for better learning
            'dropout_rate': 0.3,     # Increased dropout to prevent overfitting
            'learning_rate': 0.001,
            'batch_size': 32,        # Reduced batch size for better memory management
            'epochs': 50,            # Reduced epochs for faster training
            'l2_reg': 0.01,         # Increased regularization
            'sequence_length': 20    # Reduced sequence length
        }
        logger.info("Model configuration completed")

        # Configure data
        logger.info("Configuring data parameters...")
        data_config = DataConfig(
            symbols=[
                # Market Index
                'SPY',   # S&P 500 ETF
                
                # Mega Cap Tech
                'AAPL',  # Apple
                'MSFT',  # Microsoft
                'GOOGL', # Alphabet
                'GOOG',  # Alphabet Class C
                'AMZN',  # Amazon
                'META',  # Meta
                'NVDA',  # NVIDIA
                'AVGO',  # Broadcom
                'TSLA',  # Tesla
                'TSM',   # Taiwan Semiconductor
                
                # Semiconductor
                'AMD',   # AMD
                'INTC',  # Intel
                'QCOM',  # Qualcomm
                'MU',    # Micron
                'ARM',   # ARM Holdings
                
                # Software/Cloud
                'CRM',   # Salesforce
                'ORCL',  # Oracle
                'ADBE',  # Adobe
                'INTU',  # Intuit
                'NOW',   # ServiceNow
                
                # E-commerce/Internet
                'BABA',  # Alibaba
                'PDD',   # PDD Holdings
                'SHOP',  # Shopify
                'MELI',  # MercadoLibre
                'SE',    # Sea Limited
                
                # Fintech/Payments
                'V',     # Visa
                'MA',    # Mastercard
                'PYPL',  # PayPal
                'SQ',    # Block (Square)
                'COIN',  # Coinbase
                
                # AI/Data/Cloud
                'PLTR',  # Palantir
                'SNOW',  # Snowflake
                'MDB',   # MongoDB
                'DDOG',  # Datadog
                'NET',   # Cloudflare
                
                # Entertainment/Gaming
                'NFLX',  # Netflix
                'DIS',   # Disney
                'SONY',  # Sony
                'EA',    # Electronic Arts
                
                # EV/Future Transport
                'LI',    # Li Auto
                'RIVN',  # Rivian
                'LCID',  # Lucid
                'NIO',   # NIO
                'XPEV'   # XPeng
            ],
            market_symbols=['SPY'],  # Just SPY for market reference
            sector_etfs=[],  # No sector ETFs
            start_date='2015-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d'),
            sequence_length=60,
            n_features=None,  # Will be set dynamically
            validation_split=0.2,
            test_split=0.1,
            cross_validation_folds=5,
            scaling_method=ScalingMethod.STANDARD,
            cache_dir='data/cache',
            min_samples=1000,
            max_samples=10000
        )
        logger.info("Data configuration completed")

        # Initialize strategy
        logger.info("Initializing strategy...")
        strategy = MLStrategy(model_config=model_config, logger=logger)
        logger.info("Strategy initialization completed")

        # Fetch data
        logger.info("Fetching data...")
        data_fetcher = DataFetcher()
        stock_data = data_fetcher.fetch_multiple_stocks(
            symbols=data_config.symbols,
            start_date=data_config.start_date,
            end_date=data_config.end_date
        )
        
        if not stock_data:
            raise ValueError("No data available for any symbols")
        
        logger.info(f"Successfully fetched data for {len(stock_data)} symbols")
        
        # Log data statistics
        for symbol, data in stock_data.items():
            if data is not None and not data.empty:
                logger.info(f"{symbol}: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
            else:
                logger.warning(f"No data available for {symbol}")
        
        # Verify market data
        if 'SPY' not in stock_data or stock_data['SPY'] is None or stock_data['SPY'].empty:
            raise ValueError("Market data (SPY) is not available")
        
        logger.info("Starting model training...")
        training_duration = timedelta(hours=3)
        end_time = datetime.now() + training_duration
        logger.info(f"Starting {training_duration.total_seconds()/3600}-hour training session...")
        strategy.train_model(stock_data=stock_data, market_data={'SPY': stock_data.get('SPY')})

        logger.info("Strategy execution completed successfully")

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Error running strategy: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
