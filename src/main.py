"""Main module for running the trading strategy."""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.trading_config import TradingConfig
from src.config.ml_strategy_config import MLStrategyConfig
from src.data.data_fetcher import DataFetcher
from src.strategies.ml_strategy import MLStrategy

def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def fetch_data_parallel(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols in parallel.
    
    Args:
        symbols: List of symbols to fetch
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary mapping symbols to their data
    """
    data_fetcher = DataFetcher()
    data_dict: Dict[str, pd.DataFrame] = {}
    
    try:
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    data_fetcher.fetch_stock_data,
                    [symbol],  
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                ): symbol for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if not data.empty:
                        data_dict[symbol] = data
                except Exception as e:
                    logging.error(f"Error fetching {symbol}: {str(e)}")
                    
        return data_dict
        
    except Exception as e:
        logging.error(f"Error in parallel data fetching: {str(e)}")
        raise RuntimeError(f"Failed to fetch data in parallel: {str(e)}")

def create_models(data_dict: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> List[MLStrategy]:
    """
    Create and train models with the given data.
    
    Args:
        data_dict: Dictionary of symbol data
        start_date: Start date for training
        end_date: End date for training
        
    Returns:
        List of trained models
    """
    try:
        if not data_dict:
            raise ValueError("Empty data dictionary")
            
        # Create and train model with configuration
        config = MLStrategyConfig()  # Create default config
        strategy = MLStrategy(config)  # Pass config to MLStrategy
        
        # Prepare training data
        logging.info(f"Preparing training data for {len(data_dict)} symbols...")
        features, targets = strategy.prepare_training_data(
            pd.concat(data_dict.values()),
            list(data_dict.keys())
        )
        
        # Train the model
        if features.empty or targets.empty:
            raise ValueError("Failed to prepare training data")
            
        logging.info("Training model...")
        strategy.train_model(features, targets)
        
        logging.info("Model training completed successfully")
        return [strategy]
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise RuntimeError(f"Failed to create models: {str(e)}")

def main() -> None:
    """Main function to run the trading strategy."""
    try:
        setup_logging()
        logging.info("Starting trading strategy...")
        
        # Get configuration
        config = TradingConfig()
        
        # Get list of symbols and validate
        symbols = DataFetcher.get_symbols()[:config.max_symbols]
        if not symbols:
            raise ValueError("No symbols retrieved")
        logging.info(f"Using symbols: {symbols}")
            
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.lookback_days)
        logging.info(f"Training period: {start_date.date()} to {end_date.date()}")
        
        # Fetch data in parallel
        data_dict = fetch_data_parallel(symbols, start_date, end_date)
        if not data_dict:
            raise ValueError("No data retrieved")
        logging.info(f"Successfully fetched data for {len(data_dict)} symbols")
        
        # Create and train models
        models = create_models(data_dict, start_date, end_date)
        logging.info(f"Successfully created {len(models)} models")
        
        # Run backtests
        backtest_start = end_date - timedelta(days=config.backtest_days)
        logging.info(f"Running backtests from {backtest_start.date()} to {end_date.date()}")
        
        for i, model in enumerate(models, 1):
            try:
                # Run backtest for each symbol
                for symbol, data in data_dict.items():
                    results = model.backtest(data, [symbol])
                    
                    # Log detailed results
                    logging.info(f"\nBacktest results for Model {i}, Symbol {symbol}:")
                    logging.info("-" * 50)
                    for key, value in results.items():
                        if isinstance(value, float):
                            if key in ['total_return', 'max_drawdown', 'win_rate']:
                                logging.info(f"{key}: {value:.2%}")
                            else:
                                logging.info(f"{key}: {value:.2f}")
                        else:
                            logging.info(f"{key}: {value}")
                    logging.info("-" * 50)
                    
            except Exception as e:
                logging.error(f"Error in backtest for model {i}: {str(e)}")
        
        logging.info("Strategy execution completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
