"""Run ML trading strategy and calculate performance metrics."""
import logging
import os
from datetime import datetime, timedelta
from src.config.config import Config
from src.strategies.ml_strategy import MLStrategy
from src.data.data_fetcher import DataFetcher
from src.portfolio import Portfolio
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ML Strategy execution...")
    
    # Create configurations
    logger.info("Creating configurations...")
    config = Config()
    strategy = MLStrategy(config.model)
    logger.info("Strategy initialized")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Initialize dictionaries to store data
    stock_data = {}
    market_data = {}
    
    # Fetch stock data
    for symbol in config.stock_symbols:
        try:
            data = data_fetcher.fetch_stock_data(
                symbol,
                start_date=config.start_date,
                end_date=config.end_date
            )
            
            if not data:
                logger.error(f"No valid data fetched for {symbol}")
                continue
                
            if symbol in data and data[symbol] is not None and not data[symbol].empty:
                stock_data[symbol] = data[symbol]
                logger.info(f"Successfully processed data for {symbol}")
            else:
                logger.error(f"Invalid or empty data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    # Fetch market data
    for symbol in config.market_symbols:
        try:
            data = data_fetcher.fetch_stock_data(
                symbol,
                start_date=config.start_date,
                end_date=config.end_date
            )
            
            if not data:
                logger.error(f"No valid data fetched for {symbol}")
                continue
                
            if symbol in data and data[symbol] is not None and not data[symbol].empty:
                market_data[symbol] = data[symbol]
                logger.info(f"Successfully processed market data for {symbol}")
            else:
                logger.error(f"Invalid or empty data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    # Check if we have enough data to proceed
    if not stock_data:
        logger.error("No valid stock data fetched")
        return
        
    if not market_data:
        logger.error("No valid market data fetched")
        return
    
    # Train models
    try:
        strategy.train_models(stock_data, market_data)
        logger.info("Successfully trained models")
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        logger.error(traceback.format_exc())
        return
        
    # Run backtest
    portfolio = Portfolio(initial_capital=config.account_size)
    signals = {}
    
    # Generate trading signals
    for symbol, data in stock_data.items():
        features = strategy._calculate_technical_indicators(data)
        predictions = strategy.predict(features)
        signals[symbol] = predictions
    
    # Run backtest
    portfolio.run_backtest(signals, stock_data)
    metrics = portfolio.calculate_metrics()
    
    # Print performance metrics
    logger.info("\nPerformance Metrics:")
    logger.info("-" * 50)
    logger.info(f"CAGR: {metrics['cagr']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"Average Profit: {metrics['avg_profit']:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Final Capital: ${metrics['final_capital']:,.2f}")

if __name__ == "__main__":
    main()
