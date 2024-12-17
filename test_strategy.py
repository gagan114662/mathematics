import logging
import sys
from datetime import datetime, timedelta
from src.strategies.ml_strategy import MLStrategy
from src.utils.performance_metrics import PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_strategy() -> bool:
    """Test the ML trading strategy with performance targets."""
    try:
        # Define performance targets
        TARGET_CAGR = 25.0  # Minimum CAGR (%)
        TARGET_SHARPE = 1.0  # Minimum Sharpe ratio
        TARGET_MAX_DD = -20.0  # Maximum drawdown (%)
        TARGET_AVG_PROFIT = 1.0  # Minimum average profit per trade (%)
        
        # Initialize strategy with test symbols
        symbols = [
            'AAPL',  # Technology
            'JPM',   # Financial
            'JNJ',   # Healthcare
            'XOM',   # Energy
            'PG',    # Consumer Staples
            'HD',    # Consumer Discretionary
            'NEE',   # Utilities
            'BRK-B'  # Conglomerate
        ]
        
        start_date = '2001-01-01'
        end_date = '2023-12-31'
        
        strategy = MLStrategy(symbols, start_date, end_date)
        
        # Fetch historical data
        logger.info("\nFetching historical data...")
        data_dict = {}
        for symbol in symbols:
            try:
                data = strategy.data_fetcher.fetch_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    data_dict[symbol] = data
                    logger.info(f"✓ {symbol}: {len(data)} days of data")
                else:
                    logger.warning(f"⚠ {symbol}: No data available")
            except Exception as e:
                logger.error(f"✗ {symbol}: Error fetching data - {str(e)}")
        
        if not data_dict:
            logger.error("No data available for any symbols. Exiting...")
            return False
        
        # Train the model
        logger.info("\nTraining model...")
        try:
            strategy.train_all(data_dict)
            logger.info("✓ Model training completed")
        except Exception as e:
            logger.error(f"✗ Model training failed: {str(e)}")
            return False
        
        # Run backtest
        logger.info("\nRunning backtest...")
        try:
            results = strategy.backtest(data_dict, start_date, end_date)
            logger.info(f"✓ Backtest completed - {len(results)} days processed")
        except Exception as e:
            logger.error(f"✗ Backtest failed: {str(e)}")
            return False
        
        # Calculate performance metrics
        returns = results['daily_returns'].dropna()  # Use actual daily returns instead of portfolio value changes
        metrics = PerformanceMetrics.calculate_metrics(returns)
        
        # Log detailed performance metrics
        logger.info("\nPerformance Results:")
        logger.info("--------------------")
        logger.info(f"CAGR: {metrics['cagr']:.2f}% (Target: >{TARGET_CAGR}%)")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (Target: >{TARGET_SHARPE})")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}% (Target: >{TARGET_MAX_DD}%)")
        logger.info(f"Average Profit: {metrics['avg_profit']:.2f}% (Target: >{TARGET_AVG_PROFIT}%)")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        
        # Check if performance targets are met
        success = (
            metrics['cagr'] >= TARGET_CAGR and
            metrics['sharpe_ratio'] >= TARGET_SHARPE and
            metrics['max_drawdown'] >= TARGET_MAX_DD and
            metrics['avg_profit'] >= TARGET_AVG_PROFIT
        )
        
        if not success:
            logger.error("\n❌ Strategy needs improvement to meet performance targets.")
        else:
            logger.info("\n✨ All performance targets achieved! Strategy ready for deployment.")
        
        return success
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_strategy()
    sys.exit(0 if success else 1)
