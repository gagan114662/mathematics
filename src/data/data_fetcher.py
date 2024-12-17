"""Data fetching module."""
import logging
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
import yfinance as yf
import traceback
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataFetcher:
    """Data fetching component."""
    
    def __init__(self, config_str: str = 'default'):
        """
        Initialize DataFetcher.
        
        Args:
            config_str: Configuration string or identifier for caching
        """
        # Create a hash of the config string to use as a subdirectory
        config_hash = str(abs(hash(config_str)))[:8]  # Use first 8 chars of hash
        # Use absolute path for cache directory
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'cache', config_hash)
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """Get cache file path for a symbol."""
        return os.path.join(
            self.cache_dir,
            f"{symbol}_{start_date}_{end_date}.csv"
        )
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate fetched data.
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol
            
        Returns:
            bool: True if data is valid
        """
        try:
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns for {symbol}")
                return False
            
            # Check for empty DataFrame
            if df.empty:
                self.logger.error(f"Empty data for {symbol}")
                return False
            
            # Check for missing values
            missing = df[required_cols].isnull().sum()
            if missing.any():
                self.logger.warning(f"Missing values in {symbol}:\n{missing[missing > 0]}")
                return False
            
            # Check for invalid values
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                self.logger.error(f"Invalid price values for {symbol}")
                return False
            
            if (df['volume'] < 0).any():
                self.logger.error(f"Invalid volume values for {symbol}")
                return False
            
            # Check for data continuity
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            date_diff = df.index.to_series().diff().dt.days
            if date_diff.max() > 5:  # Allow for weekends and holidays
                self.logger.warning(f"Large gaps in data for {symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data for {symbol}: {str(e)}")
            return False
    
    def fetch_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock data with caching and validation.
        
        Args:
            symbols: Stock symbol(s)
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with stock data
        """
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            all_data = []
            for symbol in symbols:
                # Try to load from cache first
                if use_cache:
                    cache_path = self._get_cache_path(symbol, start_date, end_date)
                    if os.path.exists(cache_path):
                        self.logger.info(f"Loading cached data for {symbol}")
                        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        if self._validate_data(df, symbol):
                            all_data.append(df)
                            continue
                
                # Fetch from yfinance if not in cache or invalid
                self.logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                # Standardize column names to lowercase
                df.columns = df.columns.str.lower()
                
                if self._validate_data(df, symbol):
                    # Save to cache if valid
                    if use_cache:
                        cache_path = self._get_cache_path(symbol, start_date, end_date)
                        df.to_csv(cache_path)
                    all_data.append(df)
            
            if not all_data:
                raise ValueError("No valid data fetched for any symbol")
            
            # Combine all data
            combined_data = pd.concat(all_data, axis=1, keys=symbols)
            combined_data.index.name = 'date'
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks efficiently.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to their data
        """
        try:
            self.logger.info(f"Fetching data for {len(symbols)} symbols...")
            
            # Create ThreadPoolExecutor for parallel fetching
            results = {}
            failed_symbols = []
            
            with ThreadPoolExecutor(max_workers=5) as executor:  
                future_to_symbol = {
                    executor.submit(
                        self.fetch_stock_data,
                        symbol,
                        start_date,
                        end_date,
                        use_cache
                    ): symbol for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                            # Convert to single-level columns if MultiIndex
                            if isinstance(data.columns, pd.MultiIndex):
                                data.columns = data.columns.get_level_values(1)
                            results[symbol] = data
                        else:
                            self.logger.warning(f"No data available for {symbol}")
                            failed_symbols.append(symbol)
                    except Exception as e:
                        self.logger.error(f"Error fetching {symbol}: {str(e)}")
                        failed_symbols.append(symbol)
            
            if failed_symbols:
                self.logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
            
            if not results:
                raise ValueError("Failed to fetch data for any symbols")
            
            self.logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols)} symbols")
            
            # Validate and align data
            self._validate_data_alignment(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fetch_multiple_stocks: {str(e)}\n{traceback.format_exc()}")
            raise

    def _validate_data_alignment(self, stock_data: Dict[str, pd.DataFrame]) -> None:
        """
        Ensure all stock data is properly aligned by date.
        
        Args:
            stock_data: Dictionary of stock DataFrames
        """
        try:
            if not stock_data:
                return
            
            # Convert all indices to datetime if needed
            for symbol, df in stock_data.items():
                if not isinstance(df.index, pd.DatetimeIndex):
                    stock_data[symbol].index = pd.to_datetime(df.index)
            
            # Get all unique dates
            all_dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in stock_data.values()])))
            
            # Reindex all DataFrames to common dates and forward/backward fill
            for symbol, df in stock_data.items():
                if not df.empty:
                    # Forward fill first, then backward fill any remaining NaNs
                    stock_data[symbol] = df.reindex(all_dates).fillna(method='ffill').fillna(method='bfill')
                    
                    # Log warning if too many NaN values were filled
                    nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
                    if nan_ratio > 0.1:  # More than 10% NaN values
                        self.logger.warning(f"High ratio of NaN values ({nan_ratio:.2%}) for {symbol}")
            
            self.logger.info(f"Aligned data for {len(stock_data)} stocks")
            
        except Exception as e:
            self.logger.error(f"Error validating data alignment: {str(e)}\n{traceback.format_exc()}")
            raise

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a given symbol.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Optional[DataFrame] containing historical data or None if fetch fails
        """
        try:
            # Download data from Yahoo Finance
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # Check if we got any data
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Log data info and NaN check
            self.logger.info(f"Retrieved data shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            self.logger.info(f"Data types:\n{df.dtypes}")
            
            # Check for NaN values in each column
            nan_counts = df.isna().sum()
            if nan_counts.any():
                self.logger.warning("NaN values found in columns:")
                for col in nan_counts[nan_counts > 0].index:
                    self.logger.warning(f"{col}: {nan_counts[col]} NaN values")
            
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Ensure index is datetime with UTC timezone
            try:
                df.index = pd.to_datetime(df.index)
                dt_index = pd.DatetimeIndex(df.index)
                
                # Check if timezone is set
                if dt_index.tz is None:
                    df.index = pd.DatetimeIndex(dt_index).tz_localize('UTC')
                else:
                    df.index = pd.DatetimeIndex(dt_index).tz_convert('UTC')
                    
            except Exception as e:
                self.logger.error(f"Error converting index to DatetimeIndex for {symbol}: {str(e)}")
                return None
            
            # Handle NaN values
            # First, forward fill
            df = df.ffill()
            # Then backward fill any remaining NaNs
            df = df.bfill()
            # Finally, drop any rows that still have NaNs
            if df.isna().any().any():
                self.logger.warning("Dropping rows with NaN values")
                df = df.dropna()
                if df.empty:
                    self.logger.error("All data was NaN")
                    return None
            
            # Log final data info
            self.logger.info(f"Final data shape: {df.shape}")
            self.logger.info(f"Final columns: {df.columns.tolist()}")
            self.logger.info(f"Final data types:\n{df.dtypes}")
            self.logger.info(f"Any NaN values remaining: {df.isna().any().any()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            close_series = pd.Series(df['close'].values, index=df.index, dtype=np.float64)
            df['sma_20'] = close_series.rolling(window=20).mean()
            df['sma_50'] = close_series.rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(close_series)
            df['macd'], df['signal'] = self._calculate_macd(close_series)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(close_series)
            return df
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            close_series = pd.Series(df['close'].values, index=df.index, dtype=np.float64)
            df['sma_20'] = close_series.rolling(window=20).mean()
            df['sma_50'] = close_series.rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(close_series)
            df['macd'], df['signal'] = self._calculate_macd(close_series)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(close_series)
            return df
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    @staticmethod
    def get_symbols() -> List[str]:
        """
        Get list of S&P 500 symbols.
        
        Returns:
            List of stock symbols
        """
        try:
            # This is a placeholder - in production you would fetch from a proper source
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
            return symbols
        except Exception as e:
            logging.error(f"Error fetching symbols: {str(e)}")
            return []

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI technical indicator
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series containing RSI values
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return pd.Series(rsi, index=prices.index, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MACD technical indicator
        
        Args:
            prices: Series of price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line)
        """
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands technical indicator
        
        Args:
            prices: Series of price data
            window: Moving average window
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band) as pandas Series
        """
        try:
            # Calculate middle band (simple moving average)
            middle = pd.Series(
                prices.rolling(window=window).mean(),
                index=prices.index,
                dtype=np.float64
            )
            
            # Calculate standard deviation
            std = pd.Series(
                prices.rolling(window=window).std(),
                index=prices.index,
                dtype=np.float64
            )
            
            # Calculate upper and lower bands
            upper = pd.Series(
                middle + (std * num_std),
                index=prices.index,
                dtype=np.float64
            )
            lower = pd.Series(
                middle - (std * num_std),
                index=prices.index,
                dtype=np.float64
            )
            
            return upper, middle, lower
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def _calculate_market_cap_weights(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate weights based on market cap for each stock.
        
        Args:
            stock_data: Dictionary of stock DataFrames
            
        Returns:
            Dictionary of stock weights
        """
        try:
            weights = {}
            total_market_cap = 0
            
            for symbol, df in stock_data.items():
                if 'close' in df.columns and 'volume' in df.columns:
                    # Use last day's close * volume as a proxy for market cap
                    market_cap = df['close'].iloc[-1] * df['volume'].iloc[-1]
                    weights[symbol] = market_cap
                    total_market_cap += market_cap
            
            # Normalize weights
            if total_market_cap > 0:
                weights = {
                    symbol: cap / total_market_cap 
                    for symbol, cap in weights.items()
                }
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating market cap weights: {str(e)}")
            raise
