"""Feature engineering module for machine learning strategies."""
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field
import traceback

import numpy as np
import pandas as pd

@dataclass
class FeatureConfig:
    """Feature configuration class."""
    feature_list: List[str]
    ma_periods: List[int]
    rsi_period: int = 14
    macd_periods: Tuple[int, int, int] = (12, 26, 9)
    bb_period: int = 20
    bb_std: float = 2.0
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    window_sizes: Dict[str, int] = field(default_factory=dict)

class FeatureEngineer:
    """Feature engineering component."""
    def __init__(self, config: FeatureConfig):
        """Initialize feature engineering component.
        
        Args:
            config: Feature configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_features(self, data: pd.DataFrame, market_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Create features from input data."""
        try:
            self.logger.info("Creating features...")
            features = data.copy()
            
            # Log initial data info
            self.logger.info(f"Initial data shape: {features.shape}")
            self.logger.info(f"Initial columns: {features.columns.tolist()}")
            self.logger.info(f"Initial data types:\n{features.dtypes}")
            
            # Handle both MultiIndex and single-level columns
            if isinstance(features.columns, pd.MultiIndex):
                stock_symbol = features.columns.get_level_values(0)[0]
                self.logger.info(f"Processing features for stock: {stock_symbol}")
                close_prices = features[(stock_symbol, 'close')]
                volume = features[(stock_symbol, 'volume')] if ('volume' in features.columns.get_level_values(1)) else None
            else:
                stock_symbol = 'stock'  # Default name if no symbol in columns
                close_prices = features['close']
                volume = features['volume'] if 'volume' in features.columns else None
            
            # Handle NaN values in close prices and volume
            close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
            if volume is not None:
                volume = volume.fillna(method='ffill').fillna(method='bfill')
            
            # Create features dictionary to store all features
            feature_dict = {}
            
            # Calculate technical indicators
            if 'ma_5' in self.config.feature_list:
                feature_dict['ma_5'] = self._calculate_ma(close_prices, 5)
            if 'ma_10' in self.config.feature_list:
                feature_dict['ma_10'] = self._calculate_ma(close_prices, 10)
            if 'ma_20' in self.config.feature_list:
                feature_dict['ma_20'] = self._calculate_ma(close_prices, 20)
            if 'ma_50' in self.config.feature_list:
                feature_dict['ma_50'] = self._calculate_ma(close_prices, 50)
            
            if 'rsi' in self.config.feature_list:
                feature_dict['rsi'] = self._calculate_rsi(close_prices, self.config.rsi_period)
            
            if 'macd' in self.config.feature_list:
                macd_features = self._calculate_macd(close_prices, *self.config.macd_periods)
                for col in macd_features.columns:
                    feature_dict[f'macd_{col}'] = macd_features[col]
            
            if 'bollinger_bands' in self.config.feature_list:
                bb_features = self._calculate_bollinger_bands(close_prices, self.config.bb_period, self.config.bb_std)
                for col in bb_features.columns:
                    feature_dict[f'bb_{col}'] = bb_features[col]
            
            # Add returns and volatility features
            returns = close_prices.pct_change()
            feature_dict['returns'] = returns.fillna(0)
            feature_dict['volatility'] = returns.rolling(window=20, min_periods=1).std().fillna(method='bfill').fillna(0)
            
            # Add price momentum features
            feature_dict['momentum_1d'] = close_prices.pct_change(1).fillna(0)
            feature_dict['momentum_5d'] = close_prices.pct_change(5).fillna(0)
            feature_dict['momentum_10d'] = close_prices.pct_change(10).fillna(0)
            
            # Add volume-based features if volume data is available
            if volume is not None:
                feature_dict['volume_ma_5'] = self._calculate_ma(volume, 5)
                feature_dict['volume_ma_20'] = self._calculate_ma(volume, 20)
                feature_dict['volume_momentum'] = volume.pct_change().fillna(0)
            
            # Add market regime features if market data is provided
            if market_data is not None and isinstance(market_data, dict):
                try:
                    market_features = self._calculate_market_regime_features(market_data)
                    if not market_features.empty:
                        for col in market_features.columns:
                            feature_dict[f'market_{col}'] = market_features[col].fillna(0)
                except Exception as e:
                    self.logger.warning(f"Error calculating market regime features: {str(e)}")
                    self.logger.debug(traceback.format_exc())
            
            # Convert feature dictionary to DataFrame
            feature_df = pd.DataFrame(feature_dict, index=close_prices.index)
            
            # Add original close price and volume
            feature_df['close'] = close_prices
            if volume is not None:
                feature_df['volume'] = volume
            
            # Convert to MultiIndex columns if needed
            if isinstance(features.columns, pd.MultiIndex):
                feature_df.columns = pd.MultiIndex.from_product([[stock_symbol], feature_df.columns])
            
            # Final NaN cleanup
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Log feature information
            self.logger.info(f"Final features shape: {feature_df.shape}")
            self.logger.info(f"Final features: {feature_df.columns.tolist()}")
            self.logger.info(f"Final data types:\n{feature_df.dtypes}")
            self.logger.info(f"Any NaN values remaining: {feature_df.isna().any().any()}")
            
            # Return None if no valid data
            if feature_df.empty:
                self.logger.warning(f"No valid features created for {stock_symbol}")
                return None
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}\n{traceback.format_exc()}")
            raise
            
    def _calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate moving average."""
        ma = series.rolling(window=period, min_periods=1).mean()
        return ma.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
    def _calculate_slope(self, series: pd.Series) -> pd.Series:
        """Calculate slope of series."""
        return (series - series.shift(1)) / series.shift(1)
        
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        # Handle NaN values in input
        series = series.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate price changes
        delta = series.diff()
        
        # Create gain and loss series
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral RSI value
    
    def _calculate_macd(self, series: pd.Series, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Calculate MACD."""
        # Handle NaN values in input
        series = series.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate EMAs
        exp1 = series.ewm(span=fast_period, adjust=False, min_periods=1).mean()
        exp2 = series.ewm(span=slow_period, adjust=False, min_periods=1).mean()
        
        # Calculate MACD and signal line
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False, min_periods=1).mean()
        
        # Create DataFrame with results
        macd_df = pd.DataFrame({
            'macd': macd.fillna(0),
            'macd_signal': signal.fillna(0),
            'macd_hist': (macd - signal).fillna(0)
        })
        
        return macd_df
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        # Handle NaN values in input
        series = series.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate middle band (SMA)
        middle_band = series.rolling(window=period, min_periods=1).mean()
        
        # Calculate standard deviation
        rolling_std = series.rolling(window=period, min_periods=1).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        # Create DataFrame with results
        bb_df = pd.DataFrame({
            'bb_middle': middle_band.fillna(method='ffill').fillna(method='bfill'),
            'bb_upper': upper_band.fillna(method='ffill').fillna(method='bfill'),
            'bb_lower': lower_band.fillna(method='ffill').fillna(method='bfill')
        })
        
        return bb_df
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=14).mean()
        
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            df = data.copy()
            
            # Moving averages
            for period in self.config.ma_periods:
                df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
                
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=self.config.macd_periods[0], adjust=False).mean()
            exp2 = df['close'].ewm(span=self.config.macd_periods[1], adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.config.macd_periods[2], adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            middle_band = df['close'].rolling(window=self.config.bb_period).mean()
            std_dev = df['close'].rolling(window=self.config.bb_period).std()
            df['bb_upper'] = middle_band + (std_dev * self.config.bb_std)
            df['bb_lower'] = middle_band - (std_dev * self.config.bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / middle_band
            
            # Stochastic Oscillator
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stochastic_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df['stochastic'] = df['stochastic_k'].rolling(window=3).mean()  # %D
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # On Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price momentum
            df['momentum'] = df['close'].pct_change(periods=10)
            df['rate_of_change'] = df['close'].pct_change(periods=20)
            
            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            return df.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            raise

    def _calculate_market_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate market regime features."""
        try:
            if not market_data or 'SPY' not in market_data:
                self.logger.warning("No market data (SPY) available for regime features")
                return pd.DataFrame()
            
            spy_data = market_data['SPY']
            if spy_data is None or spy_data.empty:
                self.logger.warning("SPY data is empty")
                return pd.DataFrame()
            
            # Get SPY close prices
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_close = spy_data[('SPY', 'close')]
            else:
                spy_close = spy_data['close']
            
            # Calculate market features
            market_features = {}
            
            # Market returns
            spy_returns = spy_close.pct_change().fillna(0)
            market_features['spy_returns'] = spy_returns
            
            # Market volatility (20-day rolling)
            market_features['spy_volatility'] = spy_returns.rolling(window=20, min_periods=1).std().fillna(0)
            
            # Market momentum
            market_features['spy_momentum_1d'] = spy_close.pct_change(1).fillna(0)
            market_features['spy_momentum_5d'] = spy_close.pct_change(5).fillna(0)
            market_features['spy_momentum_10d'] = spy_close.pct_change(10).fillna(0)
            
            # Market trend (SMA ratios)
            sma_20 = spy_close.rolling(window=20, min_periods=1).mean()
            sma_50 = spy_close.rolling(window=50, min_periods=1).mean()
            market_features['spy_trend'] = (sma_20 / sma_50 - 1).fillna(0)
            
            # Market RSI
            market_features['spy_rsi'] = self._calculate_rsi(spy_close)
            
            # Create DataFrame
            market_df = pd.DataFrame(market_features, index=spy_close.index)
            
            # Fill any remaining NaN values
            market_df = market_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info("Successfully calculated market regime features")
            self.logger.info(f"Market features shape: {market_df.shape}")
            
            return market_df
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime features: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()
