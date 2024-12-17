"""Feature engineering utilities for the trading system."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from typing import Tuple, List, Dict

from src.utils.indicators import TechnicalIndicators

class FeatureEngineer:
    """Feature engineering for market data."""
    
    def __init__(self):
        """Initialize feature engineering components."""
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.indicators = TechnicalIndicators()
        
    def create_features(self, df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features from raw market data.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dictionary with feature parameters
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = df.copy()
        
        # Basic price and volume features
        df = self._create_price_features(df, config['volatility_windows'])
        df = self._create_volume_features(df, config['volume_windows'])
        
        # Technical indicators
        df = self._add_technical_indicators(df, config)
        
        # Cyclical features
        df = self.indicators.add_cyclical_features(df, date_column=None)
        
        # Interaction features
        feature_pairs = [
            ('volatility_20d', 'volume_ratio'),
            ('rsi_14', 'macd'),
            ('bb_width_20', 'volume_trend')
        ]
        df = self.indicators.add_interaction_features(df, feature_pairs)
        
        # Lag features for key indicators
        lag_columns = ['close', 'volume', 'rsi_14', 'macd']
        lag_periods = [1, 2, 3, 5]
        df = self.indicators.add_lag_features(df, lag_columns, lag_periods)
        
        # Create target
        target = (df['close'].shift(-1) / df['close'] - 1 > config['min_profit_target']).astype(int)
        
        # Select and scale features
        features = self._select_and_scale_features(df)
        
        return features, target
        
    def _create_price_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create price-based features."""
        # Returns and log returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Volatility features
        for window in windows:
            df[f'volatility_{window}d'] = df['returns'].rolling(window).std()
            df[f'volatility_ratio_{window}d'] = (
                df[f'volatility_{window}d'] / 
                df[f'volatility_{window}d'].rolling(50).mean()
            )
        
        return df
        
    def _create_volume_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create volume-based features."""
        # Scale volume
        df['volume_scaled'] = self.volume_scaler.fit_transform(
            df[['volume']].replace(0, np.nan).fillna(df['volume'].mean())
        )
        
        # Volume moving averages
        for window in windows:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            
        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['volume_trend'] = df['volume_ma_5'] / df['volume_ma_20']
        
        return df
        
    def _add_technical_indicators(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add technical indicators."""
        # RSI for multiple periods
        for period in config['rsi_periods']:
            df[f'rsi_{period}'] = self.indicators.calculate_rsi(df['close'], period)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Bollinger Bands
        for window in config['bb_windows']:
            df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
            bb_std = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + (bb_std * 2)
            df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - (bb_std * 2)
            df[f'bb_width_{window}'] = (
                (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / 
                df[f'bb_middle_{window}']
            )
        
        # ATR
        df['atr'] = self.indicators.calculate_atr(df['high'], df['low'], df['close'])
        
        # CMF
        df['cmf'] = self.indicators.calculate_cmf(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # VWAP
        df['vwap'] = self.indicators.calculate_vwap(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Ichimoku Cloud
        ichimoku = self.indicators.calculate_ichimoku(
            df['high'], df['low'], df['close']
        )
        df = pd.concat([df, ichimoku], axis=1)
        
        return df
        
    def _select_and_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and scale final feature set."""
        # Drop any rows with NaN values
        df = df.dropna()
        
        # Select features (excluding target and date-related columns)
        feature_columns = [col for col in df.columns if not col.startswith(('date', 'target'))]
        features = df[feature_columns]
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.feature_scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Apply power transform to handle skewness
        for col in scaled_features.columns:
            if scaled_features[col].skew() > 1 or scaled_features[col].skew() < -1:
                scaled_features[col] = self.power_transformer.fit_transform(
                    scaled_features[[col]]
                )
        
        return scaled_features
