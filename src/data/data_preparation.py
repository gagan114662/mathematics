"""Data loader and feature engineering module."""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PowerTransformer
from scipy.stats import yeojohnson
import logging
from .data_fetcher import DataFetcher
from src.config.ml_strategy_config import FeatureConfig

class DataPreparation:
    """Class for preparing stock data for ML model."""
    
    def __init__(self, config: FeatureConfig):
        """Initialize DataPreparation.
        
        Args:
            config: Feature configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data_fetcher = DataFetcher()

class DataLoader:
    """Handles data loading, preparation and feature engineering."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize data preparation.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.feature_scaler = RobustScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = DataFetcher()
        
    def fetch_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock data and prepare it for model training/prediction.
        
        Args:
            symbols: Stock symbol(s)
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with prepared stock data
        """
        try:
            # Fetch raw data
            raw_data = self.data_fetcher.fetch_stock_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache
            )
            
            # Handle MultiIndex columns if present
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Get the second level column names which contain the actual feature names
                raw_data.columns = raw_data.columns.get_level_values(-1).str.lower()
            else:
                # Convert column names to lowercase for case-insensitive comparison
                raw_data.columns = raw_data.columns.str.lower()
            
            # Prepare features
            features, _ = self.prepare_data(raw_data)
            
            return raw_data
            
        except Exception as e:
            self.logger.error(f"Error fetching and preparing stock data: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare data for model training or prediction.
        
        Args:
            df: Raw market data
            is_training: Whether this is for training (includes target creation)
            
        Returns:
            Tuple of features DataFrame and optional target Series
        """
        try:
            # Validate data
            self._validate_data(df)
            
            # Create features
            features = self._create_features(df)
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Scale features
            features = self._scale_features(features, is_training)
            
            # Create target if training
            target = None
            if is_training:
                target = self._create_target(df)
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data."""
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            # Get the second level column names which contain the actual feature names
            df.columns = df.columns.get_level_values(-1).str.lower()
        else:
            # Convert column names to lowercase for case-insensitive comparison
            df.columns = df.columns.str.lower()
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for sufficient data
        if len(df) < max(self.config.volatility_windows):
            raise ValueError(f"Insufficient data points. Need at least {max(self.config.volatility_windows)}")
            
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and other features."""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Volatility features
        for window in self.config.volatility_windows:
            features[f'volatility_{window}d'] = features['returns'].rolling(window).std()
            features[f'volatility_ratio_{window}d'] = (
                features[f'volatility_{window}d'] / 
                features[f'volatility_{window}d'].rolling(50).mean()
            )
        
        # Volume features
        for window in self.config.volume_windows:
            features[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            if window == 20:  # Use 20-day MA as reference
                features['volume_ratio'] = df['volume'] / features['volume_ma_20']
        
        # RSI
        features[f'rsi_{self.config.rsi_period}'] = self._calculate_rsi(df['close'], self.config.rsi_period)
        
        # MACD
        macd_features = self._calculate_macd(df['close'])
        features.update(macd_features)
        
        # Bollinger Bands
        bb_features = self._calculate_bollinger_bands(df['close'], self.config.bb_period)
        features.update(bb_features)
        
        # Momentum
        for window in self.config.momentum_windows:
            mom_features = self._calculate_momentum(df['close'], window)
            features.update(mom_features)
        
        # Additional indicators
        features['cmf'] = self._calculate_cmf(df)
        features['atr'] = self._calculate_atr(df)
        
        # Cyclical features
        features.update(self._create_cyclical_features(df))
        
        return features
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Calculate missing percentage
        missing_pct = df.isnull().mean()
        
        # Drop features with too many missing values
        columns_to_drop = missing_pct[missing_pct > self.config.missing_threshold].index
        if len(columns_to_drop) > 0:
            self.logger.warning(f"Dropping features with too many missing values: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Forward fill remaining missing values
        df = df.fillna(method='ffill')
        
        # Back fill any remaining NaNs at the start
        df = df.fillna(method='bfill')
        
        return df
        
    def _scale_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Scale features using robust scaling and power transformation."""
        if is_training:
            # Fit scalers
            scaled_data = self.feature_scaler.fit_transform(df)
        else:
            # Use pre-fit scalers
            scaled_data = self.feature_scaler.transform(df)
            
        scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        
        if self.config.use_yeo_johnson and is_training:
            # Apply Yeo-Johnson transform to highly skewed features
            for col in scaled_df.columns:
                if abs(scaled_df[col].skew()) > 1:
                    scaled_df[col] = self.power_transformer.fit_transform(
                        scaled_df[[col]]
                    )
                    
        return scaled_df
        
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable based on future returns."""
        # Calculate returns for different horizons
        horizons = [3, 5, 10, 20]
        returns = pd.DataFrame()
        
        for horizon in horizons:
            future_return = df['close'].shift(-horizon) / df['close'] - 1
            returns[f'return_{horizon}d'] = future_return
        
        # Calculate weighted average return
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_return = sum(returns[f'return_{h}d'] * w for h, w in zip(horizons, weights))
        
        # Dynamic threshold based on volatility
        vol = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        thresh_up = vol * 1.0
        
        # Create binary target
        target = (weighted_return > thresh_up).astype(int)
        
        return target
        
    # Technical indicator calculation methods
    def _calculate_rsi(self, data: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, data: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            'macd': macd,
            'macd_signal': signal,
            'macd_hist': macd - signal,
            'macd_cross': np.where(macd > signal, 1, -1)
        }
    
    def _calculate_bollinger_bands(self, data: pd.Series, window: int) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return {
            f'bb_middle_{window}': middle,
            f'bb_upper_{window}': upper,
            f'bb_lower_{window}': lower,
            f'bb_width_{window}': (upper - lower) / middle,
            f'bb_position_{window}': (data - lower) / (upper - lower)
        }
    
    def _calculate_momentum(self, data: pd.Series, window: int) -> Dict[str, pd.Series]:
        """Calculate price momentum indicators."""
        momentum = data.pct_change(window)
        return {
            f'mom_{window}d': momentum,
            f'mom_acc_{window}d': momentum.diff()
        }
    
    def _calculate_cmf(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfm * df['volume']
        return mfv.rolling(window).sum() / df['volume'].rolling(window).sum()
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create cyclical time-based features."""
        return {
            'sin_day': np.sin(2 * np.pi * df.index.dayofyear / 365.25),
            'cos_day': np.cos(2 * np.pi * df.index.dayofyear / 365.25),
            'sin_month': np.sin(2 * np.pi * df.index.month / 12),
            'cos_month': np.cos(2 * np.pi * df.index.month / 12),
            'sin_week': np.sin(2 * np.pi * df.index.dayofweek / 7),
            'cos_week': np.cos(2 * np.pi * df.index.dayofweek / 7)
        }
