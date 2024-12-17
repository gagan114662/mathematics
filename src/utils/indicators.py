"""Technical indicators for market analysis."""
import numpy as np
import pandas as pd
from typing import Optional

class TechnicalIndicators:
    """Collection of technical indicators for market analysis."""
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values as Series
        """
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series, 
                     volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: CMF period
            
        Returns:
            CMF values as Series
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * volume
        return mfv.rolling(period).sum() / volume.rolling(period).sum()

    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: Optional rolling period
            
        Returns:
            VWAP values as Series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        if period is not None:
            vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
            
        return vwap

    @staticmethod
    def calculate_fibonacci_levels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement levels.
        
        Args:
            high: High prices
            low: Low prices
            period: Period for finding swing high/low
            
        Returns:
            DataFrame with Fibonacci levels
        """
        # Find swing high and low
        rolling_high = high.rolling(period, center=True).max()
        rolling_low = low.rolling(period, center=True).min()
        
        # Calculate Fibonacci levels
        diff = rolling_high - rolling_low
        levels = pd.DataFrame(index=high.index)
        
        # Standard Fibonacci ratios
        ratios = {
            'level_0': 0.0,
            'level_236': 0.236,
            'level_382': 0.382,
            'level_500': 0.500,
            'level_618': 0.618,
            'level_786': 0.786,
            'level_1000': 1.000
        }
        
        for name, ratio in ratios.items():
            levels[name] = rolling_low + (diff * ratio)
            
        return levels

    @staticmethod
    def calculate_ichimoku(high: pd.Series, low: pd.Series, 
                          close: pd.Series) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            DataFrame with Ichimoku components
        """
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = close.shift(-26)

        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })

    @staticmethod
    def add_cyclical_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Add cyclical time-based features using sin and cos transformations.
        
        Args:
            df: DataFrame with datetime index
            date_column: Name of date column if not index
            
        Returns:
            DataFrame with added cyclical features
        """
        df = df.copy()
        
        if date_column:
            date_series = pd.to_datetime(df[date_column])
        else:
            date_series = pd.to_datetime(df.index)
            
        # Day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * date_series.dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * date_series.dt.dayofweek / 7)
        
        # Month
        df['month_sin'] = np.sin(2 * np.pi * date_series.dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * date_series.dt.month / 12)
        
        # Hour (if intraday data)
        if date_series.dt.hour.nunique() > 1:
            df['hour_sin'] = np.sin(2 * np.pi * date_series.dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * date_series.dt.hour / 24)
            
        return df

    @staticmethod
    def add_interaction_features(df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
        """
        Create interaction features between specified pairs of features.
        
        Args:
            df: DataFrame with features
            feature_pairs: List of tuples containing feature pairs to interact
            
        Returns:
            DataFrame with added interaction features
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_{feat2}_interact"
                df[interaction_name] = df[feat1] * df[feat2]
                
        return df

    @staticmethod
    def add_lag_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
        """
        Add lagged versions of specified features.
        
        Args:
            df: DataFrame with features
            columns: List of columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                    
        return df
