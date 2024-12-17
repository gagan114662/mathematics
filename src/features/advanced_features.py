"""Advanced feature engineering module with market complexity handling."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator
import yfinance as yf
import logging

class AdvancedFeatureEngineering:
    """Advanced feature engineering with market complexity handling."""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50, 100]):
        """Initialize feature engineering with lookback periods."""
        self.lookback_periods = lookback_periods
        self.scaler = RobustScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.logger = logging.getLogger(__name__)
        
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        try:
            if df.empty:
                return df
                
            # Volatility regimes using multiple lookback periods
            for period in self.lookback_periods:
                bb = BollingerBands(df['close'], window=period)
                df[f'bb_width_{period}'] = bb.bollinger_wband()
                
                # Normalized ATR for volatility scaling
                atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
                df[f'norm_atr_{period}'] = atr.average_true_range() / df['close']
                
            # Trend strength and direction
            for period in self.lookback_periods:
                adx = ADXIndicator(df['high'], df['low'], df['close'], window=period)
                df[f'adx_{period}'] = adx.adx()
                df[f'trend_strength_{period}'] = df[f'adx_{period}'] * np.sign(df['close'].diff(period))
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in market regime features: {str(e)}")
            return df
            
    def add_adaptive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features with adaptive lookback periods."""
        try:
            if df.empty:
                return df
                
            # Adaptive lookback based on volatility
            volatility = df['close'].pct_change().rolling(window=20).std()
            adaptive_window = (20 * (1 + volatility)).astype(int)
            
            # Adaptive momentum features
            df['adaptive_momentum'] = df['close'].diff(periods=adaptive_window)
            df['adaptive_rsi'] = df.groupby(level=0).apply(
                lambda x: RSIIndicator(x['close'], window=adaptive_window).rsi()
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in adaptive features: {str(e)}")
            return df
            
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between indicators."""
        try:
            if df.empty:
                return df
                
            # Combine momentum and trend indicators
            for period in self.lookback_periods:
                rsi = RSIIndicator(df['close'], window=period).rsi()
                stoch = StochasticOscillator(
                    df['high'], df['low'], df['close'], window=period
                ).stoch()
                
                # Interaction terms
                df[f'rsi_stoch_interaction_{period}'] = rsi * stoch / 100
                df[f'rsi_vol_interaction_{period}'] = (
                    rsi * df[f'norm_atr_{period}']
                )
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in interaction features: {str(e)}")
            return df
            
    def add_market_sentiment(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market sentiment indicators."""
        try:
            if df.empty:
                return df
                
            # Get market index data (e.g., SPY for US stocks)
            market_data = yf.download('SPY', 
                                    start=df.index[0], 
                                    end=df.index[-1],
                                    progress=False)
            
            if market_data is None or market_data.empty:
                self.logger.warning("No market data available")
                return df
            
            # Calculate market correlation features
            returns = df['close'].pct_change()
            market_returns = market_data['Close'].pct_change()
            
            if returns is None or market_returns is None or returns.empty or market_returns.empty:
                self.logger.warning("No returns data available")
                return df
                
            # Rolling correlation with market
            for period in self.lookback_periods:
                correlation = (
                    returns.rolling(period)
                    .corr(market_returns)
                )
                if correlation is not None:
                    df[f'market_correlation_{period}'] = correlation.fillna(method='ffill')
                
            # Market relative strength
            for period in self.lookback_periods:
                stock_momentum = returns.rolling(period).sum()
                market_momentum = market_returns.rolling(period).sum()
                if stock_momentum is not None and market_momentum is not None:
                    df[f'relative_strength_{period}'] = stock_momentum - market_momentum
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in market sentiment features: {str(e)}")
            return df
            
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using advanced imputation."""
        try:
            if df.empty:
                return df
                
            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Impute missing values
            if len(numeric_cols) > 0:
                imputed_data = self.imputer.fit_transform(df[numeric_cols])
                if imputed_data is not None:
                    df[numeric_cols] = pd.DataFrame(
                        imputed_data,
                        columns=numeric_cols,
                        index=df.index
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in missing data handling: {str(e)}")
            return df
            
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using robust scaling."""
        try:
            if df.empty:
                return df
                
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaled_data = self.scaler.fit_transform(df[numeric_cols])
                if scaled_data is not None:
                    df[numeric_cols] = pd.DataFrame(
                        scaled_data,
                        columns=numeric_cols,
                        index=df.index
                    )
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature normalization: {str(e)}")
            return df
