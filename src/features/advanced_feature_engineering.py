"""Advanced feature engineering for ML-based trading strategies."""
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from scipy import stats
import logging

class FeatureEngineer:
    """Advanced feature engineering for financial time series data."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer.
        
        Args:
            config: Dictionary containing feature engineering parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def engineer_features(self, data: DataFrame) -> DataFrame:
        """
        Engineer features from raw OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()
            
            if self.config["use_ta"]:
                df = self._add_technical_indicators(df)
                
            if self.config["use_sentiment"]:
                df = self._add_sentiment_features(df)
                
            if self.config["use_fundamentals"]:
                df = self._add_fundamental_features(df)
                
            if self.config["custom_features"]:
                df = self._add_custom_features(df)
                
            # Drop NaN values created by indicators
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            return data
            
    def _add_technical_indicators(self, data: DataFrame) -> DataFrame:
        """Add technical indicators to the dataset."""
        try:
            df = data.copy()
            params = self.config["ta_params"]
            
            if params["volume"]:
                # Volume indicators
                df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
                df["vwap"] = VolumeWeightedAveragePrice(
                    high=df["high"], 
                    low=df["low"], 
                    close=df["close"], 
                    volume=df["volume"]
                ).volume_weighted_average_price()
                
            if params["volatility"]:
                # Volatility indicators
                bb = BollingerBands(close=df["close"])
                df["bb_high"] = bb.bollinger_hband()
                df["bb_low"] = bb.bollinger_lband()
                df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]
                df["atr"] = AverageTrueRange(
                    high=df["high"], 
                    low=df["low"], 
                    close=df["close"]
                ).average_true_range()
                
            if params["trend"]:
                # Trend indicators
                df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
                df["sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
                df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
                
                macd = MACD(close=df["close"])
                df["macd"] = macd.macd()
                df["macd_signal"] = macd.macd_signal()
                df["macd_diff"] = macd.macd_diff()
                
            if params["momentum"]:
                # Momentum indicators
                df["rsi"] = RSIIndicator(close=df["close"]).rsi()
                
                stoch = StochasticOscillator(
                    high=df["high"],
                    low=df["low"],
                    close=df["close"]
                )
                df["stoch_k"] = stoch.stoch()
                df["stoch_d"] = stoch.stoch_signal()
                
            if params["others"]:
                # Price based features
                df["returns"] = df["close"].pct_change()
                df["log_returns"] = np.log1p(df["returns"])
                df["volatility"] = df["returns"].rolling(window=20).std()
                
                # Volume based features
                df["volume_ma"] = df["volume"].rolling(window=20).mean()
                df["volume_std"] = df["volume"].rolling(window=20).std()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return data
            
    def _add_sentiment_features(self, data: DataFrame) -> DataFrame:
        """Add sentiment-based features."""
        # Implement sentiment analysis if needed
        return data
        
    def _add_fundamental_features(self, data: DataFrame) -> DataFrame:
        """Add fundamental analysis features."""
        # Implement fundamental analysis if needed
        return data
        
    def _add_custom_features(self, data: DataFrame) -> DataFrame:
        """Add custom features specified in config."""
        try:
            df = data.copy()
            
            for feature in self.config["custom_features"]:
                if feature == "price_channels":
                    df["upper_channel"] = df["high"].rolling(window=20).max()
                    df["lower_channel"] = df["low"].rolling(window=20).min()
                    df["channel_width"] = (df["upper_channel"] - df["lower_channel"]) / df["close"]
                    
                elif feature == "volume_profile":
                    df["volume_price_trend"] = df["volume"] * df["returns"].fillna(0)
                    df["volume_trend"] = df["volume_price_trend"].rolling(window=20).mean()
                    
                elif feature == "statistical":
                    df["returns_skew"] = df["returns"].rolling(window=20).skew()
                    df["returns_kurt"] = df["returns"].rolling(window=20).kurt()
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding custom features: {str(e)}")
            return data
