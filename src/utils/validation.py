"""Data validation utilities for the algorithmic trading system."""
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and preprocessing utilities."""
    
    # Validation thresholds
    MISSING_VALUE_THRESHOLD = 0.2  # 20% threshold for missing values in a column
    MAX_AFFECTED_RATIO = 0.5  # Maximum ratio of columns that can have high missing values
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate price data for completeness and correctness.
        
        Args:
            df: DataFrame containing price data
            symbol: Symbol being validated
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns for {symbol}: {missing_columns}"
            
            # Check for sufficient data points
            if len(df) < 50:
                return False, f"Insufficient data points for {symbol}: {len(df)} < 50"
            
            # Check for missing values
            missing_counts = df[required_columns].isnull().sum()
            if missing_counts.any():
                logger.warning(f"Missing values in {symbol}:\n{missing_counts[missing_counts > 0]}")
            
            # Check for price anomalies
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                # Check for negative prices
                if (df[col] <= 0).any():
                    return False, f"Negative or zero prices found in {col} for {symbol}"
                
                # Check for extreme price changes (>50% in a day)
                pct_change = df[col].pct_change().abs()
                if (pct_change > 0.5).any():
                    logger.warning(f"Large price changes (>50%) detected in {symbol} {col}")
            
            # Validate high/low relationships
            if not (df['high'] >= df['low']).all():
                return False, f"Invalid high/low relationship found in {symbol}"
            
            # Validate OHLC relationships
            if not (
                (df['high'] >= df['open']).all() and
                (df['high'] >= df['close']).all() and
                (df['low'] <= df['open']).all() and
                (df['low'] <= df['close']).all()
            ):
                return False, f"Invalid OHLC relationship found in {symbol}"
            
            # Check for volume anomalies
            if (df['volume'] < 0).any():
                return False, f"Negative volume found in {symbol}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {str(e)}")
            return False, f"Validation error for {symbol}: {str(e)}"
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Preprocess price data by handling missing values and adjusting for anomalies.
        
        Args:
            df: DataFrame containing price data
            symbol: Symbol being processed
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            df = df.copy()
            
            # Forward fill missing values
            df.fillna(method='ffill', inplace=True)
            
            # Backward fill any remaining missing values at the start
            df.fillna(method='bfill', inplace=True)
            
            # Handle zero volumes
            df.loc[df['volume'] == 0, 'volume'] = df['volume'].median()
            
            # Clip extreme values (beyond 3 std from mean)
            for col in ['open', 'high', 'low', 'close']:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
            
            # Sort index
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data for {symbol}: {str(e)}")
            return df  # Return original DataFrame if preprocessing fails
    
    @staticmethod
    def validate_features(features: pd.DataFrame, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate engineered features.
        
        Args:
            features: DataFrame containing features
            symbol: Symbol being validated
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for empty features
            if features.empty:
                return False, f"Empty features DataFrame for {symbol}"
            
            # Check for infinite values
            if np.isinf(features.values).any():
                logger.error("Infinite values detected in features for %s", symbol)
                return False, f"Infinite values found in features for {symbol}"
            
            try:
                # Calculate missing value percentages (0.0 to 1.0)
                missing_mask = features.isna()
                missing_pct = missing_mask.astype(np.float64).mean()
                
                # Ensure we have a Series with float64 values
                if not isinstance(missing_pct, pd.Series):
                    missing_pct = pd.Series(missing_pct, index=features.columns)
                missing_pct = missing_pct.astype(np.float64)
                
                high_missing_mask = missing_pct > DataValidator.MISSING_VALUE_THRESHOLD
                
                if high_missing_mask.any():
                    # Get columns with high missing values and sort by percentage
                    cols_with_missing = pd.Series(missing_pct[high_missing_mask])
                    total_cols = len(features.columns)
                    affected_cols = len(cols_with_missing)
                    affected_pct = affected_cols / total_cols
                    
                    # Convert to list of tuples and sort
                    missing_items = [(str(idx), float(val)) for idx, val in cols_with_missing.items()]
                    sorted_missing = sorted(missing_items, key=lambda x: (-x[1], x[0]))
                    
                    missing_values_str = "\n\t".join([
                        f"{col}: {pct:.1%}" 
                        for col, pct in sorted_missing
                    ])
                    
                    logger.warning(
                        "High missing values (>20%%) detected in %d/%d columns (%.1f%%) for %s:\n\t%s",
                        affected_cols,
                        total_cols,
                        affected_pct * 100,
                        symbol,
                        missing_values_str
                    )
                    
                    # Return error if too many columns are affected
                    if affected_pct > 0.5:  # More than 50% of columns have high missing values
                        return False, f"Too many columns ({affected_cols}/{total_cols}, {affected_pct:.1%}) have high missing values for {symbol}"
            
            except Exception as e:
                logger.error(f"Error calculating missing values for {symbol}: {str(e)}")
                return False, f"Missing value calculation error for {symbol}: {str(e)}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating features for {symbol}: {str(e)}")
            return False, f"Feature validation error for {symbol}: {str(e)}"

def validate_data(data: pd.DataFrame, labels: Optional[pd.Series] = None) -> None:
    """Validate input data for ML model.
    
    Args:
        data: Input features or raw price data
        labels: Optional target labels
        
    Raises:
        ValueError: If data validation fails
    """
    validator = DataValidator()
    
    # Validate data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("Input data is empty")
        
    # Validate labels if provided
    if labels is not None:
        if not isinstance(labels, pd.Series):
            raise ValueError("Labels must be a pandas Series")
        if len(labels) != len(data):
            raise ValueError("Labels length must match data length")
        if labels.isnull().any():
            raise ValueError("Labels contain missing values")
