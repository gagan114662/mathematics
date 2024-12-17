"""Test feature engineering functionality."""
import unittest
import pandas as pd
import numpy as np
from src.features.feature_engineer import FeatureEngineer, FeatureConfig

class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering functionality."""
    
    def setUp(self):
        """Set up test data and feature engineer."""
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        np.random.seed(42)
        
        self.data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, size=len(dates))
        }, index=dates)
        
        # Create feature config
        self.config = FeatureConfig(
            feature_list=['ma_5', 'ma_10', 'ma_20', 'rsi', 'macd', 'bollinger_bands'],
            rsi_period=14,
            macd_periods=(12, 26, 9),
            bb_period=20,
            bb_std=2
        )
        
        self.feature_engineer = FeatureEngineer(self.config)
    
    def test_feature_creation(self):
        """Test feature creation."""
        features = self.feature_engineer.create_features(self.data)
        
        # Check if all expected features are present
        expected_features = [
            'ma_5', 'ma_5_slope', 'ma_10', 'ma_10_slope', 'ma_20', 'ma_20_slope',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'volume_change', 'volume_ma_ratio', 'price_volume_ratio', 'price_volume_ratio_ma',
            'atr'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check if there are no NaN values
        self.assertFalse(features.isnull().any().any())
        
        # Check basic feature properties
        self.assertTrue(all(features['rsi'].between(0, 100)))
        self.assertTrue(all(features['bb_upper'] >= features['bb_middle']))
        self.assertTrue(all(features['bb_lower'] <= features['bb_middle']))
    
    def test_error_handling(self):
        """Test error handling with invalid data."""
        # Test with missing columns
        invalid_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        with self.assertRaises(Exception):
            self.feature_engineer.create_features(invalid_data)

if __name__ == '__main__':
    unittest.main()
