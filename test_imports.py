"""Test if all required imports are available."""
try:
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from src.strategies.ml_strategy import MLStrategy
    from src.config.ml_strategy_config import MLStrategyConfig
    print("All imports successful!")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"Import error: {str(e)}")
except Exception as e:
    print(f"Other error: {str(e)}")
