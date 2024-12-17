"""Machine Learning Strategy Implementation."""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import optuna
import lightgbm as lgb
from numpy.typing import NDArray
import traceback
import json
from scipy import stats
import time

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

from src.config.ml_strategy_config import DataConfig, ModelConfig, FeatureConfig
from src.data.data_fetcher import DataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.risk.risk_manager import RiskManager

# Type aliases
FloatArray = NDArray[np.float64]
MLModel = Union[tf.keras.Model, tf.keras.Sequential]
ValidationType = Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
EvalSetType = List[Tuple[np.ndarray, np.ndarray]]

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    position_type: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

class MLStrategy:
    """Machine Learning trading strategy."""
    
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, feature_config: FeatureConfig):
        """Initialize ML strategy."""
        self.data_config = data_config
        self.model_config = model_config
        self.feature_config = feature_config
        self.feature_engineer = FeatureEngineer(feature_config)
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)
        
    def _prepare_data(self, stock_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            stock_data: Dictionary of stock DataFrames
            market_data: Dictionary of market DataFrames
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            # Log the structure of input data
            self.logger.info("Preparing data for training...")
            for symbol, data in stock_data.items():
                if data is not None and not data.empty:
                    self.logger.info(f"Data for {symbol}:")
                    self.logger.info(f"Shape: {data.shape}")
                    self.logger.info(f"Columns: {data.columns.tolist()}")
                    self.logger.info(f"Sample row:\n{data.iloc[0]}")
            
            # Process each stock's data
            processed_data = {}
            for symbol, data in stock_data.items():
                if data is not None and not data.empty:
                    # Ensure column names are lowercase
                    data.columns = data.columns.str.lower()
                    
                    # Calculate technical indicators
                    processed_df = self._calculate_technical_indicators(data)
                    processed_data[symbol] = processed_df
            
            # Combine all processed data
            combined_data = pd.concat(processed_data.values(), axis=0)
            
            # Prepare features and target
            features = combined_data.drop(['returns'], axis=1, errors='ignore')
            target = combined_data['returns'].shift(-1)  # Next day's returns
            
            # Remove rows with NaN values
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_idx]
            target = target[valid_idx]
            
            # Scale features
            if not hasattr(self, 'scaler'):
                self.scaler = StandardScaler()
                self.scaler.fit(features)
            
            features_scaled = self.scaler.transform(features)
            
            return features_scaled, target.values
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _init_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Initialize the LSTM model with enhanced regularization."""
        model = tf.keras.Sequential([
            # First LSTM layer
            tf.keras.layers.LSTM(128, return_sequences=True, 
                               input_shape=input_shape,
                               kernel_regularizer=tf.keras.regularizers.l2(0.02),  # Increased L2
                               recurrent_regularizer=tf.keras.regularizers.l2(0.02),
                               activity_regularizer=tf.keras.regularizers.l1(0.01)),  # Added L1
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),  # Increased dropout
            
            # Second LSTM layer
            tf.keras.layers.LSTM(64, return_sequences=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.02),
                               recurrent_regularizer=tf.keras.regularizers.l2(0.02),
                               activity_regularizer=tf.keras.regularizers.l1(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            # Third LSTM layer
            tf.keras.layers.LSTM(32,
                               kernel_regularizer=tf.keras.regularizers.l2(0.02),
                               recurrent_regularizer=tf.keras.regularizers.l2(0.02),
                               activity_regularizer=tf.keras.regularizers.l1(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            # Dense layers with strong regularization
            tf.keras.layers.Dense(16, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                activity_regularizer=tf.keras.regularizers.l1(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Dense(1)
        ])
        
        # Optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Slightly lower initial learning rate
            clipnorm=1.0,  # Gradient clipping
            clipvalue=0.5  # Value clipping
        )
        
        # Compile with Huber loss for robustness
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),  # Huber loss with custom delta
            metrics=['mae', 'mse']
        )
        
        return model

    def train_model(self, stock_data: Dict[str, pd.DataFrame], market_data: Dict[str, pd.DataFrame]) -> None:
        """Train the model."""
        try:
            # Prepare data
            self.logger.info("Preparing training data...")
            X, y = self._prepare_data(stock_data, market_data)
            
            # Split data
            split_idx = int(len(X) * (1 - self.data_config.validation_split - self.data_config.test_split))
            val_idx = int(len(X) * (1 - self.data_config.test_split))
            
            X_train, X_val, X_test = X[:split_idx], X[split_idx:val_idx], X[val_idx:]
            y_train, y_val, y_test = y[:split_idx], y[split_idx:val_idx], y[val_idx:]
            
            # Initialize model
            input_shape = (self.model_config.sequence_length, X.shape[2])
            self.model = self._init_model(input_shape)
            
            # Enhanced callbacks
            callbacks = [
                # Early stopping with longer patience but stricter min_delta
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=30,  # Increased patience
                    min_delta=1e-5,  # Stricter improvement threshold
                    restore_best_weights=True,
                    verbose=1
                ),
                # More aggressive learning rate reduction
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,  # More aggressive reduction
                    patience=15,
                    min_lr=1e-7,
                    verbose=1
                ),
                # Model checkpoint with validation improvement threshold
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='models/best_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.CSVLogger('training_log.csv'),
                EpochProgressCallback(),
                # Time-based stopping callback (3 hours = 10800 seconds)
                TimeLimit(seconds=10800, verbose=1)
            ]
            
            # Train model with class weights to handle imbalance
            self.logger.info("\nStarting 3-hour training session...")
            self.logger.info("=" * 50)
            
            # Calculate sample weights to focus on larger price movements
            sample_weights = np.abs(y_train)  # Weight by magnitude of price movement
            sample_weights = sample_weights / np.mean(sample_weights)  # Normalize weights
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1000,  # High epoch limit with early stopping
                batch_size=32,
                callbacks=callbacks,
                sample_weight=sample_weights,  # Add sample weights
                verbose=0  # Turn off default progress bar
            )
            
            # Evaluate model
            test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
            self.logger.info(f"Test loss (MSE): {test_loss:.4f}")
            self.logger.info(f"Test MAE: {test_mae:.4f}")
            
            # Save training history and create learning curves
            self._save_training_history(history)
            
            # Detailed evaluation
            self._evaluate_model(X_test, y_test)

        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            # Prepare features
            feature_cols = [col for col in features.columns if col != 'returns']
            features_scaled = self.scaler.transform(features)
            
            # Create sequence
            sequence = features_scaled[-self.data_config.sequence_length:]
            sequence = np.expand_dims(sequence, axis=0)
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            return prediction[0][0]
            
        except Exception as e:
            self.logger.error(f"Error in predict: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_training_history(self, history: tf.keras.callbacks.History) -> None:
        """Save training history."""
        try:
            history_df = pd.DataFrame(history.history)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_history_{timestamp}.csv'
            filepath = os.path.join('data', 'training_history', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            history_df.to_csv(filepath)
            self.logger.info(f"Saved training history to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving training history: {str(e)}")

    def _get_feature_names(self) -> List[str]:
        """Get feature names based on config."""
        feature_names = []
        for feature in self.feature_config.feature_list:
            if feature.startswith('ma_'):
                feature_names.extend([feature, f'{feature}_slope'])
            elif feature == 'macd':
                feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
            elif feature == 'bollinger_bands':
                feature_names.extend(['bb_middle', 'bb_upper', 'bb_lower', 'bb_width'])
            else:
                feature_names.append(feature)

        # Add basic price and volume features
        feature_names.extend(['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio'])
        return feature_names

    def _get_scaler(self) -> Optional[StandardScaler]:
        """Initialize the feature scaler.
        Returns:
            Optional scaler instance
        """
        try:
            scaling_method = self.data_config.scaling_method
            if scaling_method == 'standard':
                return StandardScaler()
            return None
        except ValueError as exc:
            self.logger.error("Error initializing scaler: %s", str(exc))
            return None

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get hyperparameters for trial."""
        if self.model_config.model_type == 'lstm':
            return {
                'lstm_units': [
                    trial.suggest_int(f'lstm_units_{i}', 32, 512)
                    for i in range(3)
                ],
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 1e-2, log=True),
                'epochs': 1000,
                'early_stopping_patience': 50,
                'reduce_lr_patience': 20
            }
        else:
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }

    def _create_lstm_model(self, sequence_length: int, input_dim: int) -> tf.keras.Model:
        """Create LSTM model."""
        try:
            lstm_units = self.model_config.lstm_units
            dropout_rate = self.model_config.dropout_rate
            learning_rate = self.model_config.learning_rate
            
            model = Sequential([
                LSTM(units=lstm_units[0],
                     input_shape=(sequence_length, input_dim),
                     return_sequences=True,
                     kernel_regularizer=tf.keras.regularizers.l2(self.model_config.l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                
                LSTM(units=lstm_units[1],
                     return_sequences=True,
                     kernel_regularizer=tf.keras.regularizers.l2(self.model_config.l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                
                LSTM(units=lstm_units[2],
                     kernel_regularizer=tf.keras.regularizers.l2(self.model_config.l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                
                Dense(units=16, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(self.model_config.l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate/2),
                
                Dense(units=1)
            ])
            
            # Use Adam optimizer with reduced learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.model_config.learning_rate)
            
            # Compile model with Huber loss for robustness
            model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.Huber(),
                         metrics=['mae'])
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating LSTM model: {str(e)}")
            raise

    def _create_lightgbm_model(self) -> lgb.LGBMRegressor:
        """Create LightGBM model."""
        try:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            self.logger.info("LightGBM model created successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating LightGBM model: {str(e)}")
            raise

    def _create_model_with_params(self, params: Dict[str, Any]) -> Optional[MLModel]:
        """Create model with parameters."""
        if self.model_config.model_type == 'lstm':
            return self._create_lstm_model(self.model_config.sequence_length, self.model_config.input_dim)
        elif self.model_config.model_type == 'lightgbm':
            return self._create_lightgbm_model()
        return None

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        try:
            # Get hyperparameters for this trial
            params = self._get_trial_params(trial)
            
            # Create and compile model with these parameters
            model = self._create_model_with_params(params)
            
            # Prepare data for walk-forward optimization
            cv_scores = []
            n_splits = 5  # Number of folds
            
            # Walk-forward cross validation
            for fold in range(n_splits):
                # Split data into train/val for this fold
                train_size = int(len(self.x_train) * 0.8)
                fold_size = train_size // n_splits
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                # Create train/val splits
                x_train_fold = np.concatenate([
                    self.x_train[:start_idx],
                    self.x_train[end_idx:train_size]
                ])
                y_train_fold = np.concatenate([
                    self.y_train[:start_idx],
                    self.y_train[end_idx:train_size]
                ])
                x_val_fold = self.x_train[start_idx:end_idx]
                y_val_fold = self.y_train[start_idx:end_idx]
                
                # Data augmentation with noise
                noise_level = trial.suggest_float('noise_level', 0.001, 0.01)
                x_train_noisy = x_train_fold + np.random.normal(0, noise_level, x_train_fold.shape)
                
                # Train model
                history = model.fit(
                    x_train_noisy, y_train_fold,
                    validation_data=(x_val_fold, y_val_fold),
                    epochs=params.get('epochs', 100),
                    batch_size=params.get('batch_size', 32),
                    verbose=0,
                    callbacks=[
                        EarlyStopping(
                            monitor='val_loss',
                            patience=params.get('early_stopping_patience', 10),
                            restore_best_weights=True,
                            mode='min'
                        ),
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6
                        ),
                        optuna.integration.TFKerasPruningCallback(
                            trial, 'val_loss'
                        )
                    ]
                )
                
                # Calculate validation metrics
                val_loss = min(history.history['val_loss'])
                train_loss = min(history.history['loss'])
                
                # Check for overfitting
                if train_loss * 0.7 > val_loss:  # If training loss is much better than validation
                    raise optuna.TrialPruned("Model is overfitting")
                
                cv_scores.append(val_loss)
            
            # Calculate mean and std of CV scores
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # Penalize high variance in cross-validation scores
            final_score = mean_score * (1 + std_score)
            
            # Prune if the score is too high
            if final_score > trial.suggest_float('max_acceptable_loss', 0.1, 0.5):
                raise optuna.TrialPruned()
            
            return -final_score  # Negative because we want to maximize
            
        except Exception as e:
            self.logger.error(f"Error in objective function: {str(e)}")
            raise

    def _optimize_hyperparameters(self, x_data: FloatArray, y_data: FloatArray) -> None:
        """Optimize model hyperparameters.
        Args:
            x_data: Training features
            y_data: Training labels
        """
        try:
            # Create study
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=30,
                    interval_steps=10
                )
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self._objective(trial),
                n_trials=50,
                timeout=3600 * 5  # 5 hours
            )
            
            # Get best parameters
            self.best_params = study.best_params
            self.logger.info("Best hyperparameters: %s", self.best_params)
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters from config."""
        if self.model_config.hyperparameters is None:
            return {}
        return self.model_config.hyperparameters

    def _fit_keras(
        self,
        model: Sequential,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: Optional[FloatArray] = None,
        y_val: Optional[FloatArray] = None
    ) -> None:
        """Fit Keras model with appropriate parameters."""
        try:
            # Initialize callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if x_val is not None else 'loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if x_val is not None else 'loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Train model with gradient clipping
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val) if x_val is not None and y_val is not None else None,
                epochs=self.model_config.epochs,
                batch_size=self.model_config.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store history for later use
            self.model.history = history
            
            # Log final metrics
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if x_val is not None else None
            
            self.logger.info(f"Training completed - Final loss: {final_loss:.4f}" + 
                           (f", Validation loss: {final_val_loss:.4f}" if final_val_loss else ""))
            
        except Exception as e:
            self.logger.error(f"Error during Keras model training: {str(e)}")
            raise

    def _fit_lightgbm(
        self,
        model: lgb.LGBMRegressor,
        X_train: FloatArray,
        y_train: FloatArray,
        X_val: Optional[FloatArray] = None,
        y_val: Optional[FloatArray] = None
    ) -> None:
        """Fit LightGBM model with early stopping.

        Args:
            model: LightGBM model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        fit_params: Dict[str, Any] = {
            'eval_metric': 'rmse',
            'callbacks': [EarlyStopping(patience=10, restore_best_weights=True)],
            'verbose': False
        }

        if X_val is not None and y_val is not None:
            fit_params['validation_data'] = (X_val, y_val)

        model.fit(X_train, y_train, **fit_params)
        self.logger.info(f"LightGBM model fitted with best iteration: {model.best_iteration_}")

    def _fit_model(
        self,
        model: Sequential,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: Optional[FloatArray] = None,
        y_val: Optional[FloatArray] = None
    ) -> None:
        """Fit the model with appropriate parameters based on model type.
        
        Args:
            model: The model to fit
            x_train: Training features
            y_train: Training labels
            x_val: Optional validation features
            y_val: Optional validation labels
        """
        try:
            if isinstance(model, Sequential):
                self._fit_keras(model, x_train, y_train, x_val, y_val)
            elif isinstance(model, lgb.LGBMRegressor):
                self._fit_lightgbm(model, x_train, y_train, x_val, y_val)
            else:
                # Default case for other model types
                model.fit(x_train, y_train)
                
            self.logger.debug(f"Model {type(model).__name__} fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting model {type(model).__name__}: {str(e)}")
            raise

    def prepare_sequences(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model."""
        try:
            self.logger.info("Preparing sequences...")
            self.logger.info(f"Input features shape: {features.shape}")
            self.logger.info(f"Input target shape: {target.shape}")
            
            # Check for NaN values in inputs
            if features.isna().any().any():
                self.logger.error("NaN values found in features before sequence preparation")
                raise ValueError("Features contain NaN values")
            if target.isna().any():
                self.logger.error("NaN values found in target before sequence preparation")
                raise ValueError("Target contains NaN values")
            
            # Convert to numpy arrays
            features_array = features.values
            target_array = target.values
            
            # Create sequences
            X, y = [], []
            for i in range(len(features_array) - self.data_config.sequence_length):
                # Extract sequence
                sequence = features_array[i:(i + self.data_config.sequence_length)]
                target = target_array[i + self.data_config.sequence_length]
                
                # Convert to numpy arrays
                X.append(sequence)
                y.append(target)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Check output shapes
            self.logger.info(f"Output X shape: {X.shape}")
            self.logger.info(f"Output y shape: {y.shape}")
            
            # Check for NaN values in outputs
            if np.isnan(X).any():
                self.logger.error("NaN values found in X after sequence preparation")
                raise ValueError("X sequences contain NaN values")
            if np.isnan(y).any():
                self.logger.error("NaN values found in y after sequence preparation")
                raise ValueError("y sequences contain NaN values")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing sequences: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def train(self) -> None:
        """Train the model."""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
                
            if self.x_train is None or self.y_train is None:
                raise ValueError("Training data not prepared")
                
            # Configure callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Train the model
            self.logger.info("Starting model training...")
            history = self.model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_val, self.y_val),
                epochs=self.model_config.epochs,
                batch_size=self.model_config.batch_size,
                callbacks=callbacks,
                verbose=2  # Show one line per epoch
            )
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            self.logger.info(f"Training completed - Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def prepare_features(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Prepare features for prediction.
        Args:
            data: Raw price data
            fit_scaler: Whether to fit the scaler on this data
        Returns:
            pd.DataFrame: Processed features
        """
        try:
            # Create features using feature engineer
            features = self.feature_engineer.create_features(data)
            
            # Drop any rows with NaN values
            features = features.dropna()
            
            # Scale features if scaler exists
            if self.scaler is not None:
                if fit_scaler:
                    features_scaled = self.scaler.fit_transform(features)
                else:
                    features_scaled = self.scaler.transform(features)
                features = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
            
            self.logger.info(f"Prepared features with shape: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            raise

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        if data is None or data.empty:
            raise ValueError("Input data is None or empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
            
        if data.isnull().any().any():
            raise ValueError("Data contains null values")

    def backtest(self, data: pd.DataFrame) -> pd.Series:
        """
        Backtest the strategy.
        Args:
            data: Historical price data
        Returns:
            pd.Series: Strategy returns
        """
        try:
            self.logger.info("Starting backtest...")
            
            # Prepare features
            features = self.prepare_features(data, fit_scaler=False)
            
            # Get predictions
            predictions = self.predict(features)
            
            # Calculate positions based on predictions
            positions = pd.Series(0, index=data.index)
            entry_threshold = getattr(self.model_config, 'entry_threshold', 0.6)
            exit_threshold = getattr(self.model_config, 'exit_threshold', -0.4)
            for i in range(len(predictions)-1):  # -1 because we can't trade on the last prediction
                if predictions[i] > entry_threshold:
                    positions.iloc[i] = 1  # Long position
                elif predictions[i] < exit_threshold:
                    positions.iloc[i] = -1  # Short position
            
            # Calculate returns
            price_returns = data['close'].pct_change()
            strategy_returns = positions.shift(1) * price_returns  # Shift positions to avoid look-ahead bias
            strategy_returns = strategy_returns.fillna(0)
            
            # Apply transaction costs
            trades = positions.diff().abs()
            transaction_costs = trades * getattr(self.model_config, 'transaction_cost', 0.001)  # Default 0.1% transaction cost
            strategy_returns = strategy_returns - transaction_costs
            
            # Store positions and returns
            self.positions = positions
            self.backtest_returns = strategy_returns
            
            # Log backtest results
            metrics = self._calculate_metrics(strategy_returns)
            self.logger.info("Backtest Results:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            return strategy_returns
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}", exc_info=True)
            return pd.Series()

    def split_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and validation sets using time-based split.
        
        Args:
            features: Feature DataFrame with datetime index
            target: Target Series with datetime index
            
        Returns:
            Tuple containing train and validation splits for features and target
        """
        # Ensure data is sorted by time
        features = features.sort_index()
        target = target.sort_index()
        
        # Calculate split point (e.g., use last 20% for validation)
        split_idx = int(len(features) * 0.8)
        
        # Split the data
        x_train = features.iloc[:split_idx]
        x_val = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_val = target.iloc[split_idx:]
        
        self.logger.info(f"Training data from {x_train.index[0]} to {x_train.index[-1]}")
        self.logger.info(f"Validation data from {x_val.index[0]} to {x_val.index[-1]}")
        
        return x_train, x_val, y_train, y_val

    def create_rolling_predictions(self, features: pd.DataFrame, window_size: int = 252) -> pd.DataFrame:
        """
        Create rolling predictions using a sliding window approach.
        
        Args:
            features: Feature DataFrame with datetime index
            window_size: Size of the rolling window in days
            
        Returns:
            DataFrame with predictions
        """
        predictions = pd.DataFrame(index=features.index)
        predictions['predicted'] = np.nan
        
        for i in range(window_size, len(features)):
            # Get training data
            train_features = features.iloc[i-window_size:i]
            train_target = features['target_1d'].iloc[i-window_size:i]
            
            # Train model on window
            self.train(train_features, train_target)
            
            # Make prediction for next point
            next_features = features.iloc[i:i+1]
            pred = self.predict(next_features)
            predictions.iloc[i] = pred[0]
            
        return predictions

    def _calculate_metrics(self, returns: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate backtest performance metrics.
        
        Args:
            returns: Series of strategy returns
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            if returns is None or len(returns) == 0:
                return {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'volatility': 0.0,
                    'calmar_ratio': 0.0
                }
            
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            trading_days = min(252, len(returns))
            annualized_return = (1 + total_return) ** (252/trading_days) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Sharpe and Sortino ratios
            risk_free_rate = self.model_config.risk.risk_free_rate
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
            sortino_ratio = excess_returns / downside_std if downside_std != 0 else 0
            
            # Drawdown analysis
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns - rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Win/Loss metrics
            winning_days = returns[returns > 0]
            losing_days = returns[returns < 0]
            win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
            
            # Average win/loss
            avg_win = winning_days.mean() if len(winning_days) > 0 else 0
            avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
            
            # Profit factor
            gross_profits = sum(p.pnl for p in self.positions if p.pnl > 0)
            gross_losses = abs(sum(p.pnl for p in self.positions if p.pnl < 0))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
            
            return {
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'volatility': float(volatility),
                'calmar_ratio': float(calmar_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            }

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the ML strategy on the given data."""
        try:
            self.logger.info("Starting ML strategy run...")
            self.logger.info(f"Input data shape: {data.shape}")
            
            # Create features
            self.logger.info("Creating features...")
            features = self.feature_engineer.create_features(data)
            self.logger.info(f"Features created. Shape: {features.shape}")
            self.logger.info(f"Feature columns: {features.columns.tolist()}")
            
            # Update model config with actual feature dimensions
            n_features = len(features.columns)
            self.logger.info(f"Setting input dimension to {n_features} based on actual features")
            self.model_config.input_dim = n_features
            self.data_config.n_features = n_features
            
            # Prepare data for training
            self.logger.info("Preparing data for training...")
            X, y = self._prepare_data(features)
            self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            
            # Split data
            self.logger.info("Splitting data into train/val/test sets...")
            train_size = int(len(X) * (1 - self.data_config.validation_split - self.data_config.test_split))
            val_size = int(len(X) * self.data_config.validation_split)
            
            self.x_train = X[:train_size]
            self.y_train = y[:train_size]
            self.x_val = X[train_size:train_size+val_size]
            self.y_val = y[train_size:train_size+val_size]
            self.x_test = X[train_size+val_size:]
            self.y_test = y[train_size+val_size:]
            
            self.logger.info(f"Train set shape: {self.x_train.shape}")
            self.logger.info(f"Validation set shape: {self.x_val.shape}")
            self.logger.info(f"Test set shape: {self.x_test.shape}")
            
            # Initialize and train model
            self.logger.info("Initializing model...")
            self.model = self._init_model()
            
            # Train model
            self.logger.info("Training model...")
            history = self.train_model()
            
            # Make predictions
            self.logger.info("Making predictions...")
            predictions = self.predict(features)
            
            # Log training metrics
            self.logger.info("Training metrics:")
            for metric, values in history.items():
                self.logger.info(f"{metric}: {values[-1]:.4f}")
            
            # Evaluate on test set
            if len(self.x_test) > 0:
                self.logger.info("Evaluating on test set...")
                test_loss = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                self.logger.info(f"Test loss: {test_loss:.4f}")
            
            # Create predictions DataFrame
            # We need to account for the sequence_length offset in the index
            valid_indices = features.index[self.data_config.sequence_length-1:]
            if len(predictions) > len(valid_indices):
                predictions = predictions[:len(valid_indices)]
            elif len(predictions) < len(valid_indices):
                valid_indices = valid_indices[:len(predictions)]
                
            predictions_df = pd.DataFrame(
                predictions, 
                index=valid_indices,
                columns=['predicted_returns']
            )
            
            self.logger.info(f"Final predictions shape: {predictions_df.shape}")
            return predictions_df
            
        except Exception as e:
            self.logger.error(f"Error in ML strategy run: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics.
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        try:
            metrics: Dict[str, float] = {}

            if self.backtest_returns is None or len(self.backtest_returns) == 0:
                return metrics

            # Basic return metrics
            total_return = (1 + self.backtest_returns).prod() - 1
            trading_days = min(252, len(self.backtest_returns))
            annualized_return = (1 + total_return) ** (252/trading_days) - 1
            
            # Risk metrics
            volatility = self.backtest_returns.std() * np.sqrt(252)
            downside_returns = self.backtest_returns[self.backtest_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Sharpe and Sortino ratios
            risk_free_rate = self.model_config.risk.risk_free_rate
            excess_returns = self.backtest_returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
            sortino_ratio = excess_returns / downside_std if downside_std != 0 else 0
            
            # Drawdown analysis
            cum_returns = (1 + self.backtest_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns - rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Win/Loss metrics
            winning_days = self.backtest_returns[self.backtest_returns > 0]
            losing_days = self.backtest_returns[self.backtest_returns < 0]
            win_rate = len(winning_days) / len(self.backtest_returns) if len(self.backtest_returns) > 0 else 0
            
            # Average win/loss
            avg_win = winning_days.mean() if len(winning_days) > 0 else 0
            avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
            
            # Profit factor
            gross_profits = sum(p.pnl for p in self.positions if p.pnl > 0)
            gross_losses = abs(sum(p.pnl for p in self.positions if p.pnl < 0))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
            
            metrics.update({
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'volatility': float(volatility),
                'calmar_ratio': float(calmar_ratio)
            })

            return metrics
            
        except (ValueError, RuntimeError) as exc:
            self.logger.error("Error calculating performance metrics: %s", str(exc))
            return {}

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the strategy.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            if not hasattr(self, 'positions') or not self.positions:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0
                }
            
            # Calculate returns
            returns = pd.Series([p.pnl for p in self.positions])
            total_return = returns.sum()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate maximum drawdown
            cumulative_returns = returns.cumsum()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns - rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Calculate win rate
            winning_trades = sum(1 for p in self.positions if p.pnl > 0)
            total_trades = len(self.positions)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Average win/loss
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
            
            # Profit factor
            gross_profits = sum(p.pnl for p in self.positions if p.pnl > 0)
            gross_losses = abs(sum(p.pnl for p in self.positions if p.pnl < 0))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }

    def _log_feature_importances(self, model: Any) -> None:
        """Log feature importances for models that support it.
        
        Args:
            model: Trained model that may have feature_importances_ attribute
        """
        if not isinstance(model, lgb.LGBMRegressor):
            return
            
        feature_importances = getattr(model, 'feature_importances_', None)
        if feature_importances is not None:
            importances = np.asarray(feature_importances, dtype=np.float64)
            feature_names = getattr(model, 'feature_name_', None)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Create list of tuples with proper typing
            feature_importance_list: List[Tuple[str, float]] = []
            for name, imp in zip(feature_names, importances):
                feature_importance_list.append((str(name), float(imp)))
            
            # Sort by importance value
            feature_importance_list.sort(key=lambda x: x[1], reverse=True)
            self.logger.info("Feature importances:")
            for feature, importance in feature_importance_list:
                self.logger.info("\t%s: %.6f", feature, importance)

    def _plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Create and save evaluation plots."""
        try:
            self.logger.info("Creating evaluation plots...")
            
            # Create directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Scatter plot of predictions vs actual values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
            plt.title('Prediction vs Actual Returns')
            plt.tight_layout()
            plt.savefig('plots/prediction_scatter.png')
            plt.close()
            
            # Distribution plot of prediction errors
            errors = y_pred - y_true
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.xlabel('Prediction Error')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Errors')
            plt.tight_layout()
            plt.savefig('plots/error_distribution.png')
            plt.close()
            
            # Calculate and log evaluation metrics
            mse = np.mean(np.square(errors))
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(mse)
            r2 = 1 - (np.sum(np.square(errors)) / np.sum(np.square(y_true - np.mean(y_true))))
            spearman_corr, _ = spearmanr(y_true, y_pred)
            
            self.logger.info("\nModel Evaluation Metrics:")
            self.logger.info(f"MSE: {mse:.6f}")
            self.logger.info(f"MAE: {mae:.6f}")
            self.logger.info(f"RMSE: {rmse:.6f}")
            self.logger.info(f"R²: {r2:.6f}")
            self.logger.info(f"Spearman Correlation: {spearman_corr:.6f}")
            
            # Save metrics to file
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'spearman_corr': float(spearman_corr)
            }
            
            with open('models/evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error in plot_prediction_analysis: {str(e)}\n{traceback.format_exc()}")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with error handling and logging."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
                
            self.logger.info(f"Making predictions on data with shape: {X.shape}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Input data contains NaN or infinite values")
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Log prediction statistics
            self.logger.info(f"Prediction statistics:")
            self.logger.info(f"Mean: {np.mean(predictions):.4f}")
            self.logger.info(f"Std: {np.std(predictions):.4f}")
            self.logger.info(f"Min: {np.min(predictions):.4f}")
            self.logger.info(f"Max: {np.max(predictions):.4f}")
            
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"Error in predict: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_training_history(self, history) -> None:
        """Save training history and create learning curves plot."""
        try:
            # Create directories if they don't exist
            os.makedirs('models', exist_ok=True)
            os.makedirs('plots', exist_ok=True)
            
            # Save history to file
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'mae': [float(x) for x in history.history['mae']],
                'val_mae': [float(x) for x in history.history['val_mae']]
            }
            
            with open('models/training_history.json', 'w') as f:
                json.dump(history_dict, f, indent=4)
            
            # Plot learning curves
            plt.figure(figsize=(12, 5))
            
            # Loss subplot
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # MAE subplot
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('plots/learning_curves.png')
            plt.close()
            
            self.logger.info("Training history and learning curves saved")
            
        except Exception as e:
            self.logger.error(f"Error saving training history: {str(e)}")

    def evaluate_predictions(self, predictions: np.ndarray, actual_returns: pd.Series) -> Dict[str, float]:
        """
        Evaluate model predictions against actual returns
        Args:
            predictions: Model predictions
            actual_returns: Actual returns
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Ensure predictions and actual returns are aligned
            predictions = predictions[:len(actual_returns)]
            
            # Calculate metrics
            mse = mean_squared_error(actual_returns, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_returns, predictions)
            r2 = r2_score(actual_returns, predictions)
            
            # Calculate directional accuracy
            correct_direction = np.sum(np.sign(predictions) == np.sign(actual_returns))
            directional_accuracy = correct_direction / len(predictions)
            
            # Calculate information coefficient (Spearman correlation)
            ic = spearmanr(predictions, actual_returns)[0]
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'information_coefficient': ic
            }
            
            # Log detailed metrics
            self.logger.info("Model Performance Metrics:")
            self.logger.info(f"MSE: {mse:.6f}")
            self.logger.info(f"RMSE: {rmse:.6f}")
            self.logger.info(f"MAE: {mae:.6f}")
            self.logger.info(f"R2 Score: {r2:.6f}")
            self.logger.info(f"Directional Accuracy: {directional_accuracy:.2%}")
            self.logger.info(f"Information Coefficient: {ic:.6f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            traceback.print_exc()
            raise

    def calculate_backtest_metrics(self, trades: List[Trade], data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics for backtest trades
        Args:
            trades: List of trades
            data: Price data DataFrame
        Returns:
            DataFrame with metrics
        """
        if not trades:
            return pd.DataFrame()
            
        # Calculate trade metrics
        trade_metrics = []
        total_return = 0
        transaction_cost = getattr(self.model_config, 'transaction_cost', 0.001)  # Default 0.1% transaction cost
        
        for trade in trades:
            returns = (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.position_type == 'short':
                returns = -returns
                
            # Apply transaction costs
            returns -= transaction_cost * 2  # Cost for both entry and exit
            
            total_return += returns
            
            trade_metrics.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_type': trade.position_type,
                'returns': returns
            })
            
        metrics_df = pd.DataFrame(trade_metrics)
        
        # Calculate strategy metrics
        if len(metrics_df) > 0:
            metrics_df['cumulative_returns'] = (1 + metrics_df['returns']).cumprod() - 1
            
            # Add summary metrics
            metrics_df.loc['Total', 'returns'] = total_return
            metrics_df.loc['Total', 'num_trades'] = len(trades)
            metrics_df.loc['Total', 'win_rate'] = len(metrics_df[metrics_df['returns'] > 0]) / len(metrics_df)
            metrics_df.loc['Total', 'avg_return'] = metrics_df['returns'].mean()
            metrics_df.loc['Total', 'std_return'] = metrics_df['returns'].std()
            
            if metrics_df['std_return'].iloc[-1] != 0:
                metrics_df.loc['Total', 'sharpe_ratio'] = (
                    metrics_df['avg_return'].iloc[-1] / metrics_df['std_return'].iloc[-1]
                ) * np.sqrt(252)  # Annualized
                
        return metrics_df

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            df = data.copy()
            
            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()
            
            # Price-based indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(4)
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price channels
            df['high_20'] = df['high'].rolling(window=20).max()
            df['low_20'] = df['low'].rolling(window=20).min()
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

class TimeLimit(tf.keras.callbacks.Callback):
    """Callback to stop training when a specified amount of time has passed."""
    
    def __init__(self, seconds=None, verbose=0):
        super(TimeLimit, self).__init__()
        self.seconds = seconds
        self.verbose = verbose
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if self.seconds is None:
            return
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print(f'\nStopping training after {elapsed_time:.2f} seconds.')

class EpochProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/1000")
        print(f"Loss: {logs.get('loss', 0):.6f}")
        print(f"MAE: {logs.get('mae', 0):.6f}")
        print(f"Val Loss: {logs.get('val_loss', 0):.6f}")
        print(f"Val MAE: {logs.get('val_mae', 0):.6f}")
        print(f"{'='*50}\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
