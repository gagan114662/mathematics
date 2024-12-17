from typing import Dict, List, Tuple, Optional, Any, Union, cast
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import numpy.typing as npt

@dataclass
class ModelConfig:
    """Configuration for model training."""
    cv_splits: int = 5
    test_size: int = 252  # One year of trading days
    early_stopping_rounds: int = 50
    n_trials: int = 50  # Number of optimization trials
    feature_importance_threshold: float = 0.5
    random_state: int = 42

class ModelTraining:
    """Handles model training and optimization."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model training.
        
        Args:
            config: Model training configuration
        """
        self.config: ModelConfig = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_importance: Optional[pd.Series] = None
        self.training_metrics: Dict[str, Any] = {}
        self.selected_features: Optional[List[str]] = None
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.last_training_time: Optional[datetime] = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train model with hyperparameter optimization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        try:
            if X.empty or y.empty:
                raise ValueError("Empty input data")
                
            if len(X) != len(y):
                raise ValueError("X and y must have same length")
                
            start_time = time.time()
            self.logger.info("Starting model training with optimization...")
            
            # Create time series splits
            tscv = TimeSeriesSplit(
                n_splits=self.config.cv_splits,
                test_size=self.config.test_size
            )
            
            # Optimize hyperparameters
            best_params = self._optimize_hyperparameters(X, y, tscv)
            
            # Train final model with best parameters
            self.model = self._train_final_model(X, y, best_params, tscv)
            
            # Update feature importance and selection
            self._update_feature_importance(list(X.columns))
            
            # Record training metrics
            self.training_metrics['training_time'] = float(time.time() - start_time)
            self.last_training_time = datetime.now()
            
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
            
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                tscv: TimeSeriesSplit) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            tscv: Time series cross-validation splitter
            
        Returns:
            Dict[str, Any]: Dictionary of optimized parameters
            
        Raises:
            RuntimeError: If optimization fails
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'scale_pos_weight': float(len(y[y == 0]) / len(y[y == 1])),
                'objective': 'binary:logistic',
                'random_state': self.config.random_state
            }
            
            scores: List[float] = []
            for train_idx, val_idx in tscv.split(X):
                X_train = X.iloc[train_idx].copy()
                X_val = X.iloc[val_idx].copy()
                y_train = y.iloc[train_idx].copy()
                y_val = y.iloc[val_idx].copy()
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
                
                y_pred = model.predict(X_val)
                score = float(f1_score(y_val, y_pred))
                scores.append(score)
            
            return float(np.mean(scores))
            
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_trials)
            
            self.logger.info(f"Best trial: {study.best_trial.value}")
            return dict(study.best_params)
        except Exception as e:
            raise RuntimeError(f"Hyperparameter optimization failed: {str(e)}")
        
    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, 
                          best_params: Dict[str, Any], tscv: TimeSeriesSplit) -> xgb.XGBClassifier:
        """
        Train final model with best parameters.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            best_params: Best hyperparameters from optimization
            tscv: Time series cross-validation splitter
            
        Returns:
            xgb.XGBClassifier: Trained XGBoost classifier
            
        Raises:
            ValueError: If training fails in all folds
        """
        cv_metrics: Dict[str, List[float]] = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'auc_roc': []
        }
        
        final_model: Optional[xgb.XGBClassifier] = None
        best_score: float = float('-inf')
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()
            
            model = xgb.XGBClassifier(**best_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )
            
            # Calculate metrics
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            cv_metrics['accuracy'].append(float(accuracy_score(y_val, y_pred)))
            cv_metrics['precision'].append(float(precision_score(y_val, y_pred)))
            cv_metrics['recall'].append(float(recall_score(y_val, y_pred)))
            cv_metrics['f1'].append(float(f1_score(y_val, y_pred)))
            cv_metrics['auc_roc'].append(float(roc_auc_score(y_val, y_prob)))
            
            # Keep the best model
            if cv_metrics['f1'][-1] > best_score:
                best_score = cv_metrics['f1'][-1]
                final_model = model
        
        if final_model is None:
            raise ValueError("Failed to train model in any fold")
            
        # Update training metrics
        self.training_metrics['cv_metrics'] = {
            metric: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            for metric, scores in cv_metrics.items()
        }
        
        return final_model
        
    def _update_feature_importance(self, feature_names: List[str]) -> None:
        """
        Update feature importance and selection.
        
        Args:
            feature_names: List of feature names
            
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Calculate feature importance
        importances = cast(npt.NDArray[np.float64], self.model.feature_importances_)
        self.feature_importance = pd.Series(
            importances,
            index=feature_names
        ).sort_values(ascending=False)
        
        # Select features
        threshold = float(self.feature_importance.median() * self.config.feature_importance_threshold)
        selector = SelectFromModel(self.model, threshold=threshold, prefit=True)
        mask = selector.get_support()
        self.selected_features = [name for name, selected in zip(feature_names, [] if mask is None else mask) if selected]
        
        # Log feature selection results
        self.logger.info(f"Selected {len(self.selected_features)} features")
        self.logger.info("Top 10 features by importance:\n" + 
                        self.feature_importance.head(10).to_string())
        
    def predict(self, X: pd.DataFrame) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]: Tuple of (predictions array, probability array)
            
        Raises:
            ValueError: If model is not trained
            RuntimeError: If prediction fails
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            return np.array(predictions, dtype=np.int32), np.array(probabilities, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
            
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training results.
        
        Returns:
            Dict[str, Any]: Dictionary containing training metrics and feature importance
        """
        if not self.training_metrics:
            return {}
            
        summary: Dict[str, Any] = {
            'training_time': float(self.training_metrics.get('training_time', 0.0)),
            'cv_metrics': self.training_metrics.get('cv_metrics', {}),
            'feature_importance': None,
            'selected_features': None,
            'last_training_time': self.last_training_time
        }
        
        if self.feature_importance is not None:
            summary['feature_importance'] = self.feature_importance.to_dict()
            
        if self.selected_features is not None:
            summary['selected_features'] = self.selected_features
            
        return summary
