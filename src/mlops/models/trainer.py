"""Model training module with versioning support."""
import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize BaseModel.
        
        Args:
            model_name: Name of the model
            **kwargs: Model hyperparameters
        """
        self.model_name = model_name
        self.model = None
        self.hyperparameters = kwargs
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model."""
        pass
    
    def save(self, filepath: Path):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info("Model saved successfully")
        
    def load(self, filepath: Path):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        logger.info(f"Loading model from {filepath}")
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info("Model loaded successfully")


class ClassificationModel(BaseModel):
    """Classification model wrapper."""
    
    def __init__(self, model_name: str, model_type: str = "random_forest", **kwargs):
        """
        Initialize ClassificationModel.
        
        Args:
            model_name: Name of the model
            model_type: Type of classification model ('random_forest', 'logistic_regression')
            **kwargs: Model hyperparameters
        """
        super().__init__(model_name, **kwargs)
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name}")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Training completed")
        
    def predict(self, X: pd.DataFrame):
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        """
        Make probability predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the classification model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")
        y_pred = self.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


class RegressionModel(BaseModel):
    """Regression model wrapper."""
    
    def __init__(self, model_name: str, model_type: str = "random_forest", **kwargs):
        """
        Initialize RegressionModel.
        
        Args:
            model_name: Name of the model
            model_type: Type of regression model ('random_forest', 'linear_regression')
            **kwargs: Model hyperparameters
        """
        super().__init__(model_name, **kwargs)
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == "linear_regression":
            self.model = LinearRegression(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name}")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Training completed")
        
    def predict(self, X: pd.DataFrame):
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the regression model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")
        y_pred = self.predict(X_test)
        
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred),
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


class ModelVersioning:
    """Manages model versions using MLflow."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize ModelVersioning.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")
        
    def log_model(self, model: BaseModel, metrics: Dict[str, float], 
                  artifacts: Optional[Dict[str, Path]] = None):
        """
        Log model with MLflow.
        
        Args:
            model: Model to log
            metrics: Evaluation metrics
            artifacts: Additional artifacts to log
        """
        with mlflow.start_run(run_name=f"{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log hyperparameters
            mlflow.log_params(model.hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model.model, "model")
            
            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(str(path), name)
                    
            logger.info(f"Model logged to MLflow: {model.model_name}")
            
    def load_model(self, run_id: str):
        """
        Load model from MLflow.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from run: {run_id}")
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded from MLflow")
        return model


class ModelTrainer:
    """Orchestrates model training pipeline."""
    
    def __init__(self, model: BaseModel, versioning: Optional[ModelVersioning] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            model: Model to train
            versioning: ModelVersioning instance for tracking
        """
        self.model = model
        self.versioning = versioning
        
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Train and evaluate model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Test metrics
        """
        logger.info("Starting training and evaluation pipeline")
        
        # Train model
        self.model.train(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = self.model.evaluate(X_val, y_val)
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Log to MLflow if versioning is enabled
        if self.versioning:
            self.versioning.log_model(self.model, test_metrics)
            
        return test_metrics
