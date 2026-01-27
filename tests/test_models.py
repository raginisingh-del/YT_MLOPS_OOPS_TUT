"""Tests for model training."""
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from mlops.models.trainer import ClassificationModel, RegressionModel


def test_classification_model():
    """Test ClassificationModel."""
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Split data
    split_idx = 80
    X_train, X_test = X_df[:split_idx], X_df[split_idx:]
    y_train, y_test = y_series[:split_idx], y_series[split_idx:]
    
    # Train model
    model = ClassificationModel(
        model_name="test_model",
        model_type="random_forest",
        n_estimators=10,
        random_state=42
    )
    
    model.train(X_train, y_train)
    assert model.is_trained
    
    # Make predictions
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
