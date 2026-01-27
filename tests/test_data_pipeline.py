"""Tests for data processing pipeline."""
import pandas as pd
import pytest
from mlops.data.pipeline import DataCleaner, DataLoader, DataPipeline, DataScaler, DataSplitter


def test_data_cleaner():
    """Test DataCleaner."""
    # Create sample data with missing values
    data = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': [5, None, 7, 8]
    })
    
    cleaner = DataCleaner(fill_strategy="mean")
    cleaned = cleaner.process(data)
    
    assert cleaned.isnull().sum().sum() == 0


def test_data_pipeline():
    """Test DataPipeline."""
    data = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': [5, None, 7, 8]
    })
    
    pipeline = DataPipeline([
        DataCleaner(fill_strategy="mean"),
    ])
    
    processed = pipeline.run(data)
    assert processed.isnull().sum().sum() == 0


def test_data_splitter():
    """Test DataSplitter."""
    X = pd.DataFrame({'a': range(100), 'b': range(100)})
    y = pd.Series(range(100))
    
    splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
    
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)


def test_data_scaler():
    """Test DataScaler."""
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
    
    scaler = DataScaler()
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert abs(X_scaled['a'].mean()) < 1e-10  # Mean should be close to 0
