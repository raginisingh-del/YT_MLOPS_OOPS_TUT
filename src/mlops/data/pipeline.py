"""Data pipeline for processing and transforming data."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data."""
        pass


class DataCleaner(DataProcessor):
    """Cleans data by handling missing values and outliers."""
    
    def __init__(self, fill_strategy: str = "mean"):
        """
        Initialize DataCleaner.
        
        Args:
            fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'drop')
        """
        self.fill_strategy = fill_strategy
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data.
        
        Args:
            data: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Cleaning data with strategy: {self.fill_strategy}")
        df = data.copy()
        
        if self.fill_strategy == "drop":
            df = df.dropna()
        else:
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    if self.fill_strategy == "mean":
                        df.loc[:, col] = df[col].fillna(df[col].mean())
                    elif self.fill_strategy == "median":
                        df.loc[:, col] = df[col].fillna(df[col].median())
                        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df


class FeatureEngineer(DataProcessor):
    """Engineers features from raw data."""
    
    def __init__(self, features_config: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            features_config: Configuration for feature engineering
        """
        self.features_config = features_config or {}
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from data.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("Engineering features")
        df = data.copy()
        
        # Example feature engineering operations
        # Can be extended based on features_config
        
        logger.info(f"Features engineered. Shape: {df.shape}")
        return df


class DataPipeline:
    """Pipeline for processing data through multiple stages."""
    
    def __init__(self, processors: Optional[list] = None):
        """
        Initialize DataPipeline.
        
        Args:
            processors: List of DataProcessor instances
        """
        self.processors = processors or []
        
    def add_processor(self, processor: DataProcessor):
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run data through all processors in the pipeline.
        
        Args:
            data: Input dataframe
            
        Returns:
            Processed dataframe
        """
        logger.info(f"Running pipeline with {len(self.processors)} processors")
        processed_data = data
        
        for i, processor in enumerate(self.processors):
            logger.info(f"Running processor {i+1}/{len(self.processors)}: {processor.__class__.__name__}")
            processed_data = processor.process(processed_data)
            
        logger.info("Pipeline completed")
        return processed_data


class DataLoader:
    """Loads and saves data from/to various sources."""
    
    @staticmethod
    def load_csv(filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Data loaded. Shape: {df.shape}")
        return df
    
    @staticmethod
    def save_csv(data: pd.DataFrame, filepath: Path, **kwargs):
        """
        Save data to CSV file.
        
        Args:
            data: Dataframe to save
            filepath: Path to save CSV file
            **kwargs: Additional arguments for pd.to_csv
        """
        logger.info(f"Saving data to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, **kwargs)
        logger.info(f"Data saved. Shape: {data.shape}")


class DataSplitter:
    """Splits data into train, validation, and test sets."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Initialize DataSplitter.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def split(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: test={self.test_size}, val={self.val_size}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test


class DataScaler:
    """Scales features for model training."""
    
    def __init__(self):
        """Initialize DataScaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Scaled dataframe
        """
        logger.info("Fitting and transforming data")
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Scaled dataframe
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        logger.info("Transforming data")
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
