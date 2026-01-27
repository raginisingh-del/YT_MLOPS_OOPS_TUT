"""Model deployment and serving module."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages registered models for deployment."""
    
    def __init__(self, registry_path: Path):
        """
        Initialize ModelRegistry.
        
        Args:
            registry_path: Path to the model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Model registry initialized at {self.registry_path}")
        
    def register_model(self, model_name: str, model_path: Path, 
                      version: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            version: Model version
            metadata: Additional metadata
        """
        if model_name not in self.models:
            self.models[model_name] = {}
            
        self.models[model_name][version] = {
            "path": model_path,
            "metadata": metadata or {},
        }
        logger.info(f"Model registered: {model_name} v{version}")
        
    def get_model_path(self, model_name: str, version: str = "latest") -> Path:
        """
        Get path to a registered model.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Path to the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
            
        if version == "latest":
            version = max(self.models[model_name].keys())
            
        if version not in self.models[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
            
        return self.models[model_name][version]["path"]
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of versions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        return list(self.models[model_name].keys())


class ModelServer:
    """Serves models for inference."""
    
    def __init__(self, model_path: Path):
        """
        Initialize ModelServer.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the model from disk."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully")
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input features
            
        Returns:
            Predictions
        """
        logger.info(f"Making predictions for {len(data)} samples")
        predictions = self.model.predict(data)
        logger.info("Predictions completed")
        return predictions
    
    def predict_batch(self, data_list: List[pd.DataFrame]) -> List[np.ndarray]:
        """
        Make batch predictions.
        
        Args:
            data_list: List of input feature dataframes
            
        Returns:
            List of predictions
        """
        logger.info(f"Making batch predictions for {len(data_list)} batches")
        predictions = [self.predict(data) for data in data_list]
        logger.info("Batch predictions completed")
        return predictions


class PredictionService:
    """High-level service for model predictions."""
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize PredictionService.
        
        Args:
            registry: ModelRegistry instance
        """
        self.registry = registry
        self.servers: Dict[str, ModelServer] = {}
        
    def load_model(self, model_name: str, version: str = "latest"):
        """
        Load a model for serving.
        
        Args:
            model_name: Name of the model
            version: Model version
        """
        model_path = self.registry.get_model_path(model_name, version)
        self.servers[model_name] = ModelServer(model_path)
        logger.info(f"Model {model_name} v{version} loaded for serving")
        
    def predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a loaded model.
        
        Args:
            model_name: Name of the model
            data: Input features
            
        Returns:
            Predictions
        """
        if model_name not in self.servers:
            raise ValueError(f"Model {model_name} not loaded. Call load_model first.")
            
        return self.servers[model_name].predict(data)
    
    def unload_model(self, model_name: str):
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.servers:
            del self.servers[model_name]
            logger.info(f"Model {model_name} unloaded")


class DeploymentManager:
    """Manages model deployment lifecycle."""
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize DeploymentManager.
        
        Args:
            registry: ModelRegistry instance
        """
        self.registry = registry
        self.deployed_models: Dict[str, str] = {}  # model_name -> version
        
    def deploy(self, model_name: str, version: str, environment: str = "production"):
        """
        Deploy a model to an environment.
        
        Args:
            model_name: Name of the model
            version: Model version
            environment: Deployment environment
        """
        # Verify model exists
        model_path = self.registry.get_model_path(model_name, version)
        
        # Mark as deployed
        deployment_key = f"{model_name}:{environment}"
        self.deployed_models[deployment_key] = version
        
        logger.info(f"Model {model_name} v{version} deployed to {environment}")
        
    def rollback(self, model_name: str, version: str, environment: str = "production"):
        """
        Rollback to a previous model version.
        
        Args:
            model_name: Name of the model
            version: Model version to rollback to
            environment: Deployment environment
        """
        self.deploy(model_name, version, environment)
        logger.info(f"Rolled back {model_name} to v{version} in {environment}")
        
    def get_deployed_version(self, model_name: str, environment: str = "production") -> str:
        """
        Get the currently deployed version.
        
        Args:
            model_name: Name of the model
            environment: Deployment environment
            
        Returns:
            Deployed version
        """
        deployment_key = f"{model_name}:{environment}"
        if deployment_key not in self.deployed_models:
            raise ValueError(f"No deployment found for {model_name} in {environment}")
        return self.deployed_models[deployment_key]
