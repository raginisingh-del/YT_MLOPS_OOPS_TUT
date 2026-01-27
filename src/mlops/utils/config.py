"""Configuration management utilities."""
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration files."""
    
    def __init__(self, config_path: Path):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            self.load()
            
    def load(self):
        """Load configuration from file."""
        logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}
        logger.info("Configuration loaded successfully")
        
    def save(self):
        """Save configuration to file."""
        logger.info(f"Saving configuration to {self.config_path}")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info("Configuration saved successfully")
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict: Dictionary to update with
        """
        self.config.update(config_dict)
