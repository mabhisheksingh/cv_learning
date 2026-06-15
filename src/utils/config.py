import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration management for training."""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to YAML config file
        """
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.set_default_config()
    
    def set_default_config(self):
        """Set default configuration values."""
        self.config = {
            'data': {
                'image_size': 224,
                'batch_size': 32,
                'num_workers': 4,
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            },
            'model': {
                'type': 'custom_cnn',
                'num_classes': 2,
                'dropout': 0.5
            },
            'training': {
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'checkpoint_interval': 5,
                'early_stopping_patience': 10
            },
            'paths': {
                'data_dir': './data',
                'checkpoint_dir': './checkpoints',
                'log_dir': './logs',
                'saved_models_dir': './saved_models'
            },
            'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        }
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def save_config(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports nested keys with dots)."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __repr__(self) -> str:
        return str(self.config)
