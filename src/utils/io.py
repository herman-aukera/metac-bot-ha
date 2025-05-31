"""
Utility functions for I/O, parsing, etc.
"""
import yaml
import os
from typing import Dict, Any

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

def get_env_var(var_name: str, default: str | None = None) -> str | None:
    """Get an environment variable, with an optional default."""
    return os.getenv(var_name, default)

# Example of how you might structure your main config loading
# This would typically be called from your main.py or pipeline setup

def load_app_config(env: str = "development") -> Dict[str, Any]:
    """
    Load application configuration based on environment.
    Looks for config.<env>.yaml (e.g., config.development.yaml)
    Also loads secrets from .env file (though these are usually handled by direnv or similar)
    """
    # Determine config file path
    # Assuming your script runs from the project root or src/
    # Adjust path as necessary if your execution context is different
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Moves up two levels from src/utils
    config_dir = os.path.join(project_root, "config")
    config_file_name = f"config.{env}.yaml"
    config_file_path = os.path.join(config_dir, config_file_name)
    
    print(f"Loading configuration from: {config_file_path}")
    config = load_yaml_config(config_file_path)
    
    # Example: Load secrets (if you were not using python-dotenv or similar)
    # For production, secrets should be managed securely (e.g., Vault, AWS Secrets Manager)
    # METACULUS_TOKEN = get_env_var("METACULUS_TOKEN")
    # if METACULUS_TOKEN:
    #     config['metaculus_api'] = config.get('metaculus_api', {})
    #     config['metaculus_api']['token'] = METACULUS_TOKEN
        
    return config

# You might also have specific parsing functions here, e.g., for question formats, LLM outputs, etc.

