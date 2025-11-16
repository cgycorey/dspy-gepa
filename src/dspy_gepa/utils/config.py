"""Configuration utilities for dspy-gepa.

Provides configuration management utilities including loading,
saving, and accessing configuration values.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .logging import get_logger


def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load LLM configuration from config file.
    
    Args:
        config_path: Path to configuration file. If None, uses default loading.
        
    Returns:
        LLM configuration dictionary
    """
    config = load_config(config_path)
    llm_config = config.get("llm", {})
    
    # Add environment variable support
    if "providers" in llm_config:
        for provider_name, provider_config in llm_config["providers"].items():
            # Support common environment variables
            if provider_name.lower() == "openai":
                if not provider_config.get("api_key"):
                    provider_config["api_key"] = os.getenv("OPENAI_API_KEY")
            elif provider_name.lower() == "anthropic":
                if not provider_config.get("api_key"):
                    provider_config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
    
    return llm_config


def get_default_llm_provider() -> str:
    """Get the default LLM provider from configuration.
    
    Returns:
        Default provider name
    """
    llm_config = load_llm_config()
    return llm_config.get("default_provider", "openai")


def get_provider_config(provider_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific LLM provider.
    
    Args:
        provider_name: Name of the provider. If None, uses default provider.
        
    Returns:
        Provider configuration dictionary
    """
    llm_config = load_llm_config()
    provider_name = provider_name or llm_config.get("default_provider", "openai")
    
    providers = llm_config.get("providers", {})
    if provider_name not in providers:
        _logger.warning(f"Provider {provider_name} not found in config")
        return {}
    
    return providers[provider_name].copy()


def is_llm_configured(provider_name: Optional[str] = None) -> bool:
    """Check if LLM is properly configured for a provider.
    
    Args:
        provider_name: Name of the provider to check. If None, checks default provider.
        
    Returns:
        True if LLM is configured, False otherwise
    """
    provider_config = get_provider_config(provider_name)
    
    if not provider_config:
        return False
    
    # Check for API key or other required authentication
    if "api_key" in provider_config:
        return bool(provider_config["api_key"])
    
    # Could check other auth methods here
    return True


_logger = get_logger(__name__)
_default_config: Optional[Dict[str, Any]] = None


def get_default_config() -> Dict[str, Any]:
    """Get default configuration settings.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "optimization": {
            "max_generations": 25,
            "population_size": 6,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8
        },
        "logging": {
            "level": "INFO",
            "include_timestamp": True
        },
        "objectives": {
            "default_weights": {
                "accuracy": 0.5,
                "efficiency": 0.3,
                "clarity": 0.2
            }
        },
        "mutation": {
            "adaptive": True,
            "strategies": ["text", "semantic", "structural"]
        },
        "selection": {
            "method": "pareto",
            "elitism_ratio": 0.1
        }
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file. If None, tries default locations.
        
    Returns:
        Configuration dictionary
    """
    global _default_config
    
    # If no path provided, try default locations
    if config_path is None:
        possible_paths = [
            "config.yaml",
            "config.yml",
            os.path.expanduser("~/.dspy_gepa/config.yaml"),
            os.path.expanduser("~/.dspy_gepa/config.yml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                _logger.info(f"Loading config from {config_path}")
                break
        
        if config_path is None:
            _logger.info("No config file found, using defaults")
            _default_config = get_default_config()
            return _default_config.copy()
    
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            _logger.warning(f"Config file not found: {config_path}, using defaults")
            _default_config = get_default_config()
            return _default_config.copy()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                loaded_config = yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        # Merge with defaults
        default_config = get_default_config()
        merged_config = merge_configs(default_config, loaded_config)
        
        _default_config = merged_config
        _logger.info(f"Configuration loaded successfully from {config_path}")
        return merged_config
        
    except Exception as e:
        _logger.error(f"Failed to load config from {config_path}: {e}")
        _logger.info("Using default configuration")
        _default_config = get_default_config()
        return _default_config.copy()


def save_config(
    config: Dict[str, Any], 
    config_path: Optional[str] = None
) -> bool:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save configuration. If None, saves to config.yaml
        
    Returns:
        True if successful, False otherwise
    """
    if config_path is None:
        config_path = "config.yaml"
    
    try:
        config_file = Path(config_path)
        
        # Create directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        _logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        _logger.error(f"Failed to save config to {config_path}: {e}")
        return False


def get_config_value(
    key: str, 
    default: Any = None, 
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """Get a configuration value using dot notation.
    
    Args:
        key: Configuration key (supports dot notation like 'optimization.max_generations')
        default: Default value if key not found
        config: Configuration dictionary to use. If None, loads current config.
        
    Returns:
        Configuration value or default
    """
    if config is None:
        global _default_config
        if _default_config is None:
            _default_config = load_config()
        config = _default_config
    
    # Support dot notation
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(
    key: str, 
    value: Any, 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Set a configuration value using dot notation.
    
    Args:
        key: Configuration key (supports dot notation)
        value: Value to set
        config: Configuration dictionary to modify. If None, creates new config.
        
    Returns:
        Modified configuration dictionary
    """
    if config is None:
        config = load_config().copy()
    
    keys = key.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the final value
    current[keys[-1]] = value
    
    return config


def merge_configs(
    base_config: Dict[str, Any], 
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with overrides
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate a configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ["optimization", "logging", "objectives"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate optimization parameters
    opt_config = config["optimization"]
    if "max_generations" in opt_config:
        mg = opt_config["max_generations"]
        if not isinstance(mg, int) or mg <= 0:
            raise ValueError("max_generations must be a positive integer")
    
    if "population_size" in opt_config:
        ps = opt_config["population_size"]
        if not isinstance(ps, int) or ps <= 0:
            raise ValueError("population_size must be a positive integer")
    
    return True


def reset_config() -> None:
    """Reset the cached configuration to force reload."""
    global _default_config
    _default_config = None
    _logger.info("Configuration cache reset")


def get_config_summary() -> str:
    """Get a summary of the current configuration.
    
    Returns:
        String summary of configuration
    """
    config = load_config()
    
    summary_parts = [
        "DSPY-GEPA Configuration Summary:",
        f"  Max generations: {get_config_value('optimization.max_generations')}",
        f"  Population size: {get_config_value('optimization.population_size')}",
        f"  Mutation rate: {get_config_value('optimization.mutation_rate')}",
        f"  Logging level: {get_config_value('logging.level')}",
    ]
    
    objectives = get_config_value('objectives.default_weights', {})
    if objectives:
        summary_parts.append(f"  Default objectives: {list(objectives.keys())}")
    
    return "\n".join(summary_parts)


_logger.debug("Configuration utilities module initialized")