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
        "llm": {
            "default_provider": "mock",
            "fallback_provider": "openai",
            "providers": {
                "mock": {
                    "enabled": True,
                    "response_delay": 0.1
                },
                "openai": {
                    "enabled": False,
                    "api_base": "https://api.openai.com/v1",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "api_key": None
                },
                "anthropic": {
                    "enabled": False,
                    "api_base": "https://api.anthropic.com",
                    "model": "claude-3-haiku-20240307",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "api_key": None
                }
            }
        },
        "experiment_tracking": {
            "enabled": False,
            "mlflow": {
                "enabled": False,
                "tracking_uri": "http://localhost:5000",
                "artifact_location": "./mlruns",
                "auto_start_server": False
            },
            "wandb": {
                "enabled": False,
                "project": "dspy-gepa-experiments",
                "entity": None
            },
            "local": {
                "enabled": True,
                "output_dir": "./experiments",
                "format": "json"
            }
        },
        "optimization": {
            "max_generations": 5,
            "population_size": 4,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
            "elitism_ratio": 0.2,
            "convergence_threshold": 0.01,
            "max_no_improvement": 3
        },
        "data": {
            "cache_dir": "./cache",
            "datasets_dir": "./datasets",
            "max_dataset_size": 1000,
            "auto_download": False,
            "use_synthetic_data": True
        },
        "evaluation": {
            "default_metrics": ["accuracy"],
            "batch_size": 8,
            "max_workers": 2,
            "timeout": 30,
            "use_mock_evaluation": True
        },
        "agent": {
            "default_max_retries": 2,
            "default_timeout": 15,
            "enable_caching": True,
            "cache_ttl": 1800,
            "max_cache_size": 100
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "./logs/dspy_gepa.log",
            "max_file_size": "10MB",
            "backup_count": 5,
            "console_output": True,
            "file_output": False
        },
        "performance": {
            "enable_profiling": False,
            "profile_dir": "./profiles",
            "memory_limit": "4GB",
            "max_execution_time": 300
        },
        "dspy": {
            "auto_install": False,
            "fallback_to_mock": True,
            "max_retries": 2,
            "timeout": 10
        },
        "error_handling": {
            "graceful_degradation": True,
            "continue_on_error": True,
            "log_errors": True,
            "raise_on_critical": False
        },
        "validation": {
            "validate_config_on_load": True,
            "validate_dependencies": True,
            "skip_optional_validation": True,
            "warn_on_missing_optional": True
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
        
        # Validate and fix configuration if enabled
        if merged_config.get("validation", {}).get("validate_config_on_load", True):
            try:
                validate_config(merged_config)
                # Adjust for service availability
                working_config = get_working_configuration()
                _default_config = working_config
                _logger.debug(f"Configuration loaded and validated from {config_path}")
                return working_config
            except Exception as e:
                _logger.warning(f"Configuration validation failed: {e}")
                if merged_config.get("error_handling", {}).get("graceful_degradation", True):
                    _logger.info("Using configuration with graceful degradation")
                    _default_config = merged_config
                    return merged_config
                else:
                    raise
        else:
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
    # Core required sections
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
    
    # Validate mutation and crossover rates
    if "mutation_rate" in opt_config:
        mr = opt_config["mutation_rate"]
        if not isinstance(mr, (int, float)) or not 0 <= mr <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
    
    if "crossover_rate" in opt_config:
        cr = opt_config["crossover_rate"]
        if not isinstance(cr, (int, float)) or not 0 <= cr <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")
    
    # Validate LLM configuration
    if "llm" in config:
        llm_config = config["llm"]
        
        # Check if default provider exists
        default_provider = llm_config.get("default_provider")
        if not default_provider:
            raise ValueError("llm.default_provider must be specified")
        
        providers = llm_config.get("providers", {})
        if default_provider not in providers:
            raise ValueError(f"Default provider '{default_provider}' not found in providers")
        
        # Validate the default provider
        provider_config = providers[default_provider]
        if not provider_config.get("enabled", True):
            _logger.warning(f"Default provider '{default_provider}' is disabled")
        
        # Check API keys for non-mock providers
        if default_provider != "mock" and default_provider != "local":
            if not provider_config.get("api_key") and not os.getenv(f"{default_provider.upper()}_API_KEY"):
                _logger.warning(f"No API key found for provider '{default_provider}'")
    
    # Validate experiment tracking configuration
    if "experiment_tracking" in config:
        tracking_config = config["experiment_tracking"]
        
        # If tracking is globally enabled, validate specific trackers
        if tracking_config.get("enabled", False):
            mlflow_config = tracking_config.get("mlflow", {})
            if mlflow_config.get("enabled", False):
                tracking_uri = mlflow_config.get("tracking_uri")
                if not tracking_uri:
                    raise ValueError("mlflow.tracking_uri must be specified when mlflow is enabled")
                
                # Warn if MLflow server might not be available
                if "localhost" in tracking_uri or "127.0.0.1" in tracking_uri:
                    _logger.warning(f"MLflow tracking URI {tracking_uri} points to localhost - ensure server is running")
            
            wandb_config = tracking_config.get("wandb", {})
            if wandb_config.get("enabled", False):
                try:
                    import wandb
                except ImportError:
                    _logger.error("wandb is enabled but not installed - install with 'pip install wandb'")
                    if config.get("validation", {}).get("raise_on_missing_optional", False):
                        raise ValueError("wandb enabled but package not installed")
    
    # Validate data configuration
    if "data" in config:
        data_config = config["data"]
        
        # Check if directories are writable
        for dir_key in ["cache_dir", "datasets_dir"]:
            dir_path = data_config.get(dir_key)
            if dir_path:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _logger.warning(f"Cannot create directory {dir_path}: {e}")
    
    # Validate logging configuration
    if "logging" in config:
        logging_config = config["logging"]
        
        # Check if log directory is writable if file logging is enabled
        if logging_config.get("file_output", False):
            log_file = logging_config.get("file")
            if log_file:
                try:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _logger.warning(f"Cannot create log directory for {log_file}: {e}")
    
    # Validate evaluation configuration
    if "evaluation" in config:
        eval_config = config["evaluation"]
        
        # Validate batch size and workers
        for key in ["batch_size", "max_workers"]:
            value = eval_config.get(key)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"evaluation.{key} must be a positive integer")
    
    _logger.info("Configuration validation completed successfully")
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
        f"  LLM provider: {get_config_value('llm.default_provider')}",
        f"  Max generations: {get_config_value('optimization.max_generations')}",
        f"  Population size: {get_config_value('optimization.population_size')}",
        f"  Mutation rate: {get_config_value('optimization.mutation_rate')}",
        f"  Logging level: {get_config_value('logging.level')}",
        f"  Experiment tracking: {get_config_value('experiment_tracking.enabled')}",
    ]
    
    # Add LLM provider status
    default_provider = get_config_value('llm.default_provider')
    provider_config = get_config_value(f'llm.providers.{default_provider}', {})
    if provider_config.get('enabled', True):
        summary_parts.append(f"  ✓ {default_provider} provider enabled")
    else:
        summary_parts.append(f"  ⚠ {default_provider} provider disabled")
    
    # Add tracking status
    tracking_enabled = get_config_value('experiment_tracking.enabled', False)
    if tracking_enabled:
        mlflow_enabled = get_config_value('experiment_tracking.mlflow.enabled', False)
        wandb_enabled = get_config_value('experiment_tracking.wandb.enabled', False)
        local_enabled = get_config_value('experiment_tracking.local.enabled', True)
        
        tracking_services = []
        if mlflow_enabled:
            tracking_services.append("MLflow")
        if wandb_enabled:
            tracking_services.append("W&B")
        if local_enabled:
            tracking_services.append("Local")
        
        summary_parts.append(f"  Active tracking: {', '.join(tracking_services)}")
    
    objectives = get_config_value('objectives.default_weights', {})
    if objectives:
        summary_parts.append(f"  Default objectives: {list(objectives.keys())}")
    
    return "\n".join(summary_parts)


def check_service_availability() -> Dict[str, bool]:
    """Check availability of optional services.
    
    Returns:
        Dictionary mapping service names to availability status
    """
    availability = {
        "mlflow": False,
        "wandb": False,
        "dspy": False,
        "openai": False,
        "anthropic": False,
        "local_tracking": True  # Always available
    }
    
    # Check MLflow
    try:
        import mlflow
        availability["mlflow"] = True
    except ImportError:
        pass
    
    # Check Weights & Biases
    try:
        import wandb
        availability["wandb"] = True
    except ImportError:
        pass
    
    # Check DSPY
    try:
        import dspy
        availability["dspy"] = True
    except ImportError:
        pass
    
    # Check OpenAI
    try:
        import openai
        availability["openai"] = True
    except ImportError:
        pass
    
    # Check Anthropic
    try:
        import anthropic
        availability["anthropic"] = True
    except ImportError:
        pass
    
    return availability


def get_working_configuration() -> Dict[str, Any]:
    """Get a configuration that works with available services.
    
    This function adjusts the configuration based on what services
    are actually available, ensuring graceful degradation.
    
    Returns:
        Working configuration dictionary
    """
    # Use the current merged config instead of calling load_config() again
    global _default_config
    if _default_config is not None:
        config = _default_config.copy()
    else:
        config = get_default_config()
    availability = check_service_availability()
    
    # Adjust experiment tracking based on availability
    if "experiment_tracking" in config:
        tracking_config = config["experiment_tracking"]
        
        # Disable MLflow if not available
        if not availability["mlflow"] and tracking_config.get("mlflow", {}).get("enabled", False):
            _logger.warning("MLflow not available, disabling MLflow tracking")
            tracking_config["mlflow"]["enabled"] = False
        
        # Disable W&B if not available
        if not availability["wandb"] and tracking_config.get("wandb", {}).get("enabled", False):
            _logger.warning("Weights & Biases not available, disabling W&B tracking")
            tracking_config["wandb"]["enabled"] = False
        
        # Ensure local tracking is enabled if others are disabled
        if not (tracking_config.get("mlflow", {}).get("enabled", False) or 
                tracking_config.get("wandb", {}).get("enabled", False)):
            tracking_config["local"]["enabled"] = True
            _logger.debug("Using local tracking for experiment logging")
    
    # Adjust LLM provider based on availability
    if "llm" in config:
        llm_config = config["llm"]
        default_provider = llm_config.get("default_provider")
        
        # Check if default provider is available
        if default_provider != "mock":
            if default_provider == "openai" and not availability["openai"]:
                _logger.warning("OpenAI not available, switching to mock provider")
                llm_config["default_provider"] = "mock"
            elif default_provider == "anthropic" and not availability["anthropic"]:
                _logger.warning("Anthropic not available, switching to mock provider")
                llm_config["default_provider"] = "mock"
    
    # Adjust DSPY configuration based on availability
    if "dspy" in config:
        dspy_config = config["dspy"]
        if not availability["dspy"] and not dspy_config.get("fallback_to_mock", True):
            _logger.warning("DSPY not available, enabling fallback to mock")
            dspy_config["fallback_to_mock"] = True
    
    return config


def validate_and_fix_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate configuration and fix common issues.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Validated and fixed configuration
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Validate basic structure
        validate_config(config)
        
        # Get working configuration based on service availability
        working_config = get_working_configuration()
        
        # Validate the working configuration
        validate_config(working_config)
        
        _logger.info("Configuration validated and optimized for available services")
        return working_config
        
    except Exception as e:
        _logger.error(f"Configuration validation failed: {e}")
        
        # Fall back to minimal working configuration
        _logger.info("Falling back to minimal configuration")
        return get_minimal_config()


def get_minimal_config() -> Dict[str, Any]:
    """Get minimal configuration that guaranteed works.
    
    Returns:
        Minimal working configuration
    """
    return {
        "llm": {
            "default_provider": "mock",
            "providers": {
                "mock": {"enabled": True, "response_delay": 0.1}
            }
        },
        "experiment_tracking": {
            "enabled": False,
            "local": {"enabled": True, "output_dir": "./experiments"}
        },
        "optimization": {
            "max_generations": 3,
            "population_size": 3,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7
        },
        "logging": {
            "level": "INFO",
            "console_output": True,
            "file_output": False
        },
        "error_handling": {
            "graceful_degradation": True,
            "continue_on_error": True
        },
        "validation": {
            "validate_config_on_load": False,
            "validate_dependencies": False
        }
    }


_logger.debug("Configuration utilities module initialized")