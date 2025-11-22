"""Simplified configuration utilities for dspy-gepa."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def configure_litellm() -> None:
    """Configure LiteLLM to handle unsupported parameters gracefully."""
    try:
        import litellm
        # Configure LiteLLM to automatically drop unsupported parameters
        litellm.drop_params = True
        
        # Log successful configuration
        print("✅ LiteLLM configured with drop_params=True")
        
    except ImportError:
        # LiteLLM not available, skip configuration
        pass
    except Exception as e:
        # Configuration failed, but continue
        print(f"⚠️  LiteLLM configuration failed: {e}")


def get_default_llm_provider() -> str:
    """Get the default LLM provider with simple detection.
    
    Returns:
        Default provider name
    """
    # Check for API keys and return real providers
    if os.getenv('OPENAI_API_KEY'):
        return "openai"
    elif os.getenv('ANTHROPIC_API_KEY'):
        return "anthropic"
    else:
        return "mock"


def is_llm_configured(provider_name: Optional[str] = None) -> bool:
    """Check if LLM is properly configured.
    
    Args:
        provider_name: Name of the provider to check
        
    Returns:
        True if LLM is configured, False otherwise
    """
    provider = provider_name or get_default_llm_provider()
    
    if provider == "openai":
        return bool(os.getenv('OPENAI_API_KEY'))
    elif provider == "anthropic":
        return bool(os.getenv('ANTHROPIC_API_KEY'))
    else:
        return True  # Mock is always available


def get_provider_config(provider_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific LLM provider.
    
    Args:
        provider_name: Name of the provider. If None, uses default provider.
        
    Returns:
        Provider configuration dictionary
    """
    provider = provider_name or get_default_llm_provider()
    
    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # Updated to gpt-4o-mini
            "api_base": "https://api.openai.com/v1"
        }
    elif provider == "anthropic":
        return {
            "provider": "anthropic",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            "api_base": "https://api.anthropic.com"
        }
    else:  # mock
        return {
            "provider": "mock",
            "response_delay": 0.1
        }


def print_llm_status() -> None:
    """Print current LLM provider status."""
    provider = get_default_llm_provider()
    configured = is_llm_configured()
    
    print("\n" + "=" * 50)
    print("🤖 LLM PROVIDER STATUS")
    print("=" * 50)
    print(f"Current Provider: {provider.upper()}")
    print(f"Configured: {'✅ Yes' if configured else '❌ No'}")
    
    if provider == "mock":
        print("To use real LLMs, set environment variables:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
    
    print("=" * 50)


def get_model_compatibility_info(model: str) -> Dict[str, Any]:
    """Get model compatibility information for structured outputs.
    
    Args:
        model: Model name to check
        
    Returns:
        Dictionary with compatibility information
    """
    # Models that support structured outputs (response_format)
    structured_output_models = {
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-1106-preview",
        "gpt-4-0125-preview", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"
    }
    
    model_lower = model.lower()
    
    return {
        "model": model,
        "supports_structured_outputs": model_lower in structured_output_models,
        "recommended_for_gepa": model_lower in {"gpt-4o-mini", "gpt-4o", "gpt-4-turbo"},
        "fallback_models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"] if model_lower not in structured_output_models else []
    }


def validate_and_suggest_model(model: str) -> str:
    """Validate model and suggest alternatives if needed.
    
    Args:
        model: Model name to validate
        
    Returns:
        Validated model name (possibly replaced with compatible alternative)
    """
    compat_info = get_model_compatibility_info(model)
    
    if not compat_info["supports_structured_outputs"]:
        # Suggest a compatible model
        if compat_info["fallback_models"]:
            suggested_model = compat_info["fallback_models"][0]
            print(f"⚠️  Model '{model}' doesn't support structured outputs. Using '{suggested_model}' instead.")
            return suggested_model
        else:
            print(f"⚠️  Model '{model}' may not support structured outputs. Consider using gpt-4o-mini.")
    
    return model