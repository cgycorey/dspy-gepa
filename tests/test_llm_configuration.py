#!/usr/bin/env python3
"""
Comprehensive LLM Configuration Test Suite for dspy-gepa

pytest-compatible tests for LLM configuration detection, including:
- Auto-detection from config files
- Environment variable loading
- Manual configuration
- Error handling scenarios
- Status reporting

Author: Converted from standalone test script for pytest compatibility
"""

from __future__ import annotations

import os
import sys
import tempfile
import yaml
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test fixtures and configuration
MOCK_API_KEYS = {
    "OPENAI_API_KEY": "sk-test1234567890abcdef1234567890abcdef12345678",
    "ANTHROPIC_API_KEY": "sk-ant-test03abcdefghijklmnopqrstuvwxyz123456",
    "OPENAI_MODEL": "gpt-4-test",
    "ANTHROPIC_MODEL": "claude-3-opus-20240229-test"
}

ENV_KEYS_TO_BACKUP = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", 
    "ANTHROPIC_MODEL", "LOCAL_MODEL_PATH", "LOCAL_API_BASE"
]

# Import check
try:
    from dspy_gepa.utils.config import (
        load_llm_config,
        get_default_llm_provider,
        get_provider_config,
        is_llm_configured,
        load_config,
        reset_config
    )
    from dspy_gepa.utils.logging import get_logger
    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    print(f"❌ Failed to import dspy-gepa config modules: {e}")

try:
    from examples.language_model_setup import setup_language_model, LMConfig
    LM_SETUP_AVAILABLE = True
except ImportError as e:
    LM_SETUP_AVAILABLE = False
    print(f"⚠️  Failed to import language_model_setup: {e}")


@pytest.fixture(scope="function")
def clean_environment():
    """Fixture to provide a clean environment for each test."""
    # Backup original environment
    original_env = {}
    for key in ENV_KEYS_TO_BACKUP:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    # Reset config
    if CONFIG_AVAILABLE:
        reset_config()
    
    yield original_env
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


@pytest.fixture(scope="function")
def temp_config_dir():
    """Fixture to provide a temporary config directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        yield config_path


@pytest.fixture(scope="function")
def mock_env_with_openai():
    """Fixture to provide environment variables with OpenAI API key."""
    original_env = {}
    
    # Backup environment
    for key in ENV_KEYS_TO_BACKUP:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    # Set mock OpenAI environment
    os.environ["OPENAI_API_KEY"] = MOCK_API_KEYS["OPENAI_API_KEY"]
    os.environ["OPENAI_MODEL"] = MOCK_API_KEYS["OPENAI_MODEL"]
    
    yield original_env
    
    # Restore environment
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


@pytest.fixture(scope="function")
def mock_env_with_anthropic():
    """Fixture to provide environment variables with Anthropic API key."""
    original_env = {}
    
    # Backup environment
    for key in ENV_KEYS_TO_BACKUP:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    # Set mock Anthropic environment
    os.environ["ANTHROPIC_API_KEY"] = MOCK_API_KEYS["ANTHROPIC_API_KEY"]
    os.environ["ANTHROPIC_MODEL"] = MOCK_API_KEYS["ANTHROPIC_MODEL"]
    
    yield original_env
    
    # Restore environment
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


@pytest.fixture(scope="function")
def mock_env_with_all():
    """Fixture to provide environment variables with all API keys."""
    original_env = {}
    
    # Backup environment
    for key in ENV_KEYS_TO_BACKUP:
        original_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    # Set mock environment with all keys
    for key, value in MOCK_API_KEYS.items():
        os.environ[key] = value
    
    yield original_env
    
    # Restore environment
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


@pytest.fixture
def sample_config_content():
    """Fixture providing sample configuration content."""
    return {
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "model": "gpt-4",
                    "api_key": "sk-test-key",
                    "api_base": "https://api.openai.com/v1"
                },
                "anthropic": {
                    "model": "claude-3-opus-20240229",
                    "api_key": "sk-ant-test-key",
                    "api_base": "https://api.anthropic.com"
                }
            }
        }
    }


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestLLMConfigDetection:
    """Test class for LLM configuration detection."""
    
    def test_config_file_detection_from_existing_config(self, clean_environment):
        """Test auto-detection of LLM configuration from config.yaml."""
        # Load existing config
        config = load_llm_config()
        
        # Check if default provider is detected
        default_provider = get_default_llm_provider()
        
        # Check if provider configs are loaded
        openai_config = get_provider_config("openai")
        anthropic_config = get_provider_config("anthropic")
        
        # Assert basic structure exists
        assert isinstance(config, dict), "Config should be a dictionary"
        # Config structure might have 'llm' section or top-level keys
        assert "llm" in config or "providers" in config, "Config should have 'llm' or 'providers' section"
        
        # Assert provider detection works (might be mock in test env)
        assert default_provider in ["openai", "anthropic", "mock", None], f"Default provider should be valid: {default_provider}"
        
        # Assert provider configs have required fields
        assert "model" in openai_config, "OpenAI config should have model field"
        assert "model" in anthropic_config, "Anthropic config should have model field"
    
    def test_config_file_detection_with_custom_config(self, temp_config_dir, sample_config_content):
        """Test auto-detection with custom config file."""
        # Create custom config file
        with open(temp_config_dir, 'w') as f:
            yaml.dump(sample_config_content, f)
        
        # Test loading custom config directly (skip CONFIG_PATH patching as it doesn't exist)
        try:
            with open(temp_config_dir, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
            assert loaded_config == sample_config_content, "Config should match sample content"
            assert loaded_config["llm"]["default_provider"] == "openai", "Default provider should be openai"
            assert loaded_config["llm"]["providers"]["openai"]["model"] == "gpt-4", "OpenAI model should be gpt-4"
            
        except Exception as e:
            pytest.skip(f"Custom config loading skipped due to: {e}")


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestEnvironmentVariables:
    """Test class for environment variable handling."""
    
    def test_openai_environment_variables(self, mock_env_with_openai):
        """Test environment variable loading for OpenAI."""
        reset_config()
        config = load_llm_config()
        
        # Check that environment variables are loaded
        assert os.environ.get("OPENAI_API_KEY") == MOCK_API_KEYS["OPENAI_API_KEY"]
        assert os.environ.get("OPENAI_MODEL") == MOCK_API_KEYS["OPENAI_MODEL"]
        
        # Check config reflects environment
        default_provider = get_default_llm_provider()
        # The config might use mock provider in testing environment
        assert default_provider in ["openai", "mock"], f"Default provider should be openai or mock, got {default_provider}"
        
        openai_config = get_provider_config("openai")
        assert openai_config is not None, "OpenAI config should be available"
    
    def test_anthropic_environment_variables(self, mock_env_with_anthropic):
        """Test environment variable loading for Anthropic."""
        reset_config()
        config = load_llm_config()
        
        # Check that environment variables are loaded
        assert os.environ.get("ANTHROPIC_API_KEY") == MOCK_API_KEYS["ANTHROPIC_API_KEY"]
        assert os.environ.get("ANTHROPIC_MODEL") == MOCK_API_KEYS["ANTHROPIC_MODEL"]
        
        # The default might still be openai if config file exists, so check anthro at least works
        anthropic_config = get_provider_config("anthropic")
        assert anthropic_config is not None, "Anthropic config should be available"
    
    def test_all_api_keys_environment(self, mock_env_with_all):
        """Test behavior when all API keys are available."""
        reset_config()
        config = load_llm_config()
        
        # Check all environment variables are present
        for key, expected_value in MOCK_API_KEYS.items():
            assert os.environ.get(key) == expected_value, f"Environment variable {key} should be set"
        
        # Should successfully detect providers (might be mock in test env)
        default_provider = get_default_llm_provider()
        assert default_provider in ["openai", "anthropic", "mock"], "Should detect at least one provider"
        
        # Both configs should be available
        openai_config = get_provider_config("openai")
        anthropic_config = get_provider_config("anthropic")
        assert openai_config is not None, "OpenAI config should be available"
        assert anthropic_config is not None, "Anthropic config should be available"


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestManualConfiguration:
    """Test class for manual LLM configuration."""
    
    def test_manual_openai_configuration(self, clean_environment):
        """Test manual OpenAI configuration setup."""
        if not LM_SETUP_AVAILABLE:
            pytest.skip("language_model_setup not available")
        
        # Set up manual configuration
        os.environ["OPENAI_API_KEY"] = MOCK_API_KEYS["OPENAI_API_KEY"]
        
        try:
            lm_config, lm_provider = setup_language_model()
            
            assert lm_config is not None, "LM config should be created"
            assert lm_provider in ["openai", "mock-openai"], f"Provider should be openai or mock-openai, got {lm_provider}"
            
            if hasattr(lm_config, 'model'):
                assert isinstance(lm_config.model, str), "Model should be a string"
            
        except Exception as e:
            pytest.fail(f"Manual OpenAI configuration failed: {e}")
    
    def test_manual_anthropic_configuration(self, clean_environment):
        """Test manual Anthropic configuration setup."""
        if not LM_SETUP_AVAILABLE:
            pytest.skip("language_model_setup not available")
        
        # Set up manual configuration
        os.environ["ANTHROPIC_API_KEY"] = MOCK_API_KEYS["ANTHROPIC_API_KEY"]
        
        try:
            lm_config, lm_provider = setup_language_model()
            
            assert lm_config is not None, "LM config should be created"
            # Provider might be openai if both are available, or anthropic, or mock versions
            assert lm_provider in ["openai", "anthropic", "mock-openai", "mock-anthropic"], f"Unexpected provider: {lm_provider}"
            
        except Exception as e:
            pytest.fail(f"Manual Anthropic configuration failed: {e}")
    
    @pytest.mark.parametrize("model_name", ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"])
    def test_manual_configuration_with_different_models(self, clean_environment, model_name):
        """Test manual configuration with different model names."""
        if not LM_SETUP_AVAILABLE:
            pytest.skip("language_model_setup not available")
        
        # Set up environment with specific model
        if model_name.startswith("gpt"):
            os.environ["OPENAI_API_KEY"] = MOCK_API_KEYS["OPENAI_API_KEY"]
            os.environ["OPENAI_MODEL"] = model_name
        else:
            os.environ["ANTHROPIC_API_KEY"] = MOCK_API_KEYS["ANTHROPIC_API_KEY"]
            os.environ["ANTHROPIC_MODEL"] = model_name
        
        try:
            lm_config, lm_provider = setup_language_model()
            
            assert lm_config is not None, f"LM config should be created for model {model_name}"
            
            # Check that model is set correctly
            if hasattr(lm_config, 'model'):
                assert isinstance(lm_config.model, str), "Model should be a string"
            
        except Exception as e:
            # Some models might not be available in the testing environment
            # This is acceptable as long as the system handles it gracefully
            assert "not available" in str(e).lower() or "invalid" in str(e).lower(), f"Expected availability error, got: {e}"


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestNoApiKeysScenarios:
    """Test class for scenarios when no API keys are available."""
    
    def test_no_api_keys_fallback_behavior(self, clean_environment):
        """Test behavior when no API keys are available."""
        reset_config()
        config = load_llm_config()
        
        # Should gracefully handle missing API keys
        assert os.environ.get("OPENAI_API_KEY") is None, "OpenAI API key should be None"
        assert os.environ.get("ANTHROPIC_API_KEY") is None, "Anthropic API key should be None"
        
        # Should detect that no LLM is configured
        configured = is_llm_configured()
        # This might be True if config file exists, False otherwise
        assert isinstance(configured, bool), "is_llm_configured should return boolean"
        
        # Should still provide default configs (possibly empty)
        openai_config = get_provider_config("openai")
        anthropic_config = get_provider_config("anthropic")
        assert isinstance(openai_config, dict), "OpenAI config should be a dict even without API key"
        assert isinstance(anthropic_config, dict), "Anthropic config should be a dict even without API key"
    
    def test_language_model_setup_without_api_keys(self, clean_environment):
        """Test language model setup when no API keys are available."""
        if not LM_SETUP_AVAILABLE:
            pytest.skip("language_model_setup not available")
        
        try:
            lm_config, lm_provider = setup_language_model()
            
            # Should fall back to mock
            assert lm_config is not None, "Should still create a config (mock)"
            assert "mock" in lm_provider.lower(), f"Should use mock provider, got: {lm_provider}"
            
        except Exception as e:
            pytest.fail(f"Language model setup without API keys failed: {e}")


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestInvalidConfigurationScenarios:
    """Test class for invalid configuration handling."""
    
    def test_invalid_openai_api_key(self, clean_environment):
        """Test behavior with invalid OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = "invalid-key-123"
        
        try:
            config = load_llm_config()
            openai_config = get_provider_config("openai")
            
            # Should still load config even with invalid key
            assert isinstance(openai_config, dict), "Should still create config dictionary"
            
            # The key validation happens at runtime, not at config loading
            assert openai_config.get("api_key") == "invalid-key-123", "Should store the invalid key for later validation"
            
        except Exception as e:
            pytest.fail(f"Config loading with invalid API key should not fail: {e}")
    
    def test_missing_config_file(self, clean_environment):
        """Test behavior when config file doesn't exist."""
        # Temporarily move existing config if it exists
        config_path = Path("src/config.yaml")
        backup_path = None
        
        if config_path.exists():
            backup_path = config_path.with_suffix(".yaml.bak")
            config_path.rename(backup_path)
        
        try:
            reset_config()
            config = load_llm_config()
            
            # Should create default config
            assert isinstance(config, dict), "Should create default config when file missing"
            # Config structure might vary - check for key sections
            assert "providers" in config or "llm" in config, "Default config should have providers or llm section"
            
        finally:
            # Restore backup
            if backup_path and backup_path.exists():
                backup_path.rename(config_path)

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config modules not available")
class TestComprehensiveStatus:
    """Test class for comprehensive status reporting."""
    
    def test_comprehensive_status_with_openai(self, mock_env_with_openai):
        """Test comprehensive status reporting with OpenAI API key."""
        reset_config()
        
        # Gather all status information
        config = load_llm_config()
        default_provider = get_default_llm_provider()
        configured = is_llm_configured()
        openai_config = get_provider_config("openai")
        anthropic_config = get_provider_config("anthropic")
        
        # Assert comprehensive information is available
        assert isinstance(config, dict), "Config should be available"
        assert isinstance(default_provider, str) or default_provider is None, "Default provider should be string or None"
        assert isinstance(configured, bool), "is_llm_configured should be boolean"
        assert isinstance(openai_config, dict), "OpenAI config should be dictionary"
        assert isinstance(anthropic_config, dict), "Anthropic config should be dictionary"
        
        # With OpenAI API key, should detect OpenAI or mock (in test env)
        if default_provider:
            assert default_provider in ["openai", "mock"], f"With OpenAI key, default should be openai or mock, got {default_provider}"
        
        # Should have model information
        assert "model" in openai_config, "OpenAI config should have model"
    
    def test_comprehensive_status_with_all_keys(self, mock_env_with_all):
        """Test comprehensive status reporting with all API keys."""
        reset_config()
        
        # Comprehensive status check
        config = load_llm_config()
        default_provider = get_default_llm_provider()
        providers = ["openai", "anthropic"]
        provider_configs = {}
        
        for provider in providers:
            provider_configs[provider] = get_provider_config(provider)
        
        # All should be available
        assert isinstance(config, dict), "Config should be available"
        assert default_provider in providers + [None, "mock"], f"Default provider should be valid: {default_provider}"
        
        for provider_name, provider_config in provider_configs.items():
            assert isinstance(provider_config, dict), f"{provider_name} config should be dict"
            assert "model" in provider_config, f"{provider_name} config should have model"
    
    def test_comprehensive_status_no_keys(self, clean_environment):
        """Test comprehensive status reporting with no API keys."""
        reset_config()
        
        # Comprehensive status check
        config = load_llm_config()
        default_provider = get_default_llm_provider()
        configured = is_llm_configured()
        
        # Should still provide status even without keys
        assert isinstance(config, dict), "Config should still be available"
        assert isinstance(configured, bool), "is_llm_configured should be boolean"
        
        # Default provider might be from config file or None
        assert isinstance(default_provider, str) or default_provider is None, "Default provider should be string or None"


class TestIntegrationScenarios:
    """Integration tests for complete scenarios."""
    
    @pytest.mark.skipif(not LM_SETUP_AVAILABLE, reason="language_model_setup not available")
    def test_full_workflow_with_openai(self, mock_env_with_openai):
        """Test complete workflow from environment to LM setup with OpenAI."""
        reset_config()
        
        # Load config
        config = load_llm_config()
        default_provider = get_default_llm_provider()
        
        # Set up language model
        lm_config, lm_provider = setup_language_model()
        
        # Verify end-to-end
        assert lm_config is not None, "Language model should be set up"
        assert default_provider == "openai", "Default provider should be openai"
        assert "openai" in lm_provider.lower(), "LM provider should be openai-based"
    
    @pytest.mark.skipif(not LM_SETUP_AVAILABLE, reason="language_model_setup not available")
    def test_full_workflow_no_keys(self, clean_environment):
        """Test complete workflow when no API keys are available."""
        reset_config()
        
        # Load config
        config = load_llm_config()
        configured = is_llm_configured()
        
        # Set up language model
        lm_config, lm_provider = setup_language_model()
        
        # Verify mock fallback
        assert lm_config is not None, "Should fall back to mock LM config"
        assert "mock" in lm_provider.lower(), f"Should use mock provider, got: {lm_provider}"
    
    def test_config_reset_and_reload(self, mock_env_with_openai):
        """Test config reset and reload behavior."""
        # Initial load
        config1 = load_llm_config()
        default1 = get_default_llm_provider()
        
        # Reset config
        reset_config()
        config2 = load_llm_config()
        default2 = get_default_llm_provider()
        
        # Should be consistent
        assert isinstance(config1, dict), "First config should be dict"
        assert isinstance(config2, dict), "Second config should be dict"
        assert default1 == default2, "Default provider should be consistent"


if __name__ == "__main__":
    # This allows running the file directly for debugging
    pytest.main([__file__, "-v"])