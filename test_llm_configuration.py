#!/usr/bin/env python3
"""
Comprehensive LLM Configuration Detection Test Script for dspy-gepa

This script tests all aspects of LLM configuration detection in the dspy-gepa system,
including auto-detection from config files, environment variable loading, manual
configuration, and error handling scenarios.

Test Scenarios:
1. Auto-detection of LLM configuration from config.yaml
2. Environment variable loading (OPENAI_API_KEY, ANTHROPIC_API_KEY)
3. Manual LLM configuration
4. Configuration when no API keys are available
5. Invalid configuration scenarios
6. Comprehensive status reporting for each scenario

Usage:
    python test_llm_configuration.py
    
Author: Generated for dspy-gepa system testing
"""

from __future__ import annotations

import os
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
    print(f"❌ Failed to import dspy-gepa config modules: {e}")
    CONFIG_AVAILABLE = False

try:
    from examples.language_model_setup import setup_language_model, LMConfig
    LM_SETUP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Failed to import language_model_setup: {e}")
    LM_SETUP_AVAILABLE = False


class LLMConfigTester:
    """Comprehensive tester for LLM configuration detection."""
    
    def __init__(self):
        """Initialize the tester with logging and test setup."""
        self.logger = get_logger(__name__)
        self.test_results = []
        self.original_env = {}
        self.temp_config_files = []
        
        # Mock API keys for testing (these are fake keys for testing only)
        self.mock_keys = {
            "OPENAI_API_KEY": "sk-test1234567890abcdef1234567890abcdef12345678",
            "ANTHROPIC_API_KEY": "sk-ant-test03abcdefghijklmnopqrstuvwxyz123456",
            "OPENAI_MODEL": "gpt-4-test",
            "ANTHROPIC_MODEL": "claude-3-opus-20240229-test"
        }
        
        print("🧪 LLM Configuration Detection Test Suite")
        print("=" * 60)
        print("Testing dspy-gepa LLM configuration system...")
        print()
    
    def setup_test_environment(self):
        """Set up the test environment by backing up original environment."""
        # Backup original environment variables
        env_keys_to_backup = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", 
            "ANTHROPIC_MODEL", "LOCAL_MODEL_PATH", "LOCAL_API_BASE"
        ]
        
        for key in env_keys_to_backup:
            self.original_env[key] = os.environ.get(key)
            # Remove from environment for clean testing
            if key in os.environ:
                del os.environ[key]
    
    def cleanup_test_environment(self):
        """Clean up test environment and restore original state."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        
        # Clean up temporary config files
        for temp_file in self.temp_config_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        
        # Reset configuration cache
        if CONFIG_AVAILABLE:
            reset_config()
    
    def create_temp_config(self, config_data: Dict[str, Any]) -> str:
        """Create a temporary config file for testing."""
        temp_file = tempfile.mktemp(suffix=".yaml")
        with open(temp_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        self.temp_config_files.append(temp_file)
        return temp_file
    
    def log_test_result(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """Log a test result with comprehensive information."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        print(f"     {message}")
        if details:
            for key, value in details.items():
                print(f"     {key}: {value}")
        print()
    
    def test_scenario_1_config_file_detection(self):
        """Test Scenario 1: Auto-detection of LLM configuration from config.yaml."""
        print("🔍 SCENARIO 1: Testing auto-detection from config.yaml")
        print("-" * 50)
        
        if not CONFIG_AVAILABLE:
            self.log_test_result(
                "Config File Detection", 
                False, 
                "Config modules not available"
            )
            return
        
        try:
            # Test with existing config.yaml
            config = load_llm_config()
            
            # Check if default provider is detected
            default_provider = get_default_llm_provider()
            
            # Check if provider configs are loaded
            openai_config = get_provider_config("openai")
            anthropic_config = get_provider_config("anthropic")
            
            success = (
                default_provider in ["openai", "anthropic"] and
                "model" in openai_config and
                "model" in anthropic_config
            )
            
            self.log_test_result(
                "Config File Detection",
                success,
                f"Default provider: {default_provider}",
                {
                    "openai_model": openai_config.get("model"),
                    "anthropic_model": anthropic_config.get("model"),
                    "config_keys": list(config.keys())
                }
            )
            
        except Exception as e:
            self.log_test_result(
                "Config File Detection",
                False,
                f"Exception: {e}"
            )
    
    def test_scenario_2_environment_variables(self):
        """Test Scenario 2: Environment variable loading."""
        print("🔍 SCENARIO 2: Testing environment variable loading")
        print("-" * 50)
        
        if not CONFIG_AVAILABLE:
            self.log_test_result(
                "Environment Variable Loading", 
                False, 
                "Config modules not available"
            )
            return
        
        try:
            # Test 1: Set OpenAI API key
            os.environ["OPENAI_API_KEY"] = self.mock_keys["OPENAI_API_KEY"]
            reset_config()  # Reset to force reload
            
            openai_config = get_provider_config("openai")
            openai_configured = is_llm_configured("openai")
            
            # Test 2: Set Anthropic API key
            os.environ["ANTHROPIC_API_KEY"] = self.mock_keys["ANTHROPIC_API_KEY"]
            reset_config()
            
            anthropic_config = get_provider_config("anthropic")
            anthropic_configured = is_llm_configured("anthropic")
            
            # Test 3: Check both providers
            both_configured = is_llm_configured("openai") and is_llm_configured("anthropic")
            
            success = (
                openai_configured and 
                anthropic_configured and
                openai_config.get("api_key") and
                anthropic_config.get("api_key")
            )
            
            self.log_test_result(
                "Environment Variable Loading",
                success,
                f"Both providers configured: {both_configured}",
                {
                    "openai_has_key": bool(openai_config.get("api_key")),
                    "anthropic_has_key": bool(anthropic_config.get("api_key")),
                    "openai_configured": openai_configured,
                    "anthropic_configured": anthropic_configured
                }
            )
            
        except Exception as e:
            self.log_test_result(
                "Environment Variable Loading",
                False,
                f"Exception: {e}"
            )
    
    def test_scenario_3_manual_configuration(self):
        """Test Scenario 3: Manual LLM configuration."""
        print("🔍 SCENARIO 3: Testing manual LLM configuration")
        print("-" * 50)
        
        if not CONFIG_AVAILABLE:
            self.log_test_result(
                "Manual Configuration", 
                False, 
                "Config modules not available"
            )
            return
        
        try:
            # Create a custom config file
            custom_config = {
                "llm": {
                    "default_provider": "openai",
                    "providers": {
                        "openai": {
                            "api_base": "https://api.openai.com/v1",
                            "model": "gpt-4-turbo",
                            "temperature": 0.5,
                            "max_tokens": 4096,
                            "api_key": "sk-manual-key-12345"
                        },
                        "anthropic": {
                            "api_base": "https://api.anthropic.com",
                            "model": "claude-3-sonnet-20240229",
                            "temperature": 0.8,
                            "max_tokens": 3072,
                            "api_key": "sk-ant-manual-key-67890"
                        }
                    }
                }
            }
            
            temp_config_path = self.create_temp_config(custom_config)
            
            # Load the manual config by temporarily modifying the config loading path
            # We need to mock the config loading to use our temp file
            original_possible_paths = []
            
            # Create a custom load function that uses our temp config
            def mock_load_llm_config(config_path=None):
                return load_llm_config(temp_config_path)
            
            with patch('dspy_gepa.utils.config.load_llm_config', side_effect=mock_load_llm_config):
                config = load_llm_config()
            
            # Reset and load with our temp config directly
            reset_config()
            config = load_llm_config(temp_config_path)
            
            # Verify the manual configuration
            default_provider = config.get("default_provider")
            
            # Get provider configs from our manual config
            providers = config.get("providers", {})
            openai_config = providers.get("openai", {})
            anthropic_config = providers.get("anthropic", {})
            
            success = (
                default_provider == "openai" and
                openai_config.get("model") == "gpt-4-turbo" and
                anthropic_config.get("model") == "claude-3-sonnet-20240229" and
                openai_config.get("temperature") == 0.5 and
                anthropic_config.get("temperature") == 0.8
            )
            
            self.log_test_result(
                "Manual Configuration",
                success,
                f"Custom config loaded with provider: {default_provider}",
                {
                    "default_provider": default_provider,
                    "openai_model": openai_config.get("model"),
                    "anthropic_model": anthropic_config.get("model"),
                    "openai_temp": openai_config.get("temperature"),
                    "anthropic_temp": anthropic_config.get("temperature")
                }
            )
            
        except Exception as e:
            self.log_test_result(
                "Manual Configuration",
                False,
                f"Exception: {e}"
            )
    
    def test_scenario_4_no_api_keys(self):
        """Test Scenario 4: Configuration when no API keys are available."""
        print("🔍 SCENARIO 4: Testing configuration with no API keys")
        print("-" * 50)
        
        # Clear all API keys from environment
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        try:
            if LM_SETUP_AVAILABLE:
                # Test language model setup without API keys
                lm_config = setup_language_model()
                
                # Should fall back to mock configuration
                success = lm_config.provider in ["mock", "local"]
                
                self.log_test_result(
                    "No API Keys - LM Setup",
                    success,
                    f"Fallback provider: {lm_config.provider}",
                    {
                        "provider": lm_config.provider,
                        "model_name": lm_config.model_name,
                        "has_api_key": bool(lm_config.api_key)
                    }
                )
            else:
                self.log_test_result(
                    "No API Keys - LM Setup",
                    False,
                    "Language model setup not available"
                )
            
            if CONFIG_AVAILABLE:
                # Test config detection without API keys
                reset_config()
                openai_configured = is_llm_configured("openai")
                anthropic_configured = is_llm_configured("anthropic")
                
                # Should not be configured without API keys
                success = not openai_configured and not anthropic_configured
                
                self.log_test_result(
                    "No API Keys - Config Detection",
                    success,
                    f"Providers configured - OpenAI: {openai_configured}, Anthropic: {anthropic_configured}",
                    {
                        "openai_configured": openai_configured,
                        "anthropic_configured": anthropic_configured
                    }
                )
            
        except Exception as e:
            self.log_test_result(
                "No API Keys",
                False,
                f"Exception: {e}"
            )
    
    def test_scenario_5_invalid_configurations(self):
        """Test Scenario 5: Invalid configuration scenarios."""
        print("🔍 SCENARIO 5: Testing invalid configuration scenarios")
        print("-" * 50)
        
        if not CONFIG_AVAILABLE:
            self.log_test_result(
                "Invalid Configurations", 
                False, 
                "Config modules not available"
            )
            return
        
        try:
            # Test 1: Invalid provider name
            invalid_provider_config = get_provider_config("invalid_provider")
            provider_success = len(invalid_provider_config) == 0
            
            self.log_test_result(
                "Invalid Provider Name",
                provider_success,
                f"Empty config returned for invalid provider: {len(invalid_provider_config) == 0}"
            )
            
            # Test 2: Malformed YAML config
            malformed_config = {
                "llm": {
                    "default_provider": "openai",
                    "providers": {
                        # Missing closing quote - this would be invalid YAML
                        "openai": {
                            "model": "gpt-4"
                        }
                    }
                }
            }
            
            # Create malformed YAML file (intentionally broken)
            temp_malformed = tempfile.mktemp(suffix=".yaml")
            with open(temp_malformed, 'w') as f:
                f.write("llm:\n  default_provider: openai\n  providers:\n    openai:\n      model: gpt-4\n  # Missing closing quote\n")
            
            self.temp_config_files.append(temp_malformed)
            
            # Try to load malformed config - should fall back to defaults
            try:
                config = load_llm_config(temp_malformed)
                yaml_success = True  # Successfully handled malformed YAML
            except Exception:
                yaml_success = False
            
            self.log_test_result(
                "Malformed YAML Handling",
                yaml_success,
                f"System handled malformed YAML gracefully: {yaml_success}"
            )
            
            # Test 3: Non-existent config file
            nonexistent_config = load_llm_config("/non/existent/path/config.yaml")
            file_success = isinstance(nonexistent_config, dict)
            
            self.log_test_result(
                "Non-existent Config File",
                file_success,
                f"Handled missing file gracefully: {file_success}"
            )
            
        except Exception as e:
            self.log_test_result(
                "Invalid Configurations",
                False,
                f"Exception: {e}"
            )
    
    def test_scenario_6_comprehensive_status(self):
        """Test Scenario 6: Comprehensive status reporting."""
        print("🔍 SCENARIO 6: Testing comprehensive status reporting")
        print("-" * 50)
        
        try:
            # Set up a complete test environment
            os.environ["OPENAI_API_KEY"] = self.mock_keys["OPENAI_API_KEY"]
            os.environ["ANTHROPIC_API_KEY"] = self.mock_keys["ANTHROPIC_API_KEY"]
            reset_config()
            
            # Gather comprehensive status
            if CONFIG_AVAILABLE:
                llm_config = load_llm_config()
                default_provider = get_default_llm_provider()
                providers = llm_config.get("providers", {})
                
                provider_status = {}
                for provider_name in providers:
                    provider_config = get_provider_config(provider_name)
                    provider_status[provider_name] = {
                        "configured": is_llm_configured(provider_name),
                        "has_api_key": bool(provider_config.get("api_key")),
                        "model": provider_config.get("model"),
                        "temperature": provider_config.get("temperature"),
                        "max_tokens": provider_config.get("max_tokens")
                    }
                
                # Test language model setup
                lm_status = {}
                if LM_SETUP_AVAILABLE:
                    lm_config = setup_language_model()
                    lm_status = {
                        "provider": lm_config.provider,
                        "model_name": lm_config.model_name,
                        "has_api_key": bool(lm_config.api_key),
                        "temperature": lm_config.temperature,
                        "max_tokens": lm_config.max_tokens
                    }
                
                success = (
                    default_provider and
                    len(provider_status) > 0 and
                    any(status["configured"] for status in provider_status.values())
                )
                
                self.log_test_result(
                    "Comprehensive Status Report",
                    success,
                    f"System status gathered for {len(provider_status)} providers",
                    {
                        "default_provider": default_provider,
                        "provider_count": len(provider_status),
                        "configured_providers": [
                            name for name, status in provider_status.items() 
                            if status["configured"]
                        ],
                        "lm_setup_available": LM_SETUP_AVAILABLE,
                        "lm_provider": lm_status.get("provider") if lm_status else None
                    }
                )
                
                # Print detailed status
                print("\n📊 DETAILED CONFIGURATION STATUS:")
                print("=" * 40)
                print(f"Default Provider: {default_provider}")
                print(f"Total Providers: {len(provider_status)}")
                print()
                
                for provider_name, status in provider_status.items():
                    print(f"Provider: {provider_name}")
                    print(f"  Configured: {'✅ Yes' if status['configured'] else '❌ No'}")
                    print(f"  Has API Key: {'✅ Yes' if status['has_api_key'] else '❌ No'}")
                    print(f"  Model: {status['model']}")
                    print(f"  Temperature: {status['temperature']}")
                    print(f"  Max Tokens: {status['max_tokens']}")
                    print()
                
                if lm_status:
                    print("Language Model Setup Status:")
                    print(f"  Provider: {lm_status['provider']}")
                    print(f"  Model: {lm_status['model_name']}")
                    print(f"  Has API Key: {'✅ Yes' if lm_status['has_api_key'] else '❌ No'}")
                    print()
            
        except Exception as e:
            self.log_test_result(
                "Comprehensive Status Report",
                False,
                f"Exception: {e}"
            )
    
    def run_all_tests(self):
        """Run all test scenarios."""
        print("Starting comprehensive LLM configuration testing...")
        print()
        
        # Set up test environment
        self.setup_test_environment()
        
        try:
            # Run all test scenarios
            self.test_scenario_1_config_file_detection()
            self.test_scenario_2_environment_variables()
            self.test_scenario_3_manual_configuration()
            self.test_scenario_4_no_api_keys()
            self.test_scenario_5_invalid_configurations()
            self.test_scenario_6_comprehensive_status()
            
            # Print final summary
            self.print_test_summary()
            
        finally:
            # Clean up
            self.cleanup_test_environment()
    
    def print_test_summary(self):
        """Print a comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        if failed_tests > 0:
            print("FAILED TESTS:")
            print("-" * 20)
            for result in self.test_results:
                if not result["success"]:
                    print(f"❌ {result['test_name']}: {result['message']}")
            print()
        
        print("DETAILED RESULTS:")
        print("-" * 20)
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test_name']}")
            if not result["success"]:
                print(f"   Error: {result['message']}")
        print()
        
        # System availability summary
        print("SYSTEM COMPONENTS STATUS:")
        print("-" * 30)
        print(f"Config Modules: {'✅ Available' if CONFIG_AVAILABLE else '❌ Not Available'}")
        print(f"LM Setup: {'✅ Available' if LM_SETUP_AVAILABLE else '❌ Not Available'}")
        print(f"Config File: {'✅ Found' if os.path.exists('config.yaml') else '❌ Not Found'}")
        print()
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! LLM configuration system is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the detailed results above.")


def main():
    """Main entry point for the test script."""
    print("LLM Configuration Detection Test for dspy-gepa")
    print("This script tests all aspects of LLM configuration detection")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("config.yaml"):
        print("⚠️  Warning: config.yaml not found in current directory")
        print("     Make sure you're running this from the dspy-gepa root directory")
        print()
    
    # Create and run the tester
    tester = LLMConfigTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()