#!/usr/bin/env python3
"""Test script for the new DSPY-GEPA user interface.

This script tests the improved user interface to ensure everything works correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("=== Testing Imports ===")
    
    try:
        # Test convenience functions
        from dspy_gepa import (
            quick_optimize, optimize_translation, get_preset, 
            list_presets, setup_openai, check_setup
        )
        print("✅ Convenience functions imported successfully")
        
        # Test help functions
        from dspy_gepa import (
            print_welcome_message, show_help, list_available_functions
        )
        print("✅ Help functions imported successfully")
        
        # Test core classes
        from dspy_gepa import SimpleGEPA, MultiObjectiveGEPAAgent
        print("✅ Core classes imported successfully")
        
        # Test error handling
        from dspy_gepa import (
            ConfigurationError, OptimizationError, set_debug_mode
        )
        print("✅ Error handling imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_help_functions():
    """Test help and guidance functions."""
    print("\n=== Testing Help Functions ===")
    
    try:
        from dspy_gepa import print_welcome_message, list_available_functions
        
        print("\n--- Welcome Message ---")
        print_welcome_message()
        
        print("\n--- Available Functions ---")
        list_available_functions()
        
        return True
        
    except Exception as e:
        print(f"❌ Help functions failed: {e}")
        return False


def test_presets():
    """Test preset functionality."""
    print("\n=== Testing Presets ===")
    
    try:
        from dspy_gepa import (
            get_preset, list_presets, get_quick_preset, get_balanced_preset
        )
        
        # List presets
        presets = list_presets()
        print("Available presets:")
        for name, description in presets.items():
            print(f"  • {name}: {description}")
        
        # Test getting presets
        quick_preset = get_quick_preset()
        print(f"\nQuick preset: {quick_preset.name}")
        print(f"Description: {quick_preset.description}")
        print(f"Max generations: {quick_preset.max_generations}")
        
        balanced_preset = get_balanced_preset()
        print(f"\nBalanced preset: {balanced_preset.name}")
        print(f"Population size: {balanced_preset.population_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Presets failed: {e}")
        return False


def test_setup_check():
    """Test setup checking functionality."""
    print("\n=== Testing Setup Check ===")
    
    try:
        from dspy_gepa import check_setup, print_llm_status
        
        status = check_setup()
        print("Setup status:")
        for key, value in status.items():
            if key != 'recommendations':
                print(f"  {key}: {'✅' if value else '❌'}")
        
        if status['recommendations']:
            print("\nRecommendations:")
            for rec in status['recommendations']:
                print(f"  • {rec}")
        
        print("\n--- LLM Status ---")
        print_llm_status()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup check failed: {e}")
        return False


def test_examples():
    """Test examples functionality."""
    print("\n=== Testing Examples ===")
    
    try:
        from dspy_gepa.examples import show_example, list_all_examples
        
        print("\n--- Available Examples ---")
        list_all_examples()
        
        print("\n--- Quick Start Example ---")
        show_example('quick_start')
        
        return True
        
    except Exception as e:
        print(f"❌ Examples failed: {e}")
        return False


def test_error_handling():
    """Test error handling improvements."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from dspy_gepa import (
            ConfigurationError, OptimizationError, set_debug_mode, format_error_with_help
        )
        
        # Test debug mode
        print("\n--- Debug Mode Test ---")
        set_debug_mode(True)
        set_debug_mode(False)
        
        # Test error formatting
        print("\n--- Error Formatting Test ---")
        config_error = ConfigurationError("Test configuration error")
        formatted = format_error_with_help(config_error)
        print("Formatted error:")
        print(formatted)
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🐶 Testing DSPY-GEPA New Interface\n")
    
    tests = [
        ("Imports", test_imports),
        ("Help Functions", test_help_functions),
        ("Presets", test_presets),
        ("Setup Check", test_setup_check),
        ("Examples", test_examples),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("=== TEST SUMMARY ===")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The new interface is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
