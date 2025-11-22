#!/usr/bin/env python3
"""Test script for the new DSPY-GEPA user interface (mock version).

This script tests the improved user interface structure without requiring
heavy dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_module_structure():
    """Test that the module structure is correct."""
    print("=== Testing Module Structure ===")
    
    # Check that key files exist
    base_path = os.path.join(os.path.dirname(__file__), 'src', 'dspy_gepa')
    
    required_files = [
        '__init__.py',
        'presets.py',
        'examples.py',
        'convenience.py',
        'user_guide.py'
    ]
    
    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True


def test_presets_module():
    """Test the presets module without dependencies."""
    print("\n=== Testing Presets Module ===")
    
    try:
        # Read the presets file to check structure
        presets_path = os.path.join(
            os.path.dirname(__file__), 'src', 'dspy_gepa', 'presets.py'
        )
        
        with open(presets_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ('OptimizationPreset class', 'class OptimizationPreset'),
            ('PresetRegistry class', 'class PresetRegistry'),
            ('get_preset function', 'def get_preset'),
            ('list_presets function', 'def list_presets'),
            ('Quick preset', "name='quick'"),
            ('Balanced preset', "name='balanced'"),
            ('Quality preset', "name='quality'"),
            ('Translation preset', "name='translation'")
        ]
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                return False
        
        print("✅ Presets module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Presets module test failed: {e}")
        return False


def test_examples_module():
    """Test the examples module structure."""
    print("\n=== Testing Examples Module ===")
    
    try:
        examples_path = os.path.join(
            os.path.dirname(__file__), 'src', 'dspy_gepa', 'examples.py'
        )
        
        with open(examples_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('Quick start example', 'def quick_start_example'),
            ('Preset configuration example', 'def preset_configuration_example'),
            ('Multi-objective example', 'def multi_objective_example'),
            ('Translation example', 'def translation_task_example'),
            ('Code generation example', 'def code_generation_example'),
            ('Error handling example', 'def error_handling_example'),
            ('Show example function', 'def show_example'),
            ('List examples function', 'def list_all_examples')
        ]
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                return False
        
        print("✅ Examples module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Examples module test failed: {e}")
        return False


def test_convenience_module():
    """Test the convenience module structure."""
    print("\n=== Testing Convenience Module ===")
    
    try:
        convenience_path = os.path.join(
            os.path.dirname(__file__), 'src', 'dspy_gepa', 'convenience.py'
        )
        
        with open(convenience_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('quick_optimize function', 'def quick_optimize'),
            ('optimize_translation function', 'def optimize_translation'),
            ('optimize_summarization function', 'def optimize_summarization'),
            ('optimize_code_generation function', 'def optimize_code_generation'),
            ('multi_objective_optimize function', 'def multi_objective_optimize'),
            ('batch_optimize function', 'def batch_optimize'),
            ('compare_prompts function', 'def compare_prompts'),
            ('setup_openai function', 'def setup_openai'),
            ('check_setup function', 'def check_setup')
        ]
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                return False
        
        print("✅ Convenience module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Convenience module test failed: {e}")
        return False


def test_user_guide_module():
    """Test the user guide module structure."""
    print("\n=== Testing User Guide Module ===")
    
    try:
        guide_path = os.path.join(
            os.path.dirname(__file__), 'src', 'dspy_gepa', 'user_guide.py'
        )
        
        with open(guide_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('print_welcome_message function', 'def print_welcome_message'),
            ('show_help function', 'def show_help'),
            ('print_setup_help function', 'def print_setup_help'),
            ('print_examples_help function', 'def print_examples_help'),
            ('print_presets_help function', 'def print_presets_help'),
            ('print_troubleshooting_help function', 'def print_troubleshooting_help'),
            ('print_best_practices function', 'def print_best_practices'),
            ('interactive_setup function', 'def interactive_setup'),
            ('list_available_functions function', 'def list_available_functions')
        ]
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                return False
        
        print("✅ User guide module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ User guide module test failed: {e}")
        return False


def test_init_module():
    """Test the main __init__.py module structure."""
    print("\n=== Testing Main __init__.py Module ===")
    
    try:
        init_path = os.path.join(
            os.path.dirname(__file__), 'src', 'dspy_gepa', '__init__.py'
        )
        
        with open(init_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('Convenience functions import', 'from .convenience import'),
            ('Presets import', 'from .presets import'),
            ('User guide import', 'from .user_guide import'),
            ('Examples import', 'from . import examples'),
            ('Error handling import', 'from .core.error_handling import'),
            ('Welcome message', 'print_welcome_message()'),
            ('Organized __all__', '# === MOST COMMON FUNCTIONS')
        ]
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                return False
        
        print("✅ Main __init__.py module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Main __init__.py test failed: {e}")
        return False


def test_error_handling_improvements():
    """Test error handling improvements."""
    print("\n=== Testing Error Handling Improvements ===")
    
    try:
        error_path = os.path.join(
            os.path.dirname(__file__), 'src', 'dspy_gepa', 'core', 'error_handling.py'
        )
        
        with open(error_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('OptimizationError class', 'class OptimizationError'),
            ('LLMError class', 'class LLMError'),
            ('Helpful error decorator', 'def handle_common_errors'),
            ('Debug mode function', 'def set_debug_mode'),
            ('Error formatting function', 'def format_error_with_help'),
            ('User-friendly suggestions', '💡 Suggestions:')
        ]
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                return False
        
        print("✅ Error handling improvements are present")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🐶 Testing DSPY-GEPA New Interface Structure\n")
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Presets Module", test_presets_module),
        ("Examples Module", test_examples_module),
        ("Convenience Module", test_convenience_module),
        ("User Guide Module", test_user_guide_module),
        ("Main __init__.py", test_init_module),
        ("Error Handling", test_error_handling_improvements)
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
    print("\n" + "="*60)
    print("=== TEST SUMMARY ===")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All structural tests passed! The new interface is properly implemented.")
        print("\n💡 Next steps:")
        print("   1. Install dependencies (dspy-ai, openai, anthropic)")
        print("   2. Set up your API key")
        print("   3. Try: from dspy_gepa import quick_optimize")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
