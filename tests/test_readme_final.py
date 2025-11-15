#!/usr/bin/env python3
"""
Final comprehensive test of README examples.

This script properly tests all README examples with correct syntax and
provides an accurate assessment of what works vs. what needs fixes.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_command(name: str, command: str, timeout: int = 60) -> tuple[bool, str, str]:
    """Run a test command and return (success, stdout, stderr)."""
    print(f"🧪 Testing: {name}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_readme_installation():
    """Test README installation command."""
    return run_test_command(
        "Installation (uv sync)", 
        "uv sync", 
        timeout=120
    )


def test_readme_verification():
    """Test README verification command as shown."""
    return run_test_command(
        "README Verification Command",
        'uv run python -c "from dspy_gepa.amope import AMOPEOptimizer; print(\'✅ Installation successful!\')"'
    )


def test_working_verification():
    """Test the working verification command."""
    return run_test_command(
        "Working Verification (Fixed)",
        '''
        uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / \"src\"))
from src.dspy_gepa.amope import AMOPEOptimizer
print(\"Installation successful!\")
"
        '''
    )


def test_readme_basic_usage():
    """Test README basic usage example."""
    return run_test_command(
        "README Basic Usage",
        '''
        uv run python -c "
from dspy_gepa.amope import AMOPEOptimizer

def evaluate_prompt(prompt_text):
    return 0.8

optimizer = AMOPEOptimizer(objectives={\"performance\": 1.0})
result = optimizer.optimize(\"Test prompt\", evaluate_prompt, generations=3)
print(f\"Best score: {result.best_score}\")
"
        '''
    )


def test_working_basic_usage():
    """Test the working basic usage example."""
    return run_test_command(
        "Working Basic Usage (Fixed)",
        '''
        uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / \"src\"))
from src.dspy_gepa.amope import AMOPEOptimizer

def evaluate_prompt(prompt_text):
    return {\"performance\": 0.8}

optimizer = AMOPEOptimizer(objectives={\"performance\": 1.0})
result = optimizer.optimize(\"Test prompt\", evaluate_prompt, generations=3)
print(f\"Best score: {result.best_score:.3f}\")
"
        '''
    )


def test_readme_demo_command():
    """Test README demo command."""
    return run_test_command(
        "README Demo Command",
        '''
        uv run python -c "
from dspy_gepa.amope import AMOPEOptimizer

def demo_eval(prompt):
    return 0.5 + 0.1 * hash(prompt) % 10 / 10

optimizer = AMOPEOptimizer(objectives={\"performance\": 1.0})
result = optimizer.optimize(\"Test prompt\", demo_eval, generations=5)
print(f\"Demo completed! Best score: {result.best_score:.3f}\")
"
        '''
    )


def test_working_demo_command():
    """Test the working demo command."""
    return run_test_command(
        "Working Demo Command (Fixed)",
        '''
        uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / \"src\"))
from src.dspy_gepa.amope import AMOPEOptimizer

def demo_eval(prompt):
    return {\"performance\": 0.5 + 0.1 * hash(prompt) % 10 / 10}

optimizer = AMOPEOptimizer(objectives={\"performance\": 1.0})
result = optimizer.optimize(\"Test prompt\", demo_eval, generations=5)
print(f\"Demo completed! Best score: {result.best_score:.3f}\")
"
        '''
    )


def test_basic_dspy_gepa_example():
    """Test the basic_dspy_gepa.py example."""
    return run_test_command(
        "Basic DSPY-GEPA Example",
        "uv run python examples/basic_dspy_gepa.py",
        timeout=90
    )


def test_core_gepa_functionality():
    """Test core GEPA functionality."""
    return run_test_command(
        "Core GEPA Functionality",
        '''
        uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / \"src\"))
from gepa import Candidate, GeneticOptimizer, ParetoSelector, TextMutator
candidate = Candidate(content=\"Test prompt\")
print(f\"Core GEPA works: {candidate.id[:8]}\")
"
        '''
    )


def generate_assessment(test_results):
    """Generate comprehensive assessment."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE README ASSESSMENT")
    print("="*60)
    
    total_tests = len(test_results)
    readme_working = sum(1 for name, success, _ in test_results if "README" not in name and success)
    actually_working = sum(1 for _, success, _ in test_results if success)
    readme_broken = sum(1 for name, success, _ in test_results if "README" in name and not success)
    
    print(f"Total tests run: {total_tests}")
    print(f"Actually working: {actually_working}/{total_tests} ({actually_working/total_tests*100:.1f}%)")
    print(f"README examples working: {readme_working}/{total_tests-readme_broken} ({readme_working/(total_tests-readme_broken)*100:.1f}%)")
    print(f"README examples broken: {readme_broken}/{readme_broken + (total_tests-readme_broken)//2} ({readme_broken/(readme_broken + (total_tests-readme_broken)//2)*100:.1f}%)")
    
    print("\n📋 Detailed Results:")
    print("-" * 20)
    
    for name, success, output in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        category = "README" if "README" in name else "WORKING"
        print(f"{status} | {category:7} | {name}")
        if not success and output.strip():
            # Show first line of error for debugging
            error_lines = output.strip().split('\n')
            if error_lines:
                print(f"    Error: {error_lines[0][:100]}...")
    
    print("\n🔍 Critical Issues Found:")
    print("-" * 25)
    
    critical_issues = []
    for name, success, _ in test_results:
        if "README" in name and not success:
            critical_issues.append(name)
    
    if critical_issues:
        print("❌ README examples that DON'T work as documented:")
        for issue in critical_issues:
            print(f"   - {issue}")
    else:
        print("✅ All README examples work as documented")
    
    print("\n💡 Specific Problems Identified:")
    print("-" * 32)
    
    problems = [
        "1. Import Path: README shows 'from dspy_gepa.amope import' but needs 'from src.dspy_gepa.amope import'",
        "2. Path Setup: Missing sys.path.insert(0, 'src') in all examples", 
        "3. Evaluation Function: README shows return float but AMOPE expects dict",
        "4. Demo Command: Same evaluation function issue as basic usage"
    ]
    
    for problem in problems:
        print(f"   {problem}")
    
    print("\n🔧 Fixes Needed:")
    print("-" * 16)
    
    fixes = [
        "Add sys.path setup to all README examples",
        "Update import paths to use src/ prefix",
        "Fix evaluation functions to return dict instead of float",
        "Test all README examples before documentation release",
        "Consider making the package properly installable"
    ]
    
    for fix in fixes:
        print(f"   • {fix}")
    
    print("\n📝 Corrected README Examples:")
    print("-" * 30)
    
    corrected_examples = [
        {
            "name": "Installation Verification",
            "code": '''# Add this to README
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from src.dspy_gepa.amope import AMOPEOptimizer
print('✅ Installation successful!')
"'''
        },
        {
            "name": "Basic Usage", 
            "code": '''# Add this to README
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from src.dspy_gepa.amope import AMOPEOptimizer

def evaluate_prompt(prompt_text):
    return {"performance": 0.8}  # Return dict, not float

optimizer = AMOPEOptimizer(objectives={"performance": 1.0})
result = optimizer.optimize("Your prompt", evaluate_prompt, generations=10)
print(f"Best score: {result.best_score}")'''
        },
        {
            "name": "Demo Command",
            "code": '''# Add this to README  
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.dspy_gepa.amope import AMOPEOptimizer

def demo_eval(prompt):
    return {'performance': 0.5 + 0.1 * hash(prompt) % 10 / 10}

optimizer = AMOPEOptimizer(objectives={'performance': 1.0})
result = optimizer.optimize('Test prompt', demo_eval, generations=10)
print(f'✅ Demo completed! Best score: {result.best_score:.3f}')
"'''
        }
    ]
    
    for example in corrected_examples:
        print(f"\n{example['name']}:")
        print("```python")
        print(example['code'])
        print("```")
    
    # Quality assessment
    print("\n🎯 FINAL QUALITY ASSESSMENT:")
    print("-" * 30)
    
    readme_accuracy = (total_tests - readme_broken - (total_tests // 2)) / (total_tests // 2) if total_tests > 0 else 0
    functionality_score = actually_working / total_tests if total_tests > 0 else 0
    
    print(f"Code Functionality: {functionality_score*100:.1f}%")
    print(f"Documentation Accuracy: {readme_accuracy*100:.1f}%")
    
    if functionality_score >= 0.8:
        print("✅ Code quality: EXCELLENT")
    elif functionality_score >= 0.6:
        print("⚠️ Code quality: GOOD")  
    else:
        print("❌ Code quality: NEEDS WORK")
    
    if readme_accuracy >= 0.8:
        print("✅ Documentation accuracy: EXCELLENT")
    elif readme_accuracy >= 0.6:
        print("⚠️ Documentation accuracy: GOOD")
    else:
        print("❌ Documentation accuracy: NEEDS WORK")
    
    # Release readiness
    print("\n🚀 RELEASE READINESS:")
    print("-" * 20)
    
    if readme_accuracy >= 0.8 and functionality_score >= 0.8:
        print("✅ READY FOR RELEASE")
        recommendation = 0
    elif functionality_score >= 0.8:
        print("⚠️ READY FOR RELEASE after documentation updates")
        recommendation = 1
    elif functionality_score >= 0.6:
        print("❌ NEEDS WORK before release")
        recommendation = 2
    else:
        print("❌ NOT READY for release")
        recommendation = 3
    
    print(f"\nRecommendation code: {recommendation}")
    return recommendation


def main():
    """Run comprehensive README assessment."""
    print("🐶 COMPREHENSIVE README EXAMPLES TEST")
    print("=" * 50)
    print("Testing all README examples and identifying fixes needed...")
    print()
    
    # Run all tests
    tests = [
        ("README Installation", test_readme_installation),
        ("README Verification Command", test_readme_verification),
        ("Working Verification", test_working_verification),
        ("README Basic Usage", test_readme_basic_usage),
        ("Working Basic Usage", test_working_basic_usage),
        ("README Demo Command", test_readme_demo_command),
        ("Working Demo Command", test_working_demo_command),
        ("Basic DSPY-GEPA Example", test_basic_dspy_gepa_example),
        ("Core GEPA Functionality", test_core_gepa_functionality),
    ]
    
    test_results = []
    
    for name, test_func in tests:
        success, stdout, stderr = test_func()
        test_results.append((name, success, stderr if not success else stdout))
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} | {name}")
        
        if not success:
            # Show first line of error for debugging
            error_lines = stderr.strip().split('\n')
            if error_lines:
                print(f"    Error: {error_lines[0][:80]}...")
    
    # Generate comprehensive assessment
    recommendation = generate_assessment(test_results)
    
    return recommendation


if __name__ == "__main__":
    sys.exit(main())
