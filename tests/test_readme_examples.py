#!/usr/bin/env python3
"""
Test script to validate README examples work exactly as documented.

This script tests all the examples from the README.md file to ensure they
work exactly as shown in the documentation. It identifies gaps between
what's documented and what actually works.
"""

import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any


class ReadmeTestResult:
    """Store results of README example tests."""
    
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.error_message = None
        self.stdout = None
        self.stderr = None
        self.execution_time = None
        self.works_as_documented = False
        self.notes = []


class ReadmeTester:
    """Test README examples comprehensively."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = []
        self.start_time = time.time()
    
    def run_command(self, command: str, timeout: int = 60) -> ReadmeTestResult:
        """Run a shell command and capture results."""
        result = ReadmeTestResult(command)
        
        try:
            start = time.time()
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            result.execution_time = time.time() - start
            
            result.stdout = process.stdout
            result.stderr = process.stderr
            result.success = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result.error_message = f"Command timed out after {timeout}s"
            result.success = False
        except Exception as e:
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def test_installation_command(self) -> ReadmeTestResult:
        """Test: uv sync command from README."""
        print("🧪 Testing installation command: uv sync")
        result = self.run_command("uv sync", timeout=120)
        
        if result.success:
            result.works_as_documented = True
            result.notes.append("✅ uv sync completed successfully")
        else:
            result.works_as_documented = False
            result.notes.append("❌ uv sync failed")
        
        return result
    
    def test_verification_command(self) -> ReadmeTestResult:
        """Test: uv run python -c \"from dspy_gepa.amope import AMOPEOptimizer; print('✅ Installation successful!')\""""
        print("🧪 Testing verification command")
        result = self.run_command(
            'uv run python -c "from dspy_gepa.amope import AMOPEOptimizer; print(\'✅ Installation successful!\')"',
            timeout=30
        )
        
        # This is expected to fail based on current import issues
        if result.success:
            result.works_as_documented = True
            result.notes.append("✅ AMOPE import works as documented")
        else:
            result.works_as_documented = False
            result.notes.append("❌ AMOPE import fails - this reveals a documentation gap")
            result.notes.append("📝 The README shows AMOPE usage but imports are broken")
        
        return result
    
    def test_basic_usage_example(self) -> ReadmeTestResult:
        """Test the basic usage example from README."""
        print("🧪 Testing basic usage example")
        
        basic_usage_code = '''
from dspy_gepa.amope import AMOPEOptimizer

# Define your evaluation function
def evaluate_prompt(prompt_text):
    # Your evaluation logic here
    # Return a score between 0 and 1
    return 0.8  # Example score

# Initialize AMOPE optimizer
optimizer = AMOPEOptimizer(
    objectives={"performance": 1.0},
    population_size=8,
    max_generations=25
)

# Run optimization
result = optimizer.optimize(
    initial_prompt="Write a concise summary of the main points.",
    evaluation_fn=evaluate_prompt,
    generations=25
)

print(f"Best prompt: {result.best_prompt}")
print(f"Best score: {result.best_score}")
'''
        
        result = self.run_command(
            f'uv run python -c "{basic_usage_code}"',
            timeout=30
        )
        
        if result.success:
            result.works_as_documented = True
            result.notes.append("✅ Basic usage example works")
        else:
            result.works_as_documented = False
            result.notes.append("❌ Basic usage example fails")
            result.notes.append("📝 Documentation shows AMOPE API that doesn't match implementation")
        
        return result
    
    def test_demo_command(self) -> ReadmeTestResult:
        """Test the demo command from troubleshooting section."""
        print("🧪 Testing demo command")
        
        demo_command = '''
uv run python -c "
from dspy_gepa.amope import AMOPEOptimizer

def demo_eval(prompt):
    return 0.5 + 0.1 * hash(prompt) % 10 / 10

optimizer = AMOPEOptimizer(objectives={'performance': 1.0})
result = optimizer.optimize('Test prompt', demo_eval, generations=10)
print(f'✅ Demo completed! Best score: {result.best_score:.3f}')
"
'''
        
        result = self.run_command(demo_command, timeout=30)
        
        if result.success:
            result.works_as_documented = True
            result.notes.append("✅ Demo command works as documented")
        else:
            result.works_as_documented = False
            result.notes.append("❌ Demo command fails")
            result.notes.append("📝 Demo shows AMOPE usage that doesn't match current implementation")
        
        return result
    
    def test_basic_dspy_gepa_example(self) -> ReadmeTestResult:
        """Test the basic_dspy_gepa.py example."""
        print("🧪 Testing basic_dspy_gepa.py example")
        
        result = self.run_command(
            "uv run python examples/basic_dspy_gepa.py",
            timeout=60
        )
        
        if result.success:
            result.works_as_documented = True
            result.notes.append("✅ basic_dspy_gepa.py runs successfully")
        else:
            # This might fail due to import issues, which is expected
            result.works_as_documented = False
            result.notes.append("❌ basic_dspy_gepa.py fails")
            if "No module named 'dspy'" in result.stderr:
                result.notes.append("📝 DSPY is not installed (expected for basic setup)")
            elif "ImportError" in result.stderr:
                result.notes.append("📝 Import issues between modules")
        
        return result
    
    def test_core_gepa_functionality(self) -> ReadmeTestResult:
        """Test what actually works with the core GEPA functionality."""
        print("🧪 Testing core GEPA functionality (what actually works)")
        
        core_functionality_test = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

# Test core GEPA imports
from gepa import Candidate, GeneticOptimizer, ParetoSelector, TextMutator

print("✅ Core GEPA imports work")

# Test basic Candidate creation
candidate = Candidate(content="Test prompt content")
print(f"✅ Candidate created: {candidate.id[:8]}")

# Test basic functionality
def simple_fitness(candidate):
    return {"accuracy": 0.8, "efficiency": 0.7}

print(f"✅ Fitness function works: {simple_fitness(candidate)}")
'''
        
        result = self.run_command(
            f'uv run python -c "{core_functionality_test}"',
            timeout=30
        )
        
        if result.success:
            result.works_as_documented = True
            result.notes.append("✅ Core GEPA functionality works")
        else:
            result.works_as_documented = False
            result.notes.append("❌ Even core GEPA functionality fails")
        
        return result
    
    def test_alternative_imports(self) -> ReadmeTestResult:
        """Test alternative import approaches."""
        print("🧪 Testing alternative import approaches")
        
        alternative_test = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

try:
    # Try direct AMOPE import
    from src.dspy_gepa.amope import AMOPEOptimizer
    print("✅ Direct AMOPE import works")
except ImportError as e:
    print(f"❌ Direct AMOPE import failed: {e}")

try:
    # Try AMOPE components
    from src.dspy_gepa.amope.adaptive_mutator import AdaptiveMutator
    print("✅ AdaptiveMutator import works")
except ImportError as e:
    print(f"❌ AdaptiveMutator import failed: {e}")
'''
        
        result = self.run_command(
            f'uv run python -c "{alternative_test}"',
            timeout=30
        )
        
        # This helps identify what imports actually work
        result.works_as_documented = True  # This is exploratory, not from README
        if result.success:
            result.notes.append("✅ Alternative import testing completed")
        else:
            result.notes.append("❌ Alternative import testing failed")
        
        return result
    
    def test_troubleshooting_commands(self) -> ReadmeTestResult:
        """Test troubleshooting commands from README."""
        print("🧪 Testing troubleshooting commands")
        
        # Test import verification
        import_test = self.run_command(
            'uv run python -c "import dspy_gepa; print(\'✅ Imports working\')"',
            timeout=30
        )
        
        # Test uv sync --reinstall suggestion
        reinstall_test = self.run_command(
            "uv sync --reinstall",
            timeout=120
        )
        
        # Create combined result
        result = ReadmeTestResult("troubleshooting_commands")
        result.success = import_test.success or reinstall_test.success
        
        if import_test.success:
            result.notes.append("✅ Basic dspy_gepa import works")
        else:
            result.notes.append("❌ Basic dspy_gepa import fails")
        
        if reinstall_test.success:
            result.notes.append("✅ uv sync --reinstall works")
        else:
            result.notes.append("❌ uv sync --reinstall fails")
        
        result.works_as_documented = result.success
        return result
    
    def test_dspy_optional_dependency(self) -> ReadmeTestResult:
        """Test DSPY optional dependency installation."""
        print("🧪 Testing DSPY optional dependency")
        
        # Test if DSPY is available
        dspy_test = self.run_command(
            'uv run python -c "import dspy; print(\'✅ DSPY available\')"',
            timeout=30
        )
        
        # Test the suggested uv add dspy command
        dspy_install = self.run_command(
            "uv add dspy",
            timeout=60
        )
        
        result = ReadmeTestResult("dspy_optional_dependency")
        result.success = dspy_test.success
        
        if dspy_test.success:
            result.works_as_documented = True
            result.notes.append("✅ DSPY is available")
        else:
            result.works_as_documented = False
            result.notes.append("❌ DSPY not available (this is optional)")
        
        if dspy_install.success:
            result.notes.append("✅ uv add dspy works")
        else:
            result.notes.append("❌ uv add dspy failed")
        
        return result
    
    def run_all_tests(self) -> Dict[str, ReadmeTestResult]:
        """Run all README example tests."""
        print("🐶 README Examples Test Suite")
        print("=" * 50)
        print("Testing all examples from README.md...")
        print()
        
        tests = [
            ("Installation", self.test_installation_command),
            ("Verification Command", self.test_verification_command),
            ("Basic Usage Example", self.test_basic_usage_example),
            ("Demo Command", self.test_demo_command),
            ("Basic DSPY-GEPA Example", self.test_basic_dspy_gepa_example),
            ("Core GEPA Functionality", self.test_core_gepa_functionality),
            ("Alternative Imports", self.test_alternative_imports),
            ("Troubleshooting Commands", self.test_troubleshooting_commands),
            ("DSPY Optional Dependency", self.test_dspy_optional_dependency),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"Running: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                self.results.append(result)
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                doc_status = "✅ DOC" if result.works_as_documented else "❌ DOC"
                print(f"  {status} | {doc_status} | {test_name}")
                
                if result.notes:
                    for note in result.notes:
                        print(f"    {note}")
                
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                result = ReadmeTestResult(test_name)
                result.success = False
                result.error_message = str(e)
                result.works_as_documented = False
                results[test_name] = result
                self.results.append(result)
            
            print()
        
        return results
    
    def generate_summary_report(self, results: Dict[str, ReadmeTestResult]) -> str:
        """Generate a comprehensive summary report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.success)
        documented_working = sum(1 for r in results.values() if r.works_as_documented)
        
        report = []
        report.append("📊 README Examples Test Summary")
        report.append("=" * 40)
        report.append(f"Total Examples Tested: {total_tests}")
        report.append(f"Passing Tests: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"Working as Documented: {documented_working}/{total_tests} ({documented_working/total_tests*100:.1f}%)")
        report.append(f"Execution Time: {time.time() - self.start_time:.2f}s")
        report.append("")
        
        # Detailed results
        report.append("📋 Detailed Results:")
        report.append("-" * 20)
        
        for name, result in results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            doc_status = "✅ DOC" if result.works_as_documented else "❌ DOC"
            report.append(f"{status} | {doc_status} | {name}")
            
            if result.error_message:
                report.append(f"    Error: {result.error_message}")
            
            if result.notes:
                for note in result.notes:
                    report.append(f"    {note}")
            report.append("")
        
        # Key findings
        report.append("🔍 Key Findings:")
        report.append("-" * 15)
        
        critical_issues = []
        for name, result in results.items():
            if not result.works_as_documented:
                critical_issues.append(name)
        
        if critical_issues:
            report.append("❌ Critical Issues (Documentation ≠ Implementation):")
            for issue in critical_issues:
                report.append(f"    - {issue}")
        else:
            report.append("✅ All examples work as documented")
        
        report.append("")
        
        # Recommendations
        report.append("💡 Recommendations:")
        report.append("-" * 18)
        
        if "Verification Command" in critical_issues:
            report.append("• Fix AMOPE import paths in src/dspy_gepa/amope/__init__.py")
            report.append("• Update README to show working imports")
        
        if "Basic Usage Example" in critical_issues:
            report.append("• Update AMOPE API documentation to match implementation")
            report.append("• Provide working basic usage example")
        
        if "Demo Command" in critical_issues:
            report.append("• Fix demo command or update documentation")
        
        if any("DSPY" in issue for issue in critical_issues):
            report.append("• Clarify DSPY as optional dependency")
            report.append("• Provide separate DSPY-specific documentation")
        
        report.append("")
        report.append("🎯 Quality Gates Status:")
        report.append("-" * 22)
        
        quality_score = (passed_tests / total_tests) * 100
        doc_score = (documented_working / total_tests) * 100
        
        if quality_score >= 80:
            report.append(f"✅ Test Quality: {quality_score:.1f}% (Excellent)")
        elif quality_score >= 60:
            report.append(f"⚠️ Test Quality: {quality_score:.1f}% (Good, needs improvement)")
        else:
            report.append(f"❌ Test Quality: {quality_score:.1f}% (Needs major fixes)")
        
        if doc_score >= 80:
            report.append(f"✅ Documentation Accuracy: {doc_score:.1f}% (Excellent)")
        elif doc_score >= 60:
            report.append(f"⚠️ Documentation Accuracy: {doc_score:.1f}% (Good, needs updates)")
        else:
            report.append(f"❌ Documentation Accuracy: {doc_score:.1f}% (Major documentation issues)")
        
        return "\n".join(report)


def main():
    """Main test execution."""
    tester = ReadmeTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 60)
    summary = tester.generate_summary_report(results)
    print(summary)
    
    # Exit with appropriate code
    total_tests = len(results)
    documented_working = sum(1 for r in results.values() if r.works_as_documented)
    
    if documented_working == total_tests:
        print("\n🎉 All README examples work as documented!")
        sys.exit(0)
    elif documented_working >= total_tests * 0.8:
        print("\n⚠️ Most examples work, but some documentation issues exist.")
        sys.exit(1)
    else:
        print("\n❌ Significant gaps between documentation and implementation.")
        sys.exit(2)


if __name__ == "__main__":
    main()
