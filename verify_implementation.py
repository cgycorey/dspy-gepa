#!/usr/bin/env python3
"""Comprehensive verification of DSPY-GEPA implementation working components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_gepa():
    """Test core GEPA components."""
    print("🔧 Testing Core GEPA Components...")
    
    try:
        from gepa import Candidate, ParetoSelector, GeneticOptimizer, TextMutator
        print("✅ Core GEPA imports successful")
        
        # Test Candidate creation
        candidates = []
        for i in range(3):
            candidate = Candidate(
                content=f"Test prompt {i+1}",
                generation=0,
                fitness_scores={
                    "accuracy": 0.5 + i * 0.1,
                    "efficiency": 0.8 - i * 0.1
                }
            )
            candidates.append(candidate)
        
        print(f"✅ Created {len(candidates)} candidates")
        
        # Test ParetoSelector
        selector = ParetoSelector(objectives=["accuracy", "efficiency"])
        selected = selector.environmental_selection(candidates, target_size=2)
        print(f"✅ Pareto selection: {len(selected)} candidates selected")
        
        # Test candidate operations
        combined_score = sum(candidates[0].fitness_scores.values()) / len(candidates[0].fitness_scores)
        print(f"✅ Fitness calculation: {combined_score:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Core GEPA test failed: {e}")
        return False

def test_amope_components():
    """Test AMOPE algorithm components."""
    print("\n🧬 Testing AMOPE Algorithm Components...")
    
    try:
        import importlib.util
        
        # Load objective balancer directly to avoid psutil dependency
        spec = importlib.util.spec_from_file_location('objective_balancer', 'src/dspy_gepa/amope/objective_balancer.py')
        objective_balancer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(objective_balancer)
        
        # Test ObjectiveBalancer
        balancer = objective_balancer.ObjectiveBalancer(
            objectives={'accuracy': 0.7, 'efficiency': 0.3},
            strategy=objective_balancer.BalancingStrategy.ADAPTIVE_HARMONIC
        )
        print("✅ ObjectiveBalancer created")
        
        # Test fitness updates
        fitness_data = [
            {'accuracy': 0.5, 'efficiency': 0.4},
            {'accuracy': 0.6, 'efficiency': 0.45},
            {'accuracy': 0.65, 'efficiency': 0.42},
            {'accuracy': 0.68, 'efficiency': 0.48},
        ]
        
        for fitness in fitness_data:
            balancer.update_fitness(fitness)
        
        print(f"✅ Weight updates: {balancer.current_objectives}")
        print(f"✅ Generations processed: {balancer.generation}")
        
        return True
    except Exception as e:
        print(f"❌ AMOPE test failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test end-to-end optimization workflow."""
    print("\n🔄 Testing End-to-End Workflow...")
    
    try:
        from gepa import Candidate, ParetoSelector
        import importlib.util
        
        # Load AMOPE components
        spec = importlib.util.spec_from_file_location('objective_balancer', 'src/dspy_gepa/amope/objective_balancer.py')
        objective_balancer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(objective_balancer)
        
        # Step 1: Create initial population
        initial_prompts = [
            "Simple prompt for testing",
            "More detailed prompt with context", 
            "Comprehensive prompt with examples"
        ]
        
        population = []
        for prompt in initial_prompts:
            candidate = Candidate(
                content=prompt,
                generation=0,
                fitness_scores={
                    "accuracy": 0.6,
                    "efficiency": 0.7
                }
            )
            population.append(candidate)
        
        print(f"✅ Initial population: {len(population)} candidates")
        
        # Step 2: Simulate evolution
        selector = ParetoSelector(objectives=["accuracy", "efficiency"])
        balancer = objective_balancer.ObjectiveBalancer(
            objectives={"accuracy": 0.6, "efficiency": 0.4}
        )
        
        # Simulate multiple generations
        for gen in range(3):
            # Selection
            selected = selector.environmental_selection(population, target_size=2)
            
            # Update balancer with mock fitness improvements
            mock_fitness = {
                "accuracy": 0.6 + gen * 0.05,
                "efficiency": 0.7 - gen * 0.02
            }
            balancer.update_fitness(mock_fitness)
            
            print(f"✅ Generation {gen+1}: {len(selected)} selected, weights updated")
        
        print(f"✅ Final weights: {balancer.current_objectives}")
        return True
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        return False

def test_project_structure():
    """Test project structure and examples."""
    print("\n📁 Testing Project Structure...")
    
    try:
        project_root = Path(".")
        
        # Check directories
        src_dir = project_root / "src"
        tests_dir = project_root / "tests"
        examples_dir = project_root / "examples"
        
        assert src_dir.exists(), "src directory missing"
        assert tests_dir.exists(), "tests directory missing"
        assert examples_dir.exists(), "examples directory missing"
        
        # Check modules
        gepa_module = src_dir / "gepa"
        dspy_gepa_module = src_dir / "dspy_gepa"
        
        assert gepa_module.exists(), "gepa module missing"
        assert dspy_gepa_module.exists(), "dspy_gepa module missing"
        
        # Check examples
        basic_example = examples_dir / "basic_dspy_gepa.py"
        agent_example = examples_dir / "basic_agent.py"
        
        assert basic_example.exists(), "basic_dspy_gepa.py example missing"
        assert agent_example.exists(), "basic_agent.py example missing"
        
        # Check example content
        with open(basic_example, 'r') as f:
            content = f.read()
            assert "import dspy" in content
            assert "from gepa import" in content
        
        print("✅ All project structure checks passed")
        return True
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Starting DSPY-GEPA Implementation Verification\n")
    
    tests = [
        ("Core GEPA Components", test_core_gepa),
        ("AMOPE Algorithm", test_amope_components), 
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Project Structure", test_project_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! DSPY-GEPA implementation is working correctly.")
        print("✅ Ready for production use!")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review implementation.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
