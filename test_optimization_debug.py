#!/usr/bin/env python3
"""Simple test to validate optimization functionality works."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dspy_gepa import GEPAAgent

def simple_evaluation_fn(prompt: str) -> dict:
    """Simple evaluation that favors longer, more specific prompts."""
    words = prompt.lower().split()
    length_score = min(1.0, len(words) / 20.0)
    
    specificity_score = 0.0
    specific_keywords = ["please", "specific", "detailed", "comprehensive", "step-by-step"]
    for keyword in specific_keywords:
        if keyword in prompt.lower():
            specificity_score += 0.1
    specificity_score = min(1.0, specificity_score)
    
    return {
        "accuracy": 0.3 + 0.4 * length_score + 0.3 * specificity_score,
        "clarity": 0.4 + 0.3 * length_score + 0.3 * specificity_score,
        "completeness": 0.2 + 0.5 * length_score + 0.3 * specificity_score
    }

def test_basic_optimization():
    """Test basic optimization functionality."""
    print("🧪 Testing basic optimization functionality...")
    
    try:
        # Create agent
        agent = GEPAAgent(
            objectives={"accuracy": 0.4, "clarity": 0.3, "completeness": 0.3},
            population_size=6,
            max_generations=3,
            verbose=False
        )
        print(f"✅ Agent created with objectives: {agent.config.objectives}")
        
        # Test evaluation (use optimizer directly)
        initial_prompt = "Write code"
        initial_score = agent.optimizer._evaluate_prompt(initial_prompt, simple_evaluation_fn)
        print(f"✅ Initial evaluation: '{initial_prompt}' -> score: {initial_score:.4f}")
        
        # Run optimization
        print("🚀 Running optimization...")
        result = agent.optimize_prompt(
            initial_prompt=initial_prompt,
            evaluation_fn=simple_evaluation_fn,
            return_summary=True
        )
        
        print(f"✅ Optimization completed in {result.optimization_time:.2f}s")
        print(f"✅ Generations: {result.generations_completed}, Evaluations: {result.total_evaluations}")
        print(f"✅ Score improvement: {result.initial_score:.4f} -> {result.best_score:.4f} (+{result.improvement:.4f})")
        print(f"✅ Initial prompt: '{initial_prompt}'")
        print(f"✅ Optimized prompt: '{result.best_prompt}'")
        
        # Validate actual improvement
        if result.best_score > result.initial_score:
            print("🎉 SUCCESS: Optimization achieved improvement!")
            return True
        else:
            print("⚠️  WARNING: No improvement detected")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_objectives():
    """Test with different objective configurations."""
    print("\n🧪 Testing multiple objective configurations...")
    
    objectives_list = [
        {"accuracy": 1.0},
        {"accuracy": 0.5, "efficiency": 0.5},
        {"accuracy": 0.4, "clarity": 0.3, "completeness": 0.3},
    ]
    
    success_count = 0
    
    for i, objectives in enumerate(objectives_list):
        try:
            agent = GEPAAgent(
                objectives=objectives,
                population_size=4,
                max_generations=2,
                verbose=False
            )
            
            initial_prompt = "test prompt"
            result = agent.optimize_prompt(
                initial_prompt=initial_prompt,
                evaluation_fn=simple_evaluation_fn,
                return_summary=True
            )
            
            if result.improvement > 0:
                print(f"✅ Objectives {i+1}: SUCCESS (+{result.improvement:.4f})")
                success_count += 1
            else:
                print(f"⚠️  Objectives {i+1}: No improvement")
                
        except Exception as e:
            print(f"❌ Objectives {i+1}: FAILED - {e}")
    
    print(f"📊 Results: {success_count}/{len(objectives_list)} objective configurations succeeded")
    return success_count > 0

def test_edge_cases():
    """Test edge cases."""
    print("\n🧪 Testing edge cases...")
    
    try:
        agent = GEPAAgent(
            objectives={"score": 1.0},
            population_size=3,
            max_generations=2,
            verbose=False
        )
        
        # Test empty prompt
        def zero_eval(prompt: str) -> dict:
            return {"score": 0.0 if not prompt.strip() else 0.5}
        
        result = agent.optimize_prompt(
            initial_prompt="",
            evaluation_fn=zero_eval,
            return_summary=True
        )
        
        print(f"✅ Empty prompt handled: '{result.best_prompt}'")
        
        # Test single word prompt
        result2 = agent.optimize_prompt(
            initial_prompt="a",
            evaluation_fn=zero_eval,
            return_summary=True
        )
        
        print(f"✅ Single word handled: '{result2.best_prompt}'")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases failed: {e}")
        return False

def main():
    """Run all tests."""
    print('=' * 60)
    print('🔬 DSPY-GEPA Optimization Functional Test')
    print('=' * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(test_basic_optimization())
    test_results.append(test_multiple_objectives())
    test_results.append(test_edge_cases())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print('\n' + '=' * 60)
    print('📊 TEST SUMMARY')
    print('=' * 60)
    print(f'Passed: {passed}/{total} tests')
    
    if passed == total:
        print('🎉 ALL TESTS PASSED! Optimization is working correctly.')
        return 0
    else:
        print('⚠️  Some tests failed. Check the output above for details.')
        return 1

if __name__ == '__main__':
    exit(main())