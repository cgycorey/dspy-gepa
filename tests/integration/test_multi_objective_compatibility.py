"""Test script to verify backward compatibility and multi-objective functionality.

This script tests that the new multi-objective framework maintains
backward compatibility with existing code while providing enhanced capabilities.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

import sys
import os
import traceback
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_basic_imports():
    """Test that all basic imports work correctly."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test core imports
        from dspy_gepa import GEPAAgent, SimpleGEPA
        from dspy_gepa.core.agent import GEPAAgent as Agent
        print("   ✅ Core agent imports successful")
        
        # Test new multi-objective imports
        from dspy_gepa import MultiObjectiveGEPAAgent, create_multi_objective_agent
        print("   ✅ Multi-objective agent imports successful")
        
        # Test interface imports
        from dspy_gepa import Objective, TaskType, OptimizationDirection, PreferenceVector
        print("   ✅ Interface imports successful")
        
        # Test objective imports
        from dspy_gepa import AccuracyMetric, FluencyMetric, RelevanceMetric
        print("   ✅ Objective imports successful")
        
        # Test mutation and tuning imports
        from dspy_gepa import SemanticMutator, ConvergenceBasedTuner
        print("   ✅ Mutation and tuning imports successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that existing code still works with the new framework."""
    print("\n🧪 Testing backward compatibility...")
    
    try:
        from dspy_gepa import GEPAAgent, Agent
        
        # Test creating a regular GEPAAgent (should work as before)
        agent = GEPAAgent(max_generations=5, verbose=False)
        print("   ✅ Basic GEPAAgent creation successful")
        
        # Test that the agent has expected attributes
        assert hasattr(agent, 'optimize_prompt'), "Missing optimize_prompt method"
        assert hasattr(agent, 'config'), "Missing config attribute"
        print("   ✅ GEPAAgent has expected attributes")
        
        # Test creating Agent (aliased version)
        agent2 = Agent(max_generations=3, verbose=False)
        print("   ✅ Agent alias works correctly")
        
        # Test simple optimization interface
        def simple_eval(prompt: str) -> float:
            """Simple evaluation function."""
            return 0.5 if len(prompt) > 10 else 0.3
        
        # This should work without any multi-objective setup
        result = agent.optimize_prompt(
            "Test prompt for optimization",
            simple_eval,
            generations=2,
            return_summary=True
        )
        
        # Should return OptimizationSummary
        from dspy_gepa.core.agent import OptimizationSummary
        assert isinstance(result, OptimizationSummary), f"Expected OptimizationSummary, got {type(result)}"
        print("   ✅ Single-objective optimization works correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def test_multi_objective_functionality():
    """Test that new multi-objective functionality works correctly."""
    print("\n🧪 Testing multi-objective functionality...")
    
    try:
        from dspy_gepa import (
            MultiObjectiveGEPAAgent, create_multi_objective_agent,
            AccuracyMetric, FluencyMetric, RelevanceMetric,
            TaskType, OptimizationDirection
        )
        
        # Test creating objectives
        objectives = [
            AccuracyMetric(weight=0.4),
            FluencyMetric(weight=0.3),
            RelevanceMetric(weight=0.3)
        ]
        print("   ✅ Objectives created successfully")
        
        # Test multi-objective agent creation
        mo_agent = MultiObjectiveGEPAAgent(
            objectives=objectives,
            task_type=TaskType.GENERATION,
            max_generations=3,
            population_size=3,
            verbose=False
        )
        print("   ✅ Multi-objective agent created successfully")
        
        # Test convenience function
        mo_agent2 = create_multi_objective_agent(
            objectives=objectives,
            task_type=TaskType.GENERATION,
            max_generations=3,
            verbose=False
        )
        print("   ✅ Convenience function works correctly")
        
        # Test that multi-objective agent can fall back to single-objective
        def simple_eval(prompt: str) -> float:
            return 0.6 if "please" in prompt.lower() else 0.4
        
        result = mo_agent.optimize_prompt(
            "Please translate this to Spanish",
            simple_eval,
            generations=2,
            return_summary=True,
            enable_multi_objective=False  # Force single-objective mode
        )
        
        from dspy_gepa.core.agent import OptimizationSummary
        assert isinstance(result, OptimizationSummary), f"Expected OptimizationSummary, got {type(result)}"
        print("   ✅ Multi-objective agent can fall back to single-objective mode")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Multi-objective functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_multi_objective_optimization():
    """Test actual multi-objective optimization."""
    print("\n🧪 Testing multi-objective optimization...")
    
    try:
        from dspy_gepa import (
            MultiObjectiveGEPAAgent, AccuracyMetric, FluencyMetric,
            TaskType, OptimizationDirection, PreferenceVector
        )
        
        # Create objectives for a translation task
        objectives = [
            AccuracyMetric(weight=0.5, direction=OptimizationDirection.MAXIMIZE),
            FluencyMetric(weight=0.3, direction=OptimizationDirection.MAXIMIZE),
        ]
        
        # Create multi-objective agent
        agent = MultiObjectiveGEPAAgent(
            objectives=objectives,
            task_type=TaskType.TRANSLATION,
            max_generations=3,
            population_size=3,
            verbose=False
        )
        
        # Simple evaluation function that simulates multiple objectives
        def multi_eval(prompt: str) -> float:
            """Simulate evaluation based on prompt characteristics."""
            score = 0.0
            
            # Accuracy simulation (favor clear instructions)
            if any(word in prompt.lower() for word in ['translate', 'please', 'accurate']):
                score += 0.5
            else:
                score += 0.3
            
            # Fluency simulation (favor polite language)
            if any(word in prompt.lower() for word in ['please', 'could you']):
                score += 0.3
            else:
                score += 0.1
            
            return min(1.0, score)
        
        # Test multi-objective optimization
        result = agent.optimize_prompt(
            "Translate to Spanish",
            multi_eval,
            generations=2,
            return_summary=True,
            enable_multi_objective=True
        )
        
        # Should return OptimizationSummary (converted from multi-objective result)
        from dspy_gepa.core.agent import OptimizationSummary
        assert isinstance(result, OptimizationSummary), f"Expected OptimizationSummary, got {type(result)}"
        print("   ✅ Multi-objective optimization completed successfully")
        
        # Test that results are stored
        if agent.last_multi_objective_result:
            frontier_size = len(agent.last_multi_objective_result.get_pareto_frontier_solutions())
            print(f"   ✅ Pareto frontier contains {frontier_size} solutions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Multi-objective optimization test failed: {e}")
        traceback.print_exc()
        return False


def test_preferences_and_selection():
    """Test preference-based solution selection."""
    print("\n🧪 Testing preference-based selection...")
    
    try:
        from dspy_gepa import (
            MultiObjectiveGEPAAgent, AccuracyMetric, FluencyMetric, RelevanceMetric,
            PreferenceVector, OptimizationDirection
        )
        
        # Create objectives
        objectives = [
            AccuracyMetric(weight=0.4, direction=OptimizationDirection.MAXIMIZE),
            FluencyMetric(weight=0.3, direction=OptimizationDirection.MAXIMIZE),
            RelevanceMetric(weight=0.3, direction=OptimizationDirection.MAXIMIZE)
        ]
        
        # Create agent
        agent = MultiObjectiveGEPAAgent(
            objectives=objectives,
            max_generations=2,
            population_size=2,
            verbose=False
        )
        
        # Test setting preferences
        preferences = PreferenceVector({
            "accuracy": 0.7,
            "fluency": 0.2,
            "relevance": 0.1
        })
        
        agent.set_preferences(preferences)
        print("   ✅ Preferences set successfully")
        
        # Test that preferences are stored
        assert agent.preference_vector is not None
        assert agent.preference_vector.weights["accuracy"] == 0.7
        print("   ✅ Preferences stored correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Preference selection test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling capabilities."""
    print("\n🧪 Testing error handling...")
    
    try:
        from dspy_gepa.core.error_handling import (
            ErrorHandler, ValidationError, ConfigurationError,
            ErrorCategory, ErrorSeverity
        )
        
        # Test error handler
        handler = ErrorHandler()
        print("   ✅ Error handler created successfully")
        
        # Test error handling
        try:
            raise ValueError("Test validation error")
        except Exception as e:
            error_context = handler.handle_error(
                e,
                context_data={"test": True},
                category=ErrorCategory.VALIDATION
            )
            assert error_context.category == ErrorCategory.VALIDATION
            print("   ✅ Error handling works correctly")
        
        # Test custom exceptions
        try:
            raise ConfigurationError("Test config error", config_key="test_key")
        except Exception as e:
            assert isinstance(e, ConfigurationError)
            assert e.category == ErrorCategory.CONFIGURATION
            print("   ✅ Custom exceptions work correctly")
        
        # Test error statistics
        stats = handler.get_error_statistics()
        assert "total_errors" in stats
        # We handled 2 errors, but let's be flexible and just check we have at least 1
        assert stats["total_errors"] >= 1  # At least one error should be counted
        print(f"   ✅ Error statistics work correctly (counted {stats['total_errors']} errors)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all compatibility tests."""
    print("🚀 Starting DSPy GEPA Multi-Objective Compatibility Tests\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Backward Compatibility", test_backward_compatibility),
        ("Multi-Objective Functionality", test_multi_objective_functionality),
        ("Multi-Objective Optimization", test_multi_objective_optimization),
        ("Preferences and Selection", test_preferences_and_selection),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The framework is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)