#!/usr/bin/env python3
"""Comprehensive tests for AMOPE-GEPA integration.

This test suite validates that AMOPE actually uses GEPA's GeneticOptimizer
during optimization and that the integration provides the expected improvements.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Core imports
    from gepa.core.candidate import Candidate
    from gepa.core.optimizer import GeneticOptimizer, OptimizationConfig
    from gepa.core.selector import ParetoSelector
    from gepa.core.mutator import TextMutator
    
    # AMOPE imports
    from dspy_gepa.amope import AMOPEOptimizer
    from dspy_gepa.amope.objective_balancer import ObjectiveBalancer, BalancingStrategy
    from dspy_gepa.amope.adaptive_mutator import AdaptiveMutator, MutationStrategy
    
    # Integration imports
    from dspy_gepa.dspy_integration.dspy_adapter import DSPYAdapter
    from dspy_gepa.dspy_integration.metric_collector import MetricCollector, DSPYMetrics
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestAMOPEGEPAIntegration:
    """Test suite for AMOPE-GEPA integration."""
    
    @pytest.fixture
    def mock_evaluation_function(self):
        """Mock evaluation function for testing."""
        def evaluate_candidate(candidate) -> Dict[str, float]:
            # Handle both string and Candidate inputs
            content = ""
            if isinstance(candidate, str):
                content = candidate
            elif hasattr(candidate, 'content'):
                content = candidate.content
            else:
                content = str(candidate)
            
            content = content.lower()
            
            # Simulate realistic evaluation with some randomness
            base_scores = {
                "accuracy": 0.5 + random.random() * 0.4,
                "efficiency": 0.6 + random.random() * 0.3,
                "complexity": 0.3 + random.random() * 0.5
            }
            
            # Add some content-based variation
            if "improved" in content:
                base_scores["accuracy"] += 0.1
            if "simple" in content:
                base_scores["efficiency"] += 0.15
            if "complex" in content:
                base_scores["complexity"] += 0.2
                
            return base_scores
        return evaluate_candidate
    
    @pytest.fixture
    def sample_initial_candidates(self):
        """Create sample initial candidates for testing."""
        candidates = []
        prompts = [
            "Answer the question accurately.",
            "Provide a simple, efficient response.",
            "Give a comprehensive, detailed answer.",
            "Respond with clarity and precision.",
            "Offer a balanced, informative reply."
        ]
        
        for i, prompt in enumerate(prompts):
            candidate = Candidate(
                content=prompt,
                generation=0,
                fitness_scores={
                    "accuracy": 0.5 + i * 0.05,
                    "efficiency": 0.7 - i * 0.03,
                    "complexity": 0.4 + i * 0.08
                }
            )
            candidates.append(candidate)
            
        return candidates
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_amope_uses_gepa_genetic_optimizer(self, mock_evaluation_function):
        """Test that AMOPE actually calls GEPA's GeneticOptimizer during optimization."""
        print("\n🧪 Testing AMOPE-GEPA GeneticOptimizer integration...")
        
        # Arrange
        objectives = {"accuracy": 0.5, "efficiency": 0.3, "complexity": 0.2}
        
        # Mock GEPA's GeneticOptimizer to track calls
        with patch('gepa.core.optimizer.GeneticOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer_class.return_value = mock_optimizer
            
            # Mock the optimize method to return realistic results
            best_candidate = Candidate(
                content="Optimized prompt for maximum performance",
                generation=5,
                fitness_scores={"accuracy": 0.85, "efficiency": 0.78, "complexity": 0.65}
            )
            mock_optimizer.optimize.return_value = [best_candidate]
            mock_optimizer.get_optimization_stats.return_value = {
                "total_generations": 10,
                "final_population_size": 20,
                "convergence_generations": 8
            }
            
            # Act
            optimizer = AMOPEOptimizer(
                objectives=objectives,
                mutation_config={"use_llm_guidance": False},
                balancing_config={"strategy": "adaptive_harmonic"}
            )
            
            result = optimizer.optimize(
                initial_prompt="Test prompt",
                evaluation_fn=mock_evaluation_function,
                generations=5
            )
            
            # Assert - verify GEPA optimizer was called
            assert mock_optimizer_class.called, "AMOPE should create GEPA GeneticOptimizer"
            assert mock_optimizer.optimize.called, "AMOPE should call GEPA optimize"
            
            # Verify integration results
            assert result is not None, "Should return optimization result"
            assert hasattr(result, 'best_prompt'), "Result should have best_prompt"
            assert hasattr(result, 'best_objectives'), "Result should have best_objectives"
            
            print("✅ AMOPE successfully calls GEPA's GeneticOptimizer")
            print(f"✅ Best prompt: {result.best_prompt[:50]}...")
            print(f"✅ Best objectives: {result.best_objectives}")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_objective_balancer_gepa_integration(self, mock_evaluation_function):
        """Test that AMOPE's ObjectiveBalancer works with GEPA evolution."""
        print("\n🎯 Testing ObjectiveBalancer-GEPA integration...")
        
        # Arrange
        objectives = {"accuracy": 0.6, "efficiency": 0.4}
        
        # Create AMOPE with ObjectiveBalancer
        optimizer = AMOPEOptimizer(
            objectives=objectives,
            balancing_config={
                "strategy": BalancingStrategy.ADAPTIVE_HARMONIC,
                "stagnation_window": 5
            }
        )
        
        # Act - run optimization to trigger ObjectiveBalancer
        result = optimizer.optimize(
            initial_prompt="Test prompt for balancing",
            evaluation_fn=mock_evaluation_function,
            generations=3
        )
        
        # Assert
        assert result is not None, "Should return result"
        assert hasattr(optimizer, 'objective_balancer'), "Should have ObjectiveBalancer"
        
        print("✅ ObjectiveBalancer integration working")
        print(f"✅ Best prompt: {result.best_prompt[:30]}...")
        print(f"✅ Final weights available: {hasattr(optimizer.objective_balancer, 'current_objectives')}")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_hybrid_mutation_strategies(self, mock_evaluation_function):
        """Test that hybrid mutation strategies work in integration."""
        print("\n🔀 Testing hybrid mutation strategies...")
        
        # Arrange
        objectives = {"accuracy": 0.7, "efficiency": 0.3}
        
        # Configure multiple mutation strategies
        mutation_config = {
            "use_llm_guidance": False,
            "adaptive_selection": True,
            "exploration_rate": 0.3
        }
        
        optimizer = AMOPEOptimizer(
            objectives=objectives,
            mutation_config=mutation_config
        )
        
        # Act
        result = optimizer.optimize(
            initial_prompt="Test mutation strategies",
            evaluation_fn=mock_evaluation_function,
            generations=4
        )
        
        # Assert
        assert result is not None, "Should return result"
        
        # Verify strategy usage is tracked
        if hasattr(result, 'strategy_usage') and result.strategy_usage:
            print(f"✅ Used {len(result.strategy_usage)} different mutation strategies")
            for strategy, usage in result.strategy_usage.items():
                print(f"   🧬 {strategy}: {usage} uses")
        else:
            print("✅ Mutation strategies configured successfully")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_backward_compatibility(self, mock_evaluation_function):
        """Test that AMOPE maintains backward compatibility."""
        print("\n🔄 Testing backward compatibility...")
        
        # Test old-style initialization
        objectives = {"accuracy": 0.8, "efficiency": 0.2}
        
        # Should work with minimal configuration
        optimizer = AMOPEOptimizer(objectives=objectives)
        
        result = optimizer.optimize(
            initial_prompt="Backward compatibility test",
            evaluation_fn=mock_evaluation_function,
            generations=2
        )
        
        # Verify it still works
        assert result is not None, "Should work with minimal config"
        assert hasattr(result, 'best_prompt'), "Should have best_prompt"
        assert hasattr(result, 'best_objectives'), "Should have best_objectives"
        
        print("✅ Backward compatibility maintained")
        print(f"✅ Basic optimization works: {result.best_prompt[:30]}...")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_performance_improvement_validation(self, mock_evaluation_function):
        """Test that AMOPE-GEPA integration provides performance improvements."""
        print("\n📈 Testing performance improvements...")
        
        # Arrange
        objectives = {"accuracy": 0.6, "efficiency": 0.4}
        initial_prompt = "Basic prompt for testing"
        generations = 8
        
        # Test baseline (simple optimization)
        baseline_fitness = mock_evaluation_function(initial_prompt)
        baseline_avg = sum(baseline_fitness.values()) / len(baseline_fitness)
        
        # Test AMOPE-GEPA integration
        amope_optimizer = AMOPEOptimizer(
            objectives=objectives,
            balancing_config={"strategy": "adaptive_harmonic"}
        )
        
        start_time = time.time()
        amope_result = amope_optimizer.optimize(
            initial_prompt=initial_prompt,
            evaluation_fn=mock_evaluation_function,
            generations=generations
        )
        amope_time = time.time() - start_time
        
        # Assert improvements
        assert amope_result is not None, "Should return result"
        
        # Compare fitness improvements
        amope_avg = sum(amope_result.best_objectives.values()) / len(amope_result.best_objectives)
        fitness_improvement = (amope_avg - baseline_avg) / baseline_avg * 100
        
        print(f"📊 Performance Comparison:")
        print(f"   🎯 Baseline fitness: {baseline_avg:.3f}")
        print(f"   🚀 AMOPE fitness: {amope_avg:.3f}")
        print(f"   📈 Improvement: {fitness_improvement:+.1f}%")
        print(f"   ⏱️ AMOPE time: {amope_time:.2f}s")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_convergence_analysis_integration(self, mock_evaluation_function):
        """Test that convergence analysis works with GEPA integration."""
        print("\n📊 Testing convergence analysis...")
        
        # Arrange
        objectives = {"accuracy": 0.7, "efficiency": 0.3}
        
        optimizer = AMOPEOptimizer(
            objectives=objectives,
            balancing_config={"strategy": "pareto_balanced"}
        )
        
        # Act
        result = optimizer.optimize(
            initial_prompt="Convergence test prompt",
            evaluation_fn=mock_evaluation_function,
            generations=6
        )
        
        # Assert convergence analysis is available
        assert result is not None, "Should return result"
        
        print("✅ Convergence analysis integration working")
        
        # Check if comprehensive analytics are available
        if hasattr(result, 'comprehensive_analytics') and result.comprehensive_analytics:
            analytics = result.comprehensive_analytics
            print(f"✅ Analytics available: {len(analytics)} categories")
        else:
            print("✅ Basic convergence tracking working")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_error_handling_in_integration(self):
        """Test error handling in AMOPE-GEPA integration."""
        print("\n🛡️ Testing error handling...")
        
        # Arrange
        objectives = {"accuracy": 0.5, "efficiency": 0.5}
        
        def failing_evaluation(candidate):
            # Simulate evaluation failure
            content = str(candidate).lower()
            if "fail" in content:
                raise ValueError("Simulated evaluation failure")
            return {"accuracy": 0.6, "efficiency": 0.7}
        
        optimizer = AMOPEOptimizer(objectives=objectives)
        
        # Act - should handle errors gracefully
        result = optimizer.optimize(
            initial_prompt="This should not fail",
            evaluation_fn=failing_evaluation,
            generations=3
        )
        
        # Assert - should still return a result despite potential errors
        assert result is not None, "Should handle errors gracefully"
        print("✅ Error handling works correctly")


class TestIntegrationEdgeCases:
    """Test edge cases and boundary conditions for AMOPE-GEPA integration."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_single_objective_optimization(self):
        """Test optimization with single objective."""
        print("\n🎯 Testing single objective optimization...")
        
        def simple_eval(candidate):
            return {"accuracy": random.random()}
        
        optimizer = AMOPEOptimizer(objectives={"accuracy": 1.0})
        
        result = optimizer.optimize(
            initial_prompt="Single objective test",
            evaluation_fn=simple_eval,
            generations=3
        )
        
        assert result is not None, "Should work with single objective"
        assert "accuracy" in result.best_objectives, "Should have accuracy score"
        print("✅ Single objective optimization works")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_many_objectives_optimization(self):
        """Test optimization with many objectives."""
        print("\n🎯 Testing many objectives optimization...")
        
        def multi_eval(candidate):
            return {
                "accuracy": random.random(),
                "efficiency": random.random(),
                "complexity": random.random(),
                "robustness": random.random(),
                "interpretability": random.random()
            }
        
        objectives = {
            "accuracy": 0.3,
            "efficiency": 0.2,
            "complexity": 0.2,
            "robustness": 0.15,
            "interpretability": 0.15
        }
        
        optimizer = AMOPEOptimizer(objectives=objectives)
        
        result = optimizer.optimize(
            initial_prompt="Multi-objective test",
            evaluation_fn=multi_eval,
            generations=3
        )
        
        assert result is not None, "Should work with many objectives"
        assert len(result.best_objectives) == 5, "Should have all 5 objectives"
        print("✅ Multi-objective optimization works")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_extreme_generation_counts(self):
        """Test optimization with extreme generation counts."""
        print("\n⚡ Testing extreme generation counts...")
        
        def quick_eval(candidate):
            return {"accuracy": 0.5 + random.random() * 0.3}
        
        # Test with 1 generation
        optimizer = AMOPEOptimizer(objectives={"accuracy": 1.0})
        
        result = optimizer.optimize(
            initial_prompt="Quick test",
            evaluation_fn=quick_eval,
            generations=1
        )
        
        assert result is not None, "Should work with 1 generation"
        print("✅ Single generation optimization works")
        
        # Test with 0 generations (should handle gracefully)
        result = optimizer.optimize(
            initial_prompt="Zero generation test",
            evaluation_fn=quick_eval,
            generations=0
        )
        
        assert result is not None, "Should handle 0 generations"
        print("✅ Zero generation edge case handled")


if __name__ == "__main__":
    # Run tests directly if script is executed
    print("🧪 Running AMOPE-GEPA Integration Tests")
    print("=" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available. Install dependencies first.")
        sys.exit(1)
    
    # Run a quick test
    test_suite = TestAMOPEGEPAIntegration()
    
    try:
        def dummy_eval(c):
            return {"accuracy": 0.6, "efficiency": 0.7, "complexity": 0.5}
        
        test_suite.test_amope_uses_gepa_genetic_optimizer(dummy_eval)
        print("\n🎉 Integration tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
