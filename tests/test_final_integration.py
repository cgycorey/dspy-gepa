#!/usr/bin/env python3
"""Final Integration Test for DSPY-GEPA Implementation

This test validates that our complete DSPY-GEPA implementation works correctly
by testing all major components and their integration. It covers:

1. Component imports and basic functionality
2. Core GEPA algorithm components  
3. AMOPE adaptive mutation and objective balancing
4. DSPY integration layer components
5. End-to-end optimization workflow
6. Example code execution

Run with: python -m pytest tests/test_final_integration.py -v
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestComponentImports:
    """Test that all major components can be imported correctly."""
    
    def test_core_gepa_imports(self):
        """Test core GEPA component imports."""
        try:
            from gepa import (
                Candidate,
                GeneticOptimizer, 
                ParetoSelector,
                TextMutator,
                ExecutionTrace,
                MutationRecord,
                OptimizationConfig
            )
            assert True  # All imports successful
        except ImportError as e:
            pytest.fail(f"Core GEPA import failed: {e}")
    
    def test_amope_imports(self):
        """Test AMOPE algorithm component imports."""
        try:
            from dspy_gepa.amope import (
                AMOPEOptimizer,
                AMOPEConfig,
                AdaptiveMutator,
                MutationStrategy,
                ObjectiveBalancer,
                BalancingStrategy,
                OptimizationResult
            )
            assert True  # All imports successful
        except ImportError as e:
            pytest.fail(f"AMOPE import failed: {e}")
    
    def test_dspy_integration_imports(self):
        """Test DSPY integration component imports."""
        try:
            from dspy_gepa.dspy_integration import (
                DSPYAdapter,
                MetricCollector,
                DSPYMetrics
            )
            assert True  # All imports successful
        except ImportError as e:
            pytest.fail(f"DSPY integration import failed: {e}")
    
    def test_main_package_imports(self):
        """Test main package imports and version info."""
        try:
            import dspy_gepa
            from dspy_gepa import get_version_info
            
            # Check version info
            version_info = get_version_info()
            assert "version" in version_info
            assert version_info["version"] == "0.1.0"
            assert "name" in version_info
            assert version_info["name"] == "DSPY-GEPA"
            
        except ImportError as e:
            pytest.fail(f"Main package import failed: {e}")


class TestCoreGEPAComponents:
    """Test core GEPA algorithm components."""
    
    def test_candidate_creation(self):
        """Test Candidate class creation and basic functionality."""
        from gepa import Candidate, ExecutionTrace
        
        # Create a basic candidate
        candidate = Candidate(
            content="Test prompt for optimization",
            generation=0,
            fitness_scores={"accuracy": 0.8, "efficiency": 0.7}
        )
        
        # Verify basic properties
        assert candidate.content == "Test prompt for optimization"
        assert candidate.generation == 0
        assert "accuracy" in candidate.fitness_scores
        assert "efficiency" in candidate.fitness_scores
        assert candidate.id is not None  # ID should be auto-generated
        assert candidate.parent_ids == []  # Default empty parent list
        
        # Test fitness score calculation
        combined_score = sum(candidate.fitness_scores.values()) / len(candidate.fitness_scores)
        assert isinstance(combined_score, float)
    
    def test_pareto_selector(self):
        """Test ParetoSelector functionality."""
        from gepa import ParetoSelector, Candidate
        
        # Create test candidates with different fitness scores
        candidates = [
            Candidate(
                content="Candidate 1",
                generation=0,
                fitness_scores={"accuracy": 0.9, "efficiency": 0.6}
            ),
            Candidate(
                content="Candidate 2", 
                generation=0,
                fitness_scores={"accuracy": 0.7, "efficiency": 0.8}
            ),
            Candidate(
                content="Candidate 3",
                generation=0,
                fitness_scores={"accuracy": 0.8, "efficiency": 0.7}
            )
        ]
        
        # Test Pareto selection
        selector = ParetoSelector(objectives=["accuracy", "efficiency"])
        selected = selector.environmental_selection(candidates, target_size=2)
        
        # Should return 2 candidates
        assert len(selected) <= 2
        assert all(isinstance(c, Candidate) for c in selected)


class TestAMOPEAlgorithm:
    """Test AMOPE adaptive algorithm components."""
    
    def test_adaptive_mutator_creation(self):
        """Test AdaptiveMutator initialization and basic functionality."""
        from dspy_gepa.amope import AdaptiveMutator, MutationStrategy
        
        # Create mutator with different strategies
        for strategy in [MutationStrategy.GRADIENT_BASED, 
                       MutationStrategy.PATTERN_BASED,
                       MutationStrategy.STATISTICAL]:
            
            mutator = AdaptiveMutator(
                strategy=strategy,
                mutation_rate=0.3,
                strength_factor=0.1
            )
            
            assert mutator.strategy == strategy
            assert mutator.mutation_rate == 0.3
            assert mutator.strength_factor == 0.1
            
            # Test basic mutation
            test_prompt = "This is a test prompt for optimization"
            result = mutator.mutate(test_prompt)
            
            # Should return MutationResult
            assert hasattr(result, 'mutated_content')
            assert hasattr(result, 'strategy_used')
            assert hasattr(result, 'confidence_score')
            assert len(result.mutated_content) > 0
    
    def test_objective_balancer(self):
        """Test ObjectiveBalancer functionality."""
        from dspy_gepa.amope import ObjectiveBalancer, BalancingStrategy
        
        # Initialize balancer with test objectives
        objectives = {"accuracy": 0.7, "efficiency": 0.3}
        balancer = ObjectiveBalancer(
            objectives=objectives,
            strategy=BalancingStrategy.ADAPTIVE_HARMONIC,
            stagnation_window=10
        )
        
        # Verify initialization
        assert balancer.objectives == objectives
        assert balancer.strategy == BalancingStrategy.ADAPTIVE_HARMONIC
        assert balancer.stagnation_window == 10
        
        # Test weight updates with mock progress data
        objective_progress = {
            "accuracy": [0.5, 0.6, 0.65, 0.68, 0.70],
            "efficiency": [0.4, 0.45, 0.42, 0.48, 0.50]
        }
        
        initial_weights = dict(balancer.objectives)
        balancer.update_weights(objective_progress)
        
        # Weights should potentially change based on progress
        assert isinstance(balancer.objectives, dict)
        assert len(balancer.objectives) == 2
    
    def test_amope_optimizer_basic(self):
        """Test AMOPEOptimizer basic functionality."""
        from dspy_gepa.amope import AMOPEOptimizer, AMOPEConfig
        
        # Create mock evaluation function
        def mock_evaluation_fn(prompt: str) -> Dict[str, float]:
            # Simulate evaluation based on prompt length (longer = better)
            length_score = min(len(prompt) / 100, 1.0)
            return {
                "accuracy": length_score * 0.8,
                "efficiency": length_score * 0.6,
                "complexity": min(length_score * 1.2, 1.0)
            }
        
        # Initialize optimizer
        optimizer = AMOPEOptimizer(
            objectives={"accuracy": 0.5, "efficiency": 0.3, "complexity": 0.2},
            population_size=3,
            max_generations=5,
            verbose=False  # Reduce output for test
        )
        
        # Run optimization
        initial_prompt = "Test prompt"
        result = optimizer.optimize(
            initial_prompt=initial_prompt,
            evaluation_fn=mock_evaluation_fn,
            generations=3
        )
        
        # Verify result structure
        assert hasattr(result, 'best_prompt')
        assert hasattr(result, 'best_score')
        assert hasattr(result, 'best_objectives')
        assert hasattr(result, 'generations_completed')
        assert hasattr(result, 'optimization_history')
        
        # Verify optimization occurred
        assert result.generations_completed > 0
        assert len(result.optimization_history) > 0
        assert len(result.best_prompt) > 0
        assert isinstance(result.best_score, float)
        assert isinstance(result.best_objectives, dict)


class TestDSPYIntegration:
    """Test DSPY integration layer components."""
    
    def test_metric_collector(self):
        """Test MetricCollector functionality."""
        from dspy_gepa.dspy_integration import MetricCollector, DSPYMetrics, ResourceUsage
        
        # Create collector
        collector = MetricCollector()
        
        # Test basic metrics creation
        metrics = DSPYMetrics(
            execution_time=1.5,
            success=True,
            total_predictions=10,
            successful_predictions=8,
            accuracy=0.8,
            estimated_cost=0.05
        )
        
        # Verify metrics structure
        assert metrics.execution_time == 1.5
        assert metrics.success is True
        assert metrics.total_predictions == 10
        assert metrics.accuracy == 0.8
        assert isinstance(metrics.resource_usage, ResourceUsage)


class TestEndToEndWorkflow:
    """Test complete end-to-end optimization workflow."""
    
    def test_complete_optimization_pipeline(self):
        """Test complete pipeline from candidate creation to optimization."""
        from gepa import Candidate, ParetoSelector
        from dspy_gepa.amope import AMOPEOptimizer, AdaptiveMutator, ObjectiveBalancer
        
        # Step 1: Create initial candidates
        initial_prompts = [
            "Simple prompt for testing",
            "More detailed prompt with context",
            "Comprehensive prompt with examples"
        ]
        
        candidates = []
        for i, prompt in enumerate(initial_prompts):
            candidate = Candidate(
                content=prompt,
                generation=0,
                fitness_scores={"accuracy": 0.5 + i * 0.1, "efficiency": 0.8 - i * 0.1}
            )
            candidates.append(candidate)
        
        assert len(candidates) == 3
        
        # Step 2: Test mutation process
        mutator = AdaptiveMutator(strategy=MutationStrategy.PATTERN_BASED)
        mutated_candidates = []
        
        for candidate in candidates:
            mutation_result = mutator.mutate(candidate.content)
            mutated_candidate = Candidate(
                content=mutation_result.mutated_content,
                generation=candidate.generation + 1,
                fitness_scores=candidate.fitness_scores.copy(),
                parent_ids=[candidate.id]
            )
            mutated_candidates.append(mutated_candidate)
        
        assert len(mutated_candidates) == 3
        
        # Step 3: Test selection process
        selector = ParetoSelector(objectives=["accuracy", "efficiency"])
        selected = selector.select(candidates + mutated_candidates, k=2)
        
        assert len(selected) <= 2
        
        # Step 4: Test AMOPE optimization integration
        def simple_evaluation(prompt: str) -> Dict[str, float]:
            # Simple evaluation based on content
            return {
                "accuracy": min(len(prompt.split()) / 10, 1.0),
                "efficiency": 0.8 - (len(prompt) / 200),
                "robustness": 0.7
            }
        
        optimizer = AMOPEOptimizer(
            objectives={"accuracy": 0.4, "efficiency": 0.4, "robustness": 0.2},
            population_size=2,
            max_generations=3,
            verbose=False
        )
        
        result = optimizer.optimize(
            initial_prompt="Test optimization workflow",
            evaluation_fn=simple_evaluation,
            generations=2
        )
        
        # Verify complete workflow success
        assert result.generations_completed > 0
        assert len(result.optimization_history) > 0
        assert isinstance(result.best_objectives, dict)


class TestExamplesValidation:
    """Test that example code can be imported and basic functionality works."""
    
    def test_basic_examples_import(self):
        """Test that example files can be imported."""
        examples_dir = Path(__file__).parent.parent / "examples"
        
        # Check that example files exist
        basic_dspy_gepa = examples_dir / "basic_dspy_gepa.py"
        basic_agent = examples_dir / "basic_agent.py"
        
        assert basic_dspy_gepa.exists(), "basic_dspy_gepa.py example missing"
        assert basic_agent.exists(), "basic_agent.py example missing"
    
    def test_example_code_structure(self):
        """Test that example code has correct structure."""
        # Read example file to check structure
        examples_dir = Path(__file__).parent.parent / "examples"
        example_file = examples_dir / "basic_dspy_gepa.py"
        
        with open(example_file, 'r') as f:
            content = f.read()
        
        # Check that example contains expected components
        assert "import dspy" in content
        assert "from gepa import" in content
        assert "from dspy_gepa.dspy_integration import" in content
        assert "class" in content  # Should contain class definitions
        assert "def" in content    # Should contain function definitions


class TestConfigurationAndEnvironment:
    """Test configuration and environment setup."""
    
    def test_project_structure(self):
        """Test that project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check key directories exist
        src_dir = project_root / "src"
        tests_dir = project_root / "tests"
        examples_dir = project_root / "examples"
        
        assert src_dir.exists(), "src directory missing"
        assert tests_dir.exists(), "tests directory missing"
        assert examples_dir.exists(), "examples directory missing"
        
        # Check key modules exist
        gepa_module = src_dir / "gepa"
        dspy_gepa_module = src_dir / "dspy_gepa"
        
        assert gepa_module.exists(), "gepa module missing"
        assert dspy_gepa_module.exists(), "dspy_gepa module missing"
    
    def test_version_consistency(self):
        """Test version consistency across modules."""
        import dspy_gepa
        from dspy_gepa.amope import __version__ as amope_version
        from dspy_gepa.dspy_integration import __version__ as integration_version
        
        # All should have consistent version
        assert dspy_gepa.__version__ == "0.1.0"
        assert amope_version == "0.1.0"
        assert integration_version == "0.1.0"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
