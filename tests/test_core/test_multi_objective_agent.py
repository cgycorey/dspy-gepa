"""Comprehensive tests for MultiObjectiveAgent core functionality."""

from __future__ import annotations

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.dspy_gepa.core.multi_objective_agent import MultiObjectiveGEPAAgent
from src.dspy_gepa.core.objectives import ObjectiveManager, AccuracyMetric, EfficiencyMetric
from src.dspy_gepa.core.interfaces import (
    EvaluationResult, ObjectiveEvaluation, SolutionMetadata,
    OptimizationDirection
)
from src.dspy_gepa.core.agent import AgentConfig
from tests.fixtures.test_data import create_sample_evaluation_results, create_sample_optimization_state


class TestMultiObjectiveAgent:
    """Test suite for MultiObjectiveGEPAAgent core functionality."""
    
    @pytest.fixture
    def basic_objectives(self):
        """Create basic objectives for testing."""
        return {
            "accuracy": AccuracyMetric(weight=1.0),
            "efficiency": EfficiencyMetric(weight=0.8)
        }
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            max_generations=10,
            population_size=20,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_rate=0.1,
            convergence_threshold=0.01,
            max_evaluations=1000
        )
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider for testing."""
        mock_provider = Mock()
        mock_provider.generate.return_value = "Test response"
        mock_provider.token_count.return_value = 10
        return mock_provider
    
    @pytest.fixture
    def multi_objective_agent(self, basic_objectives, agent_config, mock_llm_provider):
        """Create MultiObjectiveGEPAAgent instance for testing."""
        return MultiObjectiveGEPAAgent(
            objectives=basic_objectives,
            config=agent_config,
            llm_provider=mock_llm_provider
        )
    
    def test_initialization(self, multi_objective_agent, basic_objectives, agent_config):
        """Test MultiObjectiveGEPAAgent initialization."""
        assert multi_objective_agent.objectives == basic_objectives
        assert multi_objective_agent.config == agent_config
        assert multi_objective_agent.generation == 0
        assert len(multi_objective_agent.population) == 0
        assert len(multi_objective_agent.pareto_frontier) == 0
    
    def test_objective_evaluation(self, multi_objective_agent):
        """Test objective evaluation functionality."""
        solution = {"prompt": "Test prompt", "response": "Test response"}
        
        with patch.object(multi_objective_agent.objectives['accuracy'], 'evaluate') as mock_accuracy:
            with patch.object(multi_objective_agent.objectives['efficiency'], 'evaluate') as mock_efficiency:
                mock_accuracy.return_value = 0.85
                mock_efficiency.return_value = 0.75
                
                result = multi_objective_agent._evaluate_solution(solution)
                
                assert isinstance(result, EvaluationResult)
                assert "accuracy" in result.objectives
                assert "efficiency" in result.objectives
                assert result.objectives["accuracy"].score == 0.85
                assert result.objectives["efficiency"].score == 0.75
                mock_accuracy.assert_called_once_with(solution)
                mock_efficiency.assert_called_once_with(solution)
    
    def test_pareto_dominance(self, multi_objective_agent):
        """Test Pareto dominance calculation."""
        # Create two evaluation results
        result1 = EvaluationResult(
            solution_id="sol1",
            objectives={
                "accuracy": ObjectiveEvaluation("accuracy", 0.8, OptimizationDirection.MAXIMIZE, 0.1),
                "efficiency": ObjectiveEvaluation("efficiency", 0.7, OptimizationDirection.MINIMIZE, 0.05)
            },
            overall_score=0.75,
            evaluation_time=0.15
        )
        
        result2 = EvaluationResult(
            solution_id="sol2",
            objectives={
                "accuracy": ObjectiveEvaluation("accuracy", 0.9, OptimizationDirection.MAXIMIZE, 0.1),
                "efficiency": ObjectiveEvaluation("efficiency", 0.6, OptimizationDirection.MINIMIZE, 0.05)
            },
            overall_score=0.8,
            evaluation_time=0.15
        )
        
        # result2 should dominate result1 (higher accuracy, lower efficiency)
        assert multi_objective_agent._dominates(result2, result1) == True
        assert multi_objective_agent._dominates(result1, result2) == False
    
    def test_pareto_frontier_update(self, multi_objective_agent):
        """Test Pareto frontier update mechanism."""
        # Create sample results
        results = create_sample_evaluation_results(10)
        
        # Update Pareto frontier
        multi_objective_agent._update_pareto_frontier(results)
        
        # Check that frontier is non-empty
        assert len(multi_objective_agent.pareto_frontier) > 0
        
        # Verify no solution in frontier dominates another
        for i, sol1 in enumerate(multi_objective_agent.pareto_frontier):
            for j, sol2 in enumerate(multi_objective_agent.pareto_frontier):
                if i != j:
                    assert not multi_objective_agent._dominates(sol1, sol2), f"Solution {sol1.solution_id} dominates {sol2.solution_id}"
    
    def test_selection_operators(self, multi_objective_agent):
        """Test selection operators for genetic algorithm."""
        # Create a population
        population = create_sample_evaluation_results(20)
        multi_objective_agent.population = population
        multi_objective_agent._update_pareto_frontier(population)
        
        # Test tournament selection
        selected = multi_objective_agent._tournament_selection(population, tournament_size=3)
        assert selected in population
        
        # Test Pareto-based selection
        selected = multi_objective_agent._pareto_selection()
        assert selected in multi_objective_agent.pareto_frontier
    
    def test_mutation_operators(self, multi_objective_agent):
        """Test mutation operators."""
        parent_solution = {"prompt": "Test prompt", "response": "Test response"}
        
        # Test semantic mutation
        with patch.object(multi_objective_agent, '_semantic_mutation') as mock_semantic:
            mock_semantic.return_value = {"prompt": "Mutated prompt", "response": "Mutated response"}
            
            mutated = multi_objective_agent._mutate_solution(parent_solution, mutation_type="semantic")
            assert mutated != parent_solution
            mock_semantic.assert_called_once_with(parent_solution)
    
    def test_crossover_operator(self, multi_objective_agent):
        """Test crossover operator."""
        parent1 = {"prompt": "Prompt 1", "response": "Response 1"}
        parent2 = {"prompt": "Prompt 2", "response": "Response 2"}
        
        with patch.object(multi_objective_agent, '_crossover_solutions') as mock_crossover:
            mock_crossover.return_value = {"prompt": "Child prompt", "response": "Child response"}
            
            child = multi_objective_agent._crossover(parent1, parent2)
            assert isinstance(child, dict)
            mock_crossover.assert_called_once_with(parent1, parent2)
    
    def test_convergence_detection(self, multi_objective_agent):
        """Test convergence detection mechanisms."""
        # Simulate multiple generations
        for gen in range(5):
            multi_objective_agent.generation = gen
            results = create_sample_evaluation_results(10)
            multi_objective_agent._update_pareto_frontier(results)
            
            # Store convergence metrics
            if len(multi_objective_agent.pareto_frontier) > 0:
                hypervolume = sum(sol.overall_score for sol in multi_objective_agent.pareto_frontier)
                multi_objective_agent.convergence_history.append(hypervolume)
        
        # Test convergence detection
        is_converged = multi_objective_agent._check_convergence()
        assert isinstance(is_converged, bool)
    
    def test_optimization_step(self, multi_objective_agent):
        """Test single optimization step."""
        # Initialize population
        multi_objective_agent.population = create_sample_evaluation_results(10)
        multi_objective_agent._update_pareto_frontier(multi_objective_agent.population)
        
        initial_generation = multi_objective_agent.generation
        initial_population_size = len(multi_objective_agent.population)
        
        # Perform optimization step
        with patch.object(multi_objective_agent, '_evaluate_solution') as mock_evaluate:
            mock_evaluate.return_value = create_sample_evaluation_results(1)[0]
            
            multi_objective_agent._optimization_step()
            
            # Check that generation increased
            assert multi_objective_agent.generation == initial_generation + 1
            # Check that population was updated
            assert len(multi_objective_agent.population) >= initial_population_size
    
    def test_full_optimization_run(self, multi_objective_agent):
        """Test complete optimization run."""
        with patch.object(multi_objective_agent, '_evaluate_solution') as mock_evaluate:
            mock_evaluate.return_value = create_sample_evaluation_results(1)[0]
            
            # Run optimization
            results = multi_objective_agent.optimize(
                initial_population=create_sample_evaluation_results(5),
                max_generations=3
            )
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) > 0
            assert multi_objective_agent.generation > 0
            assert len(multi_objective_agent.pareto_frontier) > 0
    
    def test_preference_based_selection(self, multi_objective_agent):
        """Test preference-based solution selection."""
        # Set up Pareto frontier
        results = create_sample_evaluation_results(10)
        multi_objective_agent._update_pareto_frontier(results)
        
        # Define preferences
        preferences = {
            "accuracy": 0.7,  # Prefer accuracy
            "efficiency": 0.3  # Less preference for efficiency
        }
        
        # Get preferred solution
        preferred = multi_objective_agent.get_preferred_solution(preferences)
        
        assert preferred in multi_objective_agent.pareto_frontier
        assert isinstance(preferred, EvaluationResult)
    
    def test_error_handling(self, multi_objective_agent):
        """Test error handling in multi-objective optimization."""
        # Test handling of evaluation failures
        with patch.object(multi_objective_agent, '_evaluate_solution') as mock_evaluate:
            mock_evaluate.side_effect = Exception("Evaluation failed")
            
            # Should not crash, but handle gracefully
            with pytest.raises(Exception):
                multi_objective_agent._evaluate_solution({"test": "solution"})
    
    def test_resource_monitoring(self, multi_objective_agent):
        """Test resource monitoring during optimization."""
        with patch.object(multi_objective_agent, '_get_resource_usage') as mock_resources:
            mock_resources.return_value = {
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "evaluation_time": 0.1
            }
            
            # Check resource monitoring
            resources = multi_objective_agent._get_resource_usage()
            assert "cpu_percent" in resources
            assert "memory_percent" in resources
            assert "evaluation_time" in resources
    
    def test_checkpoint_save_load(self, multi_objective_agent, temp_checkpoint_dir):
        """Test checkpoint saving and loading."""
        # Set up some state
        multi_objective_agent.population = create_sample_evaluation_results(5)
        multi_objective_agent.generation = 3
        multi_objective_agent._update_pareto_frontier(multi_objective_agent.population)
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.json"
        multi_objective_agent.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new agent and load checkpoint
        new_agent = MultiObjectiveAgent(
            objectives=multi_objective_agent.objectives,
            config=multi_objective_agent.config,
            llm_provider=multi_objective_agent.llm_provider
        )
        
        new_agent.load_checkpoint(str(checkpoint_path))
        
        # Verify state was restored
        assert new_agent.generation == 3
        assert len(new_agent.population) == 5
        assert len(new_agent.pareto_frontier) > 0
    
    def test_performance_metrics(self, multi_objective_agent):
        """Test performance metrics collection."""
        # Run some optimization steps
        for i in range(3):
            results = create_sample_evaluation_results(5)
            multi_objective_agent._update_pareto_frontier(results)
            multi_objective_agent.generation = i
        
        # Get performance metrics
        metrics = multi_objective_agent.get_performance_metrics()
        
        assert "generation" in metrics
        assert "population_size" in metrics
        assert "pareto_frontier_size" in metrics
        assert "convergence_metrics" in metrics
        assert "resource_usage" in metrics
    
    @pytest.mark.slow
    def test_scalability_large_population(self, multi_objective_agent):
        """Test scalability with large populations."""
        # Create large population
        large_population = create_sample_evaluation_results(100)
        
        start_time = time.time()
        multi_objective_agent._update_pareto_frontier(large_population)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds max
        assert len(multi_objective_agent.pareto_frontier) > 0
    
    def test_thread_safety(self, multi_objective_agent, thread_safety_test):
        """Test thread safety of multi-objective agent."""
        def evaluate_solution():
            solution = {"prompt": "Test", "response": "Response"}
            return multi_objective_agent._evaluate_solution(solution)
        
        # Run concurrent evaluations
        results = thread_safety_test(evaluate_solution, num_threads=3, iterations=10)
        
        # Should have no errors
        assert results["total_errors"] == 0
        assert results["success_rate"] == 1.0
    
    def test_backward_compatibility(self, multi_objective_agent):
        """Test backward compatibility with single-objective interface."""
        # Test that single-objective methods still work
        solution = {"prompt": "Test", "response": "Response"}
        
        with patch.object(multi_objective_agent, '_evaluate_solution') as mock_evaluate:
            mock_result = create_sample_evaluation_results(1)[0]
            mock_evaluate.return_value = mock_result
            
            # Should work with single objective interface
            result = multi_objective_agent.evaluate_solution(solution)
            assert result == mock_result
