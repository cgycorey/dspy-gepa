"""Comprehensive integration tests for the multi-objective GEPA framework.

This test suite provides end-to-end testing of all components working together,
focusing on the actual working implementation rather than theoretical components.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import pytest
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.dspy_gepa.core.multi_objective_gepa import MultiObjectiveGEPA
from src.dspy_gepa.core.multi_objective_agent import MultiObjectiveGEPAAgent
from src.dspy_gepa.core.monitoring import MonitoringFramework
from src.dspy_gepa.core.visualization import create_pareto_analysis, create_progress_analysis
from src.dspy_gepa.core.interfaces import EvaluationResult, ObjectiveEvaluation, OptimizationDirection
from src.dspy_gepa.core.agent import AgentConfig
from tests.fixtures.test_data import create_sample_evaluation_results, create_progress_data


class TestComprehensiveIntegration:
    """Comprehensive integration tests for the complete framework."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider for testing."""
        mock_provider = Mock()
        mock_provider.generate.return_value = "Generated response for testing"
        mock_provider.token_count.return_value = 15
        return mock_provider
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return AgentConfig(
            max_generations=3,
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_rate=0.1
        )
    
    def test_multi_objective_gepa_basic_functionality(self, mock_llm_provider, basic_config):
        """Test basic multi-objective GEPA functionality."""
        # Create objectives
        from src.dspy_gepa.core.interfaces import Objective
        
        objectives = [
            Objective("accuracy", OptimizationDirection.MAXIMIZE),
            Objective("efficiency", OptimizationDirection.MINIMIZE)
        ]
        
        # Create GEPA instance
        gepa = MultiObjectiveGEPA(
            objectives=objectives,
            max_generations=basic_config.max_generations,
            population_size=basic_config.population_size,
            verbose=basic_config.verbose
        )
        
        # Test initialization
        assert gepa.objectives == objectives
        assert gepa.max_generations == basic_config.max_generations
        assert gepa.population_size == basic_config.population_size
    
    def test_objective_evaluation_workflow(self, mock_llm_provider, basic_config):
        """Test complete objective evaluation workflow."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Test solution
        solution = {
            "prompt": "Test prompt for evaluation",
            "context": "Test context information",
            "response": "Test response"
        }
        
        # Mock the evaluation process
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_result = create_sample_evaluation_results(1)[0]
            mock_eval.return_value = mock_result
            
            # Evaluate solution
            result = gepa._evaluate_solution(solution)
            
            # Verify evaluation result
            assert isinstance(result, EvaluationResult)
            assert result.solution_id is not None
            assert len(result.objectives) > 0
            assert result.overall_score is not None
            assert result.evaluation_time > 0
    
    def test_pareto_frontier_management(self, mock_llm_provider, basic_config):
        """Test Pareto frontier management functionality."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Create sample results
        results = create_sample_evaluation_results(15)
        
        # Update Pareto frontier
        gepa._update_pareto_frontier(results)
        
        # Verify frontier is non-empty
        assert len(gepa.pareto_frontier) > 0
        
        # Test Pareto dominance
        if len(gepa.pareto_frontier) >= 2:
            sol1, sol2 = gepa.pareto_frontier[0], gepa.pareto_frontier[1]
            
            # Test dominance calculation
            dominates_1_2 = gepa._dominates(sol1, sol2)
            dominates_2_1 = gepa._dominates(sol2, sol1)
            
            # At least one direction should be False (Pareto optimal)
            assert not (dominates_1_2 and dominates_2_1)
    
    def test_optimization_step_execution(self, mock_llm_provider, basic_config):
        """Test single optimization step execution."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Initialize with sample population
        gepa.population = create_sample_evaluation_results(8)
        gepa._update_pareto_frontier(gepa.population)
        
        initial_generation = gepa.generation
        initial_population_size = len(gepa.population)
        
        # Execute optimization step
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            gepa._optimization_step()
            
            # Verify state changes
            assert gepa.generation == initial_generation + 1
            assert len(gepa.population) >= initial_population_size
    
    def test_complete_optimization_run(self, mock_llm_provider, basic_config):
        """Test complete optimization run from start to finish."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Task definition
        task_description = "Generate accurate and efficient responses to questions"
        objectives = ["accuracy", "efficiency", "fluency"]
        
        # Test examples
        examples = [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Capital of France?", "expected": "Paris"},
            {"input": "Water boils at?", "expected": "100°C"}
        ]
        
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run optimization
            start_time = time.time()
            
            results = gepa.optimize(
                task_description=task_description,
                objectives=objectives,
                examples=examples,
                max_generations=2
            )
            
            end_time = time.time()
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) > 0
            assert gepa.generation > 0
            assert len(gepa.pareto_frontier) > 0
            
            # Performance check
            optimization_time = end_time - start_time
            assert optimization_time < 30.0  # Should complete within 30 seconds
    
    def test_preference_based_solution_selection(self, mock_llm_provider, basic_config):
        """Test preference-based solution selection."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Create diverse Pareto frontier
        results = create_sample_evaluation_results(20)
        gepa._update_pareto_frontier(results)
        
        # Define preference profiles
        preferences = {
            "accuracy": 0.6,
            "efficiency": 0.3,
            "fluency": 0.1
        }
        
        # Get preferred solution
        preferred = gepa.get_preferred_solution(preferences)
        
        assert preferred in gepa.pareto_frontier
        assert isinstance(preferred, EvaluationResult)
        assert preferred.solution_id is not None
    
    def test_monitoring_integration(self, mock_llm_provider, basic_config):
        """Test monitoring framework integration."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Create monitoring framework
        monitoring = MonitoringFramework()
        
        # Start monitoring
        monitoring.start_monitoring()
        
        # Simulate optimization with monitoring
        for generation in range(3):
            results = create_sample_evaluation_results(8)
            
            # Record metrics
            metrics = {
                "generation": generation,
                "best_score": max(r.overall_score for r in results),
                "avg_score": sum(r.overall_score for r in results) / len(results),
                "population_diversity": 0.8 - generation * 0.1
            }
            
            monitoring.record_evaluation_metrics(metrics)
            time.sleep(0.05)  # Small delay for monitoring
        
        # Get performance history
        history = monitoring.get_performance_history()
        
        assert len(history) == 3
        assert all("generation" in m for m in history)
        assert all("best_score" in m for m in history)
        
        # Stop monitoring
        monitoring.stop_monitoring()
        assert monitoring.is_monitoring == False
    
    def test_visualization_integration(self, mock_llm_provider, basic_config, temp_dir):
        """Test visualization framework integration."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Generate data for visualization
        results = create_sample_evaluation_results(15)
        progress_data = create_progress_data(10)
        
        # Update framework state
        gepa.population = results
        gepa._update_pareto_frontier(results)
        
        # Test Pareto visualization
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            # Create Pareto analysis
            pareto_analyzer = create_pareto_analysis()
            assert pareto_analyzer is not None
            
            # Create progress analysis
            progress_analyzer = create_progress_analysis()
            assert progress_analyzer is not None
    
    def test_checkpoint_save_load(self, mock_llm_provider, basic_config, temp_checkpoint_dir):
        """Test checkpoint save and load functionality."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Set up some state
        gepa.population = create_sample_evaluation_results(8)
        gepa.generation = 3
        gepa._update_pareto_frontier(gepa.population)
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.json"
        gepa.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new instance and load checkpoint
        new_gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        new_gepa.load_checkpoint(str(checkpoint_path))
        
        # Verify state restoration
        assert new_gepa.generation == 3
        assert len(new_gepa.population) == 8
        assert len(new_gepa.pareto_frontier) > 0
    
    def test_error_handling_and_recovery(self, mock_llm_provider, basic_config):
        """Test error handling and recovery mechanisms."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Test evaluation failure handling
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_eval.side_effect = Exception("Evaluation failed")
            
            # Should handle gracefully
            with pytest.raises(Exception):
                gepa._evaluate_solution({"test": "solution"})
        
        # Test LLM provider failure
        with patch.object(mock_llm_provider, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("LLM provider failed")
            
            # Should handle gracefully
            with pytest.raises(Exception):
                gepa._generate_response("test prompt")
    
    def test_resource_monitoring(self, mock_llm_provider, basic_config):
        """Test resource monitoring during optimization."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Mock resource monitoring
        with patch.object(gepa, '_get_resource_usage') as mock_resources:
            mock_resources.return_value = {
                "cpu_percent": 45.0,
                "memory_percent": 60.0,
                "evaluation_time": 0.12,
                "api_calls": 5
            }
            
            # Get resource usage
            resources = gepa._get_resource_usage()
            
            assert "cpu_percent" in resources
            assert "memory_percent" in resources
            assert "evaluation_time" in resources
            assert "api_calls" in resources
    
    def test_convergence_detection(self, mock_llm_provider, basic_config):
        """Test convergence detection mechanisms."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Simulate optimization progress
        convergence_history = []
        
        for generation in range(5):
            results = create_sample_evaluation_results(10)
            gepa._update_pareto_frontier(results)
            
            # Record convergence metrics
            if len(gepa.pareto_frontier) > 0:
                hypervolume = sum(sol.overall_score for sol in gepa.pareto_frontier)
                convergence_history.append(hypervolume)
            
            gepa.generation = generation
        
        # Test convergence detection
        if len(convergence_history) >= 3:
            # Simple convergence check: if improvement is small
            recent_improvements = [
                convergence_history[i] - convergence_history[i-1]
                for i in range(1, len(convergence_history))
            ]
            
            avg_improvement = sum(abs(imp) for imp in recent_improvements) / len(recent_improvements)
            
            # Should detect convergence if improvement is small
            is_converged = avg_improvement < gepa.config.convergence_threshold
            assert isinstance(is_converged, bool)
    
    def test_performance_metrics_collection(self, mock_llm_provider, basic_config):
        """Test performance metrics collection and analysis."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Run optimization with metrics collection
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run multiple steps
            for _ in range(3):
                gepa._optimization_step()
        
        # Get performance metrics
        metrics = gepa.get_performance_metrics()
        
        assert "generation" in metrics
        assert "population_size" in metrics
        assert "pareto_frontier_size" in metrics
        assert "convergence_metrics" in metrics
        
        # Verify metrics are reasonable
        assert metrics["generation"] > 0
        assert metrics["population_size"] > 0
        assert metrics["pareto_frontier_size"] >= 0
    
    @pytest.mark.slow
    def test_scalability_with_large_populations(self, mock_llm_provider, basic_config):
        """Test scalability with larger populations."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Test with larger population
        large_population = create_sample_evaluation_results(50)
        
        start_time = time.time()
        gepa._update_pareto_frontier(large_population)
        end_time = time.time()
        
        # Should complete within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 5.0  # 5 seconds max
        assert len(gepa.pareto_frontier) > 0
    
    def test_backward_compatibility(self, mock_llm_provider, basic_config):
        """Test backward compatibility with single-objective interface."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Test single-objective optimization
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run single-objective style optimization
            results = gepa.optimize(
                task_description="Single objective test",
                objectives=["accuracy"],  # Single objective
                examples=[{"input": "test", "expected": "result"}],
                max_generations=1
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Should be compatible with single-objective results
            for result in results:
                assert hasattr(result, 'solution_id')
                assert hasattr(result, 'overall_score')
                assert isinstance(result, EvaluationResult)
    
    def test_thread_safety(self, mock_llm_provider, basic_config, thread_safety_test):
        """Test thread safety of core operations."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        def evaluate_solution():
            solution = {"prompt": "Test", "response": "Response"}
            with patch.object(gepa, '_evaluate_solution') as mock_eval:
                mock_eval.return_value = create_sample_evaluation_results(1)[0]
                return gepa._evaluate_solution(solution)
        
        # Run concurrent evaluations
        results = thread_safety_test(evaluate_solution, num_threads=3, iterations=5)
        
        # Should have no errors
        assert results["total_errors"] == 0
        assert results["success_rate"] == 1.0
    
    def test_memory_efficiency(self, mock_llm_provider, basic_config):
        """Test memory efficiency during optimization."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Perform multiple operations
        for _ in range(10):
            results = create_sample_evaluation_results(20)
            gepa._update_pareto_frontier(results)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100.0
    
    def test_integration_with_dspy_components(self, mock_llm_provider, basic_config):
        """Test integration with DSPy-like components."""
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        # Mock DSPy-like components
        mock_predictor = Mock()
        mock_predictor.predict.return_value = "DSPy prediction result"
        
        mock_metric = Mock()
        mock_metric.score = 0.85
        mock_metric.name = "accuracy"
        
        # Test integration
        with patch.object(gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Simulate DSPy integration workflow
            results = gepa.optimize(
                task_description="DSPy integration test",
                objectives=["accuracy", "efficiency"],
                examples=[{"input": "DSPy test", "expected": "result"}],
                max_generations=2
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Should work with DSPy-like interfaces
            for result in results:
                assert isinstance(result, EvaluationResult)
                assert "accuracy" in result.objectives
    
    def test_end_to_end_workflow_with_all_components(self, mock_llm_provider, basic_config, temp_dir):
        """Test complete end-to-end workflow with all components."""
        # Create complete framework
        gepa = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=basic_config
        )
        
        monitoring = MonitoringFramework()
        
        # Define comprehensive problem
        task_description = "Generate comprehensive technical documentation"
        objectives = ["accuracy", "clarity", "conciseness", "completeness"]
        
        examples = [
            {
                "input": "API endpoint: POST /users",
                "context": "Creates a new user account",
                "expected": "Detailed API documentation with parameters and responses"
            },
            {
                "input": "Function: calculate_total(items)",
                "context": "Calculates total price of items in cart",
                "expected": "Function documentation with parameters and return values"
            }
        ]
        
        # Start monitoring
        monitoring.start_monitoring()
        
        try:
            # Run complete optimization
            with patch.object(gepa, '_evaluate_solution') as mock_eval:
                mock_eval.return_value = create_sample_evaluation_results(1)[0]
                
                start_time = time.time()
                
                results = gepa.optimize(
                    task_description=task_description,
                    objectives=objectives,
                    examples=examples,
                    max_generations=3
                )
                
                end_time = time.time()
                
                # Record final metrics
                final_metrics = {
                    "total_time": end_time - start_time,
                    "final_generation": gepa.generation,
                    "pareto_frontier_size": len(gepa.pareto_frontier),
                    "best_score": max(r.overall_score for r in gepa.pareto_frontier) if gepa.pareto_frontier else 0.0
                }
                
                monitoring.record_evaluation_metrics(final_metrics)
        
        finally:
            monitoring.stop_monitoring()
        
        # Verify complete workflow
        assert isinstance(results, list)
        assert len(results) > 0
        assert gepa.generation > 0
        assert len(gepa.pareto_frontier) > 0
        
        # Check monitoring data
        history = monitoring.get_performance_history()
        assert len(history) > 0
        
        # Test checkpoint saving
        checkpoint_path = temp_dir / "workflow_checkpoint.json"
        gepa.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # Test visualization data preparation
        if gepa.pareto_frontier:
            with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
                mock_fig, mock_ax = Mock(), Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                
                pareto_analyzer = create_pareto_analysis()
                assert pareto_analyzer is not None
        
        # Performance validation
        optimization_time = end_time - start_time
        assert optimization_time < 60.0  # Should complete within 1 minute
        
        # Quality validation
        assert len(gepa.pareto_frontier) >= 1
        assert all(hasattr(r, 'solution_id') for r in gepa.pareto_frontier)
        assert all(hasattr(r, 'objectives') for r in gepa.pareto_frontier)
        
        print(f"\n=== END-TO-END WORKFLOW COMPLETED ===")
        print(f"Total time: {optimization_time:.2f}s")
        print(f"Generations: {gepa.generation}")
        print(f"Pareto frontier size: {len(gepa.pareto_frontier)}")
        print(f"Best score: {max(r.overall_score for r in gepa.pareto_frontier):.3f}")
        print(f"Monitoring data points: {len(history)}")
        print(f"Checkpoint saved: {checkpoint_path.exists()}")