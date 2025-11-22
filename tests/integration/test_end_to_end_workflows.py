"""End-to-end workflow tests for the complete multi-objective GEPA framework."""

from __future__ import annotations

import pytest
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.dspy_gepa.core.multi_objective_gepa import MultiObjectiveGEPA
from src.dspy_gepa.core.multi_objective_agent import MultiObjectiveGEPAAgent
from src.dspy_gepa.core.monitoring import MonitoringFramework
from src.dspy_gepa.core.visualization import VisualizationManager
from src.dspy_gepa.dspy_integration.multi_objective_optimizer import MultiObjectiveOptimizer
from src.dspy_gepa.core.interfaces import (
    EvaluationResult, ObjectiveEvaluation, OptimizationDirection
)
from src.dspy_gepa.core.agent import AgentConfig
from tests.fixtures.test_data import create_sample_evaluation_results, create_progress_data


class TestEndToEndWorkflows:
    """Test suite for complete end-to-end workflows."""
    
    @pytest.fixture
    def complete_framework(self, mock_llm_provider):
        """Create complete framework setup for end-to-end testing."""
        config = AgentConfig(
            max_generations=5,
            population_size=15,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_rate=0.1,
            convergence_threshold=0.01
        )
        
        framework = MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=config
        )
        
        # Add monitoring
        framework.monitoring = MonitoringFramework()
        
        # Add visualization
        framework.visualization = VisualizationManager()
        
        return framework
    
    def test_complete_optimization_workflow(self, complete_framework):
        """Test complete optimization workflow from start to finish."""
        # Define problem
        task_description = "Generate answers to questions based on provided context"
        objectives = ["accuracy", "efficiency", "fluency"]
        
        # Training examples
        examples = [
            {
                "context": "Photosynthesis is the process by which plants convert sunlight into energy.",
                "question": "What is photosynthesis?",
                "expected_answer": "Process by which plants convert sunlight into energy"
            },
            {
                "context": "The Earth orbits the Sun once every 365.25 days.",
                "question": "How long does it take Earth to orbit the Sun?",
                "expected_answer": "365.25 days"
            },
            {
                "context": "Water boils at 100 degrees Celsius at sea level.",
                "question": "At what temperature does water boil at sea level?",
                "expected_answer": "100 degrees Celsius"
            }
        ]
        
        # Start monitoring
        complete_framework.monitoring.start_monitoring()
        
        with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run complete optimization
            start_time = time.time()
            
            results = complete_framework.optimize(
                task_description=task_description,
                objectives=objectives,
                examples=examples,
                max_generations=3
            )
            
            end_time = time.time()
            
            # Stop monitoring
            complete_framework.monitoring.stop_monitoring()
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) > 0
        assert complete_framework.generation > 0
        assert len(complete_framework.pareto_frontier) > 0
        
        # Check performance
        optimization_time = end_time - start_time
        assert optimization_time < 60.0  # Should complete within 60 seconds
        
        # Get monitoring data
        monitoring_data = complete_framework.monitoring.get_performance_history()
        assert len(monitoring_data) > 0
    
    def test_objective_definition_workflow(self, complete_framework):
        """Test workflow for defining and configuring objectives."""
        # Step 1: Analyze task requirements
        task_analysis = complete_framework.analyze_task_requirements(
            task="Generate technical documentation",
            constraints={"max_length": 500, "readability_level": "high"}
        )
        
        assert "recommended_objectives" in task_analysis
        assert "complexity_assessment" in task_analysis
        
        # Step 2: Configure objectives
        objectives_config = {
            "accuracy": {"weight": 1.0, "direction": "maximize"},
            "clarity": {"weight": 0.8, "direction": "maximize"},
            "conciseness": {"weight": 0.6, "direction": "minimize"},
            "technical_correctness": {"weight": 1.2, "direction": "maximize"}
        }
        
        complete_framework.configure_objectives(objectives_config)
        
        # Step 3: Validate objectives
        validation_result = complete_framework.validate_objective_configuration()
        
        assert validation_result["is_valid"] == True
        assert "objective_count" in validation_result
        assert "weight_distribution" in validation_result
    
    def test_monitoring_integration_workflow(self, complete_framework):
        """Test workflow for monitoring integration."""
        # Start monitoring with custom configuration
        monitoring_config = {
            "enable_resource_monitoring": True,
            "enable_convergence_detection": True,
            "resource_check_interval": 0.5,
            "max_cpu_percent": 75.0,
            "max_memory_percent": 85.0
        }
        
        complete_framework.monitoring.configure(monitoring_config)
        complete_framework.monitoring.start_monitoring()
        
        # Simulate optimization with monitoring
        for generation in range(3):
            results = create_sample_evaluation_results(10)
            
            # Record metrics
            metrics = {
                "generation": generation,
                "best_score": max(r.overall_score for r in results),
                "avg_score": sum(r.overall_score for r in results) / len(results),
                "population_diversity": 0.8 - generation * 0.1
            }
            
            complete_framework.monitoring.record_evaluation_metrics(metrics)
            time.sleep(0.1)  # Small delay for monitoring
        
        # Check for alerts
        alerts = complete_framework.monitoring.get_active_alerts()
        assert isinstance(alerts, list)
        
        # Get performance summary
        performance_summary = complete_framework.monitoring.get_performance_summary()
        
        assert "total_evaluations" in performance_summary
        assert "average_evaluation_time" in performance_summary
        assert "resource_usage_stats" in performance_summary
        
        complete_framework.monitoring.stop_monitoring()
    
    def test_visualization_workflow(self, complete_framework, temp_dir):
        """Test complete visualization workflow."""
        # Generate optimization data
        results = create_sample_evaluation_results(25)
        progress_data = create_progress_data(15)
        
        # Update framework state
        complete_framework.population = results
        complete_framework._update_pareto_frontier(results)
        
        # Create visualizations
        visualization_requests = [
            {
                "type": "pareto",
                "data": results,
                "config": {"x_objective": "accuracy", "y_objective": "efficiency"}
            },
            {
                "type": "progress",
                "data": progress_data,
                "config": {"metrics": ["best_accuracy", "hypervolume"]}
            },
            {
                "type": "convergence",
                "data": progress_data,
                "config": {"show_convergence_point": True}
            }
        ]
        
        # Generate all visualizations
        visualization_results = {}
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            for req in visualization_requests:
                viz_result = complete_framework.visualization.create_plot(
                    plot_type=req["type"],
                    data=req["data"],
                    customization=req["config"]
                )
                
                visualization_results[req["type"]] = viz_result
                
                # Export visualization
                export_path = temp_dir / f"{req['type']}_plot.png"
                success = complete_framework.visualization.export_plot(
                    viz_result, str(export_path)
                )
                
                assert success == True
                assert export_path.exists()
        
        # Verify all visualizations were created
        assert len(visualization_results) == 3
        assert all(result is not None for result in visualization_results.values())
    
    def test_checkpoint_resume_workflow(self, complete_framework, temp_checkpoint_dir):
        """Test workflow for checkpointing and resuming optimization."""
        # Phase 1: Initial optimization and checkpoint
        with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run initial optimization
            complete_framework.optimize(
                task_description="Test task",
                objectives=["accuracy", "efficiency"],
                examples=[{"input": "test", "expected": "result"}],
                max_generations=2
            )
            
            # Save checkpoint
            checkpoint_path = temp_checkpoint_dir / "workflow_checkpoint.json"
            complete_framework.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Record state
            saved_generation = complete_framework.generation
            saved_population_size = len(complete_framework.population)
        
        # Phase 2: Resume from checkpoint
        new_framework = MultiObjectiveGEPA(
            llm_provider=complete_framework.llm_provider,
            config=complete_framework.config
        )
        
        # Load checkpoint
        new_framework.load_checkpoint(str(checkpoint_path))
        
        # Verify state restoration
        assert new_framework.generation == saved_generation
        assert len(new_framework.population) == saved_population_size
        
        # Continue optimization
        with patch.object(new_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            new_framework.optimize(
                task_description="Test task",
                objectives=["accuracy", "efficiency"],
                examples=[{"input": "test", "expected": "result"}],
                max_generations=2,
                resume=True
            )
            
            # Should have progressed beyond saved state
            assert new_framework.generation > saved_generation
    
    def test_preference_based_selection_workflow(self, complete_framework):
        """Test workflow for preference-based solution selection."""
        # Generate diverse Pareto frontier
        results = create_sample_evaluation_results(30)
        complete_framework._update_pareto_frontier(results)
        
        # Define different preference profiles
        preference_profiles = [
            {
                "name": "accuracy_focused",
                "preferences": {"accuracy": 0.7, "efficiency": 0.2, "fluency": 0.1}
            },
            {
                "name": "balanced",
                "preferences": {"accuracy": 0.4, "efficiency": 0.3, "fluency": 0.3}
            },
            {
                "name": "efficiency_focused",
                "preferences": {"accuracy": 0.2, "efficiency": 0.7, "fluency": 0.1}
            }
        ]
        
        # Get preferred solutions for each profile
        selected_solutions = {}
        
        for profile in preference_profiles:
            preferred = complete_framework.get_preferred_solution(
                preferences=profile["preferences"]
            )
            
            selected_solutions[profile["name"]] = preferred
            
            assert preferred in complete_framework.pareto_frontier
            assert isinstance(preferred, EvaluationResult)
        
        # Verify different profiles select different solutions
        solution_ids = [sol.solution_id for sol in selected_solutions.values()]
        assert len(set(solution_ids)) >= 2  # At least 2 different solutions
    
    def test_adaptive_parameter_workflow(self, complete_framework):
        """Test workflow for adaptive parameter tuning."""
        # Initial configuration
        initial_config = {
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "population_size": 20
        }
        
        complete_framework.set_adaptive_parameters(initial_config)
        
        # Simulate optimization progress
        progress_history = []
        
        for generation in range(5):
            results = create_sample_evaluation_results(10)
            
            # Record progress
            progress_metrics = {
                "generation": generation,
                "best_score": max(r.overall_score for r in results),
                "diversity": 0.9 - generation * 0.1,
                "convergence_rate": generation * 0.02
            }
            
            progress_history.append(progress_metrics)
            
            # Adapt parameters based on progress
            adapted_params = complete_framework.adapt_parameters(progress_history)
            
            assert "mutation_rate" in adapted_params
            assert "crossover_rate" in adapted_params
            assert "population_size" in adapted_params
            
            # Parameters should change based on progress
            if generation > 2:
                # Should adapt for convergence
                assert adapted_params != initial_config
    
    def test_error_recovery_workflow(self, complete_framework):
        """Test workflow for error handling and recovery."""
        # Simulate different error scenarios
        error_scenarios = [
            {
                "type": "evaluation_failure",
                "error": Exception("Evaluation failed"),
                "recovery_expected": True
            },
            {
                "type": "resource_limit",
                "error": Exception("Resource limit exceeded"),
                "recovery_expected": True
            },
            {
                "type": "corruption",
                "error": Exception("Data corruption"),
                "recovery_expected": False
            }
        ]
        
        for scenario in error_scenarios:
            # Reset framework
            complete_framework.reset_state()
            
            # Simulate error during optimization
            with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
                if scenario["type"] == "evaluation_failure":
                    mock_eval.side_effect = scenario["error"]
                else:
                    mock_eval.return_value = create_sample_evaluation_results(1)[0]
                
                try:
                    complete_framework.optimize(
                        task_description="Test task",
                        objectives=["accuracy"],
                        examples=[{"test": "data"}],
                        max_generations=1
                    )
                    
                    recovery_successful = True
                except Exception as e:
                    recovery_successful = False
                    
                    # Check if error was handled gracefully
                    assert complete_framework.get_last_error() is not None
                
                if scenario["recovery_expected"]:
                    assert recovery_successful or complete_framework.get_last_error() is not None
                else:
                    assert not recovery_successful
    
    @pytest.mark.slow
    def test_scalability_workflow(self, complete_framework):
        """Test workflow scalability with larger problems."""
        # Large problem setup
        large_examples = [
            {
                "context": f"Context paragraph {i} with detailed information.",
                "question": f"Question {i} about the context.",
                "expected_answer": f"Answer {i} to the question."
            }
            for i in range(20)
        ]
        
        many_objectives = ["accuracy", "efficiency", "fluency", "clarity", "conciseness"]
        
        start_time = time.time()
        
        with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run scaled optimization
            results = complete_framework.optimize(
                task_description="Large scale test task",
                objectives=many_objectives,
                examples=large_examples,
                max_generations=3,
                population_size=25
            )
        
        end_time = time.time()
        
        # Performance requirements
        total_time = end_time - start_time
        assert total_time < 120.0  # Should complete within 2 minutes
        assert len(results) > 0
        assert complete_framework.generation > 0
        
        # Memory efficiency check
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Should not use excessive memory
        assert memory_mb < 500.0  # Less than 500MB
    
    def test_integration_compatibility_workflow(self, complete_framework):
        """Test workflow compatibility with external integrations."""
        # Mock external components
        external_logger = Mock()
        external_metrics = Mock()
        external_storage = Mock()
        
        # Register external integrations
        complete_framework.register_external_logger(external_logger)
        complete_framework.register_external_metrics(external_metrics)
        complete_framework.register_external_storage(external_storage)
        
        # Run optimization with external integrations
        with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            results = complete_framework.optimize(
                task_description="Integration test task",
                objectives=["accuracy", "efficiency"],
                examples=[{"test": "data"}],
                max_generations=2
            )
        
        # Verify external integrations were called
        assert external_logger.log.call_count > 0
        assert external_metrics.record.call_count > 0
        assert external_storage.save.call_count > 0
        
        # Test data export for external systems
        export_data = complete_framework.export_for_external_system(
            format="json",
            include_progress=True,
            include_pareto_frontier=True
        )
        
        assert "optimization_results" in export_data
        assert "progress_data" in export_data
        assert "pareto_frontier" in export_data
    
    def test_multi_modal_workflow(self, complete_framework):
        """Test workflow with multi-modal optimization (different types of tasks)."""
        # Define different task modalities
        task_modalities = [
            {
                "type": "text_generation",
                "task": "Generate descriptive text",
                "objectives": ["accuracy", "fluency", "creativity"]
            },
            {
                "type": "classification",
                "task": "Classify text into categories",
                "objectives": ["accuracy", "efficiency", "confidence"]
            },
            {
                "type": "summarization",
                "task": "Summarize long documents",
                "objectives": ["accuracy", "conciseness", "coverage"]
            }
        ]
        
        modality_results = {}
        
        for modality in task_modalities:
            # Configure for specific modality
            complete_framework.configure_for_modality(modality["type"])
            
            with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
                mock_eval.return_value = create_sample_evaluation_results(1)[0]
                
                # Run modality-specific optimization
                results = complete_framework.optimize(
                    task_description=modality["task"],
                    objectives=modality["objectives"],
                    examples=[{"sample": "data"}],
                    max_generations=2
                )
                
                modality_results[modality["type"]] = results
                
                assert len(results) > 0
        
        # Verify all modalities were processed
        assert len(modality_results) == 3
        assert all(len(results) > 0 for results in modality_results.values())
        
        # Compare performance across modalities
        performance_comparison = complete_framework.compare_modality_performance(modality_results)
        
        assert "best_modality" in performance_comparison
        assert "performance_ranking" in performance_comparison
        assert "efficiency_comparison" in performance_comparison
    
    def test_continuous_learning_workflow(self, complete_framework):
        """Test workflow for continuous learning and improvement."""
        # Phase 1: Initial learning
        initial_examples = [
            {"input": "question 1", "expected": "answer 1"},
            {"input": "question 2", "expected": "answer 2"}
        ]
        
        with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            initial_results = complete_framework.optimize(
                task_description="Initial learning task",
                objectives=["accuracy", "efficiency"],
                examples=initial_examples,
                max_generations=2
            )
        
        # Phase 2: Continuous learning with new data
        new_examples = [
            {"input": "question 3", "expected": "answer 3"},
            {"input": "question 4", "expected": "answer 4"},
            {"input": "question 5", "expected": "answer 5"}
        ]
        
        # Enable continuous learning mode
        complete_framework.enable_continuous_learning(
            learning_rate=0.1,
            adaptation_frequency=2
        )
        
        with patch.object(complete_framework, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            continuous_results = complete_framework.continue_optimization(
                new_examples=new_examples,
                additional_generations=3
            )
        
        # Verify learning progress
        assert len(continuous_results) > 0
        assert complete_framework.generation > 2  # Should have progressed
        
        # Check learning metrics
        learning_metrics = complete_framework.get_learning_metrics()
        
        assert "improvement_rate" in learning_metrics
        assert "adaptation_events" in learning_metrics
        assert "knowledge_retention" in learning_metrics
        
        # Should show improvement over time
        assert learning_metrics["improvement_rate"] > 0
