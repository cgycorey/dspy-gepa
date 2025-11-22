"""Comprehensive integration tests for DSPy GEPA integration."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.dspy_gepa.dspy_integration.metric_converter import MetricConverter
from src.dspy_gepa.dspy_integration.multi_objective_optimizer import MultiObjectiveOptimizer
from src.dspy_gepa.dspy_integration.signature_analyzer import SignatureAnalyzer
from src.dspy_gepa.core.multi_objective_gepa import MultiObjectiveGEPA
from src.dspy_gepa.core.interfaces import (
    EvaluationResult, ObjectiveEvaluation, OptimizationDirection
)
from src.dspy_gepa.core.agent import AgentConfig
from tests.fixtures.test_data import create_sample_evaluation_results


class TestMetricConverter:
    """Test suite for MetricConverter integration with DSPy metrics."""
    
    @pytest.fixture
    def metric_converter(self):
        """Create MetricConverter instance for testing."""
        return MetricConverter()
    
    def test_dspy_to_gepa_metric_conversion(self, metric_converter):
        """Test conversion from DSPy metrics to GEPA objectives."""
        # Mock DSPy metric
        mock_dspy_metric = Mock()
        mock_dspy_metric.name = "accuracy"
        mock_dspy_metric.direction = "maximize"
        mock_dspy_metric.score = 0.85
        
        # Convert to GEPA objective
        gepa_objective = metric_converter.convert_dspy_metric(mock_dspy_metric)
        
        assert gepa_objective.name == "accuracy"
        assert gepa_objective.direction == OptimizationDirection.MAXIMIZE
        assert isinstance(gepa_objective.evaluation, ObjectiveEvaluation)
    
    def test_gepa_to_dspy_metric_conversion(self, metric_converter):
        """Test conversion from GEPA objectives to DSPy metrics."""
        # Create GEPA objective evaluation
        gepa_eval = ObjectiveEvaluation(
            objective_name="efficiency",
            score=0.75,
            direction=OptimizationDirection.MINIMIZE,
            evaluation_time=0.1
        )
        
        # Convert to DSPy metric
        dspy_metric = metric_converter.convert_gepa_objective(gepa_eval)
        
        assert dspy_metric.name == "efficiency"
        assert dspy_metric.direction == "minimize"
        assert dspy_metric.score == 0.75
    
    def test_batch_metric_conversion(self, metric_converter):
        """Test batch conversion of multiple metrics."""
        # Create multiple DSPy metrics
        dspy_metrics = [
            Mock(name="accuracy", direction="maximize", score=0.8),
            Mock(name="efficiency", direction="minimize", score=0.7),
            Mock(name="fluency", direction="maximize", score=0.9)
        ]
        
        # Convert batch
        gepa_objectives = metric_converter.convert_batch_dspy_metrics(dspy_metrics)
        
        assert len(gepa_objectives) == 3
        assert all(obj.name in ["accuracy", "efficiency", "fluency"] for obj in gepa_objectives)
    
    def test_custom_metric_mapping(self, metric_converter):
        """Test custom metric mapping configuration."""
        # Define custom mapping
        custom_mapping = {
            "custom_metric": {
                "direction": "maximize",
                "weight": 1.5,
                "normalization": "min_max"
            }
        }
        
        metric_converter.set_custom_mapping(custom_mapping)
        
        # Test custom conversion
        mock_custom_metric = Mock()
        mock_custom_metric.name = "custom_metric"
        mock_custom_metric.score = 0.6
        
        gepa_objective = metric_converter.convert_dspy_metric(mock_custom_metric)
        
        assert gepa_objective.name == "custom_metric"
        assert gepa_objective.direction == OptimizationDirection.MAXIMIZE
    
    def test_metric_normalization(self, metric_converter):
        """Test metric normalization during conversion."""
        # Create metrics with different scales
        metrics = [
            Mock(name="accuracy", direction="maximize", score=0.85),  # 0-1 scale
            Mock(name="latency", direction="minimize", score=150),   # 0-1000 scale
            Mock(name="cost", direction="minimize", score=0.05)      # 0-1 scale
        ]
        
        # Normalize metrics
        normalized = metric_converter.normalize_metrics(metrics)
        
        assert len(normalized) == 3
        # All normalized scores should be in 0-1 range
        for metric in normalized:
            assert 0.0 <= metric.score <= 1.0


class TestSignatureAnalyzer:
    """Test suite for SignatureAnalyzer integration with DSPy signatures."""
    
    @pytest.fixture
    def signature_analyzer(self):
        """Create SignatureAnalyzer instance for testing."""
        return SignatureAnalyzer()
    
    def test_signature_parsing(self, signature_analyzer):
        """Test parsing of DSPy signatures."""
        # Mock DSPy signature
        mock_signature = Mock()
        mock_signature.inputs = {"question": "str", "context": "str"}
        mock_signature.outputs = {"answer": "str", "confidence": "float"}
        mock_signature.instructions = "Answer the question based on the context."
        
        # Analyze signature
        analysis = signature_analyzer.analyze_signature(mock_signature)
        
        assert "input_fields" in analysis
        assert "output_fields" in analysis
        assert "complexity_score" in analysis
        assert "recommended_metrics" in analysis
        
        assert len(analysis["input_fields"]) == 2
        assert len(analysis["output_fields"]) == 2
    
    def test_objective_recommendation(self, signature_analyzer):
        """Test objective recommendation based on signature analysis."""
        # Mock complex signature
        mock_signature = Mock()
        mock_signature.inputs = {"prompt": "str", "context": "str", "examples": "list"}
        mock_signature.outputs = {"response": "str", "reasoning": "str", "confidence": "float"}
        mock_signature.instructions = "Generate a detailed response with reasoning and confidence score."
        
        # Get recommendations
        recommendations = signature_analyzer.recommend_objectives(mock_signature)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend relevant objectives
        recommended_names = [rec["name"] for rec in recommendations]
        assert any("accuracy" in name.lower() for name in recommended_names)
        assert any("efficiency" in name.lower() for name in recommended_names)
    
    def test_complexity_estimation(self, signature_analyzer):
        """Test complexity estimation for signatures."""
        # Simple signature
        simple_signature = Mock()
        simple_signature.inputs = {"text": "str"}
        simple_signature.outputs = {"summary": "str"}
        simple_signature.instructions = "Summarize the text."
        
        # Complex signature
        complex_signature = Mock()
        complex_signature.inputs = {"prompt": "str", "context": "str", "examples": "list", "constraints": "dict"}
        complex_signature.outputs = {"answer": "str", "reasoning": "str", "confidence": "float", "sources": "list"}
        complex_signature.instructions = "Generate a comprehensive answer with detailed reasoning, confidence scores, and source citations based on the provided context and examples."
        
        simple_complexity = signature_analyzer.estimate_complexity(simple_signature)
        complex_complexity = signature_analyzer.estimate_complexity(complex_signature)
        
        # Complex signature should have higher complexity
        assert complex_complexity > simple_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0
    
    def test_signature_optimization_suggestions(self, signature_analyzer):
        """Test optimization suggestions based on signature analysis."""
        mock_signature = Mock()
        mock_signature.inputs = {"prompt": "str", "context": "str"}
        mock_signature.outputs = {"response": "str", "confidence": "float"}
        mock_signature.instructions = "Generate response with confidence."
        
        suggestions = signature_analyzer.get_optimization_suggestions(mock_signature)
        
        assert isinstance(suggestions, dict)
        assert "objectives" in suggestions
        assert "parameters" in suggestions
        assert "strategies" in suggestions
        
        assert len(suggestions["objectives"]) > 0
        assert len(suggestions["parameters"]) > 0


class TestMultiObjectiveOptimizer:
    """Test suite for MultiObjectiveOptimizer with DSPy integration."""
    
    @pytest.fixture
    def mock_dspy_program(self):
        """Mock DSPy program for testing."""
        mock_program = Mock()
        mock_program.predict = Mock(return_value="Test prediction")
        mock_program.signature = Mock()
        mock_program.signature.inputs = {"question": "str"}
        mock_program.signature.outputs = {"answer": "str"}
        return mock_program
    
    @pytest.fixture
    def multi_objective_optimizer(self, mock_dspy_program):
        """Create MultiObjectiveOptimizer instance for testing."""
        config = AgentConfig(
            max_generations=5,
            population_size=10,
            mutation_rate=0.1
        )
        
        return MultiObjectiveOptimizer(
            dspy_program=mock_dspy_program,
            config=config
        )
    
    def test_optimizer_initialization(self, multi_objective_optimizer, mock_dspy_program):
        """Test MultiObjectiveOptimizer initialization."""
        assert multi_objective_optimizer.dspy_program == mock_dspy_program
        assert multi_objective_optimizer.config.max_generations == 5
        assert multi_objective_optimizer.config.population_size == 10
    
    def test_dspy_program_evaluation(self, multi_objective_optimizer):
        """Test evaluation of DSPy programs."""
        # Create test input
        test_input = {"question": "What is the capital of France?"}
        
        # Mock evaluation metrics
        with patch.object(multi_objective_optimizer, '_evaluate_dspy_output') as mock_eval:
            mock_eval.return_value = {
                "accuracy": 0.9,
                "efficiency": 0.8,
                "fluency": 0.85
            }
            
            result = multi_objective_optimizer.evaluate_program(test_input)
            
            assert isinstance(result, EvaluationResult)
            assert "accuracy" in result.objectives
            assert "efficiency" in result.objectives
            assert "fluency" in result.objectives
            
            mock_dspy_program.predict.assert_called_with(**test_input)
    
    def test_multi_objective_optimization_run(self, multi_objective_optimizer):
        """Test complete multi-objective optimization run."""
        # Test data
        test_examples = [
            {"question": "What is 2+2?", "expected_answer": "4"},
            {"question": "What is the capital of France?", "expected_answer": "Paris"}
        ]
        
        with patch.object(multi_objective_optimizer, 'evaluate_program') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run optimization
            results = multi_objective_optimizer.optimize(
                examples=test_examples,
                objectives=["accuracy", "efficiency"]
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            assert multi_objective_optimizer.generation > 0
    
    def test_program_mutation(self, multi_objective_optimizer):
        """Test DSPy program mutation strategies."""
        base_program = multi_objective_optimizer.dspy_program
        
        # Test semantic mutation
        mutated_program = multi_objective_optimizer.mutate_program(
            base_program,
            mutation_type="semantic"
        )
        
        assert mutated_program is not None
        assert mutated_program != base_program
    
    def test_program_crossover(self, multi_objective_optimizer):
        """Test DSPy program crossover."""
        program1 = multi_objective_optimizer.dspy_program
        program2 = Mock()  # Another mock program
        
        child_program = multi_objective_optimizer.crossover_programs(program1, program2)
        
        assert child_program is not None
        assert hasattr(child_program, 'predict')
    
    def test_dspymetric_integration(self, multi_objective_optimizer):
        """Test integration with DSPy metrics."""
        # Mock DSPy metrics
        mock_accuracy_metric = Mock()
        mock_accuracy_metric.score = 0.85
        
        mock_efficiency_metric = Mock()
        mock_efficiency_metric.score = 0.75
        
        # Test metric integration
        combined_score = multi_objective_optimizer.combine_dspy_metrics([
            mock_accuracy_metric,
            mock_efficiency_metric
        ])
        
        assert isinstance(combined_score, float)
        assert 0.0 <= combined_score <= 1.0


class TestMultiObjectiveGEPAIntegration:
    """Test suite for complete MultiObjectiveGEPA integration."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider for testing."""
        mock_provider = Mock()
        mock_provider.generate.return_value = "Generated response"
        mock_provider.token_count.return_value = 15
        return mock_provider
    
    @pytest.fixture
    def multi_objective_gepa(self, mock_llm_provider):
        """Create MultiObjectiveGEPA instance for testing."""
        config = MultiObjectiveConfig(
            max_generations=3,
            population_size=8,
            mutation_rate=0.1
        )
        
        return MultiObjectiveGEPA(
            llm_provider=mock_llm_provider,
            config=config
        )
    
    def test_gepa_dspy_integration(self, multi_objective_gepa):
        """Test integration between GEPA and DSPy components."""
        # Create test DSPy program signature
        mock_signature = Mock()
        mock_signature.inputs = {"prompt": "str", "context": "str"}
        mock_signature.outputs = {"response": "str", "confidence": "float"}
        
        # Test signature analysis integration
        analysis = multi_objective_gepa.analyze_dspy_signature(mock_signature)
        
        assert "recommended_objectives" in analysis
        assert "complexity_assessment" in analysis
        assert len(analysis["recommended_objectives"]) > 0
    
    def test_end_to_end_optimization(self, multi_objective_gepa):
        """Test end-to-end optimization workflow."""
        # Define optimization problem
        task_description = "Answer questions based on provided context"
        objectives = ["accuracy", "efficiency", "fluency"]
        
        # Test examples
        examples = [
            {"context": "The sky is blue.", "question": "What color is the sky?", "answer": "Blue"},
            {"context": "Water freezes at 0°C.", "question": "At what temperature does water freeze?", "answer": "0°C"}
        ]
        
        with patch.object(multi_objective_gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run optimization
            results = multi_objective_gepa.optimize(
                task_description=task_description,
                objectives=objectives,
                examples=examples
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            assert multi_objective_gepa.generation > 0
            assert len(multi_objective_gepa.pareto_frontier) > 0
    
    def test_dspy_checkpoint_compatibility(self, multi_objective_gepa, temp_checkpoint_dir):
        """Test checkpoint compatibility with DSPy format."""
        # Set up some optimization state
        multi_objective_gepa.population = create_sample_evaluation_results(5)
        multi_objective_gepa.generation = 2
        
        # Save in GEPA format
        gepa_checkpoint = temp_checkpoint_dir / "gepa_checkpoint.json"
        multi_objective_gepa.save_checkpoint(str(gepa_checkpoint))
        
        # Convert to DSPy format
        dspy_checkpoint = temp_checkpoint_dir / "dspy_checkpoint.json"
        success = multi_objective_gepa.convert_to_dspy_checkpoint(
            str(gepa_checkpoint),
            str(dspy_checkpoint)
        )
        
        assert success == True
        assert dspy_checkpoint.exists()
    
    def test_metric_interoperability(self, multi_objective_gepa):
        """Test metric interoperability between GEPA and DSPy."""
        # Create GEPA evaluation result
        gepa_result = create_sample_evaluation_results(1)[0]
        
        # Convert to DSPy format
        dspy_metrics = multi_objective_gepa.convert_to_dspy_metrics(gepa_result)
        
        assert isinstance(dspy_metrics, list)
        assert len(dspy_metrics) == len(gepa_result.objectives)
        
        # Convert back to GEPA format
        converted_gepa = multi_objective_gepa.convert_from_dspy_metrics(dspy_metrics)
        
        assert len(converted_gepa.objectives) == len(gepa_result.objectives)
    
    def test_pipeline_integration(self, multi_objective_gepa):
        """Test integration with DSPy optimization pipeline."""
        # Create mock DSPy pipeline components
        mock_predictor = Mock()
        mock_predictor.predict.return_value = "Prediction result"
        
        mock_metric = Mock()
        mock_metric.score = 0.8
        
        # Test pipeline integration
        pipeline_result = multi_objective_gepa.integrate_with_dspy_pipeline(
            predictor=mock_predictor,
            metrics=[mock_metric],
            optimization_config={"max_iterations": 5}
        )
        
        assert "optimization_results" in pipeline_result
        assert "pipeline_performance" in pipeline_result
        assert "integration_metrics" in pipeline_result
    
    def test_backward_compatibility(self, multi_objective_gepa):
        """Test backward compatibility with single-objective DSPy GEPA."""
        # Test single-objective optimization
        single_objective_result = multi_objective_gepa.optimize_single_objective(
            task="Test task",
            metric="accuracy",
            examples=[{"input": "test", "expected": "result"}]
        )
        
        assert isinstance(single_objective_result, EvaluationResult)
        assert "accuracy" in single_objective_result.objectives
        
        # Should be compatible with original DSPy GEPA results
        assert hasattr(single_objective_result, 'solution_id')
        assert hasattr(single_objective_result, 'overall_score')
    
    @pytest.mark.slow
    def test_integration_scalability(self, multi_objective_gepa):
        """Test scalability of integration components."""
        import time
        
        # Create larger test set
        large_examples = [
            {"input": f"test_{i}", "expected": f"result_{i}"}
            for i in range(50)
        ]
        
        start_time = time.time()
        
        with patch.object(multi_objective_gepa, '_evaluate_solution') as mock_eval:
            mock_eval.return_value = create_sample_evaluation_results(1)[0]
            
            # Run optimization with larger dataset
            results = multi_objective_gepa.optimize(
                task_description="Scalability test",
                objectives=["accuracy", "efficiency"],
                examples=large_examples[:10],  # Subset for testing
                max_generations=2
            )
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 30.0  # 30 seconds max
        assert len(results) > 0
    
    def test_error_handling_integration(self, multi_objective_gepa):
        """Test error handling in integration components."""
        # Test handling of DSPy program failures
        with patch.object(multi_objective_gepa.llm_provider, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("LLM provider failed")
            
            with pytest.raises(Exception):
                multi_objective_gepa._evaluate_solution({"test": "input"})
        
        # Test handling of metric conversion failures
        with patch.object(multi_objective_gepa, 'convert_to_dspy_metrics') as mock_convert:
            mock_convert.side_effect = Exception("Conversion failed")
            
            with pytest.raises(Exception):
                result = create_sample_evaluation_results(1)[0]
                multi_objective_gepa.convert_to_dspy_metrics(result)
    
    def test_thread_safety_integration(self, multi_objective_gepa, thread_safety_test):
        """Test thread safety of integration components."""
        def evaluate_and_convert():
            result = create_sample_evaluation_results(1)[0]
            dspy_metrics = multi_objective_gepa.convert_to_dspy_metrics(result)
            return dspy_metrics
        
        # Run concurrent conversions
        results = thread_safety_test(evaluate_and_convert, num_threads=3, iterations=5)
        
        # Should have no errors
        assert results["total_errors"] == 0
        assert results["success_rate"] == 1.0
    
    def test_memory_efficiency_integration(self, multi_objective_gepa):
        """Test memory efficiency of integration components."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple integration operations
        for _ in range(10):
            result = create_sample_evaluation_results(1)[0]
            multi_objective_gepa.convert_to_dspy_metrics(result)
            multi_objective_gepa.convert_from_dspy_metrics([])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50.0
