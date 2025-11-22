"""Comprehensive tests for monitoring and analysis components."""

from __future__ import annotations

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.dspy_gepa.core.monitoring import MonitoringFramework
from src.dspy_gepa.core.analysis import ConvergenceDetector
from src.dspy_gepa.core.analysis import ProgressAnalyzer, PerformanceAnalyzer
from src.dspy_gepa.core.interfaces import (
    EvaluationResult, ObjectiveEvaluation, OptimizationDirection
)
from tests.fixtures.test_data import create_sample_evaluation_results, create_progress_data, MockProcess


class TestMonitoringFramework:
    """Test suite for MonitoringFramework."""
    
    @pytest.fixture
    def monitoring_framework(self):
        """Create MonitoringFramework instance for testing."""
        return MonitoringFramework()
    
    def test_initialization(self, monitoring_framework):
        """Test MonitoringFramework initialization."""
        assert monitoring_framework.is_monitoring == False
        assert len(monitoring_framework.metrics_history) == 0
    
    def test_resource_monitoring(self, monitoring_framework, mock_psutil):
        """Test resource monitoring functionality."""
        # Start monitoring
        monitoring_framework.start_monitoring()
        
        # Get resource metrics
        metrics = monitoring_framework.get_resource_metrics()
        
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "memory_used_mb" in metrics
        assert "timestamp" in metrics
        
        # Stop monitoring
        monitoring_framework.stop_monitoring()
        assert monitoring_framework.is_monitoring == False
    
    def test_resource_limit_enforcement(self, monitoring_framework, mock_psutil):
        """Test resource limit enforcement."""
        # Mock resource usage above limits
        with patch.object(mock_psutil.Process.return_value, 'cpu_percent', return_value=95.0):
            with patch.object(mock_psutil.Process.return_value, 'memory_percent', return_value=95.0):
                
                monitoring_framework.start_monitoring()
                
                # Check if limits are exceeded
                limits_exceeded = monitoring_framework.check_resource_limits()
                
                assert limits_exceeded["cpu_exceeded"] == True
                assert limits_exceeded["memory_exceeded"] == True
                assert limits_exceeded["overall_exceeded"] == True
                
                monitoring_framework.stop_monitoring()
    
    def test_convergence_detection(self, monitoring_framework):
        """Test convergence detection functionality."""
        # Create convergence detector
        detector = ConvergenceDetector()
        
        # Add some performance history
        for i in range(10):
            results = create_sample_evaluation_results(5)
            metrics = {
                "hypervolume": 10.0 + i * 0.5,
                "best_score": 0.8 + i * 0.01,
                "diversity": 1.0 - i * 0.05
            }
            detector.update_metrics(metrics)
        
        # Check convergence
        is_converged = detector.check_convergence()
        assert isinstance(is_converged, bool)
        
        # Get convergence statistics
        stats = detector.get_convergence_stats()
        assert "method" in stats
        assert "is_converged" in stats
        assert "convergence_generation" in stats
        assert "convergence_metrics" in stats
    
    def test_performance_tracking(self, monitoring_framework):
        """Test performance tracking over time."""
        monitoring_framework.start_monitoring()
        
        # Simulate multiple evaluation cycles
        for i in range(5):
            results = create_sample_evaluation_results(10)
            
            # Record evaluation metrics
            eval_metrics = {
                "generation": i,
                "population_size": len(results),
                "evaluation_time": 0.1 + i * 0.01,
                "best_score": max(r.overall_score for r in results),
                "avg_score": sum(r.overall_score for r in results) / len(results)
            }
            
            monitoring_framework.record_evaluation_metrics(eval_metrics)
            time.sleep(0.01)  # Small delay
        
        # Get performance history
        history = monitoring_framework.get_performance_history()
        
        assert len(history) == 5
        assert all("generation" in metrics for metrics in history)
        assert all("best_score" in metrics for metrics in history)
        assert all("timestamp" in metrics for metrics in history)
        
        monitoring_framework.stop_monitoring()
    
    def test_alert_system(self, monitoring_framework):
        """Test alert system for critical events."""
        alerts_captured = []
        
        def test_alert_handler(alert):
            alerts_captured.append(alert)
        
        # Register alert handler
        monitoring_framework.register_alert_handler(test_alert_handler)
        
        # Trigger resource alert
        monitoring_framework._trigger_alert(
            alert_type="resource_limit",
            severity="warning",
            message="CPU usage exceeded limit",
            details={"cpu_percent": 85.0, "limit": 80.0}
        )
        
        assert len(alerts_captured) == 1
        assert alerts_captured[0]["alert_type"] == "resource_limit"
        assert alerts_captured[0]["severity"] == "warning"
    
    def test_checkpoint_integration(self, monitoring_framework, temp_checkpoint_dir):
        """Test checkpoint integration with monitoring data."""
        monitoring_framework.start_monitoring()
        
        # Record some metrics
        for i in range(3):
            metrics = {
                "generation": i,
                "best_score": 0.8 + i * 0.05,
                "evaluation_time": 0.1
            }
            monitoring_framework.record_evaluation_metrics(metrics)
        
        # Save monitoring state to checkpoint
        checkpoint_path = temp_checkpoint_dir / "monitoring_checkpoint.json"
        monitoring_framework.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Load monitoring state
        new_framework = MonitoringFramework(config=monitoring_framework.config)
        new_framework.load_checkpoint(str(checkpoint_path))
        
        assert len(new_framework.metrics_history) == 3
        
        monitoring_framework.stop_monitoring()


class TestProgressAnalyzer:
    """Test suite for ProgressAnalyzer."""
    
    @pytest.fixture
    def progress_analyzer(self):
        """Create ProgressAnalyzer instance for testing."""
        return ProgressAnalyzer()
    
    def test_progress_calculation(self, progress_analyzer):
        """Test progress calculation across generations."""
        # Create progress data
        progress_data = create_progress_data(20)
        
        # Calculate progress metrics
        progress_metrics = progress_analyzer.calculate_progress(progress_data)
        
        assert "overall_progress" in progress_metrics
        assert "accuracy_improvement" in progress_metrics
        assert "efficiency_improvement" in progress_metrics
        assert "hypervolume_growth" in progress_metrics
        assert "convergence_trend" in progress_metrics
        
        # Check that progress is between 0 and 1
        assert 0.0 <= progress_metrics["overall_progress"] <= 1.0
    
    def test_trend_analysis(self, progress_analyzer):
        """Test trend analysis for optimization metrics."""
        # Create trending data
        generations = list(range(10))
        scores = [0.5 + i * 0.03 for i in generations]  # Improving trend
        
        trend_analysis = progress_analyzer.analyze_trend(generations, scores)
        
        assert "slope" in trend_analysis
        assert "direction" in trend_analysis
        assert "confidence" in trend_analysis
        assert "r_squared" in trend_analysis
        
        # Should detect positive trend
        assert trend_analysis["direction"] == "improving"
        assert trend_analysis["slope"] > 0
    
    def test_plateau_detection(self, progress_analyzer):
        """Test plateau detection in optimization progress."""
        # Create plateau data (improvement then stagnation)
        scores = [0.5 + i * 0.05 for i in range(5)] + [0.75] * 10  # Plateau after 5 generations
        generations = list(range(len(scores)))
        
        plateau_info = progress_analyzer.detect_plateau(generations, scores)
        
        assert "is_plateau" in plateau_info
        assert "plateau_start" in plateau_info
        assert "plateau_duration" in plateau_info
        
        # Should detect plateau
        assert plateau_info["is_plateau"] == True
        assert plateau_info["plateau_start"] == 5
        assert plateau_info["plateau_duration"] == 10
    
    def test_estimation_completion_time(self, progress_analyzer):
        """Test completion time estimation."""
        # Create progress data with known trend
        progress_data = create_progress_data(10)
        
        # Estimate completion time
        time_estimate = progress_analyzer.estimate_completion_time(
            progress_data,
            target_score=0.95,
            current_generation=10
        )
        
        assert "estimated_generations" in time_estimate
        assert "estimated_time_seconds" in time_estimate
        assert "confidence" in time_estimate
        
        # Should provide reasonable estimate
        assert time_estimate["estimated_generations"] > 0
        assert time_estimate["estimated_time_seconds"] > 0
        assert 0.0 <= time_estimate["confidence"] <= 1.0


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer."""
    
    @pytest.fixture
    def performance_analyzer(self):
        """Create PerformanceAnalyzer instance for testing."""
        return PerformanceAnalyzer()
    
    def test_efficiency_analysis(self, performance_analyzer):
        """Test efficiency analysis of optimization process."""
        # Create sample evaluation results
        results = create_sample_evaluation_results(20)
        
        efficiency_metrics = performance_analyzer.analyze_efficiency(results)
        
        assert "evaluation_efficiency" in efficiency_metrics
        assert "convergence_efficiency" in efficiency_metrics
        assert "resource_efficiency" in efficiency_metrics
        assert "overall_efficiency_score" in efficiency_metrics
        
        # Check that efficiency scores are reasonable
        assert 0.0 <= efficiency_metrics["overall_efficiency_score"] <= 1.0
    
    def test_bottleneck_detection(self, performance_analyzer):
        """Test bottleneck detection in optimization pipeline."""
        # Create performance data with bottlenecks
        performance_data = {
            "evaluation_times": [0.1, 0.15, 0.2, 0.5, 0.12, 0.11],  # Spike at index 3
            "mutation_times": [0.01, 0.02, 0.01, 0.02, 0.01, 0.01],
            "selection_times": [0.001, 0.001, 0.002, 0.001, 0.001, 0.001]
        }
        
        bottlenecks = performance_analyzer.detect_bottlenecks(performance_data)
        
        assert "primary_bottleneck" in bottlenecks
        assert "bottleneck_severity" in bottlenecks
        assert "recommendations" in bottlenecks
        
        # Should identify evaluation as bottleneck
        assert bottlenecks["primary_bottleneck"] == "evaluation"
    
    def test_quality_diversity_analysis(self, performance_analyzer):
        """Test quality-diversity trade-off analysis."""
        # Create diverse population
        results = create_sample_evaluation_results(30)
        
        qd_analysis = performance_analyzer.analyze_quality_diversity(results)
        
        assert "quality_score" in qd_analysis
        assert "diversity_score" in qd_analysis
        assert "balance_score" in qd_analysis
        assert "recommendations" in qd_analysis
        
        # Check that scores are in valid range
        assert 0.0 <= qd_analysis["quality_score"] <= 1.0
        assert 0.0 <= qd_analysis["diversity_score"] <= 1.0
        assert 0.0 <= qd_analysis["balance_score"] <= 1.0
    
    def test_performance_comparison(self, performance_analyzer):
        """Test performance comparison between different runs."""
        # Create performance data for two runs
        run1_data = {
            "best_scores": [0.5, 0.6, 0.7, 0.75, 0.8],
            "evaluation_times": [10, 20, 30, 40, 50],
            "resource_usage": [100, 120, 140, 160, 180]
        }
        
        run2_data = {
            "best_scores": [0.45, 0.55, 0.65, 0.72, 0.78],
            "evaluation_times": [8, 15, 25, 35, 45],
            "resource_usage": [90, 110, 130, 150, 170]
        }
        
        comparison = performance_analyzer.compare_performance(run1_data, run2_data)
        
        assert "winner" in comparison
        assert "score_improvement" in comparison
        assert "time_improvement" in comparison
        assert "resource_improvement" in comparison
        assert "overall_better" in comparison
        
        # Should declare run1 as winner (better final score)
        assert comparison["winner"] == "run1"
    
    @pytest.mark.slow
    def test_scalability_analysis(self, performance_analyzer):
        """Test scalability analysis with different population sizes."""
        scalability_results = []
        
        for pop_size in [10, 20, 50, 100]:
            results = create_sample_evaluation_results(pop_size)
            
            start_time = time.time()
            analysis = performance_analyzer.analyze_efficiency(results)
            end_time = time.time()
            
            scalability_results.append({
                "population_size": pop_size,
                "analysis_time": end_time - start_time,
                "efficiency_score": analysis["overall_efficiency_score"]
            })
        
        # Check that analysis time scales reasonably
        for i in range(1, len(scalability_results)):
            prev_time = scalability_results[i-1]["analysis_time"]
            curr_time = scalability_results[i]["analysis_time"]
            prev_size = scalability_results[i-1]["population_size"]
            curr_size = scalability_results[i]["population_size"]
            
            # Time should not grow disproportionately
            time_ratio = curr_time / prev_time
            size_ratio = curr_size / prev_size
            
            assert time_ratio < size_ratio * 2  # Allow some overhead
    
    def test_thread_safety(self, performance_analyzer, thread_safety_test):
        """Test thread safety of performance analysis."""
        def analyze_performance():
            results = create_sample_evaluation_results(10)
            return performance_analyzer.analyze_efficiency(results)
        
        # Run concurrent analyses
        results = thread_safety_test(analyze_performance, num_threads=3, iterations=5)
        
        # Should have no errors
        assert results["total_errors"] == 0
        assert results["success_rate"] == 1.0
    
    def test_memory_usage(self, performance_analyzer):
        """Test memory usage during analysis."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple analyses
        for _ in range(10):
            results = create_sample_evaluation_results(50)
            performance_analyzer.analyze_efficiency(results)
            performance_analyzer.analyze_quality_diversity(results)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100.0
