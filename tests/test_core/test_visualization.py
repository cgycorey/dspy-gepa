"""Comprehensive tests for visualization capabilities."""

from __future__ import annotations

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import os

from src.dspy_gepa.core.visualization import (
    create_pareto_plot, create_progress_plot, create_comparison_plot
)
from src.dspy_gepa.core.interfaces import (
    EvaluationResult, ObjectiveEvaluation, OptimizationDirection
)
from tests.fixtures.test_data import create_sample_evaluation_results, create_progress_data


class TestVisualizationFunctions:
    """Test suite for visualization functions."""


class TestParetoVisualization:
    """Test suite for Pareto visualization."""
    
    def test_2d_pareto_plot(self):
        """Test 2D Pareto frontier plot."""
        results = create_sample_evaluation_results(20)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                results,
                x_objective="accuracy",
                y_objective="efficiency",
                plot_type="2d"
            )
            
            assert plot_result is not None
            mock_ax.scatter.assert_called()
            mock_ax.set_xlabel.assert_called_with("accuracy")
            mock_ax.set_ylabel.assert_called_with("efficiency")
    
    def test_3d_pareto_plot(self):
        """Test 3D Pareto frontier plot."""
        results = create_sample_evaluation_results(20)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_plt.figure.return_value = mock_fig
            
            plot_result = create_pareto_plot(
                results,
                x_objective="accuracy",
                y_objective="efficiency",
                z_objective="complexity",
                plot_type="3d"
            )
            
            assert plot_result is not None
            mock_ax.scatter.assert_called()
            mock_ax.set_xlabel.assert_called_with("accuracy")
            mock_ax.set_ylabel.assert_called_with("efficiency")
            mock_ax.set_zlabel.assert_called_with("complexity")
    
    def test_parallel_coordinates_plot(self):
        """Test parallel coordinates plot for multi-objective visualization."""
        results = create_sample_evaluation_results(15)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                results,
                objectives=["accuracy", "efficiency", "complexity"],
                plot_type="parallel_coordinates"
            )
            
            assert plot_result is not None
            mock_ax.plot.assert_called()
    
    def test_radar_chart(self):
        """Test radar chart for objective comparison."""
        results = create_sample_evaluation_results(10)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                results[:5],  # First 5 solutions
                objectives=["accuracy", "efficiency", "complexity"],
                plot_type="radar"
            )
            
            assert plot_result is not None
            mock_ax.plot.assert_called()
    
    def test_heatmap_visualization(self):
        """Test heatmap visualization for objective correlations."""
        results = create_sample_evaluation_results(20)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                results,
                objectives=["accuracy", "efficiency", "complexity"],
                plot_type="heatmap"
            )
            
            assert plot_result is not None
            mock_ax.imshow.assert_called()
    
    def test_pareto_frontier_highlighting(self):
        """Test highlighting of Pareto frontier in plots."""
        results = create_sample_evaluation_results(20)
        
        # Identify Pareto frontier
        frontier = results[:8]  # Assume first 8 are on frontier
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                results,
                x_objective="accuracy",
                y_objective="efficiency",
                frontier_solutions=frontier,
                plot_type="2d"
            )
            
            assert plot_result is not None
            # Should have two scatter calls: one for population, one for frontier
            assert mock_ax.scatter.call_count >= 2


class TestProgressVisualization:
    """Test suite for progress visualization."""
    
    def test_line_plot_creation(self):
        """Test line plot for progress tracking."""
        progress_data = create_progress_data(20)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_progress_plot(
                progress_data,
                metrics=["best_accuracy", "avg_efficiency"],
                plot_type="line"
            )
            
            assert plot_result is not None
            mock_ax.plot.assert_called()
            mock_ax.legend.assert_called()
    
    def test_convergence_plot(self):
        """Test convergence visualization."""
        progress_data = create_progress_data(15)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_progress_plot(
                progress_data,
                metrics=["best_accuracy"],
                plot_type="convergence",
                convergence_point=10
            )
            
            assert plot_result is not None
            mock_ax.plot.assert_called()
            mock_ax.axvline.assert_called_with(10, color='red', linestyle='--')
    
    def test_multi_axis_plot(self):
        """Test multi-axis plot for different scales."""
        progress_data = create_progress_data(20)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_fig.add_subplot.return_value = mock_ax1
            mock_ax1.twinx.return_value = mock_ax2
            mock_plt.figure.return_value = mock_fig
            
            plot_result = create_progress_plot(
                progress_data,
                metrics=["best_accuracy", "hypervolume"],
                plot_type="multi_axis"
            )
            
            assert plot_result is not None
            mock_ax1.plot.assert_called()
            mock_ax2.plot.assert_called()
    
    def test_subplots_dashboard(self):
        """Test dashboard with multiple subplots."""
        progress_data = create_progress_data(20)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig = Mock()
            mock_axes = [Mock() for _ in range(4)]  # 2x2 subplot grid
            mock_fig.subplots.return_value = (mock_fig, mock_axes)
            
            plot_result = create_progress_plot(
                progress_data,
                metrics=["best_accuracy", "avg_efficiency", "hypervolume", "diversity"],
                plot_type="dashboard"
            )
            
            assert plot_result is not None
            assert mock_fig.subplots.call_count == 1
    
    def test_animated_progress(self):
        """Test animated progress visualization."""
        progress_data = create_progress_data(10)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            # Mock animation
            with patch('src.dspy_gepa.core.visualization.FuncAnimation') as mock_animation:
                mock_anim = Mock()
                mock_animation.return_value = mock_anim
                
                plot_result = create_progress_plot(
                    progress_data,
                    metrics=["best_accuracy"],
                    plot_type="animated"
                )
                
                assert plot_result is not None
                mock_animation.assert_called()


class TestConvenienceFunctions:
    """Test suite for convenience plotting functions."""
    
    def test_create_pareto_plot(self):
        """Test convenience function for Pareto plots."""
        results = create_sample_evaluation_results(15)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                results,
                x_objective="accuracy",
                y_objective="efficiency",
                plot_type="2d"
            )
            
            assert plot_result is not None
    
    def test_create_progress_plot(self):
        """Test convenience function for progress plots."""
        progress_data = create_progress_data(15)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_progress_plot(
                progress_data,
                metrics=["best_accuracy", "hypervolume"],
                plot_type="line"
            )
            
            assert plot_result is not None
    
    def test_create_comparison_plot(self):
        """Test convenience function for comparison plots."""
        run1_data = create_progress_data(10)
        run2_data = create_progress_data(10)
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_comparison_plot(
                {"run1": run1_data, "run2": run2_data},
                metric="best_accuracy",
                plot_type="line"
            )
            
            assert plot_result is not None
    
    def test_text_fallback_plots(self, mock_matplotlib, mock_plotly):
        """Test text-based fallback plots."""
        results = create_sample_evaluation_results(5)
        progress_data = create_progress_data(5)
        
        # Pareto plot fallback
        pareto_text = create_pareto_plot(
            results,
            x_objective="accuracy",
            y_objective="efficiency",
            backend="text"
        )
        
        assert isinstance(pareto_text, str)
        assert "accuracy" in pareto_text
        assert "efficiency" in pareto_text
        
        # Progress plot fallback
        progress_text = create_progress_plot(
            progress_data,
            metrics=["best_accuracy"],
            backend="text"
        )
        
        assert isinstance(progress_text, str)
        assert "best_accuracy" in progress_text
    
    @pytest.mark.slow
    def test_large_dataset_visualization(self):
        """Test visualization with large datasets."""
        # Create large dataset
        large_results = create_sample_evaluation_results(500)
        
        start_time = time.time()
        
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_fig, mock_ax = Mock(), Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            plot_result = create_pareto_plot(
                large_results,
                x_objective="accuracy",
                y_objective="efficiency"
            )
            
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert plot_result is not None
    
    def test_error_handling(self):
        """Test error handling in visualization functions."""
        # Test with invalid data
        with pytest.raises((ValueError, TypeError)):
            create_pareto_plot(
                None,
                x_objective="accuracy",
                y_objective="efficiency"
            )
        
        # Test with missing objectives
        results = create_sample_evaluation_results(5)
        
        with pytest.raises((ValueError, KeyError)):
            create_pareto_plot(
                results,
                x_objective="nonexistent_objective",
                y_objective="efficiency"
            )
    
    def test_thread_safety(self, thread_safety_test):
        """Test thread safety of visualization functions."""
        def create_plot():
            results = create_sample_evaluation_results(10)
            
            with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
                mock_fig, mock_ax = Mock(), Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                
                return create_pareto_plot(
                    results,
                    x_objective="accuracy",
                    y_objective="efficiency"
                )
        
        # Run concurrent plot creation
        results = thread_safety_test(create_plot, num_threads=3, iterations=5)
        
        # Should have no errors
        assert results["total_errors"] == 0
        assert results["success_rate"] == 1.0
