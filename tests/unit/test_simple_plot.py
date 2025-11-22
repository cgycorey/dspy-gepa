"""Simple test to verify plot saving functionality."""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dspy_gepa.core.visualization import ParetoFrontierVisualizer, VisualizationConfig, VisualizationType
from dspy_gepa.core.interfaces import EvaluationResult, OptimizationDirection, ObjectiveEvaluation
import random

def test_simple_plot():
    """Test basic plot saving functionality."""
    create_simple_test()

def create_simple_test():
    """Create a simple test with actual plot saving."""
    print("Creating simple visualization test...")
    
    # Create sample solutions
    solutions = []
    for i in range(20):
        objectives = {
            "accuracy": ObjectiveEvaluation(
                objective_name="accuracy",
                score=random.uniform(0.6, 0.95),
                direction=OptimizationDirection.MAXIMIZE
            ),
            "token_usage": ObjectiveEvaluation(
                objective_name="token_usage",
                score=random.uniform(100, 500),
                direction=OptimizationDirection.MINIMIZE
            ),
        }
        
        solution = EvaluationResult(
            solution_id=f"sol_{i:03d}",
            objectives=objectives,
            overall_score=None,
            metadata=None
        )
        solutions.append(solution)
    
    # Create objectives for the visualizer
    from dspy_gepa.core.objectives import AccuracyMetric, TokenUsageMetric
    objectives = [
        AccuracyMetric(weight=1.0, direction=OptimizationDirection.MAXIMIZE),
        TokenUsageMetric(weight=1.0, direction=OptimizationDirection.MINIMIZE),
    ]
    
    # Configure visualizer
    config = VisualizationConfig(
        figure_size=(10, 8),
        interactive=False,
        show_grid=True,
        show_legend=True
    )
    
    # Create visualizer
    visualizer = ParetoFrontierVisualizer(config)
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    # Test 2D scatter plot
    try:
        print("Creating 2D scatter plot...")
        result = visualizer.visualize(
            solutions, objectives, VisualizationType.SCATTER_2D,
            title="Pareto Frontier - 2D Scatter Plot",
            save_path="test_outputs/pareto_2d_scatter.png",
            show_plot=False
        )
        print("✓ 2D scatter plot created successfully")
    except Exception as e:
        print(f"✗ 2D scatter plot failed: {e}")
    
    # Test parallel coordinates plot
    try:
        print("Creating parallel coordinates plot...")
        result = visualizer.visualize(
            solutions, objectives, VisualizationType.PARALLEL_COORDINATES,
            title="Pareto Frontier - Parallel Coordinates",
            save_path="test_outputs/pareto_parallel_coordinates.png",
            show_plot=False
        )
        print("✓ Parallel coordinates plot created successfully")
    except Exception as e:
        print(f"✗ Parallel coordinates plot failed: {e}")
    
    # Test heatmap
    try:
        print("Creating heatmap...")
        result = visualizer.visualize(
            solutions, objectives, VisualizationType.HEATMAP,
            title="Pareto Frontier - Heatmap",
            save_path="test_outputs/pareto_heatmap.png",
            show_plot=False
        )
        print("✓ Heatmap created successfully")
    except Exception as e:
        print(f"✗ Heatmap failed: {e}")
    
    print("\nTest completed. Check test_outputs directory for generated plots.")

if __name__ == "__main__":
    create_simple_test()