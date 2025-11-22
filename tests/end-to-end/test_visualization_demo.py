"""Demo script to test the visualization framework.

This script creates sample data and demonstrates the visualization capabilities
of the multi-objective optimization framework.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

import random
import sys
import os
from typing import List, Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dspy_gepa.core.visualization import (
    ParetoFrontierVisualizer,
    OptimizationProgressVisualizer,
    VisualizationConfig,
    VisualizationType,
    create_pareto_analysis,
    create_progress_analysis,
)
from dspy_gepa.core.interfaces import EvaluationResult, Objective, OptimizationDirection, ObjectiveEvaluation
from dspy_gepa.core.objectives import AccuracyMetric, TokenUsageMetric, ExecutionTimeMetric


def create_sample_solutions(num_solutions: int = 50) -> List[EvaluationResult]:
    """Create sample solutions for testing."""
    solutions = []
    
    for i in range(num_solutions):
        # Create a simple solution with random objective values
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
            "complexity": ObjectiveEvaluation(
                objective_name="complexity",
                score=random.uniform(0.1, 0.8),
                direction=OptimizationDirection.MINIMIZE
            ),
        }
        
        solution = EvaluationResult(
            solution_id=f"sol_{i:03d}",
            objectives=objectives,
            overall_score=None,  # Not used for visualization
            metadata=None
        )
        solutions.append(solution)
    
    return solutions


def create_sample_objectives() -> List[Objective]:
    """Create sample objectives for testing."""
    return [
        AccuracyMetric(weight=1.0, direction=OptimizationDirection.MAXIMIZE),
        TokenUsageMetric(weight=1.0, direction=OptimizationDirection.MINIMIZE),
        # Note: We'll use a custom objective for complexity
    ]


def create_sample_progress_data() -> Dict[str, List[float]]:
    """Create sample progress data for testing."""
    generations = 30
    
    # Simulate convergence with some noise
    progress_data = {
        "best_accuracy": [],
        "avg_accuracy": [],
        "best_efficiency": [],
        "avg_efficiency": [],
        "cpu_usage": [],
        "memory_usage": [],
    }
    
    for gen in range(generations):
        # Simulate improvement over time with diminishing returns
        improvement_factor = 1 - (gen / generations) * 0.5
        noise = random.uniform(-0.02, 0.02)
        
        progress_data["best_accuracy"].append(0.95 - improvement_factor * 0.25 + noise)
        progress_data["avg_accuracy"].append(0.85 - improvement_factor * 0.35 + noise * 0.5)
        progress_data["best_efficiency"].append(0.90 - improvement_factor * 0.40 + noise)
        progress_data["avg_efficiency"].append(0.75 - improvement_factor * 0.45 + noise * 0.5)
        
        # Simulate resource usage
        progress_data["cpu_usage"].append(50 + random.uniform(-10, 20))
        progress_data["memory_usage"].append(2048 + random.uniform(-200, 400))
    
    return progress_data


def test_pareto_visualization():
    """Test Pareto frontier visualization."""
    print("\n" + "="*50)
    print("TESTING PARETO FRONTIER VISUALIZATION")
    print("="*50)
    
    # Create sample data
    solutions = create_sample_solutions(30)
    objectives = create_sample_objectives()
    
    # Add complexity objective manually
    from dspy_gepa.core.objectives import TokenUsageMetric
    complexity_obj = TokenUsageMetric(
        weight=1.0,
        direction=OptimizationDirection.MINIMIZE
    )
    # Rename it to complexity for demonstration
    complexity_obj.name = "complexity"
    complexity_obj.description = "Model complexity (lower is better)"
    objectives.append(complexity_obj)
    
    # Configuration
    config = VisualizationConfig(
        figure_size=(12, 8),
        interactive=False,  # Set to True if you have plotly installed
        show_grid=True,
        show_legend=True
    )
    
    # Create visualizer
    visualizer = ParetoFrontierVisualizer(config)
    
    # Test different visualization types
    viz_types = [
        VisualizationType.SCATTER_2D,
        VisualizationType.PARALLEL_COORDINATES,
        VisualizationType.HEATMAP,
    ]
    
    if len(objectives) >= 3:
        viz_types.extend([VisualizationType.SCATTER_3D, VisualizationType.RADAR_CHART])
    
    for viz_type in viz_types:
        try:
            print(f"\nTesting {viz_type.value}...")
            result = visualizer.visualize(
                solutions, objectives, viz_type,
                title=f"Test {viz_type.value.replace('_', ' ').title()}",
                save_path=f"test_pareto_{viz_type.value}.png",
                show_plot=False  # Set to True to display plots
            )
            print(f"✓ {viz_type.value} visualization created successfully")
        except Exception as e:
            print(f"✗ {viz_type.value} visualization failed: {e}")
    
    # Test comprehensive analysis
    try:
        print("\nTesting comprehensive Pareto analysis...")
        results = create_pareto_analysis(
            solutions, objectives, config,
            save_dir="test_outputs"
        )
        print(f"✓ Comprehensive analysis created with {len(results)} visualizations")
    except Exception as e:
        print(f"✗ Comprehensive analysis failed: {e}")


def test_progress_visualization():
    """Test optimization progress visualization."""
    print("\n" + "="*50)
    print("TESTING OPTIMIZATION PROGRESS VISUALIZATION")
    print("="*50)
    
    # Create sample progress data
    progress_data = create_sample_progress_data()
    
    # Configuration
    config = VisualizationConfig(
        figure_size=(12, 6),
        interactive=False,  # Set to True if you have plotly installed
        show_grid=True,
        show_legend=True
    )
    
    # Create visualizer
    visualizer = OptimizationProgressVisualizer(config)
    
    # Test different visualization types
    viz_types = [
        VisualizationType.CONVERGENCE_PLOT,
        VisualizationType.OBJECTIVE_TRAJECTORIES,
        VisualizationType.RESOURCE_USAGE,
    ]
    
    for viz_type in viz_types:
        try:
            print(f"\nTesting {viz_type.value}...")
            result = visualizer.visualize(
                progress_data, viz_type,
                title=f"Test {viz_type.value.replace('_', ' ').title()}",
                save_path=f"test_progress_{viz_type.value}.png",
                show_plot=False  # Set to True to display plots
            )
            print(f"✓ {viz_type.value} visualization created successfully")
        except Exception as e:
            print(f"✗ {viz_type.value} visualization failed: {e}")
    
    # Test comprehensive analysis
    try:
        print("\nTesting comprehensive progress analysis...")
        results = create_progress_analysis(
            progress_data, config,
            save_dir="test_outputs"
        )
        print(f"✓ Comprehensive analysis created with {len(results)} visualizations")
    except Exception as e:
        print(f"✗ Comprehensive analysis failed: {e}")


def test_text_fallbacks():
    """Test text-based fallbacks when plotting libraries aren't available."""
    print("\n" + "="*50)
    print("TESTING TEXT-BASED FALLBACKS")
    print("="*50)
    
    # Create sample data
    solutions = create_sample_solutions(10)
    progress_data = create_sample_progress_data()
    
    # Create objectives
    objectives = create_sample_objectives()
    
    # Configuration with text-only mode
    config = VisualizationConfig(interactive=False)
    
    # Test text reports
    try:
        print("\nTesting Pareto text report...")
        visualizer = ParetoFrontierVisualizer(config)
        
        # Temporarily disable plotting libraries to test fallback
        import dspy_gepa.core.visualization as viz_module
        original_matplotlib = viz_module.MATPLOTLIB_AVAILABLE
        original_plotly = viz_module.PLOTLY_AVAILABLE
        
        viz_module.MATPLOTLIB_AVAILABLE = False
        viz_module.PLOTLY_AVAILABLE = False
        
        result = visualizer.visualize(
            solutions, objectives, VisualizationType.SCATTER_2D,
            show_plot=False
        )
        
        # Restore original values
        viz_module.MATPLOTLIB_AVAILABLE = original_matplotlib
        viz_module.PLOTLY_AVAILABLE = original_plotly
        
        print("✓ Text-based Pareto report generated successfully")
        
    except Exception as e:
        print(f"✗ Text-based Pareto report failed: {e}")
    
    try:
        print("\nTesting Progress text report...")
        visualizer = OptimizationProgressVisualizer(config)
        
        # Temporarily disable plotting libraries to test fallback
        viz_module.MATPLOTLIB_AVAILABLE = False
        viz_module.PLOTLY_AVAILABLE = False
        
        result = visualizer.visualize(
            progress_data, VisualizationType.CONVERGENCE_PLOT,
            show_plot=False
        )
        
        # Restore original values
        viz_module.MATPLOTLIB_AVAILABLE = original_matplotlib
        viz_module.PLOTLY_AVAILABLE = original_plotly
        
        print("✓ Text-based Progress report generated successfully")
        
    except Exception as e:
        print(f"✗ Text-based Progress report failed: {e}")


def main():
    """Run all visualization tests."""
    print("DSPY-GEPA Visualization Framework Demo")
    print("=====================================")
    
    # Check available libraries
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not available (will use text fallbacks)")
    
    try:
        import plotly
        print("✓ plotly available")
    except ImportError:
        print("✗ plotly not available (will use matplotlib fallbacks)")
    
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        print("✗ numpy not available (will use pure Python calculations)")
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    # Run tests
    test_pareto_visualization()
    test_progress_visualization()
    test_text_fallbacks()
    
    print("\n" + "="*50)
    print("DEMO COMPLETED")
    print("Check the 'test_outputs' directory for generated plots.")
    print("="*50)


if __name__ == "__main__":
    main()