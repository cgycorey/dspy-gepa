"""Visualization framework for multi-objective optimization results.

This module provides comprehensive visualization capabilities for multi-objective
optimization results, including Pareto frontier visualization and optimization
progress tracking. It supports both static (matplotlib) and interactive (plotly)
visualizations with graceful fallbacks to text-based reports.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from .interfaces import EvaluationResult, Objective, OptimizationDirection
from ..utils.logging import get_logger


_logger = get_logger(__name__)

# Optional imports with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    _logger.info("matplotlib not available, falling back to text-based reports")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    _logger.info("plotly not available, static plots only")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    _logger.info("numpy not available, using pure Python calculations")


class VisualizationType(Enum):
    """Supported visualization types."""
    SCATTER_2D = "scatter_2d"
    SCATTER_3D = "scatter_3d"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR_CHART = "radar_chart"
    HEATMAP = "heatmap"
    CONVERGENCE_PLOT = "convergence_plot"
    OBJECTIVE_TRAJECTORIES = "objective_trajectories"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    style: str = "default"
    color_palette: List[str] = None
    interactive: bool = True
    save_format: str = "png"
    show_grid: bool = True
    show_legend: bool = True
    title_font_size: int = 14
    label_font_size: int = 12
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
            ]


class TextReportGenerator:
    """Fallback text-based report generator when plotting libraries aren't available."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def generate_pareto_report(
        self,
        solutions: List[EvaluationResult],
        objectives: List[Objective],
        title: str = "Pareto Frontier Analysis"
    ) -> str:
        """Generate text-based Pareto frontier report."""
        lines = [f"\n{title}", "=" * len(title)]
        
        if not solutions:
            lines.append("No solutions to display.")
            return "\n".join(lines)
        
        lines.append(f"\nTotal Solutions: {len(solutions)}")
        lines.append(f"Objectives: {', '.join(obj.name for obj in objectives)}")
        
        # Extract objective values
        obj_values = []
        for solution in solutions:
            values = [solution.get_objective_score(obj.name) for obj in objectives]
            obj_values.append(values)
        
        # Statistics for each objective
        lines.append("\nObjective Statistics:")
        lines.append("-" * 20)
        for i, obj in enumerate(objectives):
            values = [v[i] for v in obj_values]
            if values:
                lines.append(f"{obj.name}:")
                lines.append(f"  Min: {min(values):.4f}")
                lines.append(f"  Max: {max(values):.4f}")
                lines.append(f"  Mean: {statistics.mean(values):.4f}")
                lines.append(f"  Std: {statistics.stdev(values) if len(values) > 1 else 0.0:.4f}")
        
        # Top solutions
        lines.append("\nTop 5 Solutions (by first objective):")
        lines.append("-" * 35)
        sorted_solutions = sorted(
            solutions,
            key=lambda s: s.get_objective_score(objectives[0].name),
            reverse=objectives[0].direction == OptimizationDirection.MAXIMIZE
        )
        
        for i, solution in enumerate(sorted_solutions[:5]):
            lines.append(f"\nSolution {i+1} (ID: {solution.solution_id[:8]}):")
            for obj in objectives:
                value = solution.get_objective_score(obj.name)
                lines.append(f"  {obj.name}: {value:.4f}")
        
        return "\n".join(lines)
    
    def generate_progress_report(
        self,
        progress_data: Dict[str, List[float]],
        title: str = "Optimization Progress Report"
    ) -> str:
        """Generate text-based progress report."""
        lines = [f"\n{title}", "=" * len(title)]
        
        if not progress_data:
            lines.append("No progress data to display.")
            return "\n".join(lines)
        
        for metric, values in progress_data.items():
            if not values:
                continue
            
            lines.append(f"\n{metric}:")
            lines.append(f"  Initial: {values[0]:.4f}")
            lines.append(f"  Final: {values[-1]:.4f}")
            
            if len(values) > 1:
                improvement = values[-1] - values[0]
                lines.append(f"  Improvement: {improvement:+.4f}")
                lines.append(f"  Best: {max(values):.4f}")
                lines.append(f"  Worst: {min(values):.4f}")
        
        return "\n".join(lines)

class ParetoFrontierVisualizer:
    """Visualizer for Pareto frontier analysis with support for multiple chart types."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.text_gen = TextReportGenerator(self.config)
    


class OptimizationProgressVisualizer:
    """Visualizer for optimization progress tracking over time."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.text_gen = TextReportGenerator(self.config)
    
    def plot_progress(
        self,
        progress_data: Dict[str, List[float]],
        title: str = "Optimization Progress"
    ) -> None:
        """Plot optimization progress over generations."""
        if not progress_data:
            print("No progress data to visualize")
            return
        
        # Fallback to text report if no plotting libraries
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            report = self.text_gen.generate_progress_report(progress_data, title)
            print(report)
            return
        
        # Simple matplotlib implementation if available
        if MATPLOTLIB_AVAILABLE:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                for metric, values in progress_data.items():
                    generations = list(range(len(values)))
                    ax.plot(generations, values, label=metric, marker='o')
                
                ax.set_xlabel('Generation')
                ax.set_ylabel('Score')
                ax.set_title(title)
                ax.legend()
                ax.grid(self.config.show_grid)
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error creating progress plot: {e}")
                # Fallback to text report
                report = self.text_gen.generate_progress_report(progress_data, title)
                print(report)


def create_pareto_analysis(config: VisualizationConfig = None) -> ParetoFrontierVisualizer:
    """Factory function to create Pareto frontier visualizer."""
    return ParetoFrontierVisualizer(config)


def create_progress_analysis(config: VisualizationConfig = None) -> OptimizationProgressVisualizer:
    """Factory function to create optimization progress visualizer."""
    return OptimizationProgressVisualizer(config)


# Additional classes for architecture compliance

class VisualizationEngine:
    """Main visualization engine for multi-objective optimization."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.pareto_plotter = ParetoFrontierPlotter(self.config)
        self.convergence_plotter = ConvergencePlotter(self.config)
        self.performance_plotter = PerformancePlotter(self.config)
    
    def create_plot(self, plot_type: str, data: Any, **kwargs) -> Optional[str]:
        """Create a plot of the specified type."""
        if plot_type == "pareto_2d":
            return self.pareto_plotter.plot_2d_frontier(data, **kwargs)
        elif plot_type == "pareto_3d":
            return self.pareto_plotter.plot_3d_frontier(data, **kwargs)
        elif plot_type == "convergence":
            return self.convergence_plotter.plot_convergence(data, **kwargs)
        elif plot_type == "performance":
            return self.performance_plotter.plot_performance(data, **kwargs)
        else:
            _logger.error(f"Unknown plot type: {plot_type}")
            return None
    
    def save_plot(self, plot_path: str, data: Any, plot_type: str, **kwargs) -> bool:
        """Save a plot to file."""
        result = self.create_plot(plot_type, data, save_path=plot_path, **kwargs)
        return result is not None
    
    def show_plot(self, plot_type: str, data: Any, **kwargs) -> None:
        """Display a plot."""
        self.create_plot(plot_type, data, show=True, **kwargs)


class ParetoFrontierPlotter:
    """Specialized plotter for Pareto frontiers."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_2d_frontier(self, frontier: List[EvaluationResult], obj1: str, obj2: str,
                         save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Create 2D Pareto frontier plot."""
        if not frontier:
            _logger.warning("Empty frontier provided")
            return None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                # Extract objective values
                obj1_values = []
                obj2_values = []
                
                for solution in frontier:
                    obj1_score = solution.get_objective_score(obj1)
                    obj2_score = solution.get_objective_score(obj2)
                    if obj1_score is not None and obj2_score is not None:
                        obj1_values.append(obj1_score)
                        obj2_values.append(obj2_score)
                
                if not obj1_values or not obj2_values:
                    _logger.warning(f"No valid scores for objectives {obj1}, {obj2}")
                    return None
                
                plt.figure(figsize=self.config.figure_size)
                plt.scatter(obj1_values, obj2_values, alpha=0.7, s=self.config.point_size)
                plt.xlabel(obj1, fontsize=self.config.font_size)
                plt.ylabel(obj2, fontsize=self.config.font_size)
                plt.title(f"Pareto Frontier: {obj1} vs {obj2}", fontsize=self.config.font_size + 2)
                plt.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return save_path or "plot_generated"
                
            except Exception as e:
                _logger.error(f"Error creating 2D plot: {e}")
                return None
        else:
            _logger.warning("Matplotlib not available for 2D plotting")
            return None
    
    def plot_3d_frontier(self, frontier: List[EvaluationResult], obj1: str, obj2: str, obj3: str,
                         save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Create 3D Pareto frontier plot."""
        if not frontier:
            _logger.warning("Empty frontier provided")
            return None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                # Extract objective values
                obj1_values = []
                obj2_values = []
                obj3_values = []
                
                for solution in frontier:
                    obj1_score = solution.get_objective_score(obj1)
                    obj2_score = solution.get_objective_score(obj2)
                    obj3_score = solution.get_objective_score(obj3)
                    if all(score is not None for score in [obj1_score, obj2_score, obj3_score]):
                        obj1_values.append(obj1_score)
                        obj2_values.append(obj2_score)
                        obj3_values.append(obj3_score)
                
                if not obj1_values:
                    _logger.warning(f"No valid scores for objectives {obj1}, {obj2}, {obj3}")
                    return None
                
                fig = plt.figure(figsize=self.config.figure_size)
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(obj1_values, obj2_values, obj3_values, alpha=0.7, s=self.config.point_size)
                ax.set_xlabel(obj1, fontsize=self.config.font_size)
                ax.set_ylabel(obj2, fontsize=self.config.font_size)
                ax.set_zlabel(obj3, fontsize=self.config.font_size)
                ax.set_title(f"Pareto Frontier: {obj1}, {obj2}, {obj3}", fontsize=self.config.font_size + 2)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return save_path or "plot_generated"
                
            except Exception as e:
                _logger.error(f"Error creating 3D plot: {e}")
                return None
        else:
            _logger.warning("Matplotlib not available for 3D plotting")
            return None
    
    def plot_parallel_coordinates(self, frontier: List[EvaluationResult], objectives: List[str],
                                  save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Create parallel coordinates plot."""
        if not frontier or not objectives:
            _logger.warning("Empty frontier or objectives provided")
            return None
        
        try:
            import pandas as pd
            
            # Prepare data
            data = []
            for solution in frontier:
                row = {'solution': solution.solution_id}
                for obj in objectives:
                    score = solution.get_objective_score(obj)
                    row[obj] = score if score is not None else 0.0
                data.append(row)
            
            df = pd.DataFrame(data)
            
            if MATPLOTLIB_AVAILABLE:
                from pandas.plotting import parallel_coordinates
                
                plt.figure(figsize=self.config.figure_size)
                parallel_coordinates(df, 'solution', colormap='viridis')
                plt.title("Pareto Frontier - Parallel Coordinates", fontsize=self.config.font_size + 2)
                plt.xticks(rotation=45, fontsize=self.config.font_size)
                plt.yticks(fontsize=self.config.font_size)
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return save_path or "plot_generated"
            
        except ImportError:
            _logger.warning("Pandas not available for parallel coordinates plot")
            return None
        except Exception as e:
            _logger.error(f"Error creating parallel coordinates plot: {e}")
            return None


class ConvergencePlotter:
    """Specialized plotter for convergence analysis."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_convergence(self, history: Dict[str, List[float]], 
                         save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Plot convergence metrics over time."""
        if not history:
            _logger.warning("Empty history provided")
            return None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                fig, axes = plt.subplots(len(history), 1, figsize=(10, 4 * len(history)))
                if len(history) == 1:
                    axes = [axes]
                
                for i, (metric_name, values) in enumerate(history.items()):
                    axes[i].plot(values, linewidth=2)
                    axes[i].set_title(f"{metric_name} Convergence", fontsize=self.config.font_size)
                    axes[i].set_xlabel("Generation", fontsize=self.config.font_size)
                    axes[i].set_ylabel(metric_name, fontsize=self.config.font_size)
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return save_path or "plot_generated"
                
            except Exception as e:
                _logger.error(f"Error creating convergence plot: {e}")
                return None
        else:
            _logger.warning("Matplotlib not available for convergence plotting")
            return None
    
    def plot_hypervolume(self, hypervolume_history: List[float],
                        save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Plot hypervolume over generations."""
        return self.plot_convergence({"hypervolume": hypervolume_history}, save_path, show)
    
    def plot_diversity(self, diversity_history: List[float],
                      save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Plot diversity metrics over generations."""
        return self.plot_convergence({"diversity": diversity_history}, save_path, show)


class PerformancePlotter:
    """Specialized plotter for performance analysis."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_performance(self, frontier: List[EvaluationResult],
                         save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Plot performance metrics for solutions."""
        if not frontier:
            _logger.warning("Empty frontier provided")
            return None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                # Collect all objectives
                all_objectives = set()
                for solution in frontier:
                    all_objectives.update(solution.objectives.keys())
                
                if not all_objectives:
                    _logger.warning("No objectives found in frontier")
                    return None
                
                # Create subplots for each objective
                fig, axes = plt.subplots(1, len(all_objectives), figsize=(4 * len(all_objectives), 6))
                if len(all_objectives) == 1:
                    axes = [axes]
                
                for i, obj_name in enumerate(sorted(all_objectives)):
                    values = []
                    for solution in frontier:
                        score = solution.get_objective_score(obj_name)
                        if score is not None:
                            values.append(score)
                    
                    if values:
                        axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
                        axes[i].set_title(f"{obj_name} Distribution", fontsize=self.config.font_size)
                        axes[i].set_xlabel(obj_name, fontsize=self.config.font_size)
                        axes[i].set_ylabel("Frequency", fontsize=self.config.font_size)
                        axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return save_path or "plot_generated"
                
            except Exception as e:
                _logger.error(f"Error creating performance plot: {e}")
                return None
        else:
            _logger.warning("Matplotlib not available for performance plotting")
            return None
    
    def plot_objectives(self, frontier: List[EvaluationResult], objectives: List[str],
                       save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Plot specific objectives."""
        return self.plot_performance(frontier, save_path, show)
    
    def plot_comparison(self, solutions: List[EvaluationResult], metric: str,
                       save_path: Optional[str] = None, show: bool = False) -> Optional[str]:
        """Compare solutions on a specific metric."""
        if not solutions:
            _logger.warning("No solutions provided")
            return None
        
        if MATPLOTLIB_AVAILABLE:
            try:
                solution_names = []
                metric_values = []
                
                for solution in solutions:
                    score = solution.get_objective_score(metric)
                    if score is not None:
                        solution_names.append(solution.solution_id[:10])  # Truncate for display
                        metric_values.append(score)
                
                if not metric_values:
                    _logger.warning(f"No valid scores for metric {metric}")
                    return None
                
                plt.figure(figsize=self.config.figure_size)
                plt.bar(solution_names, metric_values, alpha=0.7)
                plt.title(f"Solution Comparison: {metric}", fontsize=self.config.font_size + 2)
                plt.xlabel("Solution", fontsize=self.config.font_size)
                plt.ylabel(metric, fontsize=self.config.font_size)
                plt.xticks(rotation=45, fontsize=self.config.font_size)
                plt.yticks(fontsize=self.config.font_size)
                plt.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return save_path or "plot_generated"
                
            except Exception as e:
                _logger.error(f"Error creating comparison plot: {e}")
                return None
        else:
            _logger.warning("Matplotlib not available for comparison plotting")
            return None
