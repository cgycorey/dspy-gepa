"""Core components for dspy-gepa.

This package contains core framework components:
- agent: High-level agent interface
- visualization: Multi-objective optimization visualization framework

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from .agent import GEPAAgent
from .visualization import (
    ParetoFrontierVisualizer,
    OptimizationProgressVisualizer,
    VisualizationConfig,
    VisualizationType,
    create_pareto_analysis,
    create_progress_analysis,
)

__all__ = [
    "GEPAAgent",
    "ParetoFrontierVisualizer",
    "OptimizationProgressVisualizer",
    "VisualizationConfig",
    "VisualizationType",
    "create_pareto_analysis",
    "create_progress_analysis",
]