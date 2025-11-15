"""DSPY-GEPA Integration Layer.

This package provides the integration between DSPY programs and the GEPA (Genetic-Pareto Algorithm)
framework. It enables evolutionary optimization of DSPY programs by converting them to GEPA
candidates and applying genetic operators for prompt engineering and program optimization.

Main Components:
- DSPYAdapter: Main adapter for converting DSPY programs to/from GEPA candidates
- MetricCollector: Performance metrics collection and tracking
- ProgramParser: Parse and analyze DSPY programs for optimization

Example Usage:
    from dspy_gepa.dspy_integration import DSPYAdapter, MetricCollector
    
    # Create adapter
    adapter = DSPYAdapter()
    
    # Convert DSPY program to GEPA candidate
    candidate = adapter.dspy_to_candidate(program)
    
    # Convert back to DSPY program
    program = adapter.candidate_to_dspy(candidate)
    
    # Collect metrics
    collector = MetricCollector()
    metrics = collector.evaluate_program(program, dataset)
"""

from .dspy_adapter import DSPYAdapter
from .metric_collector import MetricCollector, DSPYMetrics

# TODO: Add program_parser when implemented
# from .program_parser import ProgramParser, DSPYProgramInfo

__all__ = [
    "DSPYAdapter",
    "MetricCollector", 
    "DSPYMetrics",
    # TODO: Add when implemented
    # "ProgramParser",
    # "DSPYProgramInfo",
]

__version__ = "0.1.0"