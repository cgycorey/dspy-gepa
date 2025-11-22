"""DSPy integration layer for multi-objective GEPA optimization.

This package provides seamless integration between DSPy modules and the
multi-objective GEPA optimization framework, enabling automatic metric
conversion, signature-aware optimization, and enhanced DSPy workflows.

Components:
- MultiObjectiveOptimizer: Main optimizer for DSPy modules
- MetricConverter: Automatic conversion of DSPy metrics
- SignatureAnalyzer: Signature-aware optimization strategies

Example Usage:
    ```python
    from dspy_gepa.dspy_integration import MultiObjectiveOptimizer
    from dspy_gepa.core import AccuracyMetric, EfficiencyMetric
    
    optimizer = MultiObjectiveOptimizer()
    result = optimizer.optimize_module(
        module=module,
        trainset=train_data,
        objectives=[AccuracyMetric(), EfficiencyMetric()]
    )
    ```
"""

from .multi_objective_optimizer import MultiObjectiveOptimizer
from .metric_converter import DSPyMetricConverter
from .signature_analyzer import DSPySignatureAnalyzer

__all__ = [
    "MultiObjectiveOptimizer",
    "DSPyMetricConverter", 
    "DSPySignatureAnalyzer"
]