"""Metric converter for automatic DSPy to GEPA metric conversion.

This module provides automatic conversion of DSPy metrics to the multi-objective
format used by GEPA, enabling seamless integration between DSPy evaluation
functions and the multi-objective optimization framework.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from ..core.interfaces import Objective, MetricConverter, OptimizationDirection
from ..core.objectives import (
    AccuracyMetric, TokenUsageMetric, ExecutionTimeMetric,
    FluencyMetric, RelevanceMetric, CompositeMetric
)
from ..utils.logging import get_logger


_logger = get_logger(__name__)


@dataclass
class MetricConversionConfig:
    """Configuration for metric conversion."""
    normalize_scores: bool = True
    infer_direction: bool = True
    default_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    cache_conversions: bool = True
    validate_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.default_direction, OptimizationDirection):
            raise ValueError("default_direction must be an OptimizationDirection enum")


class DSPyMetricConverter(MetricConverter):
    """Converter for DSPy metrics to multi-objective format.
    
    This class provides automatic conversion of DSPy evaluation functions
    to the multi-objective format used by GEPA, handling direction inference,
    normalization, and composite metric creation.
    """
    
    def __init__(self, config: Optional[MetricConversionConfig] = None):
        """Initialize the metric converter.
        
        Args:
            config: Conversion configuration
        """
        self.config = config or MetricConversionConfig()
        self.conversion_cache: Dict[str, Objective] = {}
        
        # Registry of known DSPy metrics and their properties
        self.known_metrics = {
            'evaluate_exact_match': {
                'direction': OptimizationDirection.MAXIMIZE,
                'range': (0.0, 1.0),
                'category': 'accuracy'
            },
            'evaluate_f1': {
                'direction': OptimizationDirection.MAXIMIZE,
                'range': (0.0, 1.0),
                'category': 'accuracy'
            },
            'evaluate_bleu': {
                'direction': OptimizationDirection.MAXIMIZE,
                'range': (0.0, 1.0),
                'category': 'fluency'
            },
            'evaluate_rouge': {
                'direction': OptimizationDirection.MAXIMIZE,
                'range': (0.0, 1.0),
                'category': 'relevance'
            },
            'evaluate_exact_match_strict': {
                'direction': OptimizationDirection.MAXIMIZE,
                'range': (0.0, 1.0),
                'category': 'accuracy'
            },
            'evaluatesemantic_similarity': {
                'direction': OptimizationDirection.MAXIMIZE,
                'range': (0.0, 1.0),
                'category': 'relevance'
            }
        }
        
        _logger.info("DSPyMetricConverter initialized")
    
    def dspy_to_multi_obj(self, dspy_metric: Callable, objective_name: str) -> Objective:
        """Convert a DSPy metric to multi-objective format.
        
        Args:
            dspy_metric: DSPy evaluation metric function
            objective_name: Name for the resulting objective
            
        Returns:
            Converted Objective instance
        """
        # Check cache first
        if self.config.cache_conversions:
            cache_key = f"{objective_name}_{id(dspy_metric)}"
            if cache_key in self.conversion_cache:
                return self.conversion_cache[cache_key]
        
        try:
            # Analyze the DSPy metric
            metric_info = self._analyze_dspy_metric(dspy_metric)
            
            # Determine optimization direction
            direction = self._determine_direction(dspy_metric, metric_info)
            
            # Create evaluation wrapper
            eval_wrapper = self._create_evaluation_wrapper(dspy_metric, metric_info)
            
            # Create the objective
            objective = Objective(
                name=objective_name,
                weight=1.0,
                direction=direction,
                description=f"Converted from DSPy metric: {dspy_metric.__name__}"
            )
            
            # Override evaluate method
            objective.evaluate = eval_wrapper
            
            # Cache the result
            if self.config.cache_conversions:
                self.conversion_cache[cache_key] = objective
            
            _logger.info(f"Converted DSPy metric '{dspy_metric.__name__}' to objective '{objective_name}'")
            return objective
            
        except Exception as e:
            _logger.error(f"Failed to convert DSPy metric: {e}")
            # Return a default objective
            return Objective(
                name=objective_name,
                weight=1.0,
                direction=self.config.default_direction,
                description=f"Default objective (conversion failed for {dspy_metric.__name__})"
            )
    
    def aggregate_dspy_metrics(self, metrics: List[Callable], weights: List[float]) -> List[Objective]:
        """Aggregate multiple DSPy metrics into multi-objective format.
        
        Args:
            metrics: List of DSPy metrics to convert
            weights: Weights for each metric
            
        Returns:
            List of converted objectives
        """
        if len(metrics) != len(weights):
            raise ValueError("Number of metrics must match number of weights")
        
        objectives = []
        for i, (metric, weight) in enumerate(zip(metrics, weights)):
            objective_name = f"dspy_metric_{i}_{metric.__name__}"
            objective = self.dspy_to_multi_obj(metric, objective_name)
            objective.weight = weight
            objectives.append(objective)
        
        _logger.info(f"Aggregated {len(metrics)} DSPy metrics into {len(objectives)} objectives")
        return objectives
    
    def create_composite_metric(self, task_metrics: Dict[str, Callable]) -> List[Objective]:
        """Create composite objective from task-specific metrics.
        
        Args:
            task_metrics: Dictionary mapping task names to DSPy metrics
            
        Returns:
            List of objectives for the composite metric
        """
        objectives = []
        
        for task_name, metric in task_metrics.items():
            objective_name = f"{task_name}_objective"
            objective = self.dspy_to_multi_obj(metric, objective_name)
            objectives.append(objective)
        
        # Create a composite objective that combines all task metrics
        if len(objectives) > 1:
            composite_name = f"composite_{'_'.join(task_metrics.keys())}"
            composite_objective = self._create_composite_objective(objectives, composite_name)
            objectives.append(composite_objective)
        
        _logger.info(f"Created {len(objectives)} objectives from {len(task_metrics)} task metrics")
        return objectives
    
    def _analyze_dspy_metric(self, metric: Callable) -> Dict[str, Any]:
        """Analyze a DSPy metric to determine its properties."""
        metric_info = {
            'name': getattr(metric, '__name__', 'unknown'),
            'docstring': getattr(metric, '__doc__', ''),
            'signature': inspect.signature(metric),
            'known_metric': metric.__name__ in self.known_metrics
        }
        
        # Extract information from known metrics
        if metric_info['known_metric']:
            metric_info.update(self.known_metrics[metric.__name__])
        
        # Analyze docstring for hints
        if metric_info['docstring']:
            metric_info.update(self._analyze_docstring(metric_info['docstring']))
        
        return metric_info
    
    def _determine_direction(self, metric: Callable, metric_info: Dict[str, Any]) -> OptimizationDirection:
        """Determine the optimization direction for a metric."""
        if not self.config.infer_direction:
            return self.config.default_direction
        
        # Use known metric information
        if metric_info['known_metric']:
            return metric_info['direction']
        
        # Analyze name for hints
        name_lower = metric_info['name'].lower()
        if any(keyword in name_lower for keyword in ['loss', 'error', 'distance', 'cost']):
            return OptimizationDirection.MINIMIZE
        elif any(keyword in name_lower for keyword in ['accuracy', 'score', 'f1', 'precision', 'recall']):
            return OptimizationDirection.MAXIMIZE
        
        # Check docstring
        if metric_info['docstring']:
            doc_lower = metric_info['docstring'].lower()
            if any(keyword in doc_lower for keyword in ['lower is better', 'minimize', 'reduce']):
                return OptimizationDirection.MINIMIZE
            elif any(keyword in doc_lower for keyword in ['higher is better', 'maximize', 'increase']):
                return OptimizationDirection.MAXIMIZE
        
        # Default to configured direction
        return self.config.default_direction
    
    def _create_evaluation_wrapper(self, metric: Callable, metric_info: Dict[str, Any]) -> Callable:
        """Create an evaluation wrapper for the DSPy metric."""
        def evaluate(program: Any, dataset: List[Any], **kwargs) -> float:
            """Evaluate a program using the DSPy metric."""
            try:
                # Prepare evaluation arguments
                eval_kwargs = self._prepare_evaluation_args(metric, dataset, **kwargs)
                
                # Call the DSPy metric
                if hasattr(metric, '__call__'):
                    result = metric(program, dataset, **eval_kwargs)
                else:
                    result = metric(program, dataset)
                
                # Normalize result if needed
                if self.config.normalize_scores:
                    result = self._normalize_score(result, metric_info)
                
                # Validate result
                if self.config.validate_metrics:
                    result = self._validate_score(result)
                
                return float(result)
                
            except Exception as e:
                _logger.warning(f"Evaluation failed for metric {metric.__name__}: {e}")
                return 0.0
        
        return evaluate
    
    def _prepare_evaluation_args(self, metric: Callable, dataset: List[Any], **kwargs) -> Dict[str, Any]:
        """Prepare arguments for metric evaluation."""
        # Extract relevant arguments from kwargs based on metric signature
        sig = inspect.signature(metric)
        valid_params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                valid_params[param_name] = kwargs[param_name]
        
        return valid_params
    
    def _normalize_score(self, score: Any, metric_info: Dict[str, Any]) -> float:
        """Normalize a score to [0, 1] range."""
        try:
            score_float = float(score)
            
            # Use known metric range if available
            if 'range' in metric_info:
                min_val, max_val = metric_info['range']
                if max_val > min_val:
                    normalized = (score_float - min_val) / (max_val - min_val)
                    return max(0.0, min(1.0, normalized))
            
            # Default normalization strategies
            if 0.0 <= score_float <= 1.0:
                return score_float  # Already normalized
            elif score_float > 0:
                return min(1.0, score_float / 100.0)  # Assume percentage
            else:
                return 0.0
                
        except (ValueError, TypeError):
            return 0.0
    
    def _validate_score(self, score: float) -> float:
        """Validate and correct a score."""
        if not isinstance(score, (int, float)):
            return 0.0
        
        if score < 0:
            return 0.0
        elif score > 1 and self.config.normalize_scores:
            return min(1.0, score)
        
        return float(score)
    
    def _analyze_docstring(self, docstring: str) -> Dict[str, Any]:
        """Analyze docstring for metric properties."""
        doc_lower = docstring.lower()
        
        analysis = {}
        
        # Look for range information
        if 'range: 0-1' in doc_lower or 'normalized' in doc_lower:
            analysis['range'] = (0.0, 1.0)
        elif 'percentage' in doc_lower:
            analysis['range'] = (0.0, 100.0)
        
        # Look for direction hints
        if 'higher is better' in doc_lower or 'maximize' in doc_lower:
            analysis['direction'] = OptimizationDirection.MAXIMIZE
        elif 'lower is better' in doc_lower or 'minimize' in doc_lower:
            analysis['direction'] = OptimizationDirection.MINIMIZE
        
        return analysis
    
    def _create_composite_objective(self, objectives: List[Objective], name: str) -> Objective:
        """Create a composite objective from multiple objectives."""
        def evaluate_composite(program: Any, dataset: List[Any], **kwargs) -> float:
            """Evaluate composite objective as weighted average."""
            total_score = 0.0
            total_weight = 0.0
            
            for objective in objectives:
                try:
                    score = objective.evaluate(program, dataset, **kwargs)
                    weight = objective.weight
                    total_score += score * weight
                    total_weight += weight
                except Exception as e:
                    _logger.warning(f"Composite evaluation failed for {objective.name}: {e}")
                    continue
            
            return total_score / total_weight if total_weight > 0 else 0.0
        
        composite = Objective(
            name=name,
            weight=1.0,
            direction=OptimizationDirection.MAXIMIZE,
            description=f"Composite objective combining {len(objectives)} metrics"
        )
        composite.evaluate = evaluate_composite
        
        return composite
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get statistics about metric conversions."""
        return {
            "cache_size": len(self.conversion_cache),
            "known_metrics_count": len(self.known_metrics),
            "config": {
                "normalize_scores": self.config.normalize_scores,
                "infer_direction": self.config.infer_direction,
                "cache_conversions": self.config.cache_conversions,
                "validate_metrics": self.config.validate_metrics
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the conversion cache."""
        self.conversion_cache.clear()
        _logger.info("Conversion cache cleared")
    
    def register_known_metric(self, name: str, properties: Dict[str, Any]) -> None:
        """Register a new known metric with its properties.
        
        Args:
            name: Metric name
            properties: Metric properties including direction, range, category
        """
        if 'direction' not in properties:
            properties['direction'] = self.config.default_direction
        
        self.known_metrics[name] = properties
        _logger.info(f"Registered known metric: {name}")
    
    def __repr__(self) -> str:
        """String representation of the converter."""
        return (
            f"DSPyMetricConverter("
            f"known_metrics={len(self.known_metrics)}, "
            f"cache_size={len(self.conversion_cache)})"
        )