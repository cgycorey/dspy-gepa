"""Common objective implementations for multi-objective GEPA optimization.

This module provides concrete objective implementations that can be used
out of the box for common optimization scenarios.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass

from .interfaces import Objective, OptimizationDirection
from ..utils.logging import get_logger


_logger = get_logger(__name__)


class AccuracyMetric(Objective):
    """Objective that measures task accuracy."""
    
    def __init__(
        self,
        weight: float = 1.0,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        accuracy_fn: Optional[Callable[[Any, List[Any]], float]] = None
    ):
        """Initialize accuracy metric.
        
        Args:
            weight: Relative importance weight
            direction: Optimization direction
            accuracy_fn: Custom accuracy function (program, dataset) -> score
        """
        super().__init__("accuracy", weight, direction, "Task accuracy measurement")
        self.accuracy_fn = accuracy_fn or self._default_accuracy_fn
    
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate accuracy of the program."""
        try:
            return self.accuracy_fn(program, dataset)
        except Exception as e:
            _logger.warning(f"Accuracy evaluation failed: {e}")
            return 0.0
    
    def _default_accuracy_fn(self, program: Any, dataset: List[Any]) -> float:
        """Default accuracy implementation."""
        if not dataset:
            return 0.0
        
        correct = 0
        total = 0
        
        for example in dataset:
            try:
                # Simple accuracy check for string outputs
                expected = getattr(example, 'expected', getattr(example, 'answer', ''))
                
                if hasattr(program, '__call__'):
                    actual = program(**example.inputs)
                elif isinstance(program, str):
                    # For prompt strings, this is a simplified evaluation
                    actual = "mock_output"  # Would need LLM call in practice
                else:
                    actual = str(program)
                
                if isinstance(actual, dict):
                    actual = actual.get('answer', actual.get('output', str(actual)))
                elif hasattr(actual, 'answer'):
                    actual = actual.answer
                else:
                    actual = str(actual)
                
                if str(actual).lower().strip() == str(expected).lower().strip():
                    correct += 1
                
                total += 1
                
            except Exception:
                total += 1
        
        return correct / total if total > 0 else 0.0


class FluencyMetric(Objective):
    """Objective that measures output fluency and language quality."""
    
    def __init__(
        self,
        weight: float = 1.0,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        language_model: Optional[Any] = None
    ):
        """Initialize fluency metric.
        
        Args:
            weight: Relative importance weight
            direction: Optimization direction
            language_model: Optional LM for fluency scoring
        """
        super().__init__("fluency", weight, direction, "Language fluency and quality")
        self.language_model = language_model
    
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate fluency of program outputs."""
        if not dataset:
            return 0.0
        
        fluency_scores = []
        
        for example in dataset:
            try:
                # Generate output
                if hasattr(program, '__call__') and callable(program):
                    try:
                        output = program(**example.inputs)
                    except:
                        output = str(program)
                else:
                    output = str(program)
                
                if isinstance(output, dict):
                    text = output.get('answer', output.get('output', str(output)))
                elif hasattr(output, 'answer'):
                    text = output.answer
                else:
                    text = str(output)
                
                # Calculate fluency score
                score = self._calculate_fluency(text)
                fluency_scores.append(score)
                
            except Exception as e:
                _logger.warning(f"Fluency evaluation failed for example: {e}")
                fluency_scores.append(0.5)  # Neutral score
        
        return sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.0
    
    def _calculate_fluency(self, text: str) -> float:
        """Calculate fluency score for text."""
        if not text or not isinstance(text, str):
            return 0.0
        
        score = 0.5  # Base score
        
        # Length penalty/reward
        length = len(text.split())
        if 5 <= length <= 50:  # Good length range
            score += 0.2
        elif length < 5:
            score -= 0.2
        elif length > 100:
            score -= 0.1
        
        # Punctuation and capitalization
        if text[0].isupper() if text else False:
            score += 0.1
        
        if text.strip().endswith(('.', '!', '?')):
            score += 0.1
        
        # Repetition penalty
        words = text.lower().split()
        if len(set(words)) / len(words) < 0.7:  # High repetition
            score -= 0.2
        
        # Grammar heuristics
        grammar_issues = 0
        if '  ' in text:  # Double spaces
            grammar_issues += 1
        if text.count(',') > text.count('.') + 2:  # Too many commas
            grammar_issues += 1
        
        score -= grammar_issues * 0.1
        
        return max(0.0, min(1.0, score))


class RelevanceMetric(Objective):
    """Objective that measures output relevance to input and context."""
    
    def __init__(
        self,
        weight: float = 1.0,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        relevance_fn: Optional[Callable[[str, str], float]] = None
    ):
        """Initialize relevance metric.
        
        Args:
            weight: Relative importance weight
            direction: Optimization direction
            relevance_fn: Custom relevance function
        """
        super().__init__("relevance", weight, direction, "Output relevance to input")
        self.relevance_fn = relevance_fn or self._default_relevance_fn
    
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate relevance of program outputs."""
        if not dataset:
            return 0.0
        
        relevance_scores = []
        
        for example in dataset:
            try:
                # Get input text
                input_text = self._extract_input_text(example)
                
                # Generate output
                if hasattr(program, '__call__') and callable(program):
                    try:
                        output = program(**example.inputs)
                    except:
                        output = str(program)
                else:
                    output = str(program)
                
                if isinstance(output, dict):
                    output_text = output.get('answer', output.get('output', str(output)))
                elif hasattr(output, 'answer'):
                    output_text = output.answer
                else:
                    output_text = str(output)
                
                # Calculate relevance
                relevance = self.relevance_fn(input_text, output_text)
                relevance_scores.append(relevance)
                
            except Exception as e:
                _logger.warning(f"Relevance evaluation failed for example: {e}")
                relevance_scores.append(0.5)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def _extract_input_text(self, example: Any) -> str:
        """Extract input text from example."""
        if hasattr(example, 'inputs'):
            inputs = example.inputs
            if isinstance(inputs, dict):
                return ' '.join(str(v) for v in inputs.values())
            elif isinstance(inputs, (list, tuple)):
                return ' '.join(str(v) for v in inputs)
            else:
                return str(inputs)
        
        return str(example)
    
    def _default_relevance_fn(self, input_text: str, output_text: str) -> float:
        """Default relevance calculation based on keyword overlap."""
        if not input_text or not output_text:
            return 0.0
        
        input_words = set(re.findall(r'\b\w+\b', input_text.lower()))
        output_words = set(re.findall(r'\b\w+\b', output_text.lower()))
        
        if not input_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(input_words.intersection(output_words))
        relevance = overlap / len(input_words)
        
        # Additional factors
        if len(output_words) > 0:
            output_coverage = overlap / len(output_words)
            relevance = (relevance + output_coverage) / 2
        
        return min(1.0, relevance)


class TokenUsageMetric(Objective):
    """Objective that measures token usage (to be minimized)."""
    
    def __init__(
        self,
        weight: float = 1.0,
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
        cost_per_token: float = 0.0001,
        tokenizer: Optional[Any] = None
    ):
        """Initialize token usage metric.
        
        Args:
            weight: Relative importance weight  
            direction: Optimization direction (should be MINIMIZE)
            cost_per_token: Cost per token for cost calculation
            tokenizer: Optional tokenizer for accurate token counting
        """
        super().__init__("token_usage", weight, direction, "Token usage efficiency")
        self.cost_per_token = cost_per_token
        self.tokenizer = tokenizer
    
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate token usage."""
        if not dataset:
            return 0.0
        
        total_tokens = 0
        
        for example in dataset:
            try:
                # Estimate tokens for input
                input_text = self._extract_input_text(example)
                input_tokens = self._count_tokens(input_text)
                
                # Estimate tokens for output (simplified)
                if hasattr(program, '__call__') and callable(program):
                    try:
                        output = program(**example.inputs)
                    except:
                        output = str(program)
                else:
                    output = str(program)
                
                if isinstance(output, dict):
                    output_text = output.get('answer', output.get('output', str(output)))
                elif hasattr(output, 'answer'):
                    output_text = output.answer
                else:
                    output_text = str(output)
                
                output_tokens = self._count_tokens(output_text)
                
                total_tokens += input_tokens + output_tokens
                
            except Exception as e:
                _logger.warning(f"Token counting failed for example: {e}")
                total_tokens += 100  # Default estimate
        
        # Normalize by dataset size
        avg_tokens = total_tokens / len(dataset) if dataset else 0
        
        # Convert to cost if needed
        cost = avg_tokens * self.cost_per_token
        
        return cost
    
    def _extract_input_text(self, example: Any) -> str:
        """Extract input text from example."""
        if hasattr(example, 'inputs'):
            inputs = example.inputs
            if isinstance(inputs, dict):
                return ' '.join(str(v) for v in inputs.values())
            elif isinstance(inputs, (list, tuple)):
                return ' '.join(str(v) for v in inputs)
            else:
                return str(inputs)
        
        return str(example)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: approximate tokens (4 chars per token on average)
        return max(1, len(text) // 4)


class ExecutionTimeMetric(Objective):
    """Objective that measures execution time (to be minimized)."""
    
    def __init__(
        self,
        weight: float = 1.0,
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
        time_budget: float = 10.0  # seconds
    ):
        """Initialize execution time metric.
        
        Args:
            weight: Relative importance weight
            direction: Optimization direction (should be MINIMIZE)
            time_budget: Time budget for normalization
        """
        super().__init__("execution_time", weight, direction, "Execution time efficiency")
        self.time_budget = time_budget
    
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate execution time."""
        if not dataset:
            return 0.0
        
        total_time = 0.0
        
        for example in dataset:
            try:
                start_time = time.time()
                
                if hasattr(program, '__call__'):
                    result = program(**example.inputs)
                else:
                    # For non-callable programs, simulate minimal time
                    time.sleep(0.001)  # 1ms minimum
                    result = str(program)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
            except Exception as e:
                _logger.warning(f"Execution time measurement failed for example: {e}")
                total_time += 1.0  # Penalty time
        
        # Average time per example
        avg_time = total_time / len(dataset) if dataset else 0
        
        # Normalize by time budget
        normalized_time = avg_time / self.time_budget
        
        return normalized_time


class CompositeMetric(Objective):
    """Objective that combines multiple metrics."""
    
    def __init__(
        self,
        name: str,
        metrics: List[Objective],
        weights: Optional[List[float]] = None,
        aggregation_fn: str = "weighted_average"  # "weighted_average", "min", "max"
    ):
        """Initialize composite metric.
        
        Args:
            name: Name of the composite metric
            metrics: List of metrics to combine
            weights: Weights for each metric
            aggregation_fn: How to aggregate metrics
        """
        if not metrics:
            raise ValueError("At least one metric must be provided")
        
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        self.aggregation_fn = aggregation_fn
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        
        # Determine direction based on metrics
        max_count = sum(1 for m in metrics if m.direction == OptimizationDirection.MAXIMIZE)
        direction = OptimizationDirection.MAXIMIZE if max_count > len(metrics) / 2 else OptimizationDirection.MINIMIZE
        
        super().__init__(name, 1.0, direction, f"Composite metric combining {len(metrics)} metrics")
    
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate composite metric."""
        scores = []
        
        for metric in self.metrics:
            try:
                score = metric.evaluate(program, dataset, **kwargs)
                scores.append(score)
            except Exception as e:
                _logger.warning(f"Metric {metric.name} evaluation failed: {e}")
                scores.append(0.0)
        
        if not scores:
            return 0.0
        
        # Aggregate scores
        if self.aggregation_fn == "weighted_average":
            return sum(s * w for s, w in zip(scores, self.weights))
        elif self.aggregation_fn == "min":
            return min(scores)
        elif self.aggregation_fn == "max":
            return max(scores)
        else:
            return sum(scores) / len(scores)
    
    def add_metric(self, metric: Objective, weight: float = 1.0) -> None:
        """Add a new metric to the composite."""
        self.metrics.append(metric)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]


# Common objective combinations
def create_default_task_objectives(
    task_type: str = "general",
    weights: Optional[Dict[str, float]] = None
) -> List[Objective]:
    """Create a default set of objectives for a task type.
    
    Args:
        task_type: Type of task ("translation", "generation", "classification", etc.)
        weights: Custom weights for objectives
        
    Returns:
        List of objectives
    """
    default_weights = {
        "accuracy": 0.4,
        "fluency": 0.2,
        "relevance": 0.2,
        "token_usage": 0.1,
        "execution_time": 0.1
    }
    
    if weights:
        default_weights.update(weights)
    
    objectives = [
        AccuracyMetric(weight=default_weights["accuracy"]),
        FluencyMetric(weight=default_weights["fluency"]),
        RelevanceMetric(weight=default_weights["relevance"]),
        TokenUsageMetric(weight=default_weights["token_usage"]),
        ExecutionTimeMetric(weight=default_weights["execution_time"])
    ]
    
    # Task-specific adjustments
    if task_type == "translation":
        objectives[0].weight = 0.5  # Higher weight for accuracy
        objectives[1].weight = 0.3  # Fluency more important for translation
        objectives[2].weight = 0.1
        objectives[3].weight = 0.05
        objectives[4].weight = 0.05
    
    elif task_type == "code_generation":
        objectives[0].weight = 0.6  # Accuracy very important for code
        objectives[1].weight = 0.1
        objectives[2].weight = 0.1
        objectives[3].weight = 0.1
        objectives[4].weight = 0.1
    
    elif task_type == "summarization":
        objectives[0].weight = 0.3
        objectives[1].weight = 0.3  # Fluency important for summaries
        objectives[2].weight = 0.3  # Relevance very important
        objectives[3].weight = 0.05
        objectives[4].weight = 0.05
    
    return objectives


def create_efficiency_focused_objectives() -> List[Objective]:
    """Create objectives focused on efficiency."""
    return [
        AccuracyMetric(weight=0.3),
        FluencyMetric(weight=0.2),
        TokenUsageMetric(weight=0.3, direction=OptimizationDirection.MINIMIZE),
        ExecutionTimeMetric(weight=0.2, direction=OptimizationDirection.MINIMIZE)
    ]


def create_quality_focused_objectives() -> List[Objective]:
    """Create objectives focused on output quality."""
    return [
        AccuracyMetric(weight=0.4),
        FluencyMetric(weight=0.3),
        RelevanceMetric(weight=0.3)
    ]


# Additional classes for architecture compliance

@dataclass
class TaskMetrics:
    """Container for task-related metrics."""
    accuracy_metric: Optional[AccuracyMetric] = None
    fluency_metric: Optional[FluencyMetric] = None
    relevance_metric: Optional[RelevanceMetric] = None
    
    def __post_init__(self):
        """Initialize default metrics if not provided."""
        if self.accuracy_metric is None:
            self.accuracy_metric = AccuracyMetric()
        if self.fluency_metric is None:
            self.fluency_metric = FluencyMetric()
        if self.relevance_metric is None:
            self.relevance_metric = RelevanceMetric()
    
    def get_all_objectives(self) -> List[Objective]:
        """Get all task objectives."""
        objectives = []
        if self.accuracy_metric:
            objectives.append(self.accuracy_metric)
        if self.fluency_metric:
            objectives.append(self.fluency_metric)
        if self.relevance_metric:
            objectives.append(self.relevance_metric)
        return objectives
    
    def evaluate_task(self, program: Any, dataset: List[Any]) -> Dict[str, float]:
        """Evaluate all task metrics."""
        results = {}
        
        if self.accuracy_metric:
            try:
                results["accuracy"] = self.accuracy_metric.evaluate(program, dataset)
            except Exception as e:
                _logger.warning(f"Accuracy evaluation failed: {e}")
                results["accuracy"] = 0.0
        
        if self.fluency_metric:
            try:
                results["fluency"] = self.fluency_metric.evaluate(program, dataset)
            except Exception as e:
                _logger.warning(f"Fluency evaluation failed: {e}")
                results["fluency"] = 0.0
        
        if self.relevance_metric:
            try:
                results["relevance"] = self.relevance_metric.evaluate(program, dataset)
            except Exception as e:
                _logger.warning(f"Relevance evaluation failed: {e}")
                results["relevance"] = 0.0
        
        return results


@dataclass
class ResourceMetrics:
    """Container for resource-related metrics."""
    token_usage_metric: Optional[TokenUsageMetric] = None
    execution_time_metric: Optional[ExecutionTimeMetric] = None
    
    def __post_init__(self):
        """Initialize default metrics if not provided."""
        if self.token_usage_metric is None:
            self.token_usage_metric = TokenUsageMetric()
        if self.execution_time_metric is None:
            self.execution_time_metric = ExecutionTimeMetric()
    
    def get_all_objectives(self) -> List[Objective]:
        """Get all resource objectives."""
        objectives = []
        if self.token_usage_metric:
            objectives.append(self.token_usage_metric)
        if self.execution_time_metric:
            objectives.append(self.execution_time_metric)
        return objectives
    
    def evaluate_resources(self, program: Any, dataset: List[Any]) -> Dict[str, float]:
        """Evaluate all resource metrics."""
        results = {}
        
        if self.token_usage_metric:
            try:
                results["token_usage"] = self.token_usage_metric.evaluate(program, dataset)
            except Exception as e:
                _logger.warning(f"Token usage evaluation failed: {e}")
                results["token_usage"] = float('inf')
        
        if self.execution_time_metric:
            try:
                results["execution_time"] = self.execution_time_metric.evaluate(program, dataset)
            except Exception as e:
                _logger.warning(f"Execution time evaluation failed: {e}")
                results["execution_time"] = float('inf')
        
        return results