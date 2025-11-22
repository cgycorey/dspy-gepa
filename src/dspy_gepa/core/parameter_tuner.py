"""Dynamic parameter tuner for multi-objective GEPA optimization.

This module provides intelligent parameter adjustment strategies that
adapt optimization parameters based on convergence analysis, performance
feedback, and resource utilization patterns.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import time
import math
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import random

from .interfaces import (
    ParameterTuner, ConvergenceDetector, OptimizationDirection,
    EvaluationResult
)
from ..utils.logging import get_logger


_logger = get_logger(__name__)


@dataclass
class ParameterBounds:
    """Bounds for parameter values."""
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    
    def __post_init__(self):
        """Validate bounds."""
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value ({self.min_value}) must be less than max_value ({self.max_value})")
        
        if self.step_size is None:
            self.step_size = (self.max_value - self.min_value) / 20.0


@dataclass
class ParameterDefinition:
    """Definition of a tunable parameter."""
    name: str
    default_value: float
    bounds: ParameterBounds
    description: str
    adjustment_strategy: str = "adaptive"  # "adaptive", "linear", "exponential", "oscillating"
    importance_weight: float = 1.0
    
    def clamp_value(self, value: float) -> float:
        """Clamp value to bounds."""
        return max(self.bounds.min_value, min(self.bounds.max_value, value))


@dataclass
class TuningHistory:
    """History of parameter tuning operations."""
    generation: int
    old_parameters: Dict[str, float]
    new_parameters: Dict[str, float]
    performance_change: float
    convergence_indicators: Dict[str, float]
    tuning_reason: str
    timestamp: float = field(default_factory=time.time)


class ConvergenceBasedTuner(ParameterTuner):
    """Parameter tuner based on convergence analysis."""
    
    def __init__(
        self,
        parameter_definitions: Optional[List[ParameterDefinition]] = None,
        tuning_frequency: int = 5,  # Tune every N generations
        performance_window: int = 10,  # Window for performance analysis
        min_improvement_threshold: float = 0.01,  # Minimum improvement to consider successful
        max_adjustment_factor: float = 0.3,  # Maximum parameter adjustment per tuning
        verbose: bool = True
    ):
        """Initialize convergence-based parameter tuner.
        
        Args:
            parameter_definitions: List of tunable parameters
            tuning_frequency: How often to tune parameters (in generations)
            performance_window: Window size for performance analysis
            min_improvement_threshold: Minimum improvement for successful tuning
            max_adjustment_factor: Maximum factor for parameter adjustments
            verbose: Whether to log tuning decisions
        """
        self.tuning_frequency = tuning_frequency
        self.performance_window = performance_window
        self.min_improvement_threshold = min_improvement_threshold
        self.max_adjustment_factor = max_adjustment_factor
        self.verbose = verbose
        
        # Default parameter definitions
        self.parameter_definitions = parameter_definitions or self._get_default_parameters()
        
        # Create parameter lookup
        self.param_map = {param.name: param for param in self.parameter_definitions}
        
        # Tuning history and performance tracking
        self.tuning_history: List[TuningHistory] = []
        self.performance_history: deque = deque(maxlen=performance_window * 2)
        self.convergence_history: deque = deque(maxlen=performance_window * 2)
        self.last_tuning_generation = -1
        
        # Statistics
        self.successful_tunings = 0
        self.failed_tunings = 0
    
    def _get_default_parameters(self) -> List[ParameterDefinition]:
        """Get default parameter definitions for GEPA optimization."""
        return [
            ParameterDefinition(
                name="population_size",
                default_value=20.0,
                bounds=ParameterBounds(5.0, 100.0, 5.0),
                description="Population size for genetic optimization",
                adjustment_strategy="adaptive",
                importance_weight=1.0
            ),
            ParameterDefinition(
                name="mutation_rate",
                default_value=0.1,
                bounds=ParameterBounds(0.01, 0.5, 0.01),
                description="Mutation rate for genetic operations",
                adjustment_strategy="oscillating",
                importance_weight=1.2
            ),
            ParameterDefinition(
                name="crossover_rate",
                default_value=0.7,
                bounds=ParameterBounds(0.3, 0.95, 0.05),
                description="Crossover rate for genetic operations",
                adjustment_strategy="adaptive",
                importance_weight=0.8
            ),
            ParameterDefinition(
                name="elitism_rate",
                default_value=0.2,
                bounds=ParameterBounds(0.05, 0.5, 0.05),
                description="Rate of elite solution preservation",
                adjustment_strategy="adaptive",
                importance_weight=0.6
            ),
            ParameterDefinition(
                name="selection_pressure",
                default_value=2.0,
                bounds=ParameterBounds(1.0, 5.0, 0.1),
                description="Selection pressure coefficient",
                adjustment_strategy="linear",
                importance_weight=0.9
            )
        ]
    
    def should_adjust_parameters(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if parameters should be adjusted."""
        generation = optimization_state.get("generation", 0)
        
        # Check if it's time to tune
        if generation - self.last_tuning_generation < self.tuning_frequency:
            return False
        
        # Check if we have enough performance data
        if len(self.performance_history) < self.performance_window:
            return False
        
        # Check if convergence indicators suggest tuning is needed
        convergence_indicators = self._extract_convergence_indicators(optimization_state)
        
        # Tune if converging too quickly or not converging at all
        if convergence_indicators.get("convergence_rate", 0.0) > 0.8:
            if self.verbose:
                _logger.info("High convergence rate detected - triggering parameter tuning")
            return True
        
        if convergence_indicators.get("improvement_rate", 0.0) < self.min_improvement_threshold:
            if self.verbose:
                _logger.info("Low improvement rate detected - triggering parameter tuning")
            return True
        
        # Tune if diversity is too low
        if convergence_indicators.get("diversity_score", 1.0) < 0.3:
            if self.verbose:
                _logger.info("Low diversity detected - triggering parameter tuning")
            return True
        
        return False
    
    def adjust_parameters(self, current_params: Dict[str, Any], optimization_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust optimization parameters based on current state."""
        generation = optimization_state.get("generation", 0)
        
        # Extract performance and convergence metrics
        performance_metrics = self._extract_performance_metrics(optimization_state)
        convergence_indicators = self._extract_convergence_indicators(optimization_state)
        
        # Store current state in history
        self.performance_history.append(performance_metrics)
        self.convergence_history.append(convergence_indicators)
        
        # Calculate parameter adjustments
        new_params = current_params.copy()
        adjustments_made = {}
        
        for param_name, param_def in self.param_map.items():
            if param_name in current_params:
                current_value = float(current_params[param_name])
                new_value = self._calculate_parameter_adjustment(
                    param_def, current_value, performance_metrics, convergence_indicators
                )
                
                if abs(new_value - current_value) > 1e-6:
                    new_value = param_def.clamp_value(new_value)
                    new_params[param_name] = new_value
                    adjustments_made[param_name] = {
                        "old": current_value,
                        "new": new_value,
                        "change": (new_value - current_value) / current_value * 100
                    }
        
        # Record tuning operation
        if adjustments_made:
            performance_change = self._estimate_performance_change(
                current_params, new_params, performance_metrics
            )
            
            tuning_record = TuningHistory(
                generation=generation,
                old_parameters=current_params,
                new_parameters=new_params,
                performance_change=performance_change,
                convergence_indicators=convergence_indicators,
                tuning_reason=self._determine_tuning_reason(convergence_indicators)
            )
            
            self.tuning_history.append(tuning_record)
            self.last_tuning_generation = generation
            
            if self.verbose:
                _logger.info(f"Parameters tuned at generation {generation}: {adjustments_made}")
        
        return new_params
    
    def _extract_performance_metrics(self, optimization_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from optimization state."""
        metrics = {}
        
        # Extract best scores for each objective
        if "metrics" in optimization_state:
            opt_metrics = optimization_state["metrics"]
            for key, value in opt_metrics.items():
                if key.startswith("best_"):
                    metrics[key] = float(value)
        
        # Calculate composite performance score
        if metrics:
            metrics["composite_score"] = sum(metrics.values()) / len(metrics)
        else:
            metrics["composite_score"] = 0.0
        
        # Extract hypervolume if available
        metrics["hypervolume"] = float(optimization_state.get("metrics", {}).get("hypervolume", 0.0))
        
        # Extract frontier size
        metrics["frontier_size"] = float(optimization_state.get("frontier_size", 0))
        
        return metrics
    
    def _extract_convergence_indicators(self, optimization_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract convergence indicators from optimization state."""
        indicators = {}
        
        # Calculate convergence rate based on recent performance
        if len(self.performance_history) >= 3:
            recent_scores = [h.get("composite_score", 0.0) for h in list(self.performance_history)[-3:]]
            if len(recent_scores) >= 2:
                improvement_rate = abs(recent_scores[-1] - recent_scores[-2]) / max(recent_scores[-2], 1e-6)
                indicators["improvement_rate"] = improvement_rate
                
                # Calculate convergence rate (inverse of improvement)
                convergence_rate = 1.0 - min(1.0, improvement_rate * 10)
                indicators["convergence_rate"] = convergence_rate
        
        # Extract diversity metrics
        if "metrics" in optimization_state:
            opt_metrics = optimization_state["metrics"]
            indicators["diversity_score"] = opt_metrics.get("objective_variance", 0.5)
            indicators["pairwise_distance"] = opt_metrics.get("pairwise_distance", 0.5)
        
        # Calculate stagnation indicator
        if len(self.performance_history) >= self.performance_window:
            recent_scores = [h.get("composite_score", 0.0) for h in list(self.performance_history)[-self.performance_window:]]
            if recent_scores:
                score_variance = sum((s - sum(recent_scores)/len(recent_scores))**2 for s in recent_scores) / len(recent_scores)
                stagnation = 1.0 - min(1.0, score_variance * 10)
                indicators["stagnation"] = stagnation
        
        # Generation-based convergence
        generation = optimization_state.get("generation", 0)
        max_generations = optimization_state.get("max_generations", 100)
        generation_progress = generation / max(max_generations, 1)
        indicators["generation_progress"] = generation_progress
        
        return indicators
    
    def _calculate_parameter_adjustment(
        self,
        param_def: ParameterDefinition,
        current_value: float,
        performance_metrics: Dict[str, float],
        convergence_indicators: Dict[str, float]
    ) -> float:
        """Calculate the adjusted value for a parameter."""
        strategy = param_def.adjustment_strategy
        adjustment_factor = 0.0
        
        if strategy == "adaptive":
            adjustment_factor = self._adaptive_adjustment(
                param_def, current_value, performance_metrics, convergence_indicators
            )
        elif strategy == "linear":
            adjustment_factor = self._linear_adjustment(
                param_def, convergence_indicators
            )
        elif strategy == "exponential":
            adjustment_factor = self._exponential_adjustment(
                param_def, convergence_indicators
            )
        elif strategy == "oscillating":
            adjustment_factor = self._oscillating_adjustment(
                param_def, current_value, convergence_indicators
            )
        
        # Apply adjustment with bounds
        max_change = current_value * self.max_adjustment_factor
        actual_change = max(-max_change, min(max_change, adjustment_factor))
        
        return current_value + actual_change
    
    def _adaptive_adjustment(
        self,
        param_def: ParameterDefinition,
        current_value: float,
        performance_metrics: Dict[str, float],
        convergence_indicators: Dict[str, float]
    ) -> float:
        """Calculate adaptive parameter adjustment."""
        adjustment = 0.0
        
        # Adjust based on convergence rate
        convergence_rate = convergence_indicators.get("convergence_rate", 0.0)
        if convergence_rate > 0.7:  # Converging too fast
            if param_def.name == "mutation_rate":
                adjustment += current_value * 0.2  # Increase mutation to add diversity
            elif param_def.name == "population_size":
                adjustment += (param_def.bounds.step_size or 1.0) * 2  # Increase population
            elif param_def.name == "selection_pressure":
                adjustment -= current_value * 0.1  # Reduce selection pressure
        
        # Adjust based on improvement rate
        improvement_rate = convergence_indicators.get("improvement_rate", 0.0)
        if improvement_rate < self.min_improvement_threshold:  # Not improving enough
            if param_def.name == "mutation_rate":
                adjustment += current_value * 0.3  # Increase mutation significantly
            elif param_def.name == "crossover_rate":
                adjustment -= current_value * 0.2  # Reduce crossover to preserve good solutions
        
        # Adjust based on diversity
        diversity_score = convergence_indicators.get("diversity_score", 0.5)
        if diversity_score < 0.3:  # Low diversity
            if param_def.name == "mutation_rate":
                adjustment += current_value * 0.4  # Increase mutation
            elif param_def.name == "population_size":
                adjustment += (param_def.bounds.step_size or 1.0) * 3  # Increase population
        
        # Adjust based on hypervolume progress
        hypervolume = performance_metrics.get("hypervolume", 0.0)
        if hypervolume > 0:  # Making progress in multi-objective space
            if param_def.name == "elitism_rate":
                adjustment += current_value * 0.1  # Increase elitism to preserve good solutions
        
        return adjustment * param_def.importance_weight
    
    def _linear_adjustment(
        self,
        param_def: ParameterDefinition,
        convergence_indicators: Dict[str, float]
    ) -> float:
        """Calculate linear parameter adjustment based on generation progress."""
        generation_progress = convergence_indicators.get("generation_progress", 0.0)
        
        # Linear adjustment based on generation progress
        if param_def.name in ["mutation_rate"]:
            # Decrease mutation rate over time
            return -(param_def.bounds.step_size or 0.01) * generation_progress
        elif param_def.name in ["elitism_rate", "selection_pressure"]:
            # Increase elitism and selection pressure over time
            return (param_def.bounds.step_size or 0.01) * generation_progress * 0.5
        else:
            return 0.0
    
    def _exponential_adjustment(
        self,
        param_def: ParameterDefinition,
        convergence_indicators: Dict[str, float]
    ) -> float:
        """Calculate exponential parameter adjustment."""
        convergence_rate = convergence_indicators.get("convergence_rate", 0.0)
        
        # Exponential adjustment based on convergence
        if convergence_rate > 0.5:
            if param_def.name == "mutation_rate":
                # Exponential increase in mutation when converging
                factor = math.exp(convergence_rate) - 1.0
                base_value = param_def.default_value
                return base_value * factor * 0.5
        
        return 0.0
    
    def _oscillating_adjustment(
        self,
        param_def: ParameterDefinition,
        current_value: float,
        convergence_indicators: Dict[str, float]
    ) -> float:
        """Calculate oscillating parameter adjustment."""
        generation_progress = convergence_indicators.get("generation_progress", 0.0)
        
        # Create oscillation based on generation progress
        oscillation = math.sin(generation_progress * 2 * math.pi * 3)  # 3 oscillations per optimization
        
        if param_def.name == "mutation_rate":
            # Oscillate mutation rate around default
            default_value = param_def.default_value
            return (default_value - current_value) * 0.1 + oscillation * (param_def.bounds.step_size or 0.01)
        
        return 0.0
    
    def _estimate_performance_change(
        self,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> float:
        """Estimate the expected performance change from parameter adjustments."""
        # Simple heuristic-based estimation
        change_score = 0.0
        
        for param_name in old_params:
            if param_name in new_params and param_name in self.param_map:
                old_val = float(old_params[param_name])
                new_val = float(new_params[param_name])
                param_def = self.param_map[param_name]
                relative_change = abs(new_val - old_val) / max(old_val, 1e-6)
                
                # Weight by parameter importance
                weighted_change = relative_change * param_def.importance_weight
                change_score += weighted_change
        
        return change_score
    
    def _determine_tuning_reason(self, convergence_indicators: Dict[str, float]) -> str:
        """Determine the primary reason for parameter tuning."""
        reasons = []
        
        convergence_rate = convergence_indicators.get("convergence_rate", 0.0)
        if convergence_rate > 0.8:
            reasons.append("high_convergence")
        
        improvement_rate = convergence_indicators.get("improvement_rate", 0.0)
        if improvement_rate < self.min_improvement_threshold:
            reasons.append("low_improvement")
        
        diversity_score = convergence_indicators.get("diversity_score", 0.5)
        if diversity_score < 0.3:
            reasons.append("low_diversity")
        
        stagnation = convergence_indicators.get("stagnation", 0.0)
        if stagnation > 0.7:
            reasons.append("stagnation")
        
        return reasons[0] if reasons else "scheduled"
    
    def get_tuning_metrics(self) -> List[str]:
        """Get the metrics this tuner uses for decisions."""
        return [
            "convergence_rate",
            "improvement_rate", 
            "diversity_score",
            "hypervolume",
            "generation_progress",
            "stagnation"
        ]
    
    def get_tuning_statistics(self) -> Dict[str, Any]:
        """Get statistics about tuning operations."""
        total_tunings = self.successful_tunings + self.failed_tunings
        success_rate = self.successful_tunings / max(total_tunings, 1)
        
        return {
            "total_tunings": total_tunings,
            "successful_tunings": self.successful_tunings,
            "failed_tunings": self.failed_tunings,
            "success_rate": success_rate,
            "last_tuning_generation": self.last_tuning_generation,
            "tuning_history_size": len(self.tuning_history),
            "parameters_tuned": len(self.param_map)
        }
    
    def get_parameter_history(self, param_name: str) -> List[Tuple[int, float]]:
        """Get the history of values for a specific parameter."""
        if param_name not in self.param_map:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        history = []
        for tuning_record in self.tuning_history:
            if param_name in tuning_record.new_parameters:
                history.append((
                    tuning_record.generation,
                    tuning_record.new_parameters[param_name]
                ))
        
        return history


class ResourceAwareTuner(ParameterTuner):
    """Parameter tuner that considers resource constraints and utilization."""
    
    def __init__(
        self,
        base_tuner: ParameterTuner,
        resource_limits: Optional[Dict[str, float]] = None,
        resource_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize resource-aware tuner.
        
        Args:
            base_tuner: Base tuner for primary parameter adjustments
            resource_limits: Limits for various resources (evaluations, time, memory)
            resource_weights: Weights for resource considerations
        """
        self.base_tuner = base_tuner
        self.resource_limits = resource_limits or {}
        self.resource_weights = resource_weights or {
            "evaluations": 1.0,
            "time": 0.8,
            "memory": 0.6
        }
        
        # Resource usage tracking
        self.resource_usage_history: List[Dict[str, float]] = []
    
    def should_adjust_parameters(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if parameters should be adjusted considering resources."""
        # Check base tuner first
        if not self.base_tuner.should_adjust_parameters(optimization_state):
            return False
        
        # Check resource constraints
        current_usage = self._get_current_resource_usage(optimization_state)
        
        for resource, limit in self.resource_limits.items():
            if current_usage.get(resource, 0) > limit:
                _logger.warning(f"Resource limit exceeded for {resource}: {current_usage[resource]} > {limit}")
                return False
        
        return True
    
    def adjust_parameters(self, current_params: Dict[str, Any], optimization_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters with resource awareness."""
        # Get base adjustment
        new_params = self.base_tuner.adjust_parameters(current_params, optimization_state)
        
        # Apply resource-aware modifications
        resource_usage = self._get_current_resource_usage(optimization_state)
        
        # Reduce population if evaluations are limited
        if "evaluations" in self.resource_limits:
            eval_usage = resource_usage.get("evaluations", 0)
            eval_limit = self.resource_limits["evaluations"]
            
            if eval_usage > eval_limit * 0.8:  # Approaching limit
                if "population_size" in new_params:
                    reduction_factor = 0.8  # Reduce by 20%
                    new_params["population_size"] = max(
                        5.0,  # Minimum population
                        new_params["population_size"] * reduction_factor
                    )
        
        # Reduce mutation if time is limited
        if "time" in self.resource_limits:
            time_usage = resource_usage.get("time", 0)
            time_limit = self.resource_limits["time"]
            
            if time_usage > time_limit * 0.8:  # Approaching limit
                if "mutation_rate" in new_params:
                    reduction_factor = 0.9  # Reduce by 10%
                    new_params["mutation_rate"] = max(
                        0.01,  # Minimum mutation rate
                        new_params["mutation_rate"] * reduction_factor
                    )
        
        # Record resource usage
        self.resource_usage_history.append(resource_usage)
        
        return new_params
    
    def _get_current_resource_usage(self, optimization_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract current resource usage from optimization state."""
        usage = {}
        
        # Evaluation count
        usage["evaluations"] = float(optimization_state.get("evaluation_count", 0))
        
        # Time elapsed
        if "start_time" in optimization_state:
            usage["time"] = time.time() - optimization_state["start_time"]
        
        # Memory usage (simplified)
        usage["memory"] = float(optimization_state.get("population_size", 20) * 10)  # KB estimate
        
        return usage
    
    def get_tuning_metrics(self) -> List[str]:
        """Get metrics including resource considerations."""
        base_metrics = self.base_tuner.get_tuning_metrics()
        resource_metrics = ["evaluations", "time", "memory"]
        return base_metrics + resource_metrics


class CompositeParameterTuner(ParameterTuner):
    """Composite tuner that combines multiple tuning strategies."""
    
    def __init__(
        self,
        tuners: List[ParameterTuner],
        tuner_weights: Optional[List[float]] = None,
        decision_strategy: str = "majority_vote"  # "majority_vote", "weighted_vote", "first_agree"
    ):
        """Initialize composite parameter tuner.
        
        Args:
            tuners: List of tuners to combine
            tuner_weights: Weights for each tuner
            decision_strategy: How to combine tuner decisions
        """
        self.tuners = tuners
        self.tuner_weights = tuner_weights or [1.0] * len(tuners)
        self.decision_strategy = decision_strategy
        
        # Normalize weights
        total_weight = sum(self.tuner_weights)
        if total_weight > 0:
            self.tuner_weights = [w / total_weight for w in self.tuner_weights]
    
    def should_adjust_parameters(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if parameters should be adjusted using composite decision."""
        votes = []
        weighted_votes = 0.0
        
        for tuner, weight in zip(self.tuners, self.tuner_weights):
            vote = tuner.should_adjust_parameters(optimization_state)
            votes.append(vote)
            weighted_votes += weight * (1.0 if vote else 0.0)
        
        if self.decision_strategy == "majority_vote":
            return sum(votes) > len(votes) / 2
        elif self.decision_strategy == "weighted_vote":
            return weighted_votes > 0.5
        elif self.decision_strategy == "first_agree":
            return any(votes)
        
        return False
    
    def adjust_parameters(self, current_params: Dict[str, Any], optimization_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters using composite strategy."""
        if self.decision_strategy == "first_agree":
            # Use the first tuner that suggests adjustment
            for tuner in self.tuners:
                if tuner.should_adjust_parameters(optimization_state):
                    return tuner.adjust_parameters(current_params, optimization_state)
            return current_params
        else:
            # Average the adjustments from all tuners
            adjustments = []
            for tuner in self.tuners:
                if tuner.should_adjust_parameters(optimization_state):
                    adjusted = tuner.adjust_parameters(current_params, optimization_state)
                    adjustments.append(adjusted)
            
            if not adjustments:
                return current_params
            
            # Average the parameter values
            new_params = current_params.copy()
            for param_name in current_params:
                values = [adj.get(param_name, current_params[param_name]) for adj in adjustments]
                new_params[param_name] = sum(values) / len(values)
            
            return new_params
    
    def get_tuning_metrics(self) -> List[str]:
        """Get combined metrics from all tuners."""
        all_metrics = set()
        for tuner in self.tuners:
            all_metrics.update(tuner.get_tuning_metrics())
        return list(all_metrics)


# Additional classes for architecture compliance

class DynamicParameterTuner(ParameterTuner):
    """Main dynamic parameter tuner that combines multiple strategies."""
    
    def __init__(self, tuners: Optional[List[ParameterTuner]] = None,
                 combination_strategy: str = "weighted_average",
                 verbose: bool = True):
        super().__init__()
        self.tuners = tuners or []
        self.combination_strategy = combination_strategy
        self.verbose = verbose
        self.tuning_history = []
        self.performance_history = []
        
        # Initialize with default tuners if none provided
        if not self.tuners:
            from .analysis import ParetoStabilityDetector
            
            self.tuners = [
                ConvergenceBasedTuner(
                    convergence_detector=ParetoStabilityDetector(),
                    verbose=verbose
                )
            ]
    
    def adjust_parameters(self, current_params: Dict[str, Any],
                         optimization_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters based on multiple strategies."""
        adjustments = []
        
        # Get adjustments from all tuners
        for tuner in self.tuners:
            try:
                adjustment = tuner.adjust_parameters(current_params, optimization_metrics)
                adjustments.append(adjustment)
            except Exception as e:
                _logger.warning(f"Tuner {type(tuner).__name__} failed: {e}")
        
        if not adjustments:
            return current_params
        
        # Combine adjustments based on strategy
        if self.combination_strategy == "weighted_average":
            return self._weighted_average_combination(current_params, adjustments)
        elif self.combination_strategy == "conservative":
            return self._conservative_combination(current_params, adjustments)
        elif self.combination_strategy == "aggressive":
            return self._aggressive_combination(current_params, adjustments)
        else:
            return self._simple_average_combination(current_params, adjustments)
    
    def _weighted_average_combination(self, current_params: Dict[str, Any],
                                    adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine adjustments using weighted average."""
        new_params = current_params.copy()
        
        for param_name in current_params:
            values = []
            weights = []
            
            for i, adj in enumerate(adjustments):
                if param_name in adj:
                    values.append(adj[param_name])
                    # Weight based on recency (more recent tuners get higher weight)
                    weights.append(i + 1)
            
            if values:
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    new_params[param_name] = weighted_sum / total_weight
        
        return new_params
    
    def _conservative_combination(self, current_params: Dict[str, Any],
                                 adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine adjustments conservatively (smaller changes)."""
        new_params = current_params.copy()
        
        for param_name in current_params:
            values = []
            for adj in adjustments:
                if param_name in adj:
                    values.append(adj[param_name])
            
            if values:
                # Use the value closest to current (most conservative)
                current_value = current_params[param_name]
                closest_value = min(values, key=lambda x: abs(x - current_value))
                new_params[param_name] = closest_value
        
        return new_params
    
    def _aggressive_combination(self, current_params: Dict[str, Any],
                               adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine adjustments aggressively (larger changes)."""
        new_params = current_params.copy()
        
        for param_name in current_params:
            values = []
            for adj in adjustments:
                if param_name in adj:
                    values.append(adj[param_name])
            
            if values:
                # Use the value farthest from current (most aggressive)
                current_value = current_params[param_name]
                farthest_value = max(values, key=lambda x: abs(x - current_value))
                new_params[param_name] = farthest_value
        
        return new_params
    
    def _simple_average_combination(self, current_params: Dict[str, Any],
                                   adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine adjustments using simple average."""
        new_params = current_params.copy()
        
        for param_name in current_params:
            values = []
            for adj in adjustments:
                if param_name in adj:
                    values.append(adj[param_name])
            
            if values:
                new_params[param_name] = sum(values) / len(values)
        
        return new_params
    
    def get_tuning_metrics(self) -> List[str]:
        """Get metrics required for tuning."""
        all_metrics = set()
        for tuner in self.tuners:
            all_metrics.update(tuner.get_tuning_metrics())
        return list(all_metrics)
    
    def get_tuning_statistics(self) -> Dict[str, Any]:
        """Get statistics about tuning performance."""
        return {
            "total_adjustments": len(self.tuning_history),
            "average_adjustment_magnitude": self._calculate_average_magnitude(),
            "tuner_count": len(self.tuners),
            "combination_strategy": self.combination_strategy,
            "recent_performance": self.performance_history[-10:] if self.performance_history else []
        }
    
    def _calculate_average_magnitude(self) -> float:
        """Calculate average magnitude of parameter adjustments."""
        if not self.tuning_history:
            return 0.0
        
        magnitudes = []
        for adjustment in self.tuning_history:
            for param_name, change in adjustment.items():
                magnitudes.append(abs(change))
        
        return sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
    
    def record_adjustment(self, adjustment: Dict[str, Any], performance: float):
        """Record an adjustment and its performance impact."""
        self.tuning_history.append(adjustment)
        self.performance_history.append(performance)
        
        # Keep history manageable
        if len(self.tuning_history) > 1000:
            self.tuning_history.pop(0)
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
    
    def add_tuner(self, tuner: ParameterTuner):
        """Add a new tuner to the combination."""
        self.tuners.append(tuner)
        _logger.info(f"Added tuner: {type(tuner).__name__}")
    
    def remove_tuner(self, tuner_type: type) -> bool:
        """Remove a tuner of the specified type."""
        for i, tuner in enumerate(self.tuners):
            if isinstance(tuner, tuner_type):
                removed = self.tuners.pop(i)
                _logger.info(f"Removed tuner: {type(removed).__name__}")
                return True
        return False
    
    def set_combination_strategy(self, strategy: str):
        """Set the combination strategy for adjustments."""
        valid_strategies = ["weighted_average", "conservative", "aggressive", "simple_average"]
        if strategy in valid_strategies:
            self.combination_strategy = strategy
            _logger.info(f"Set combination strategy to: {strategy}")
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Valid options: {valid_strategies}")
    
    def reset_history(self):
        """Reset tuning history."""
        self.tuning_history.clear()
        self.performance_history.clear()
        _logger.info("Reset tuning history")