"""Objective balancer for AMOPE algorithm.

This module implements dynamic objective weight adjustment to escape local
optima and balance multiple objectives during optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


@dataclass
class ObjectiveInfo:
    """Information about an optimization objective."""
    name: str
    weight: float
    target_value: Optional[float] = None
    is_maximization: bool = True
    stagnation_threshold: float = 0.01
    improvement_window: int = 10


@dataclass
class StagnationMetrics:
    """Metrics for detecting objective stagnation."""
    stagnation_score: float
    improvement_rate: float
    trend_direction: str
    last_significant_change: int


class BalancingStrategy(Enum):
    """Strategies for objective balancing."""
    STAGNATION_FOCUS = "stagnation_focus"
    BALANCED_IMPROVEMENT = "balanced_improvement"
    PARETO_BALANCED = "pareto_balanced"
    ADAPTIVE_HARMONIC = "adaptive_harmonic"


class ObjectiveBalancer:
    """Dynamically adjusts objective weights to escape local optima."""
    
    def __init__(self, objectives: Dict[str, float], 
                 strategy: BalancingStrategy = BalancingStrategy.ADAPTIVE_HARMONIC,
                 stagnation_window: int = 15,
                 min_weight: float = 0.1,
                 max_weight: float = 3.0):
        """
        Initialize the objective balancer.
        
        Args:
            objectives: Dictionary of objective names to initial weights
            strategy: Balancing strategy to use
            stagnation_window: Window size for detecting stagnation
            min_weight: Minimum allowed weight for any objective
            max_weight: Maximum allowed weight for any objective
        """
        self.initial_objectives = objectives.copy()
        self.current_objectives = objectives.copy()
        self.strategy = strategy
        self.stagnation_window = stagnation_window
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Tracking
        self.fitness_history: List[Dict[str, float]] = []
        self.weight_history: List[Dict[str, float]] = []
        self.stagnation_scores: Dict[str, float] = {}
        self.improvement_rates: Dict[str, float] = {}
        self.generation = 0
        
        # Advanced parameters
        self.adaptation_rate = 0.1
        self.exploration_boost = 1.5
        self.convergence_threshold = 0.005
    
    def update_fitness(self, fitness_values: Dict[str, float]) -> None:
        """Update fitness history and recalculate weights."""
        self.fitness_history.append(fitness_values.copy())
        self._update_stagnation_metrics()
        self._adjust_weights()
        self.generation += 1
    
    def _update_stagnation_metrics(self) -> None:
        """Update stagnation metrics for all objectives."""
        if len(self.fitness_history) < 2:
            return
        
        window_size = min(self.stagnation_window, len(self.fitness_history))
        recent_history = self.fitness_history[-window_size:]
        
        for objective_name in self.current_objectives.keys():
            metrics = self._calculate_stagnation_metrics(objective_name, recent_history)
            self.stagnation_scores[objective_name] = metrics.stagnation_score
            self.improvement_rates[objective_name] = metrics.improvement_rate
    
    def _calculate_stagnation_metrics(self, objective_name: str, 
                                    history: List[Dict[str, float]]) -> StagnationMetrics:
        """Calculate stagnation metrics for a specific objective."""
        values = [h.get(objective_name, 0.0) for h in history]
        
        if len(values) < 2:
            return StagnationMetrics(0.0, 0.0, "stable", 0)
        
        # Calculate improvement rate
        if len(values) >= 3:
            # Use linear regression to determine trend
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            improvement_rate = coeffs[0]
        else:
            improvement_rate = values[-1] - values[0]
        
        # Determine trend direction
        if abs(improvement_rate) < self.convergence_threshold:
            trend_direction = "stable"
        elif improvement_rate > 0:
            trend_direction = "improving"
        else:
            trend_direction = "degrading"
        
        # Calculate stagnation score (0 = no stagnation, 1 = complete stagnation)
        if len(values) >= 5:
            # Calculate variance and trend consistency
            variance = np.var(values)
            trend_consistency = 1.0 - (abs(improvement_rate) / (np.mean(values) + 1e-6))
            stagnation_score = min(1.0, trend_consistency * (1.0 - variance))
        else:
            stagnation_score = 0.5
        
        # Find last significant change
        last_significant_change = 0
        for i in range(len(values) - 1, 0, -1):
            if abs(values[i] - values[i-1]) > self.convergence_threshold:
                last_significant_change = len(values) - i
                break
        
        return StagnationMetrics(
            stagnation_score=stagnation_score,
            improvement_rate=improvement_rate,
            trend_direction=trend_direction,
            last_significant_change=last_significant_change
        )
    
    def _adjust_weights(self) -> None:
        """Adjust objective weights based on current strategy."""
        if self.strategy == BalancingStrategy.STAGNATION_FOCUS:
            self._apply_stagnation_focus()
        elif self.strategy == BalancingStrategy.BALANCED_IMPROVEMENT:
            self._apply_balanced_improvement()
        elif self.strategy == BalancingStrategy.PARETO_BALANCED:
            self._apply_pareto_balanced()
        else:  # ADAPTIVE_HARMONIC
            self._apply_adaptive_harmonic()
        
        # Normalize weights
        self._normalize_weights()
        
        # Store history
        self.weight_history.append(self.current_objectives.copy())
    
    def _apply_stagnation_focus(self) -> None:
        """Focus on stagnant objectives."""
        total_stagnation = sum(self.stagnation_scores.values())
        
        if total_stagnation < 0.1:
            return  # No significant stagnation
        
        for objective_name, base_weight in self.initial_objectives.items():
            stagnation_score = self.stagnation_scores.get(objective_name, 0.0)
            
            # Increase weight for stagnant objectives
            if stagnation_score > 0.3:
                weight_multiplier = 1.0 + (stagnation_score * self.exploration_boost)
            else:
                weight_multiplier = 1.0 - (stagnation_score * 0.5)
            
            self.current_objectives[objective_name] = base_weight * weight_multiplier
    
    def _apply_balanced_improvement(self) -> None:
        """Balance improvement across all objectives."""
        for objective_name, base_weight in self.initial_objectives.items():
            improvement_rate = self.improvement_rates.get(objective_name, 0.0)
            
            # Adjust based on improvement rate
            if improvement_rate > 0:
                # Objectives improving well can maintain or slightly reduce weight
                weight_adjustment = 1.0 - (improvement_rate * 0.1)
            else:
                # Objectives not improving need more attention
                weight_adjustment = 1.0 + abs(improvement_rate) * 0.5
            
            self.current_objectives[objective_name] = base_weight * weight_adjustment
    
    def _apply_pareto_balanced(self) -> None:
        """Balance weights considering Pareto front dynamics."""
        # This would require knowledge of the current Pareto front
        # For now, use a simplified approach
        
        avg_improvement = np.mean(list(self.improvement_rates.values()))
        
        for objective_name, base_weight in self.initial_objectives.items():
            improvement_rate = self.improvement_rates.get(objective_name, 0.0)
            stagnation_score = self.stagnation_scores.get(objective_name, 0.0)
            
            # Balance between stagnation and improvement
            if improvement_rate < avg_improvement and stagnation_score > 0.2:
                # Needs more attention
                weight_multiplier = 1.2
            elif improvement_rate > avg_improvement * 1.5:
                # Can afford less attention
                weight_multiplier = 0.8
            else:
                # Maintain balance
                weight_multiplier = 1.0
            
            self.current_objectives[objective_name] = base_weight * weight_multiplier
    
    def _apply_adaptive_harmonic(self) -> None:
        """Apply adaptive harmonic balancing."""
        for objective_name, base_weight in self.initial_objectives.items():
            improvement_rate = self.improvement_rates.get(objective_name, 0.0)
            stagnation_score = self.stagnation_scores.get(objective_name, 0.0)
            
            # Calculate harmonic adjustment
            harmonic_factor = 1.0
            
            # Stagnation adjustment
            if stagnation_score > 0.3:
                harmonic_factor *= (1.0 + stagnation_score)
            
            # Improvement adjustment
            if improvement_rate < -0.01:  # Degrading
                harmonic_factor *= 1.5
            elif improvement_rate > 0.05:  # Improving well
                harmonic_factor *= 0.9
            
            # Apply adaptive rate
            self.current_objectives[objective_name] = base_weight + \
                (harmonic_factor - 1.0) * base_weight * self.adaptation_rate
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0 while respecting bounds."""
        # Apply bounds
        for objective_name in self.current_objectives:
            self.current_objectives[objective_name] = max(
                self.min_weight, 
                min(self.max_weight, self.current_objectives[objective_name])
            )
        
        # Normalize to sum to 1.0
        total_weight = sum(self.current_objectives.values())
        if total_weight > 0:
            for objective_name in self.current_objectives:
                self.current_objectives[objective_name] /= total_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current objective weights."""
        return self.current_objectives.copy()
    
    def get_stagnation_report(self) -> Dict[str, Dict[str, float]]:
        """Get detailed stagnation metrics for all objectives."""
        report = {}
        for objective_name in self.current_objectives.keys():
            report[objective_name] = {
                "stagnation_score": self.stagnation_scores.get(objective_name, 0.0),
                "improvement_rate": self.improvement_rates.get(objective_name, 0.0),
                "current_weight": self.current_objectives.get(objective_name, 0.0),
                "weight_change": self.current_objectives.get(objective_name, 0.0) - \
                               self.initial_objectives.get(objective_name, 0.0)
            }
        return report
    
    def detect_convergence_stagnation(self) -> bool:
        """Detect if the entire optimization is stagnating."""
        if len(self.fitness_history) < self.stagnation_window:
            return False
        
        # Check if all objectives are stagnating
        all_stagnating = all(
            score > 0.6 for score in self.stagnation_scores.values()
        )
        
        # Check if weight changes are minimal
        recent_weights = self.weight_history[-5:] if len(self.weight_history) >= 5 else self.weight_history
        if len(recent_weights) < 2:
            return False
        
        weight_variance = 0.0
        for objective_name in self.current_objectives.keys():
            weight_values = [w.get(objective_name, 0.0) for w in recent_weights]
            weight_variance += np.var(weight_values)
        
        weight_variance /= len(self.current_objectives)
        
        return all_stagnating and weight_variance < 0.01
    
    def reset_weights(self) -> None:
        """Reset weights to initial values."""
        self.current_objectives = self.initial_objectives.copy()
        self.fitness_history.clear()
        self.weight_history.clear()
        self.stagnation_scores.clear()
        self.improvement_rates.clear()
        self.generation = 0
    
    def suggest_exploration_boost(self) -> List[str]:
        """Suggest which objectives need exploration boosts."""
        suggestions = []
        
        for objective_name, stagnation_score in self.stagnation_scores.items():
            if stagnation_score > 0.5:
                suggestions.append(objective_name)
        
        return suggestions