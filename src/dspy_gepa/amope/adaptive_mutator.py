"""Adaptive mutator for AMOPE algorithm.

This module implements adaptive mutation strategies that dynamically
select the best mutation approach based on performance analysis
and optimization progress.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Import GEPA core components
try:
    from gepa.core.mutator import TextMutator, MutationStrategy as BaseMutationStrategy
    from gepa.core.candidate import Candidate
except ImportError:
    # Try development import
    try:
        from src.gepa.core.mutator import TextMutator, MutationStrategy as BaseMutationStrategy
        from src.gepa.core.candidate import Candidate
    except ImportError:
        # Define placeholder classes for development
        class TextMutator:
            pass
        class BaseMutationStrategy:
            pass
        class Candidate:
            def __init__(self, content="", **kwargs):
                self.content = content
                for k, v in kwargs.items():
                    setattr(self, k, v)


@dataclass
class MutationResult:
    """Result of a mutation operation with tracking."""
    mutated_content: str
    strategy_used: str
    confidence_score: float
    estimated_improvement: float
    computation_cost: float


class MutationStrategy(Enum):
    """Available mutation strategies."""
    GRADIENT_BASED = "gradient_based"
    LLM_GUIDED = "llm_guided"
    PATTERN_BASED = "pattern_based"
    STATISTICAL = "statistical"


class PerformanceAnalyzer:
    """Analyzes performance data to guide mutation strategy selection."""
    
    def __init__(self):
        self.gradient_window = 5
        self.convergence_threshold = 0.01
    
    def analyze_gradient(self, fitness_history: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze performance gradient characteristics."""
        if len(fitness_history) < 2:
            return {"slope": 0.0, "variance": 0.0, "trend": "stable"}
        
        # Calculate gradients for each objective
        gradients = {}
        for objective in fitness_history[0].keys():
            values = [h[objective] for h in fitness_history[-self.gradient_window:]]
            if len(values) < 2:
                gradients[objective] = 0.0
            else:
                # Simple gradient calculation
                gradient = (values[-1] - values[0]) / len(values)
                gradients[objective] = gradient
        
        # Calculate overall gradient characteristics
        all_gradients = list(gradients.values())
        slope = np.mean(all_gradients)
        variance = np.var(all_gradients) if len(all_gradients) > 1 else 0.0
        
        # Determine trend
        if abs(slope) < self.convergence_threshold:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "degrading"
        
        return {
            "slope": slope,
            "variance": variance,
            "trend": trend,
            "gradients": gradients
        }
    
    def detect_convergence_stage(self, population_metrics: Dict[str, Any]) -> str:
        """Detect current convergence stage."""
        diversity = population_metrics.get("diversity", 0.5)
        improvement_rate = population_metrics.get("improvement_rate", 0.1)
        
        if diversity > 0.7 and improvement_rate > 0.05:
            return "exploration"
        elif diversity < 0.3 and improvement_rate < 0.01:
            return "converged"
        elif improvement_rate < 0.02:
            return "exploitation"
        else:
            return "balanced"


class GradientBasedMutation:
    """Mutation strategy based on performance gradients."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def mutate(self, candidate: Candidate, context: Optional[Dict[str, Any]] = None) -> str:
        """Apply gradient-based mutation."""
        content = candidate.content
        
        # Analyze gradient information from context
        gradient_info = context.get("gradient_analysis", {}) if context else {}
        
        # Apply gradient-informed modifications
        if gradient_info.get("trend") == "stable":
            # Apply small, directed changes
            mutations = self._apply_directed_mutations(content)
        elif gradient_info.get("trend") == "degrading":
            # Apply more significant changes
            mutations = self._apply_recovery_mutations(content)
        else:
            # Apply progressive improvements
            mutations = self._apply_progressive_mutations(content)
        
        return mutations
    
    def _apply_directed_mutations(self, content: str) -> str:
        """Apply small, directed mutations."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def' in line or 'class' in line:
                # Minor modifications to function/class definitions
                if random.random() < 0.3:
                    lines[i] = self._tweak_line(line)
        return '\n'.join(lines)
    
    def _apply_recovery_mutations(self, content: str) -> str:
        """Apply larger changes for recovery."""
        # More aggressive mutations
        return self._major_restructure(content)
    
    def _apply_progressive_mutations(self, content: str) -> str:
        """Apply progressive improvements."""
        # Moderate improvements
        return self._moderate_enhancement(content)
    
    def _tweak_line(self, line: str) -> str:
        """Apply minor tweak to a line."""
        # Simple line modification logic
        return line
    
    def _major_restructure(self, content: str) -> str:
        """Major restructuring of content."""
        return content
    
    def _moderate_enhancement(self, content: str) -> str:
        """Moderate enhancement of content."""
        return content


class StatisticalMutation:
    """Statistical mutation for exploration."""
    
    def mutate(self, candidate: Candidate, context: Optional[Dict[str, Any]] = None) -> str:
        """Apply statistical mutation."""
        content = candidate.content
        
        # Apply random variations
        mutation_types = ['substitution', 'insertion', 'deletion', 'reordering']
        chosen_type = random.choice(mutation_types)
        
        if chosen_type == 'substitution':
            return self._substitution_mutation(content)
        elif chosen_type == 'insertion':
            return self._insertion_mutation(content)
        elif chosen_type == 'deletion':
            return self._deletion_mutation(content)
        else:  # reordering
            return self._reordering_mutation(content)
    
    def _substitution_mutation(self, content: str) -> str:
        """Substitute parts of content."""
        return content
    
    def _insertion_mutation(self, content: str) -> str:
        """Insert new content."""
        return content
    
    def _deletion_mutation(self, content: str) -> str:
        """Delete parts of content."""
        return content
    
    def _reordering_mutation(self, content: str) -> str:
        """Reorder content sections."""
        return content


class AdaptiveMutator(TextMutator):
    """Enhanced mutator with adaptive strategy selection."""
    
    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        
        # Initialize mutation strategies
        self.mutation_strategies = {
            MutationStrategy.GRADIENT_BASED: GradientBasedMutation(llm_client),
            MutationStrategy.STATISTICAL: StatisticalMutation(),
        }
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategy_performance = {k: [] for k in MutationStrategy}
        self.generation_count = 0
        
        # Strategy selection parameters
        self.exploration_rate = 0.3
        self.performance_window = 10
    
    def select_adaptive_strategy(self, candidate: Candidate, 
                                population_metrics: Dict[str, Any],
                                fitness_history: List[Dict[str, float]]) -> MutationStrategy:
        """Select mutation strategy based on current context."""
        
        # Analyze current state
        convergence_stage = self.performance_analyzer.detect_convergence_stage(population_metrics)
        gradient_analysis = self.performance_analyzer.analyze_gradient(fitness_history)
        
        # Strategy selection logic
        if convergence_stage == "exploration":
            # Favor statistical mutation for diversity
            return MutationStrategy.STATISTICAL
        elif convergence_stage == "exploitation":
            # Favor gradient-based for focused improvement
            return MutationStrategy.GRADIENT_BASED
        elif convergence_stage == "converged":
            # Try statistical mutation to escape local optima
            return MutationStrategy.STATISTICAL
        else:  # balanced
            # Choose based on gradient characteristics
            if gradient_analysis["variance"] > 0.1:
                return MutationStrategy.STATISTICAL
            else:
                return MutationStrategy.GRADIENT_BASED
    
    def mutate_with_adaptation(self, candidate: Candidate, 
                              context: Optional[Dict[str, Any]] = None) -> MutationResult:
        """Perform mutation with adaptive strategy selection."""
        
        # Extract context information
        population_metrics = context.get("population_metrics", {}) if context else {}
        fitness_history = context.get("fitness_history", []) if context else []
        gradient_analysis = self.performance_analyzer.analyze_gradient(fitness_history)
        
        # Select strategy
        strategy = self.select_adaptive_strategy(candidate, population_metrics, fitness_history)
        
        # Apply mutation
        mutator = self.mutation_strategies[strategy]
        mutated_content = mutator.mutate(candidate, context)
        
        # Calculate metrics
        confidence_score = self._calculate_confidence(strategy, population_metrics)
        estimated_improvement = self._estimate_improvement(strategy, gradient_analysis)
        computation_cost = self._estimate_cost(strategy)
        
        # Track performance
        self.strategy_performance[strategy].append({
            "generation": self.generation_count,
            "confidence": confidence_score
        })
        
        self.generation_count += 1
        
        return MutationResult(
            mutated_content=mutated_content,
            strategy_used=strategy.value,
            confidence_score=confidence_score,
            estimated_improvement=estimated_improvement,
            computation_cost=computation_cost
        )
    
    def _calculate_confidence(self, strategy: MutationStrategy, 
                            population_metrics: Dict[str, Any]) -> float:
        """Calculate confidence score for strategy selection."""
        base_confidence = 0.7
        
        # Adjust based on historical performance
        if self.strategy_performance[strategy]:
            recent_performance = self.strategy_performance[strategy][-5:]
            avg_confidence = np.mean([p["confidence"] for p in recent_performance])
            base_confidence = 0.7 * base_confidence + 0.3 * avg_confidence
        
        return min(1.0, base_confidence)
    
    def _estimate_improvement(self, strategy: MutationStrategy, 
                            gradient_analysis: Dict[str, float]) -> float:
        """Estimate potential improvement."""
        base_improvement = 0.1
        
        if strategy == MutationStrategy.GRADIENT_BASED:
            # Higher potential when gradient is clear
            base_improvement *= (1 + abs(gradient_analysis.get("slope", 0)))
        
        return min(1.0, base_improvement)
    
    def _estimate_cost(self, strategy: MutationStrategy) -> float:
        """Estimate computational cost."""
        costs = {
            MutationStrategy.GRADIENT_BASED: 0.3,
            MutationStrategy.STATISTICAL: 0.1,
        }
        return costs.get(strategy, 0.2)
    
    def get_strategy_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for each strategy."""
        stats = {}
        for strategy, performance_list in self.strategy_performance.items():
            if performance_list:
                confidences = [p["confidence"] for p in performance_list]
                stats[strategy.value] = {
                    "usage_count": len(performance_list),
                    "avg_confidence": np.mean(confidences),
                    "confidence_std": np.std(confidences),
                    "success_rate": np.mean([c > 0.5 for c in confidences])
                }
            else:
                stats[strategy.value] = {
                    "usage_count": 0,
                    "avg_confidence": 0.0,
                    "confidence_std": 0.0,
                    "success_rate": 0.0
                }
        
        return stats