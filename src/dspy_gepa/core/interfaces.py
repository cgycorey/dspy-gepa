"""Core interfaces and abstractions for multi-objective GEPA optimization.

This module defines the fundamental interfaces and abstract base classes
that form the foundation of the multi-objective optimization framework.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import time


class OptimizationDirection(Enum):
    """Direction of optimization for an objective."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class TaskType(Enum):
    """Supported task types for optimization."""
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    QA = "qa"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    CUSTOM = "custom"


@dataclass
class SolutionMetadata:
    """Metadata associated with a solution."""
    generation: int
    parent_ids: List[str]
    mutation_type: Optional[str] = None
    evaluation_time: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None
    convergence_info: Optional[Dict[str, Any]] = None
    custom_data: Optional[Dict[str, Any]] = None


@dataclass
class ObjectiveEvaluation:
    """Result of evaluating a solution against an objective."""
    objective_name: str
    score: float
    direction: OptimizationDirection
    metadata: Optional[Dict[str, Any]] = None
    evaluation_time: Optional[float] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result for a solution."""
    solution_id: str
    objectives: Dict[str, ObjectiveEvaluation]
    overall_score: Optional[float] = None
    evaluation_time: Optional[float] = None
    metadata: Optional[SolutionMetadata] = None
    
    def get_objective_score(self, objective_name: str) -> Optional[float]:
        """Get score for a specific objective."""
        return self.objectives.get(objective_name, ObjectiveEvaluation(
            objective_name, 0.0, OptimizationDirection.MAXIMIZE
        )).score
    
    def is_dominated_by(self, other: 'EvaluationResult') -> bool:
        """Check if this evaluation is dominated by another."""
        worse_in_any = False
        
        for obj_name, eval_result in self.objectives.items():
            if obj_name not in other.objectives:
                continue
                
            other_eval = other.objectives[obj_name]
            
            if eval_result.direction == OptimizationDirection.MAXIMIZE:
                if eval_result.score < other_eval.score:
                    worse_in_any = True
                elif eval_result.score > other_eval.score:
                    return False  # Better in at least one objective
            else:  # MINIMIZE
                if eval_result.score > other_eval.score:
                    worse_in_any = True
                elif eval_result.score < other_eval.score:
                    return False  # Better in at least one objective
        
        return worse_in_any


@dataclass
class PreferenceVector:
    """User preferences for solution selection."""
    weights: Dict[str, float]
    constraints: Optional[Dict[str, Tuple[float, float]]] = None
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total_weight = sum(abs(w) for w in self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}


@dataclass
class Objective(ABC):
    """Abstract base class for optimization objectives."""
    name: str
    weight: float = 1.0
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    description: Optional[str] = None
    
    def __post_init__(self):
        """Initialize description if not provided."""
        if self.description is None:
            self.description = self.name
    
    @abstractmethod
    def evaluate(self, program: Any, dataset: List[Any], **kwargs) -> float:
        """Evaluate a program against this objective.
        
        Args:
            program: The program/prompt to evaluate
            dataset: Dataset for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Normalized score (typically 0.0 to 1.0)
        """
        pass
    
    def validate_score(self, score: float) -> float:
        """Validate and normalize a score."""
        if not isinstance(score, (int, float)):
            raise ValueError(f"Score must be numeric, got {type(score)}")
        if score < 0:
            raise ValueError(f"Score must be non-negative, got {score}")
        return float(score)


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    def __init__(self, name: str, weight: float = 1.0, task_types: Optional[List[TaskType]] = None):
        """Initialize mutation operator.
        
        Args:
            name: Unique identifier for the operator
            weight: Relative selection weight
            task_types: Task types this operator applies to
        """
        self.name = name
        self.weight = weight
        self.task_types = task_types or list(TaskType)
    
    @abstractmethod
    def mutate(self, solution: Any, **kwargs) -> Any:
        """Apply mutation to a solution.
        
        Args:
            solution: The solution to mutate
            **kwargs: Additional mutation parameters
            
        Returns:
            Mutated solution
        """
        pass
    
    def can_handle_task(self, task_type: TaskType) -> bool:
        """Check if this operator can handle the given task type."""
        return task_type in self.task_types
    
    def get_mutation_rate(self, generation: int, convergence_metrics: Dict[str, float]) -> float:
        """Get dynamic mutation rate based on optimization state."""
        return 0.1  # Default mutation rate
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this mutator (optional)."""
        return {}


class ParameterTuner(ABC):
    """Abstract base class for parameter tuning strategies."""
    
    @abstractmethod
    def should_adjust_parameters(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if parameters should be adjusted."""
        pass
    
    @abstractmethod
    def adjust_parameters(self, current_params: Dict[str, Any], optimization_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust optimization parameters based on current state."""
        pass
    
    @abstractmethod
    def get_tuning_metrics(self) -> List[str]:
        """Get the metrics this tuner uses for decisions."""
        pass


class ConvergenceDetector(ABC):
    """Abstract base class for convergence detection."""
    
    @abstractmethod
    def has_converged(self, optimization_state: Dict[str, Any]) -> bool:
        """Check if optimization has converged."""
        pass
    
    @abstractmethod
    def get_convergence_metrics(self) -> List[str]:
        """Get the metrics used for convergence detection."""
        pass
    
    @abstractmethod
    def get_convergence_score(self, optimization_state: Dict[str, Any]) -> float:
        """Get a convergence score (0.0 = no convergence, 1.0 = fully converged)."""
        pass


class OptimalStoppingEstimator(ABC):
    """Abstract base class for optimal stopping estimation.
    
    Determines when to stop optimization to maximize expected utility
    while balancing exploration vs exploitation and resource constraints.
    """
    
    @abstractmethod
    def should_stop(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if optimization should be stopped.
        
        Args:
            optimization_state: Current optimization state including metrics,
                               generation info, resource usage, etc.
        
        Returns:
            True if optimization should stop, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stopping_confidence(self) -> float:
        """Get confidence level for the stopping decision.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_stopping_reason(self) -> str:
        """Get human-readable reason for stopping recommendation.
        
        Returns:
            String explaining why stopping was recommended
        """
        pass
    
    @abstractmethod
    def get_predicted_improvement(self) -> float:
        """Get predicted improvement from continuing optimization.
        
        Returns:
            Expected improvement score (can be negative if degradation expected)
        """
        pass


class SelectionStrategy(ABC):
    """Abstract base class for solution selection strategies."""
    
    @abstractmethod
    def select_parents(self, population: List[EvaluationResult], num_parents: int) -> List[EvaluationResult]:
        """Select parent solutions for reproduction."""
        pass
    
    @abstractmethod
    def select_survivors(self, population: List[EvaluationResult], num_survivors: int) -> List[EvaluationResult]:
        """Select solutions to survive to next generation."""
        pass


class ParetoFrontierManager(ABC):
    """Abstract base class for Pareto frontier management."""
    
    @abstractmethod
    def update_frontier(self, candidate: EvaluationResult) -> bool:
        """Update Pareto frontier with a new candidate.
        
        Returns:
            True if candidate was added to frontier
        """
        pass
    
    @abstractmethod
    def get_frontier(self) -> List[EvaluationResult]:
        """Get current Pareto frontier."""
        pass
    
    @abstractmethod
    def select_solution(self, preference: Optional[PreferenceVector] = None) -> Optional[EvaluationResult]:
        """Select a solution from the frontier based on preferences."""
        pass
    
    @abstractmethod
    def prune_frontier(self, max_size: int) -> None:
        """Prune frontier to maximum size using diversity metrics."""
        pass
    
    @abstractmethod
    def calculate_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """Calculate hypervolume indicator for the frontier."""
        pass


class MultiObjectiveOptimizer(ABC):
    """Abstract base class for multi-objective optimizers."""
    
    @abstractmethod
    def optimize(
        self,
        program: Any,
        trainset: List[Any],
        objectives: List[Objective],
        **kwargs
    ) -> Any:
        """Run multi-objective optimization.
        
        Args:
            program: Initial program to optimize
            trainset: Training dataset
            objectives: List of objectives to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimization result
        """
        pass
    
    @abstractmethod
    def get_pareto_frontier(self) -> ParetoFrontierManager:
        """Get the current Pareto frontier."""
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history and metrics."""
        pass


class MetricConverter(ABC):
    """Abstract base class for metric conversion between DSPy and GEPA formats."""
    
    @abstractmethod
    def dspy_to_multi_obj(self, dspy_metric: Callable, objective_name: str) -> Objective:
        """Convert DSPy metric to multi-objective format."""
        pass
    
    @abstractmethod
    def aggregate_dspy_metrics(self, metrics: List[Callable], weights: List[float]) -> List[Objective]:
        """Aggregate multiple DSPy metrics into multi-objective format."""
        pass
    
    @abstractmethod
    def create_composite_metric(self, task_metrics: Dict[str, Callable]) -> List[Objective]:
        """Create composite objective from task-specific metrics."""
        pass


class ResourceMonitor(ABC):
    """Abstract base class for resource monitoring."""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return resource usage data."""
        pass
    
    @abstractmethod
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        pass
    
    @abstractmethod
    def check_resource_limits(self, limits: Dict[str, Any]) -> bool:
        """Check if current usage exceeds specified limits."""
        pass


class CheckpointManager(ABC):
    """Abstract base class for checkpoint and recovery."""
    
    @abstractmethod
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_id: Optional[str] = None) -> str:
        """Save optimization state to checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load optimization state from checkpoint."""
        pass
    
    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        pass


class OptimizationLogger(ABC):
    """Abstract base class for optimization logging."""
    
    @abstractmethod
    def log_generation_start(self, generation: int, population_size: int) -> None:
        """Log the start of a generation."""
        pass
    
    @abstractmethod
    def log_generation_end(self, generation: int, metrics: Dict[str, Any]) -> None:
        """Log the end of a generation."""
        pass
    
    @abstractmethod
    def log_evaluation(self, solution_id: str, results: EvaluationResult) -> None:
        """Log solution evaluation results."""
        pass
    
    @abstractmethod
    def log_mutation(self, solution_id: str, mutation_type: str, result_id: str) -> None:
        """Log mutation operation."""
        pass
    
    @abstractmethod
    def log_convergence(self, generation: int, convergence_score: float) -> None:
        """Log convergence information."""
        pass
    
    @abstractmethod
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization logs."""
        pass


@dataclass
class CandidateSolution:
    """Represents a candidate solution in multi-objective optimization."""
    solution_id: str
    program: Any
    objectives: Dict[str, ObjectiveEvaluation]
    generation: int
    parent_solutions: List[str]
    metadata: SolutionMetadata
    
    def dominates(self, other: 'CandidateSolution') -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        # This solution dominates other if it's better or equal in all objectives
        # and strictly better in at least one objective
        at_least_one_better = False
        
        for obj_name in self.objectives:
            if obj_name in other.objectives:
                self_score = self.objectives[obj_name].score
                other_score = other.objectives[obj_name].score
                
                # Check objective direction
                self_direction = self.objectives[obj_name].direction
                
                if self_direction == OptimizationDirection.MAXIMIZE:
                    if self_score < other_score:
                        return False  # Not better in this objective
                    elif self_score > other_score:
                        at_least_one_better = True
                else:  # MINIMIZE
                    if self_score > other_score:
                        return False  # Not better in this objective
                    elif self_score < other_score:
                        at_least_one_better = True
            else:
                # Other solution doesn't have this objective, assume dominance
                at_least_one_better = True
        
        return at_least_one_better
    
    def crowding_distance(self, frontier: List['CandidateSolution']) -> float:
        """Calculate crowding distance for diversity preservation."""
        if not frontier or len(frontier) <= 2:
            return float('inf')
        
        distance = 0.0
        
        # Calculate crowding distance for each objective
        for obj_name in self.objectives:
            # Extract scores for this objective
            scores = []
            solutions_with_scores = []
            
            for solution in frontier:
                if obj_name in solution.objectives:
                    scores.append(solution.objectives[obj_name].score)
                    solutions_with_scores.append(solution)
            
            if len(scores) <= 2:
                continue
            
            # Sort by score
            sorted_pairs = sorted(zip(scores, solutions_with_scores), key=lambda x: x[0])
            sorted_scores, sorted_solutions = zip(*sorted_pairs)
            
            # Find this solution's position
            try:
                idx = sorted_solutions.index(self)
            except ValueError:
                continue
            
            # Boundary solutions get infinite distance
            if idx == 0 or idx == len(sorted_scores) - 1:
                return float('inf')
            
            # Calculate normalized distance
            score_range = sorted_scores[-1] - sorted_scores[0]
            if score_range > 0:
                distance += (sorted_scores[idx + 1] - sorted_scores[idx - 1]) / score_range
        
        return distance
    
    def get_objective_score(self, objective_name: str) -> Optional[float]:
        """Get the score for a specific objective."""
        if objective_name in self.objectives:
            return self.objectives[objective_name].score
        return None
    
    def get_objectives_summary(self) -> Dict[str, float]:
        """Get summary of all objective scores."""
        return {name: eval.score for name, eval in self.objectives.items()}
    
    def is_feasible(self) -> bool:
        """Check if solution satisfies all constraints."""
        # Simple feasibility check - can be extended
        return len(self.objectives) > 0
    
    def __hash__(self) -> int:
        """Make solution hashable for use in sets/dicts."""
        return hash(self.solution_id)
    
    def __eq__(self, other) -> bool:
        """Check equality based on solution ID."""
        if not isinstance(other, CandidateSolution):
            return False
        return self.solution_id == other.solution_id