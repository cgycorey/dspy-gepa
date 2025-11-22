"""Multi-objective GEPA implementation.

This module provides the core multi-objective optimization framework
that extends the existing GEPA capabilities with Pareto frontier management
and sophisticated multi-objective optimization strategies.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

from .interfaces import (
    Objective, OptimizationDirection, TaskType, PreferenceVector,
    EvaluationResult, ObjectiveEvaluation, SolutionMetadata,
    ParetoFrontierManager, MultiObjectiveOptimizer, MutationOperator,
    ParameterTuner, ConvergenceDetector, SelectionStrategy,
    ResourceMonitor, CheckpointManager, OptimizationLogger,
    OptimalStoppingEstimator
)
from .analysis import (
    ParetoStabilityDetector, HypervolumeConvergenceDetector,
    DiversityConvergenceDetector, ImprovementPlateauDetector,
    StatisticalTrendEstimator, ResourceBoundedEstimator,
    DiminishingReturnsEstimator
)
from .monitoring import MonitoringFramework
from ..utils.logging import get_logger


_logger = get_logger(__name__)


@dataclass
class OptimizationState:
    """Current state of the optimization process."""
    generation: int
    population: List[EvaluationResult]
    frontier: List[EvaluationResult]
    metrics: Dict[str, float]
    parameter_history: List[Dict[str, Any]]
    convergence_history: List[float]
    resource_usage: Dict[str, Any]
    start_time: float
    evaluation_count: int = 0
    
    def get_best_solution(self, objective_name: str) -> Optional[EvaluationResult]:
        """Get best solution for a specific objective."""
        if not self.population:
            return None
        
        direction = None
        # Find the direction for this objective
        for result in self.population:
            if objective_name in result.objectives:
                direction = result.objectives[objective_name].direction
                break
        
        if direction is None:
            return None
        
        if direction == OptimizationDirection.MAXIMIZE:
            return max(self.population, key=lambda r: r.get_objective_score(objective_name) or 0)
        else:
            return min(self.population, key=lambda r: r.get_objective_score(objective_name) or float('inf'))
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Calculate diversity metrics for the current population."""
        if len(self.population) < 2:
            return {"pairwise_distance": 0.0, "objective_variance": 0.0}
        
        # Simple diversity metrics
        objective_scores = defaultdict(list)
        for result in self.population:
            for obj_name, obj_eval in result.objectives.items():
                objective_scores[obj_name].append(obj_eval.score)
        
        # Calculate average variance across objectives
        variances = []
        for obj_name, scores in objective_scores.items():
            if len(scores) > 1:
                mean_score = sum(scores) / len(scores)
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                variances.append(variance)
        
        return {
            "objective_variance": sum(variances) / len(variances) if variances else 0.0,
            "pairwise_distance": self._calculate_pairwise_distance()
        }
    
    def _calculate_pairwise_distance(self) -> float:
        """Calculate average pairwise distance in objective space."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i, result1 in enumerate(self.population):
            for result2 in self.population[i+1:]:
                distance = self._objective_distance(result1, result2)
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _objective_distance(self, r1: EvaluationResult, r2: EvaluationResult) -> float:
        """Calculate Euclidean distance between two results in objective space."""
        # Get all objective names
        obj_names = set(r1.objectives.keys()) | set(r2.objectives.keys())
        
        distance_squared = 0.0
        for obj_name in obj_names:
            score1 = r1.get_objective_score(obj_name) or 0.0
            score2 = r2.get_objective_score(obj_name) or 0.0
            distance_squared += (score1 - score2) ** 2
        
        return distance_squared ** 0.5
    
    def add_solution(self, solution: EvaluationResult) -> None:
        """Add a solution to the population."""
        self.population.append(solution)
        self.evaluation_count += 1
    
    def update_frontier(self, frontier: List[EvaluationResult]) -> None:
        """Update the Pareto frontier."""
        self.frontier = frontier.copy()


class ParetoFrontier(ParetoFrontierManager):
    """Implementation of Pareto frontier management."""
    
    def __init__(self, max_size: Optional[int] = None):
        """Initialize Pareto frontier.
        
        Args:
            max_size: Maximum size of frontier (None for unlimited)
        """
        self.max_size = max_size
        self._frontier: List[EvaluationResult] = []
        self._solution_index: Dict[str, int] = {}  # solution_id -> index in frontier
        _logger.debug(f"ParetoFrontier initialized with max_size={max_size}")
    
    def update_frontier(self, candidate: EvaluationResult) -> bool:
        """Update Pareto frontier with a new candidate."""
        _logger.debug(f"Updating frontier with candidate {candidate.solution_id}")
        
        # Check if candidate is dominated by existing frontier
        dominated = False
        for i, existing in enumerate(self._frontier):
            if candidate.is_dominated_by(existing):
                dominated = True
                _logger.debug(f"Candidate {candidate.solution_id} is dominated by {existing.solution_id}")
                break
        
        if dominated:
            return False
        
        # Remove existing solutions that are dominated by candidate
        to_remove = []
        for i, existing in enumerate(self._frontier):
            if existing.is_dominated_by(candidate):
                to_remove.append(i)
                _logger.debug(f"Candidate {candidate.solution_id} dominates {existing.solution_id}")
        
        # Remove dominated solutions (in reverse order to maintain indices)
        for i in reversed(to_remove):
            removed = self._frontier.pop(i)
            del self._solution_index[removed.solution_id]
        
        # Add candidate to frontier
        self._frontier.append(candidate)
        self._solution_index[candidate.solution_id] = len(self._frontier) - 1
        
        # Prune if necessary
        if self.max_size and len(self._frontier) > self.max_size:
            self.prune_frontier(self.max_size)
        
        _logger.debug(f"Frontier updated. Size: {len(self._frontier)}")
        return True
    
    def get_frontier(self) -> List[EvaluationResult]:
        """Get current Pareto frontier."""
        return self._frontier.copy()
    
    def select_solution(self, preference: Optional[PreferenceVector] = None) -> Optional[EvaluationResult]:
        """Select a solution from the frontier based on preferences."""
        if not self._frontier:
            return None
        
        if preference is None:
            # Return a random solution from the frontier
            return random.choice(self._frontier)
        
        # Calculate weighted score for each solution
        best_solution = None
        best_score = float('-inf')
        
        for solution in self._frontier:
            score = 0.0
            for obj_name, weight in preference.weights.items():
                obj_score = solution.get_objective_score(obj_name) or 0.0
                score += weight * obj_score
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution
    
    def prune_frontier(self, max_size: int) -> None:
        """Prune frontier to maximum size using diversity metrics."""
        if len(self._frontier) <= max_size:
            return
        
        _logger.debug(f"Pruning frontier from {len(self._frontier)} to {max_size}")
        
        # Calculate crowding distances
        crowding_distances = self._calculate_crowding_distances()
        
        # Sort by crowding distance (keep diverse solutions)
        sorted_indices = sorted(
            range(len(self._frontier)),
            key=lambda i: crowding_distances[i],
            reverse=True
        )
        
        # Keep top max_size solutions
        keep_indices = set(sorted_indices[:max_size])
        new_frontier = [self._frontier[i] for i in keep_indices]
        
        # Update frontier and index
        self._frontier = new_frontier
        self._solution_index = {
            solution.solution_id: i 
            for i, solution in enumerate(self._frontier)
        }
        
        _logger.debug(f"Frontier pruned. New size: {len(self._frontier)}")
    
    def _calculate_crowding_distances(self) -> List[float]:
        """Calculate crowding distance for each solution in the frontier."""
        if len(self._frontier) == 0:
            return []
        
        n = len(self._frontier)
        distances = [0.0] * n
        
        # Get all objective names
        all_objectives = set()
        for solution in self._frontier:
            all_objectives.update(solution.objectives.keys())
        
        # Calculate crowding distance for each objective
        for obj_name in all_objectives:
            # Sort solutions by this objective
            sorted_solutions = sorted(
                enumerate(self._frontier),
                key=lambda x: x[1].get_objective_score(obj_name) or 0
            )
            
            # Get objective scores
            scores = [sol.get_objective_score(obj_name) or 0 for _, sol in sorted_solutions]
            
            # Handle edge cases
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                continue
            
            # Set boundary points to infinity
            distances[sorted_solutions[0][0]] = float('inf')
            distances[sorted_solutions[-1][0]] = float('inf')
            
            # Calculate crowding distance for interior points
            for i in range(1, n - 1):
                if distances[sorted_solutions[i][0]] != float('inf'):
                    distance_contribution = (
                        scores[i + 1] - scores[i - 1]
                    ) / (max_score - min_score)
                    distances[sorted_solutions[i][0]] += distance_contribution
        
        # Replace infinity with a large number
        for i in range(n):
            if distances[i] == float('inf'):
                distances[i] = 1e9
        
        return distances
    
    def calculate_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """Calculate hypervolume indicator for the frontier."""
        if not self._frontier:
            return 0.0
        
        # Get all objective names and directions
        all_objectives = {}
        for solution in self._frontier:
            for obj_name, obj_eval in solution.objectives.items():
                all_objectives[obj_name] = obj_eval.direction
        
        # Set reference point if not provided
        if reference_point is None:
            reference_point = {}
            for obj_name, direction in all_objectives.items():
                if direction == OptimizationDirection.MAXIMIZE:
                    reference_point[obj_name] = 0.0  # Worse than any real solution
                else:
                    reference_point[obj_name] = 1.0  # Worse than any real solution
        
        # Simple hypervolume calculation (for small frontier sizes)
        # This is a simplified implementation - for production use a more efficient algorithm
        hypervolume = 0.0
        
        for solution in self._frontier:
            volume = 1.0
            for obj_name, direction in all_objectives.items():
                score = solution.get_objective_score(obj_name) or 0.0
                ref_value = reference_point.get(obj_name, 0.0)
                
                if direction == OptimizationDirection.MAXIMIZE:
                    contribution = max(0, score - ref_value)
                else:
                    contribution = max(0, ref_value - score)
                
                volume *= contribution
            
            hypervolume += volume
        
        return hypervolume


class MultiObjectiveGEPA(MultiObjectiveOptimizer):
    """Main multi-objective GEPA optimizer."""
    
    def __init__(
        self,
        objectives: List[Objective],
        max_generations: int = 30,
        population_size: int = 20,
        mutation_operators: Optional[List[MutationOperator]] = None,
        parameter_tuner: Optional[ParameterTuner] = None,
        convergence_detector: Optional[ConvergenceDetector] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[OptimizationLogger] = None,
        verbose: bool = True,
        random_seed: Optional[int] = None
    ):
        """Initialize multi-objective GEPA optimizer.
        
        Args:
            objectives: List of objectives to optimize
            max_generations: Maximum number of generations
            population_size: Population size for optimization
            mutation_operators: List of mutation operators
            parameter_tuner: Parameter tuning strategy
            convergence_detector: Convergence detection strategy
            selection_strategy: Selection strategy for evolution
            resource_monitor: Resource monitoring
            checkpoint_manager: Checkpoint management
            logger: Optimization logging
            verbose: Whether to print progress
            random_seed: Random seed for reproducibility
        """
        self.objectives = objectives
        self.max_generations = max_generations
        self.population_size = population_size
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
        
        # Initialize components
        self.pareto_frontier = ParetoFrontier(max_size=population_size * 2)
        self.mutation_operators = mutation_operators or []
        self.parameter_tuner = parameter_tuner
        self.convergence_detector = convergence_detector
        self.selection_strategy = selection_strategy
        self.resource_monitor = resource_monitor
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        
        # Optimization state
        self.optimization_state: Optional[OptimizationState] = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        _logger.info(f"MultiObjectiveGEPA initialized with {len(objectives)} objectives")
        if self.verbose:
            print(f"🚀 Multi-Objective GEPA initialized")
            print(f"   Objectives: {len(objectives)}")
            print(f"   Max generations: {max_generations}")
            print(f"   Population size: {population_size}")
    
    def optimize(
        self,
        program: Any,
        trainset: List[Any],
        objectives: Optional[List[Objective]] = None,
        **kwargs
    ) -> OptimizationState:
        """Run multi-objective optimization.
        
        Args:
            program: Initial program to optimize
            trainset: Training dataset
            objectives: Optional override of objectives
            **kwargs: Additional optimization parameters
            
        Returns:
            Final optimization state
        """
        if objectives:
            self.objectives = objectives
        
        if self.verbose:
            print(f"\n🎯 Starting multi-objective optimization...")
            print(f"   Objectives: {[obj.name for obj in self.objectives]}")
            print(f"   Training examples: {len(trainset)}")
        
        # Start resource monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        # Initialize optimization state
        self.optimization_state = OptimizationState(
            generation=0,
            population=[],
            frontier=[],
            metrics={},
            parameter_history=[],
            convergence_history=[],
            resource_usage={},
            start_time=time.time()
        )
        
        # Initialize current parameter state
        self.current_parameters = {
            "population_size": self.population_size,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "elitism_rate": 0.2
        }
        
        try:
            # Create initial population
            self._create_initial_population(program, trainset)
            
            # Run optimization generations
            for generation in range(self.max_generations):
                self.optimization_state.generation = generation
                
                if self.verbose:
                    print(f"\n📊 Generation {generation + 1}/{self.max_generations}")
                
                # Evaluate current population
                self._evaluate_population(trainset)
                
                # Update Pareto frontier
                self._update_pareto_frontier()
                
                # Calculate generation metrics
                self._calculate_generation_metrics()
                
                # Log generation
                if self.logger:
                    self.logger.log_generation_end(
                        generation, 
                        self.optimization_state.metrics
                    )
                
                # Check convergence
                if self._should_stop():
                    if self.verbose:
                        print(f"✅ Convergence detected at generation {generation + 1}")
                    break
                
                # Create next generation
                if generation < self.max_generations - 1:
                    self._create_next_generation()
                
                # Save checkpoint
                if self.checkpoint_manager and generation % 5 == 0:
                    checkpoint_id = f"gen_{generation}"
                    self.checkpoint_manager.save_checkpoint(
                        self._get_checkpoint_state(), 
                        checkpoint_id
                    )
            
            # Finalize optimization
            self._finalize_optimization()
            
            if self.verbose:
                self._print_final_results()
            
            return self.optimization_state
            
        finally:
            # Stop resource monitoring
            if self.resource_monitor:
                self.optimization_state.resource_usage = self.resource_monitor.stop_monitoring()
    
    def get_pareto_frontier(self) -> ParetoFrontierManager:
        """Get the current Pareto frontier."""
        return self.pareto_frontier
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history and metrics."""
        return self.optimization_history.copy()
    
    def _create_initial_population(self, program: Any, trainset: List[Any]) -> None:
        """Create initial population from the seed program."""
        if self.verbose:
            print(f"🌱 Creating initial population...")
        
        if not self.optimization_state:
            raise RuntimeError("Optimization state not initialized")
        
        population = []
        
        # Create variations of the initial program
        for i in range(self.population_size):
            solution_id = f"gen_0_sol_{i}"
            
            # Apply simple mutations to create diversity
            if i == 0:
                # Keep original as first solution
                candidate_program = program
                mutation_type = "original"
            else:
                candidate_program = self._apply_simple_mutation(program)
                mutation_type = "simple_mutation"
            
            # Evaluate the candidate
            evaluation_result = self._evaluate_solution(candidate_program, trainset, solution_id)
            evaluation_result.metadata = SolutionMetadata(
                generation=0,
                parent_ids=[],
                mutation_type=mutation_type
            )
            
            population.append(evaluation_result)
        
        self.optimization_state.population = population
        
        if self.verbose:
            print(f"   Created {len(population)} initial solutions")
    
    def _apply_simple_mutation(self, program: Any) -> Any:
        """Apply simple mutation to create program variation."""
        # This is a placeholder - in practice, you'd use the mutation operators
        # For now, return the program unchanged (real mutations would be applied later)
        return program
    
    def _evaluate_population(self, trainset: List[Any]) -> None:
        """Evaluate all solutions in the current population."""
        if not self.optimization_state:
            raise RuntimeError("Optimization state not initialized")
            
        for solution in self.optimization_state.population:
            if solution.evaluation_time is None:
                # Re-evaluate if not already evaluated
                updated_result = self._evaluate_solution(
                    # Extract program from solution (placeholder logic)
                    solution,
                    trainset,
                    solution.solution_id
                )
                solution.objectives = updated_result.objectives
                solution.evaluation_time = updated_result.evaluation_time
    
    def _evaluate_solution(self, program: Any, trainset: List[Any], solution_id: str) -> EvaluationResult:
        """Evaluate a solution against all objectives."""
        start_time = time.time()
        
        objectives = {}
        for objective in self.objectives:
            try:
                score = objective.evaluate(program, trainset)
                objectives[objective.name] = ObjectiveEvaluation(
                    objective_name=objective.name,
                    score=objective.validate_score(score),
                    direction=objective.direction,
                    evaluation_time=time.time() - start_time
                )
            except Exception as e:
                _logger.warning(f"Failed to evaluate objective {objective.name} for {solution_id}: {e}")
                objectives[objective.name] = ObjectiveEvaluation(
                    objective_name=objective.name,
                    score=0.0,
                    direction=objective.direction,
                    evaluation_time=time.time() - start_time
                )
        
        evaluation_time = time.time() - start_time
        if self.optimization_state:
            self.optimization_state.evaluation_count += 1
        
        return EvaluationResult(
            solution_id=solution_id,
            objectives=objectives,
            evaluation_time=evaluation_time
        )
    
    def _update_pareto_frontier(self) -> None:
        """Update Pareto frontier with current population."""
        if not self.optimization_state:
            raise RuntimeError("Optimization state not initialized")
            
        for solution in self.optimization_state.population:
            self.pareto_frontier.update_frontier(solution)
        
        self.optimization_state.frontier = self.pareto_frontier.get_frontier()
    
    def _calculate_generation_metrics(self) -> None:
        """Calculate metrics for the current generation."""
        if not self.optimization_state:
            raise RuntimeError("Optimization state not initialized")
            
        state = self.optimization_state
        
        # Basic metrics
        state.metrics = {
            "generation": state.generation,
            "population_size": len(state.population),
            "frontier_size": len(state.frontier),
            "evaluation_count": state.evaluation_count,
            "elapsed_time": time.time() - state.start_time
        }
        
        # Diversity metrics
        diversity_metrics = state.get_diversity_metrics()
        state.metrics.update(diversity_metrics)
        
        # Hypervolume
        if state.frontier:
            state.metrics["hypervolume"] = self.pareto_frontier.calculate_hypervolume()
        else:
            state.metrics["hypervolume"] = 0.0
        
        # Best scores for each objective
        for obj in self.objectives:
            best_solution = state.get_best_solution(obj.name)
            if best_solution:
                score = best_solution.get_objective_score(obj.name)
                if score is not None:
                    state.metrics[f"best_{obj.name}"] = score
        
        # Store in history
        self.optimization_history.append(state.metrics.copy())
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop."""
        # Check convergence
        if self.convergence_detector and self.optimization_state:
            if self.convergence_detector.has_converged({
                "population": self.optimization_state.population,
                "frontier": self.optimization_state.frontier,
                "metrics": self.optimization_state.metrics,
                "history": self.optimization_history
            }):
                return True
        
        # Check resource limits
        if self.resource_monitor:
            current_usage = self.resource_monitor.get_current_usage()
            # Simple resource check (can be customized)
            if current_usage.get("evaluations", 0) > 1000:  # Example limit
                return True
        
        return False
    
    def _create_next_generation(self) -> None:
        """Create the next generation using selection and mutation."""
        # This is a simplified implementation
        # In practice, you'd use the selection strategy and mutation operators
        
        if not self.optimization_state:
            raise RuntimeError("Optimization state not initialized")
            
        current_population = self.optimization_state.population
        new_population = []
        
        # Elite selection: keep best solutions from frontier
        elite_size = min(len(self.optimization_state.frontier), self.population_size // 4)
        if elite_size > 0:
            elite_solutions = self.optimization_state.frontier[:elite_size]
            new_population.extend(elite_solutions)
        
        # Fill remaining slots with mutated solutions
        while len(new_population) < self.population_size:
            # Select parent
            parent = random.choice(current_population)
            
            # Apply mutation (placeholder)
            child_solution_id = f"gen_{self.optimization_state.generation + 1}_sol_{len(new_population)}"
            child_evaluation = self._evaluate_solution(
                parent,  # Placeholder: should be actual mutated program
                [],  # Placeholder: should be actual trainset
                child_solution_id
            )
            
            child_evaluation.metadata = SolutionMetadata(
                generation=self.optimization_state.generation + 1,
                parent_ids=[parent.solution_id],
                mutation_type="genetic_mutation"
            )
            
            new_population.append(child_evaluation)
        
        self.optimization_state.population = new_population
    
    def _finalize_optimization(self) -> None:
        """Finalize the optimization process."""
        # Final evaluation of frontier
        self._update_pareto_frontier()
        
        # Calculate final metrics
        self._calculate_generation_metrics()
        
        # Save final checkpoint
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                self._get_checkpoint_state(),
                "final"
            )
    
    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        if not self.optimization_state:
            return {}
        
        return {
            "generation": self.optimization_state.generation,
            "population_size": len(self.optimization_state.population),
            "frontier_size": len(self.optimization_state.frontier),
            "metrics": self.optimization_state.metrics,
            "evaluation_count": self.optimization_state.evaluation_count,
            "objectives": [{"name": obj.name, "weight": obj.weight, "direction": obj.direction.value} for obj in self.objectives],
            "parameters": getattr(self, 'current_parameters', {}),
            "timestamp": time.time()
        }
    
    def _print_final_results(self) -> None:
        """Print final optimization results."""
        if not self.optimization_state:
            return
        
        print(f"\n🎉 Multi-objective optimization completed!")
        print(f"   Generations: {self.optimization_state.generation + 1}")
        print(f"   Evaluations: {self.optimization_state.evaluation_count}")
        print(f"   Frontier size: {len(self.optimization_state.frontier)}")
        print(f"   Hypervolume: {self.optimization_state.metrics.get('hypervolume', 0.0):.4f}")
        
        if self.optimization_state.frontier:
            print(f"\n📈 Best scores by objective:")
            for obj in self.objectives:
                best_solution = self.optimization_state.get_best_solution(obj.name)
                if best_solution:
                    score = best_solution.get_objective_score(obj.name)
                    print(f"   {obj.name}: {score:.4f}")


# Additional classes for architecture compliance

@dataclass
class ParetoFrontier:
    """Represents a Pareto frontier of non-dominated solutions."""
    solutions: List[EvaluationResult] = field(default_factory=list)
    hypervolume: float = 0.0
    generation: int = 0
    
    def update(self, candidate: EvaluationResult, objectives: List[Objective]) -> bool:
        """Update frontier with a candidate solution."""
        # Check if candidate is dominated
        dominated = False
        to_remove = []
        
        for existing in self.solutions:
            if self._dominates(existing, candidate, objectives):
                dominated = True
                break
            elif self._dominates(candidate, existing, objectives):
                to_remove.append(existing)
        
        # Remove dominated solutions
        for solution in to_remove:
            self.solutions.remove(solution)
        
        # Add candidate if not dominated
        if not dominated:
            self.solutions.append(candidate)
            return True
        
        return False
    
    def get_frontier(self) -> List[EvaluationResult]:
        """Get all solutions on the Pareto frontier."""
        return self.solutions.copy()
    
    def calculate_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """Calculate hypervolume of the Pareto frontier."""
        if not self.solutions:
            return 0.0
        
        # Simple hypervolume approximation
        try:
            import numpy as np
            
            # Extract objective values
            objective_names = set()
            for solution in self.solutions:
                objective_names.update(solution.objectives.keys())
            
            if not objective_names:
                return 0.0
            
            # Create matrix of objective values
            objective_matrix = []
            for solution in self.solutions:
                row = []
                for obj_name in objective_names:
                    score = solution.get_objective_score(obj_name)
                    row.append(score if score is not None else 0.0)
                objective_matrix.append(row)
            
            if not objective_matrix:
                return 0.0
            
            objective_matrix = np.array(objective_matrix)
            
            # Normalize to [0, 1] range
            min_vals = objective_matrix.min(axis=0)
            max_vals = objective_matrix.max(axis=0)
            ranges = max_vals - min_vals + 1e-8
            normalized = (objective_matrix - min_vals) / ranges
            
            # Simple hypervolume estimation
            hypervolume = 0.0
            for point in normalized:
                volume = 1.0
                for coord in point:
                    volume *= coord
                hypervolume += volume
            
            return hypervolume / len(normalized)
            
        except ImportError:
            # Fallback simple calculation
            return len(self.solutions) / 100.0
    
    def _dominates(self, sol1: EvaluationResult, sol2: EvaluationResult, 
                   objectives: List[Objective]) -> bool:
        """Check if sol1 dominates sol2."""
        at_least_one_better = False
        
        for obj in objectives:
            score1 = sol1.get_objective_score(obj.name)
            score2 = sol2.get_objective_score(obj.name)
            
            if score1 is None or score2 is None:
                continue
            
            if obj.direction == OptimizationDirection.MAXIMIZE:
                if score1 < score2:
                    return False
                elif score1 > score2:
                    at_least_one_better = True
            else:  # MINIMIZE
                if score1 > score2:
                    return False
                elif score1 < score2:
                    at_least_one_better = True
        
        return at_least_one_better


class CandidateSolution:
    """Represents a candidate solution in multi-objective optimization."""
    
    def __init__(self, solution_id: str, program: Any, objectives: Dict[str, float],
                 generation: int, parent_solutions: List[str] = None,
                 metadata: Dict[str, Any] = None):
        self.solution_id = solution_id
        self.program = program
        self.objectives = objectives
        self.generation = generation
        self.parent_solutions = parent_solutions or []
        self.metadata = metadata or {}
    
    def dominates(self, other: 'CandidateSolution') -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        at_least_one_better = False
        
        for obj_name, self_score in self.objectives.items():
            if obj_name in other.objectives:
                other_score = other.objectives[obj_name]
                
                if self_score < other_score:
                    return False  # Not better in this objective
                elif self_score > other_score:
                    at_least_one_better = True
            else:
                at_least_one_better = True
        
        return at_least_one_better
    
    def crowding_distance(self, frontier: 'ParetoFrontier') -> float:
        """Calculate crowding distance for diversity preservation."""
        solutions = frontier.get_frontier()
        if not solutions or len(solutions) <= 2:
            return float('inf')
        
        distance = 0.0
        
        # Calculate crowding distance for each objective
        for obj_name in self.objectives:
            # Extract scores for this objective
            scores = []
            solutions_with_scores = []
            
            for solution in solutions:
                if hasattr(solution, 'objectives') and obj_name in solution.objectives:
                    score = solution.objectives[obj_name]
                    scores.append(score)
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
        return self.objectives.get(objective_name)
    
    def __hash__(self) -> int:
        """Make solution hashable for use in sets/dicts."""
        return hash(self.solution_id)
    
    def __eq__(self, other) -> bool:
        """Check equality based on solution ID."""
        if not isinstance(other, CandidateSolution):
            return False
        return self.solution_id == other.solution_id