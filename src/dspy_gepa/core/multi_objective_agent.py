"""Multi-objective GEPA Agent implementation.

This module provides an enhanced version of the GEPAAgent that supports
multi-objective optimization while maintaining backward compatibility
with existing single-objective functionality.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass

from .agent import GEPAAgent, AgentConfig, OptimizationSummary
from .multi_objective_gepa import MultiObjectiveGEPA, OptimizationState
from .mutation_engine import CompositeMutator, SemanticMutator, TaskSpecificMutator, AdaptiveRateMutator
from .parameter_tuner import ConvergenceBasedTuner, ResourceAwareTuner
from .interfaces import (
    Objective, TaskType, PreferenceVector, EvaluationResult,
    ParetoFrontierManager, MutationOperator, ParameterTuner
)
from ..utils.logging import get_logger


_logger = get_logger(__name__)


@dataclass
class MultiObjectiveOptimizationResult:
    """Result from multi-objective optimization."""
    optimization_state: OptimizationState
    pareto_frontier: ParetoFrontierManager
    optimization_history: List[Dict[str, Any]]
    time_elapsed: float
    
    def get_best_solution(self, objective_name: str) -> Optional[EvaluationResult]:
        """Get best solution for a specific objective."""
        return self.optimization_state.get_best_solution(objective_name)
    
    def get_balanced_solution(self) -> Optional[EvaluationResult]:
        """Get a balanced solution considering all objectives."""
        return self.pareto_frontier.select_solution()
    
    def get_preferred_solution(self, preferences: PreferenceVector) -> Optional[EvaluationResult]:
        """Get solution based on user preferences."""
        return self.pareto_frontier.select_solution(preferences)
    
    def get_pareto_frontier_solutions(self) -> List[EvaluationResult]:
        """Get all solutions on the Pareto frontier."""
        return self.pareto_frontier.get_frontier()
    
    def get_num_objectives(self) -> int:
        """Get number of objectives optimized."""
        return len(self.optimization_state.metrics)
    
    def get_hypervolume(self) -> float:
        """Get hypervolume of the Pareto frontier."""
        return self.pareto_frontier.calculate_hypervolume()
    
    def to_summary(self) -> OptimizationSummary:
        """Convert to single-objective summary for backward compatibility."""
        if not self.optimization_state.frontier:
            return OptimizationSummary(
                best_prompt="",
                best_score=0.0,
                initial_score=0.0,
                generations_completed=self.optimization_state.generation,
                total_evaluations=self.optimization_state.evaluation_count,
                optimization_time=self.time_elapsed,
                improvement=0.0
            )
        
        # Use the first frontier solution as "best" for compatibility
        best_solution = self.optimization_state.frontier[0]
        best_score = 0.0
        
        # Calculate average score across all objectives
        if best_solution.objectives:
            scores = [obj.score for obj in best_solution.objectives.values()]
            best_score = sum(scores) / len(scores)
        
        return OptimizationSummary(
            best_prompt=getattr(best_solution, 'solution_id', ''),
            best_score=best_score,
            initial_score=0.0,
            generations_completed=self.optimization_state.generation,
            total_evaluations=self.optimization_state.evaluation_count,
            optimization_time=self.time_elapsed,
            improvement=best_score
        )


class MultiObjectiveGEPAAgent(GEPAAgent):
    """Enhanced GEPAAgent with multi-objective optimization capabilities.
    
    This agent extends the original GEPAAgent to support multi-objective
    optimization while maintaining full backward compatibility.
    """
    
    def __init__(
        self,
        objectives: Optional[List[Objective]] = None,
        task_type: TaskType = TaskType.CUSTOM,
        max_generations: Optional[int] = None,
        population_size: Optional[int] = None,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
        mutation_operators: Optional[List[MutationOperator]] = None,
        parameter_tuner: Optional[ParameterTuner] = None,
        enable_multi_objective: bool = True,
        preference_vector: Optional[PreferenceVector] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any
    ):
        """Initialize Multi-Objective GEPAAgent.
        
        Args:
            objectives: List of objectives for multi-objective optimization
            task_type: Type of task being optimized
            max_generations: Maximum number of generations
            population_size: Population size for optimization
            config: Agent configuration
            mutation_operators: Custom mutation operators
            parameter_tuner: Parameter tuning strategy
            enable_multi_objective: Whether to use multi-objective optimization
            preference_vector: User preferences for solution selection
            **kwargs: Additional arguments passed to parent class
        """
        # Handle verbose parameter
        if verbose is not None:
            if config is None:
                config = {}
            if isinstance(config, dict):
                config['verbose'] = verbose
            else:
                # Update the config object if it's already an AgentConfig
                config.verbose = verbose
        
        # Initialize parent agent first
        super().__init__(
            max_generations=max_generations,
            population_size=population_size,
            config=config,
            **kwargs
        )
        
        self.task_type = task_type
        self.enable_multi_objective = enable_multi_objective
        self.preference_vector = preference_vector
        
        # Multi-objective components
        self.objectives = objectives or []
        self.mutation_operators = mutation_operators or self._create_default_mutators()
        self.parameter_tuner = parameter_tuner or self._create_default_parameter_tuner()
        
        # Multi-objective optimizer
        self.multi_optimizer: Optional[MultiObjectiveGEPA] = None
        
        # Results storage
        self.last_multi_objective_result: Optional[MultiObjectiveOptimizationResult] = None
        
        if self.enable_multi_objective and self.objectives:
            self._initialize_multi_objective_optimizer()
        
        if self.config.verbose:
            multi_obj_status = "enabled" if self.enable_multi_objective else "disabled"
            print(f"🎯 Multi-objective optimization: {multi_obj_status}")
            if self.enable_multi_objective:
                print(f"   Objectives: {len(self.objectives)}")
                print(f"   Task type: {task_type.value}")
    
    def _create_default_mutators(self) -> List[MutationOperator]:
        """Create default mutation operators based on task type."""
        mutators = [
            SemanticMutator(weight=1.0),
            TaskSpecificMutator(task_types=[self.task_type]),
            AdaptiveRateMutator(base_mutation_rate=0.1)
        ]
        
        return mutators
    
    def _create_default_parameter_tuner(self) -> ParameterTuner:
        """Create default parameter tuner."""
        base_tuner = ConvergenceBasedTuner(
            tuning_frequency=5,
            verbose=self.config.verbose
        )
        
        # Wrap with resource awareness if needed
        return ResourceAwareTuner(
            base_tuner=base_tuner,
            resource_limits={
                "evaluations": getattr(self.config, 'max_generations', 25) * getattr(self.config, 'population_size', 6) * 2,
                "time": 300.0  # 5 minutes
            }
        )
    
    def _initialize_multi_objective_optimizer(self) -> None:
        """Initialize the multi-objective optimizer."""
        if not self.objectives:
            raise ValueError("Objectives must be specified for multi-objective optimization")
        
        self.multi_optimizer = MultiObjectiveGEPA(
            objectives=self.objectives,
            max_generations=getattr(self.config, 'max_generations', 25),
            population_size=getattr(self.config, 'population_size', 6),
            mutation_operators=self.mutation_operators,
            parameter_tuner=self.parameter_tuner,
            verbose=getattr(self, 'verbose', True)
        )
        
        _logger.info("Multi-objective optimizer initialized")
    
    def optimize_prompt(
        self,
        initial_prompt: str,
        evaluation_fn: Callable[[str], float],
        generations: Optional[int] = None,
        return_summary: bool = True,
        enable_multi_objective: Optional[bool] = None
    ) -> Union[str, OptimizationSummary]:
        """Optimize a prompt using either single or multi-objective optimization.
        
        Args:
            initial_prompt: Starting prompt for optimization
            evaluation_fn: Function that evaluates a prompt and returns a score
            generations: Number of generations to run (overrides config)
            return_summary: Whether to return full summary
            enable_multi_objective: Override multi-objective setting
            
        Returns:
            Optimization result (type depends on configuration)
        """
        # Determine which optimization mode to use
        use_multi_objective = enable_multi_objective if enable_multi_objective is not None else self.enable_multi_objective
        
        if use_multi_objective and self.multi_optimizer and self.objectives:
            return self._optimize_prompt_multi_objective(
                initial_prompt, evaluation_fn, generations, return_summary
            )
        else:
            # Fall back to single-objective optimization
            return super().optimize_prompt(initial_prompt, evaluation_fn, generations, return_summary)
    
    def _optimize_prompt_multi_objective(
        self,
        initial_prompt: str,
        evaluation_fn: Callable[[str], float],
        generations: Optional[int] = None,
        return_summary: bool = True
    ) -> Union[str, OptimizationSummary, MultiObjectiveOptimizationResult]:
        """Optimize prompt using multi-objective optimization."""
        if not self.multi_optimizer:
            raise RuntimeError("Multi-objective optimizer not initialized")
        
        if self.config.verbose:
            print(f"\n🎯 Starting multi-objective prompt optimization...")
            print(f"   Objectives: {[obj.name for obj in self.objectives]}")
        
        start_time = time.time()
        
        # Create evaluation dataset (mock for prompt optimization)
        # In practice, this would be real evaluation data
        evaluation_dataset = [{"prompt": initial_prompt}]
        
        try:
            # Run multi-objective optimization
            optimization_state = self.multi_optimizer.optimize(
                program=initial_prompt,
                trainset=evaluation_dataset,
                max_generations=generations or getattr(self.config, 'max_generations', 25)
            )
            
            optimization_time = time.time() - start_time
            
            # Create result object
            result = MultiObjectiveOptimizationResult(
                optimization_state=optimization_state,
                pareto_frontier=self.multi_optimizer.get_pareto_frontier(),
                optimization_history=self.multi_optimizer.get_optimization_history(),
                time_elapsed=optimization_time
            )
            
            self.last_multi_objective_result = result
            
            if self.config.verbose:
                print(f"\n✅ Multi-objective optimization completed!")
                print(f"   Generations: {optimization_state.generation}")
                print(f"   Evaluations: {optimization_state.evaluation_count}")
                print(f"   Frontier size: {len(result.get_pareto_frontier_solutions())}")
                print(f"   Hypervolume: {result.get_hypervolume():.4f}")
            
            # Return based on requested format
            if not return_summary:
                # Return the best solution based on preferences
                if self.preference_vector:
                    solution = result.get_preferred_solution(self.preference_vector)
                else:
                    solution = result.get_balanced_solution()
                
                return getattr(solution, 'solution_id', '') if solution else initial_prompt
            
            # Convert to standard summary for backward compatibility
            return result.to_summary()
            
        except Exception as e:
            _logger.error(f"Multi-objective optimization failed: {e}")
            if self.config.verbose:
                print(f"❌ Multi-objective optimization failed: {e}")
            
            # Fall back to single-objective
            if self.config.verbose:
                print("🔄 Falling back to single-objective optimization...")
            
            return super().optimize_prompt(initial_prompt, evaluation_fn, generations, return_summary)
    
    def add_objective(self, objective: Objective) -> None:
        """Add a new optimization objective."""
        self.objectives.append(objective)
        
        # Reinitialize optimizer if needed
        if self.enable_multi_objective:
            self._initialize_multi_objective_optimizer()
        
        _logger.info(f"Added objective: {objective.name}")
    
    def remove_objective(self, objective_name: str) -> bool:
        """Remove an optimization objective."""
        for i, obj in enumerate(self.objectives):
            if obj.name == objective_name:
                removed_obj = self.objectives.pop(i)
                
                # Reinitialize optimizer if needed
                if self.enable_multi_objective:
                    if self.objectives:
                        self._initialize_multi_objective_optimizer()
                    else:
                        self.multi_optimizer = None
                
                _logger.info(f"Removed objective: {objective_name}")
                return True
        
        return False
    
    def set_preferences(self, preferences: PreferenceVector) -> None:
        """Set user preferences for solution selection."""
        self.preference_vector = preferences
        _logger.info(f"Set preferences with {len(preferences.weights)} objectives")
    
    def get_pareto_frontier(self) -> Optional[ParetoFrontierManager]:
        """Get the current Pareto frontier."""
        if self.multi_optimizer:
            return self.multi_optimizer.get_pareto_frontier()
        return None
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get comprehensive optimization insights."""
        base_insights = super().get_optimization_insights()
        
        if not self.enable_multi_objective or not self.last_multi_objective_result:
            return base_insights
        
        # Add multi-objective specific insights
        mo_insights = {
            "multi_objective_enabled": True,
            "objectives_count": len(self.objectives),
            "objectives": [obj.name for obj in self.objectives],
            "pareto_frontier_size": len(self.last_multi_objective_result.get_pareto_frontier_solutions()),
            "hypervolume": self.last_multi_objective_result.get_hypervolume(),
            "optimization_history_size": len(self.last_multi_objective_result.optimization_history)
        }
        
        # Add objective-specific best scores
        for obj in self.objectives:
            best_solution = self.last_multi_objective_result.get_best_solution(obj.name)
            if best_solution:
                mo_insights[f"best_{obj.name}"] = best_solution.get_objective_score(obj.name)
        
        # Merge insights
        base_insights.update(mo_insights)
        
        return base_insights
    
    def enable_multi_objective_mode(self, objectives: List[Objective]) -> None:
        """Enable multi-objective optimization mode."""
        self.objectives = objectives
        self.enable_multi_objective = True
        
        if objectives:
            self._initialize_multi_objective_optimizer()
        
        _logger.info(f"Multi-objective mode enabled with {len(objectives)} objectives")
    
    def disable_multi_objective_mode(self) -> None:
        """Disable multi-objective optimization."""
        self.enable_multi_objective = False
        self.multi_optimizer = None
        
        _logger.info("Multi-objective mode disabled")
    
    def get_tuning_statistics(self) -> Optional[Dict[str, Any]]:
        """Get parameter tuning statistics."""
        if self.parameter_tuner and hasattr(self.parameter_tuner, 'get_tuning_statistics'):
            try:
                return self.parameter_tuner.get_tuning_statistics()
            except:
                pass
        return None
    
    def get_mutation_statistics(self) -> Optional[Dict[str, Any]]:
        """Get mutation operator statistics."""
        if self.multi_optimizer and hasattr(self.multi_optimizer, 'mutation_operators'):
            stats = {}
            try:
                for mutator in self.multi_optimizer.mutation_operators:
                    if hasattr(mutator, 'get_usage_stats'):
                        stats[mutator.name] = mutator.get_usage_stats()
                return stats
            except:
                pass
        return None
    
    def export_optimization_state(self) -> Optional[Dict[str, Any]]:
        """Export current optimization state for analysis."""
        if not self.last_multi_objective_result:
            return None
        
        return {
            "optimization_state": self.last_multi_objective_result.optimization_state,
            "pareto_frontier": self.last_multi_objective_result.get_pareto_frontier_solutions(),
            "history": self.last_multi_objective_result.optimization_history,
            "objectives": [
                {
                    "name": obj.name,
                    "weight": obj.weight,
                    "direction": obj.direction.value,
                    "description": obj.description
                }
                for obj in self.objectives
            ],
            "task_type": self.task_type.value,
            "preferences": (
                {
                    "weights": self.preference_vector.weights,
                    "constraints": self.preference_vector.constraints
                }
                if self.preference_vector else None
            )
        }
    
    def __repr__(self) -> str:
        """String representation of the multi-objective agent."""
        multi_obj_indicator = "🎯" if self.enable_multi_objective else "🔧"
        llm_indicator = "🤖" if self.is_llm_available() else "🔧"
        
        return (
            f"{multi_obj_indicator} MultiObjectiveGEPAAgent{llm_indicator}("
            f"generations={self.config.max_generations}, "
            f"population={self.config.population_size}, "
            f"objectives={len(self.objectives)})"
        )


# Convenience function for creating multi-objective agents
def create_multi_objective_agent(
    objectives: List[Objective],
    task_type: TaskType = TaskType.CUSTOM,
    max_generations: int = 30,
    population_size: int = 20,
    verbose: bool = True,
    **kwargs: Any
) -> MultiObjectiveGEPAAgent:
    """Create a multi-objective GEPA agent with sensible defaults.
    
    Args:
        objectives: List of objectives to optimize
        task_type: Type of task being optimized
        max_generations: Maximum number of generations
        population_size: Population size for optimization
        verbose: Whether to show progress information
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Configured MultiObjectiveGEPAAgent instance
    """
    return MultiObjectiveGEPAAgent(
        objectives=objectives,
        task_type=task_type,
        max_generations=max_generations,
        population_size=population_size,
        verbose=verbose,
        **kwargs
    )


# Backward compatibility aliases
MOGEPAAgent = MultiObjectiveGEPAAgent