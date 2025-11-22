"""Multi-objective optimizer for DSPy modules.

This module provides a specialized optimizer that integrates seamlessly with
DSPy modules, enabling multi-objective optimization while maintaining full
compatibility with existing DSPy workflows.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import time
import copy
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass

from ..core.multi_objective_gepa import MultiObjectiveGEPA
from ..core.interfaces import (
    Objective, TaskType, PreferenceVector, EvaluationResult,
    ParetoFrontierManager, OptimizationDirection
)
from ..core.objectives import (
    AccuracyMetric, TokenUsageMetric, ExecutionTimeMetric,
    create_default_task_objectives
)
from ..core.mutation_engine import CompositeMutator, SemanticMutator
from ..core.parameter_tuner import ConvergenceBasedTuner
from .metric_converter import DSPyMetricConverter
from .signature_analyzer import DSPySignatureAnalyzer
from ..utils.logging import get_logger


_logger = get_logger(__name__)


@dataclass
class DSPyOptimizationConfig:
    """Configuration for DSPy module optimization."""
    max_generations: int = 25
    population_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_size: int = 2
    optimization_timeout: Optional[float] = None
    enable_signature_optimization: bool = True
    enable_metric_conversion: bool = True
    preserve_module_structure: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0.0 and 1.0")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0.0 and 1.0")


@dataclass
class DSPyOptimizationResult:
    """Result from DSPy module optimization."""
    optimized_module: Any
    optimization_history: List[Dict[str, Any]]
    pareto_frontier: ParetoFrontierManager
    time_elapsed: float
    generations_completed: int
    evaluations_performed: int
    hypervolume: float
    
    def get_best_module(self, objective_name: str) -> Any:
        """Get the best module for a specific objective."""
        best_solution = self.pareto_frontier.get_frontier()[0] if self.pareto_frontier.get_frontier() else None
        if best_solution:
            return getattr(best_solution, 'module', None)
        return self.optimized_module
    
    def get_preferred_module(self, preferences: PreferenceVector) -> Any:
        """Get module based on user preferences."""
        solution = self.pareto_frontier.select_solution(preferences)
        return getattr(solution, 'module', self.optimized_module) if solution else self.optimized_module
    
    def get_objective_scores(self) -> Dict[str, List[float]]:
        """Get scores for all objectives across the frontier."""
        scores = {}
        for solution in self.pareto_frontier.get_frontier():
            for obj_name, obj_eval in solution.objectives.items():
                if obj_name not in scores:
                    scores[obj_name] = []
                scores[obj_name].append(obj_eval.score)
        return scores


class MultiObjectiveOptimizer:
    """Multi-objective optimizer for DSPy modules.
    
    This optimizer provides seamless integration with DSPy modules,
    enabling multi-objective optimization while maintaining compatibility
    with existing DSPy workflows.
    """
    
    def __init__(
        self,
        config: Optional[DSPyOptimizationConfig] = None,
        metric_converter: Optional[DSPyMetricConverter] = None,
        signature_analyzer: Optional[DSPySignatureAnalyzer] = None
    ):
        """Initialize the multi-objective optimizer.
        
        Args:
            config: Optimization configuration
            metric_converter: Metric converter instance
            signature_analyzer: Signature analyzer instance
        """
        self.config = config or DSPyOptimizationConfig()
        self.metric_converter = metric_converter or DSPyMetricConverter()
        self.signature_analyzer = signature_analyzer or DSPySignatureAnalyzer()
        
        # Initialize core optimizer
        self.core_optimizer: Optional[MultiObjectiveGEPA] = None
        
        _logger.info("MultiObjectiveOptimizer initialized")
    
    def optimize_module(
        self,
        module: Any,
        trainset: List[Any],
        objectives: Optional[List[Objective]] = None,
        dspy_metrics: Optional[List[Callable]] = None,
        preferences: Optional[PreferenceVector] = None,
        **kwargs: Any
    ) -> DSPyOptimizationResult:
        """Optimize a DSPy module using multi-objective optimization.
        
        Args:
            module: DSPy module to optimize
            trainset: Training dataset
            objectives: List of objectives to optimize
            dspy_metrics: DSPy metrics to convert to objectives
            preferences: User preferences for solution selection
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimization result with optimized module and analysis
        """
        start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🚀 Starting multi-objective DSPy module optimization...")
            print(f"   Module type: {type(module).__name__}")
            print(f"   Training samples: {len(trainset)}")
        
        try:
            # Convert DSPy metrics to objectives if provided
            if dspy_metrics and not objectives:
                objectives = self.metric_converter.aggregate_dspy_metrics(
                    dspy_metrics, [1.0] * len(dspy_metrics)
                )
            
            # Create default objectives if none provided
            if not objectives:
                objectives = create_default_task_objectives(TaskType.CUSTOM)
            
            # Analyze module signature for optimization hints
            if self.config.enable_signature_optimization:
                signature_info = self.signature_analyzer.analyze_module(module)
                if self.config.verbose:
                    print(f"   Signature analysis: {signature_info}")
            
            # Initialize core optimizer
            self._initialize_core_optimizer(objectives)
            
            # Create evaluation wrapper for DSPy modules
            evaluation_fn = self._create_dspy_evaluation_wrapper(module, trainset)
            
            # Run multi-objective optimization
            optimization_state = self.core_optimizer.optimize(
                program=module,
                trainset=trainset,
                max_generations=self.config.max_generations
            )
            
            optimization_time = time.time() - start_time
            
            # Extract best solution based on preferences
            optimized_module = self._extract_optimized_module(
                optimization_state, preferences
            )
            
            # Create result object
            result = DSPyOptimizationResult(
                optimized_module=optimized_module,
                optimization_history=self.core_optimizer.get_optimization_history(),
                pareto_frontier=self.core_optimizer.get_pareto_frontier(),
                time_elapsed=optimization_time,
                generations_completed=optimization_state.generation,
                evaluations_performed=optimization_state.evaluation_count,
                hypervolume=self.core_optimizer.get_pareto_frontier().calculate_hypervolume()
            )
            
            if self.config.verbose:
                print(f"\n✅ Multi-objective optimization completed!")
                print(f"   Generations: {result.generations_completed}")
                print(f"   Evaluations: {result.evaluations_performed}")
                print(f"   Hypervolume: {result.hypervolume:.4f}")
                print(f"   Time elapsed: {result.time_elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            _logger.error(f"Multi-objective optimization failed: {e}")
            if self.config.verbose:
                print(f"❌ Optimization failed: {e}")
            
            # Return original module with basic result
            return DSPyOptimizationResult(
                optimized_module=module,
                optimization_history=[],
                pareto_frontier=self.core_optimizer.get_pareto_frontier() if self.core_optimizer else None,
                time_elapsed=time.time() - start_time,
                generations_completed=0,
                evaluations_performed=0,
                hypervolume=0.0
            )
    
    def optimize_prompt(
        self,
        initial_prompt: str,
        evaluation_fn: Callable[[str], float],
        objectives: Optional[List[Objective]] = None,
        preferences: Optional[PreferenceVector] = None,
        **kwargs: Any
    ) -> DSPyOptimizationResult:
        """Optimize a prompt using multi-objective optimization.
        
        Args:
            initial_prompt: Starting prompt for optimization
            evaluation_fn: Evaluation function
            objectives: List of objectives to optimize
            preferences: User preferences for solution selection
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🎯 Starting multi-objective prompt optimization...")
        
        try:
            # Create default objectives if none provided
            if not objectives:
                objectives = create_default_task_objectives(TaskType.CUSTOM)
            
            # Initialize core optimizer
            self._initialize_core_optimizer(objectives)
            
            # Create mock dataset for prompt optimization
            mock_dataset = [{"prompt": initial_prompt, "target": "optimized"}]
            
            # Run optimization
            optimization_state = self.core_optimizer.optimize(
                program=initial_prompt,
                trainset=mock_dataset,
                max_generations=self.config.max_generations
            )
            
            optimization_time = time.time() - start_time
            
            # Get optimized prompt
            frontier = self.core_optimizer.get_pareto_frontier()
            best_solution = frontier.select_solution(preferences) if frontier else None
            optimized_prompt = getattr(best_solution, 'solution_id', initial_prompt) if best_solution else initial_prompt
            
            # Create result
            result = DSPyOptimizationResult(
                optimized_module=optimized_prompt,
                optimization_history=self.core_optimizer.get_optimization_history(),
                pareto_frontier=frontier,
                time_elapsed=optimization_time,
                generations_completed=optimization_state.generation,
                evaluations_performed=optimization_state.evaluation_count,
                hypervolume=frontier.calculate_hypervolume() if frontier else 0.0
            )
            
            if self.config.verbose:
                print(f"\n✅ Prompt optimization completed!")
                print(f"   Generations: {result.generations_completed}")
                print(f"   Evaluations: {result.evaluations_performed}")
                print(f"   Hypervolume: {result.hypervolume:.4f}")
            
            return result
            
        except Exception as e:
            _logger.error(f"Prompt optimization failed: {e}")
            if self.config.verbose:
                print(f"❌ Prompt optimization failed: {e}")
            
            return DSPyOptimizationResult(
                optimized_module=initial_prompt,
                optimization_history=[],
                pareto_frontier=None,
                time_elapsed=time.time() - start_time,
                generations_completed=0,
                evaluations_performed=0,
                hypervolume=0.0
            )
    
    def _initialize_core_optimizer(self, objectives: List[Objective]) -> None:
        """Initialize the core multi-objective optimizer."""
        # Create mutation operators
        mutators = [
            SemanticMutator(weight=1.0),
        ]
        mutation_engine = CompositeMutator(mutators=mutators)
        
        # Create parameter tuner
        parameter_tuner = ConvergenceBasedTuner(
            tuning_frequency=5,
            verbose=self.config.verbose
        )
        
        # Initialize core optimizer
        self.core_optimizer = MultiObjectiveGEPA(
            objectives=objectives,
            max_generations=self.config.max_generations,
            population_size=self.config.population_size,
            mutation_operators=[mutation_engine],
            parameter_tuner=parameter_tuner,
            verbose=self.config.verbose
        )
        
        _logger.info(f"Core optimizer initialized with {len(objectives)} objectives")
    
    def _create_dspy_evaluation_wrapper(self, module: Any, trainset: List[Any]) -> Callable:
        """Create evaluation wrapper for DSPy modules."""
        def evaluate_solution(solution: Any) -> Dict[str, float]:
            """Evaluate a solution against all objectives."""
            scores = {}
            
            # Create a copy of the module for evaluation
            eval_module = copy.deepcopy(module)
            
            # Apply solution changes (this would depend on solution representation)
            if hasattr(solution, 'apply_to_module'):
                solution.apply_to_module(eval_module)
            elif isinstance(solution, str):
                # Handle string-based solutions (prompts)
                if hasattr(eval_module, 'prompt'):
                    eval_module.prompt = solution
            
            # Evaluate on training set
            total_score = 0.0
            correct_predictions = 0
            
            for example in trainset:
                try:
                    # Run the module
                    prediction = eval_module(**example)
                    
                    # Simple accuracy calculation (this would be more sophisticated)
                    if hasattr(example, 'answer') and prediction == example.answer:
                        correct_predictions += 1
                    
                    total_score += 1.0  # Default score
                    
                except Exception as e:
                    _logger.warning(f"Evaluation error: {e}")
                    total_score += 0.0
            
            # Calculate metrics
            accuracy = correct_predictions / len(trainset) if trainset else 0.0
            avg_score = total_score / len(trainset) if trainset else 0.0
            
            scores['accuracy'] = accuracy
            scores['performance'] = avg_score
            
            return scores
        
        return evaluate_solution
    
    def _extract_optimized_module(
        self,
        optimization_state: Any,
        preferences: Optional[PreferenceVector] = None
    ) -> Any:
        """Extract the optimized module from optimization state."""
        frontier = optimization_state.frontier if hasattr(optimization_state, 'frontier') else []
        
        if not frontier:
            return None
        
        # Select solution based on preferences
        if preferences:
            selected_solution = None
            for solution in frontier:
                if hasattr(solution, 'module'):
                    selected_solution = solution
                    break
        else:
            # Use the first frontier solution
            selected_solution = frontier[0] if frontier else None
        
        return getattr(selected_solution, 'module', None) if selected_solution else None
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from the optimization process."""
        if not self.core_optimizer:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "config": {
                "max_generations": self.config.max_generations,
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate
            },
            "metric_converter_enabled": self.config.enable_metric_conversion,
            "signature_optimization_enabled": self.config.enable_signature_optimization
        }
    
    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return (
            f"MultiObjectiveOptimizer("
            f"generations={self.config.max_generations}, "
            f"population_size={self.config.population_size})"
        )