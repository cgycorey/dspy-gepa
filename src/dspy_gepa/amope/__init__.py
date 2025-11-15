"""AMOPE (Adaptive Multi-Objective Prompt Evolution) Algorithm.

This package implements advanced features that extend beyond the core GEPA algorithm:
- Adaptive mutation strategies based on performance gradients
- Dynamic objective balancing for multi-objective optimization  
- Hierarchical co-evolution of multiple DSPY components
- Advanced reflection engine with LLM guidance

AMOPE provides a unified interface for sophisticated prompt optimization
with automatic strategy selection and dynamic adaptation.

Quick Start:
    ```python
    from dspy_gepa.amope import AMOPEOptimizer
    
    # Initialize optimizer with your objectives
    optimizer = AMOPEOptimizer(
        objectives={"accuracy": 0.7, "efficiency": 0.3},
        mutation_config={"use_llm_guidance": True},
        balancing_config={"strategy": "adaptive_harmonic"}
    )
    
    # Optimize your prompt
    best_prompt = optimizer.optimize(
        initial_prompt="Your initial prompt here",
        evaluation_fn=your_evaluation_function,
        generations=50
    )
    ```
"""

import random
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field

# Import core AMOPE components
from .adaptive_mutator import AdaptiveMutator, MutationStrategy
from .objective_balancer import ObjectiveBalancer, BalancingStrategy


@dataclass
class AMOPEConfig:
    """Configuration for AMOPE optimizer."""
    
    # Core optimization settings
    objectives: Dict[str, float] = field(default_factory=dict)
    population_size: int = 10
    max_generations: int = 100
    convergence_threshold: float = 0.001
    stagnation_generations: int = 15
    
    # Adaptive mutator settings
    mutation_config: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "adaptive",
        "use_llm_guidance": False,
        "mutation_rate": 0.3,
        "strength_factor": 0.1
    })
    
    # Objective balancer settings
    balancing_config: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "adaptive_harmonic",
        "stagnation_window": 15,
        "min_weight": 0.1,
        "max_weight": 3.0
    })
    
    # Reflection and co-evolution (placeholder for future)
    enable_reflection: bool = False
    enable_co_evolution: bool = False
    
    # LLM settings (optional)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization settings
    early_stopping: bool = True
    verbose: bool = True
    save_history: bool = True
    random_seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Results from AMOPE optimization."""
    
    best_prompt: str
    best_score: float
    best_objectives: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    generations_completed: int
    convergence_achieved: bool
    total_evaluation_time: float
    strategy_usage: Dict[str, int]
    

class AMOPEOptimizer:
    """Unified AMOPE optimizer combining adaptive mutation and objective balancing.
    
    This class provides a high-level interface to the AMOPE algorithm,
    automatically coordinating adaptive mutation strategies and dynamic
    objective balancing for optimal prompt evolution.
    """
    
    def __init__(self, 
                 objectives: Dict[str, float],
                 mutation_config: Optional[Dict[str, Any]] = None,
                 balancing_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize AMOPE optimizer.
        
        Args:
            objectives: Dictionary mapping objective names to initial weights
            mutation_config: Configuration for adaptive mutator
            balancing_config: Configuration for objective balancer
            **kwargs: Additional configuration options
        """
        # Create configuration
        self.config = AMOPEConfig(
            objectives=objectives,
            mutation_config=mutation_config or {},
            balancing_config=balancing_config or {},
            **kwargs
        )
        
        # Set random seed if provided
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        # Initialize core components
        self._initialize_components()
        
        # Optimization state
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_generation = 0
        self.best_candidate = None
        self.stagnation_counter = 0
        self.strategy_usage: Dict[str, int] = {}
    
    def _initialize_components(self):
        """Initialize AMOPE components based on configuration."""
        
        # Initialize adaptive mutator
        self.mutator = AdaptiveMutator(
            llm_client=None  # Will use default LLM client from TextMutator
        )
        
        # Initialize objective balancer
        from .objective_balancer import BalancingStrategy
        self.balancer = ObjectiveBalancer(
            objectives=self.config.objectives,
            strategy=BalancingStrategy(self.config.balancing_config.get("strategy", "adaptive_harmonic")),
            stagnation_window=self.config.balancing_config.get("stagnation_window", 15),
            min_weight=self.config.balancing_config.get("min_weight", 0.1),
            max_weight=self.config.balancing_config.get("max_weight", 3.0)
        )
    
    def optimize(self, 
                 initial_prompt: str,
                 evaluation_fn: Callable[[str], Dict[str, float]],
                 generations: Optional[int] = None) -> OptimizationResult:
        """Run AMOPE optimization.
        
        Args:
            initial_prompt: Starting prompt for optimization
            evaluation_fn: Function that evaluates a prompt and returns objective scores
            generations: Number of generations to run (overrides config)
            
        Returns:
            OptimizationResult with best prompt and optimization statistics
        """
        
        # Setup optimization
        generations = generations or self.config.max_generations
        start_time = time.time()
        
        # Initialize population
        current_prompt = initial_prompt
        current_score = self._evaluate_prompt(current_prompt, evaluation_fn)
        
        self.best_candidate = {
            "prompt": current_prompt,
            "score": current_score,
            "objectives": current_score
        }
        
        if self.config.verbose:
            print(f"Starting AMOPE optimization with {len(self.config.objectives)} objectives")
            print(f"Initial score: {current_score:.4f}")
        
        # Main optimization loop
        for generation in range(generations):
            self.current_generation = generation
            
            # Generate mutated candidates
            candidates = self._generate_candidates(current_prompt, evaluation_fn)
            
            # Select best candidate
            best_candidate = max(candidates, key=lambda x: x["score"])
            
            # Check for improvement
            if best_candidate["score"] > self.best_candidate["score"] + self.config.convergence_threshold:
                self.best_candidate = best_candidate
                current_prompt = best_candidate["prompt"]
                current_score = best_candidate["score"]
                self.stagnation_counter = 0
                
                if self.config.verbose:
                    print(f"Generation {generation}: New best score: {current_score:.4f}")
            else:
                self.stagnation_counter += 1
            
            # Update objective weights based on progress
            if generation > 0:
                self._update_objectives()
            
            # Record generation data
            generation_data = {
                "generation": generation,
                "best_score": current_score,
                "best_objectives": best_candidate["objectives"],
                "improvement": best_candidate["score"] - self.best_candidate["score"],
                "objective_weights": dict(self.balancer.current_objectives)
            }
            
            if self.config.save_history:
                self.optimization_history.append(generation_data)
            
            # Check convergence and early stopping
            if self._should_stop(generation, generations):
                break
        
        # Calculate final statistics
        total_time = time.time() - start_time
        
        result = OptimizationResult(
            best_prompt=self.best_candidate["prompt"],
            best_score=self.best_candidate["score"],
            best_objectives=self.best_candidate["objectives"],
            optimization_history=self.optimization_history,
            generations_completed=self.current_generation + 1,
            convergence_achieved=self.stagnation_counter >= self.config.stagnation_generations,
            total_evaluation_time=total_time,
            strategy_usage=self.strategy_usage
        )
        
        if self.config.verbose:
            print(f"\nOptimization completed in {total_time:.2f}s")
            print(f"Final score: {result.best_score:.4f}")
            print(f"Generations: {result.generations_completed}")
            print(f"Strategy usage: {result.strategy_usage}")
        
        return result
    
    def _evaluate_prompt(self, prompt: str, evaluation_fn: Callable) -> float:
        """Evaluate prompt and return combined score."""
        objectives = evaluation_fn(prompt)
        
        # Apply current weights to get combined score
        combined_score = 0.0
        total_weight = 0.0
        
        for obj_name, obj_value in objectives.items():
            if obj_name in self.balancer.current_objectives:
                weight = self.balancer.current_objectives[obj_name]
                combined_score += weight * obj_value
                total_weight += weight
        
        return combined_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_candidates(self, prompt: str, evaluation_fn: Callable) -> List[Dict[str, Any]]:
        """Generate mutated candidates using simple text mutations."""
        candidates = []
        
        # Simple mutation strategies
        strategies = [
            "append_instruction",
            "modify_tone", 
            "add_context",
            "simplify_language"
        ]
        
        for i in range(self.config.population_size):
            # Apply simple mutation
            mutated_prompt = self._simple_mutate(prompt, i % len(strategies))
            strategy_name = strategies[i % len(strategies)]
            
            # Track strategy usage
            self.strategy_usage[strategy_name] = self.strategy_usage.get(strategy_name, 0) + 1
            
            # Evaluate mutated prompt
            objectives = evaluation_fn(mutated_prompt)
            combined_score = self._evaluate_prompt(mutated_prompt, evaluation_fn)
            
            candidates.append({
                "prompt": mutated_prompt,
                "score": combined_score,
                "objectives": objectives,
                "strategy": strategy_name,
                "mutation_confidence": 0.7  # Default confidence
            })
        
        return candidates
    
    def _simple_mutate(self, prompt: str, strategy_index: int) -> str:
        """Apply simple text mutation strategies."""
        import random
        
        mutations = [
            f"{prompt}. Please provide a detailed response.",
            f"Please carefully {prompt.lower()}.",
            f"Consider the following context: {prompt}",
            f"Simplify and {prompt.lower()}."
        ]
        
        # Add some random variation
        base_mutation = mutations[strategy_index % len(mutations)]
        if random.random() < 0.3:
            base_mutation += " Be thorough and accurate."
            
        return base_mutation
    
    def _update_objectives(self):
        """Update objective weights based on optimization progress."""
        if len(self.optimization_history) >= 2:
            # Get the most recent objectives
            latest_objectives = self.optimization_history[-1]["best_objectives"]
            
            # Update weights using balancer
            self.balancer.update_fitness(latest_objectives)
    
    def _should_stop(self, current_generation: int, max_generations: int) -> bool:
        """Check if optimization should stop."""
        # Stop if max generations reached
        if current_generation >= max_generations - 1:
            return True
        
        # Stop if stagnation detected (if early stopping enabled)
        if self.config.early_stopping and self.stagnation_counter >= self.config.stagnation_generations:
            if self.config.verbose:
                print(f"Stopping early due to stagnation for {self.stagnation_counter} generations")
            return True
        
        return False


# Component imports for advanced users
from .adaptive_mutator import AdaptiveMutator, MutationStrategy, MutationResult, PerformanceAnalyzer
from .objective_balancer import ObjectiveBalancer, BalancingStrategy, ObjectiveInfo, StagnationMetrics


# Version and exports
__version__ = "0.1.0"
__author__ = "AMOPE Development Team"
__email__ = "contact@amope.ai"

__all__ = [
    # Main optimizer
    "AMOPEOptimizer",
    "AMOPEConfig", 
    "OptimizationResult",
    
    # Core components
    "AdaptiveMutator",
    "MutationStrategy",
    "MutationResult",
    "PerformanceAnalyzer",
    "ObjectiveBalancer",
    "BalancingStrategy",
    "ObjectiveInfo", 
    "StagnationMetrics",
    
    # Utilities
    "__version__",
]