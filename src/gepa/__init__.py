"""GEPA (Genetic-Pareto Algorithm) for prompt optimization and text component evolution.

This package implements the GEPA algorithm described in the research paper,
which combines genetic algorithms with Pareto optimization for evolving
prompts, code, and other text components using multi-objective optimization.

Core Components:
- Candidate: Represents text components with fitness scores and metadata
- GeneticOptimizer: Main evolutionary loop orchestrator
- ParetoSelector: Multi-objective selection using Pareto dominance
- TextMutator: LLM-driven mutation strategies
"""

# Import local GEPA components with explicit relative imports
try:
    from .core.candidate import Candidate, ExecutionTrace, MutationRecord
    from .core.mutator import TextMutator, MutationStrategy, LLMReflectionMutator
    from .core.optimizer import GeneticOptimizer, OptimizationConfig
    from .core.selector import ParetoSelector
except ImportError:
    # Fallback for when running as installed package
    from gepa.core.candidate import Candidate, ExecutionTrace, MutationRecord
    from gepa.core.mutator import TextMutator, MutationStrategy, LLMReflectionMutator
    from gepa.core.optimizer import GeneticOptimizer, OptimizationConfig
    from gepa.core.selector import ParetoSelector

__version__ = "0.1.0"
__author__ = "GEPA Development Team"

__all__ = [
    # Core classes
    "Candidate",
    "GeneticOptimizer", 
    "ParetoSelector",
    "TextMutator",
    
    # Supporting classes
    "ExecutionTrace",
    "MutationRecord",
    "OptimizationConfig",
    "MutationStrategy",
    "LLMReflectionMutator",
]

# Quick start example
__example__ = """
# Basic usage of GEPA
from gepa import Candidate, GeneticOptimizer, ParetoSelector, TextMutator

def fitness_function(candidate: Candidate) -> Dict[str, float]:
    # Evaluate your candidate here
    return {"accuracy": 0.8, "efficiency": 0.7}

# Initialize optimizer
optimizer = GeneticOptimizer(
    objectives=["accuracy", "efficiency"],
    fitness_function=fitness_function
)

# Run optimization
initial_candidates = ["Prompt 1", "Prompt 2", "Prompt 3"]
best_candidates = optimizer.optimize(initial_candidates)

print(f"Found {len(best_candidates)} optimal solutions")
"""