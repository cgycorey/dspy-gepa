"""Genetic optimizer for GEPA algorithm.

This module implements the GeneticOptimizer class which orchestrates the main
evolutionary loop in the GEPA (Genetic-Pareto Algorithm) framework. It combines
genetic evolution with Pareto-optimal selection and LLM-driven mutations.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .candidate import Candidate, ExecutionTrace
from .mutator import TextMutator
from .selector import ParetoSelector


class OptimizationConfig:
    """Configuration for genetic optimization."""
    
    def __init__(self,
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.8,
                 crossover_rate: float = 0.6,
                 tournament_size: int = 2,
                 elite_size: int = 5,
                 early_stop_generations: int = 20,
                 fitness_threshold: float = 0.95):
        """Initialize optimization configuration.
        
        Args:
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of selection tournaments
            elite_size: Number of elite candidates to preserve
            early_stop_generations: Generations to wait for improvement before early stopping
            fitness_threshold: Threshold for considering optimization successful
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.early_stop_generations = early_stop_generations
        self.fitness_threshold = fitness_threshold


class GeneticOptimizer:
    """Main genetic optimizer for the GEPA algorithm.
    
    This class orchestrates the evolutionary process, managing population
    evolution, fitness evaluation, and convergence tracking.
    
    Attributes:
        config: Optimization configuration
        selector: Pareto selector for candidate selection
        mutator: Text mutator for applying mutations
        fitness_function: Function to evaluate candidate fitness
        objectives: List of optimization objectives
        population: Current population of candidates
        generation_history: History of best candidates per generation
        logger: Logger for optimization progress
    """
    
    def __init__(self,
                 objectives: List[str],
                 fitness_function: Callable[[Candidate], Dict[str, float]],
                 config: Optional[OptimizationConfig] = None,
                 llm_client: Optional[Any] = None,
                 maximize_objectives: Optional[Dict[str, bool]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the genetic optimizer.
        
        Args:
            objectives: List of objective names to optimize
            fitness_function: Function to evaluate candidate fitness
            config: Optimization configuration
            llm_client: LLM client for intelligent mutations
            maximize_objectives: Dictionary indicating whether to maximize each objective
            logger: Logger for progress tracking
        """
        self.config = config or OptimizationConfig()
        self.fitness_function = fitness_function
        self.objectives = objectives
        
        # Initialize components
        self.selector = ParetoSelector(objectives, maximize_objectives)
        self.mutator = TextMutator(llm_client=llm_client)
        
        # State management
        self.population: List[Candidate] = []
        self.generation_history: List[List[Candidate]] = []
        self.best_candidates: List[Candidate] = []
        self.current_generation = 0
        
        # Convergence tracking
        self.no_improvement_count = 0
        self.best_fitness_history: Dict[str, List[float]] = {obj: [] for obj in objectives}
        
        # Logging
        self.logger = logger or self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger("GeneticOptimizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def initialize_population(self, initial_candidates: List[str]) -> None:
        """Initialize the population with candidate content.
        
        Args:
            initial_candidates: List of initial candidate content strings
        """
        self.logger.info(f"Initializing population with {len(initial_candidates)} candidates")
        
        self.population = []
        for i, content in enumerate(initial_candidates):
            candidate = Candidate(
                content=content,
                generation=0,
                metadata={"source": "initial", "index": i}
            )
            self.population.append(candidate)
        
        # Evaluate initial fitness
        self._evaluate_population()
        
        # Record generation
        self.generation_history.append(self.population.copy())
        self.current_generation = 0
        
        self.logger.info(f"Population initialized: {len(self.population)} candidates")
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for all candidates in the population."""
        self.logger.info(f"Evaluating fitness for generation {self.current_generation}")
        
        for candidate in self.population:
            try:
                # Evaluate fitness
                fitness_scores = self.fitness_function(candidate)
                candidate.set_fitness_scores(fitness_scores)
                
                # Record execution trace
                trace = ExecutionTrace(
                    execution_time=0.0,  # Could be measured if needed
                    success=True,
                    metrics=fitness_scores
                )
                candidate.add_execution_trace(trace)
                
            except Exception as e:
                self.logger.error(f"Error evaluating candidate {candidate.id}: {e}")
                
                # Record failed evaluation
                trace = ExecutionTrace(
                    execution_time=0.0,
                    success=False,
                    error=str(e)
                )
                candidate.add_execution_trace(trace)
    
    def _select_parents(self) -> List[Candidate]:
        """Select parent candidates for reproduction."""
        num_parents = min(
            int(self.config.population_size * self.config.crossover_rate),
            len(self.population)
        )
        
        return self.selector.select_parents(
            self.population, 
            num_parents, 
            self.config.tournament_size
        )
    
    def _crossover(self, parent1: Candidate, parent2: Candidate) -> Candidate:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent candidate
            parent2: Second parent candidate
            
        Returns:
            Child candidate
        """
        # Simple content crossover: combine parts of both parents
        content1_lines = parent1.content.split('\n')
        content2_lines = parent2.content.split('\n')
        
        # Randomly select cutting point
        min_lines = min(len(content1_lines), len(content2_lines))
        if min_lines <= 1:
            # If content has only one line, use character-based crossover
            content1_chars = parent1.content
            content2_chars = parent2.content
            min_chars = min(len(content1_chars), len(content2_chars))
            if min_chars <= 1:
                # If content is very short, just return parent1
                child_content = parent1.content
            else:
                cut_point = random.randint(1, min_chars - 1)
                child_content = content1_chars[:cut_point] + content2_chars[cut_point:]
        else:
            # Use line-based crossover for multi-line content
            cut_point = random.randint(1, min_lines - 1)
            child_content = '\n'.join(content1_lines[:cut_point] + content2_lines[cut_point:])
        
        # Combine content
        child_content = '\n'.join(content1_lines[:cut_point] + content2_lines[cut_point:])
        
        # Create child candidate
        child = Candidate(
            content=child_content,
            generation=self.current_generation + 1,
            parent_ids=[parent1.id, parent2.id],
            metadata={
                "crossover_type": "simple",
                "parent1_id": parent1.id,
                "parent2_id": parent2.id,
                "cut_point": cut_point
            }
        )
        
        return child
    
    def _mutate_offspring(self, offspring: List[Candidate], 
                         context: Optional[Dict[str, Any]] = None) -> List[Candidate]:
        """Apply mutations to offspring.
        
        Args:
            offspring: List of offspring candidates
            context: Context for mutations
            
        Returns:
            List of mutated offspring
        """
        return self.mutator.batch_mutate(
            offspring, 
            self.config.mutation_rate,
            context
        )
    
    def _environmental_selection(self) -> List[Candidate]:
        """Perform environmental selection to form next generation."""
        # Combine current population and offspring
        all_candidates = self.population.copy()
        
        # Select candidates for next generation
        selected = self.selector.environmental_selection(
            all_candidates,
            self.config.population_size
        )
        
        # Mark elite candidates
        pareto_front = self.selector.get_pareto_front(selected)
        for i, candidate in enumerate(selected):
            candidate.is_elite = candidate in pareto_front[:self.config.elite_size]
        
        return selected
    
    def _convergence_check(self) -> bool:
        """Check if optimization has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if self.current_generation == 0:
            return False
        
        # Check for improvement in best fitness
        improved = False
        
        for obj in self.objectives:
            if len(self.best_fitness_history[obj]) >= 2:
                current_best = self.best_fitness_history[obj][-1]
                previous_best = self.best_fitness_history[obj][-2]
                
                # Check if we've reached the threshold
                maximize = self.selector.maximize_objectives.get(obj, True)
                threshold_reached = (
                    (maximize and current_best >= self.config.fitness_threshold) or
                    (not maximize and current_best <= self.config.fitness_threshold)
                )
                
                if threshold_reached:
                    self.logger.info(f"Convergence: {obj} reached threshold {self.config.fitness_threshold}")
                    return True
                
                # Check for improvement
                improvement = abs(current_best - previous_best) > 0.001
                if improvement:
                    improved = True
        
        # Update no improvement counter
        if improved:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Check early stopping criterion
        if self.no_improvement_count >= self.config.early_stop_generations:
            self.logger.info(f"Early stopping: no improvement for {self.no_improvement_count} generations")
            return True
        
        return False
    
    def _update_best_candidates(self) -> None:
        """Update the record of best candidates."""
        if not self.population:
            return
        
        # Get Pareto front (best candidates)
        pareto_front = self.selector.get_pareto_front(self.population)
        
        # Update best candidates
        self.best_candidates = pareto_front
        
        # Update fitness history
        for obj in self.objectives:
            if pareto_front:
                best_score = max(
                    (c.get_fitness(obj) for c in pareto_front if c.get_fitness(obj) is not None),
                    default=0.0
                )
                self.best_fitness_history[obj].append(best_score)
            else:
                self.best_fitness_history[obj].append(0.0)
    
    def evolve_generation(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Evolve one generation.
        
        Args:
            context: Context for mutations (reflection, feedback, etc.)
            
        Returns:
            True if evolution should continue, False if converged
        """
        self.logger.info(f"Evolving generation {self.current_generation + 1}")
        
        # Select parents
        parents = self._select_parents()
        self.logger.info(f"Selected {len(parents)} parents")
        
        # Create offspring through crossover
        offspring: List[Candidate] = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            if random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2)
                offspring.append(child)
        
        self.logger.info(f"Generated {len(offspring)} offspring through crossover")
        
        # Apply mutations
        if offspring:
            offspring = self._mutate_offspring(offspring, context)
            self.logger.info(f"Applied mutations to {len(offspring)} offspring")
        
        # Combine with current population
        self.population.extend(offspring)
        
        # Evaluate fitness for new candidates
        if offspring:
            for candidate in offspring:
                try:
                    fitness_scores = self.fitness_function(candidate)
                    candidate.set_fitness_scores(fitness_scores)
                    
                    trace = ExecutionTrace(
                        execution_time=0.0,
                        success=True,
                        metrics=fitness_scores
                    )
                    candidate.add_execution_trace(trace)
                except Exception as e:
                    self.logger.error(f"Error evaluating offspring {candidate.id}: {e}")
        
        # Environmental selection
        self.population = self._environmental_selection()
        
        # Update generation
        self.current_generation += 1
        self.generation_history.append(self.population.copy())
        
        # Update best candidates and check convergence
        self._update_best_candidates()
        
        # Log generation stats
        self._log_generation_stats()
        
        # Check convergence
        should_continue = not self._convergence_check()
        
        return should_continue
    
    def _log_generation_stats(self) -> None:
        """Log statistics for the current generation."""
        if not self.population:
            return
        
        pareto_front = self.selector.get_pareto_front(self.population)
        
        stats = {
            "generation": self.current_generation,
            "population_size": len(self.population),
            "pareto_front_size": len(pareto_front),
            "no_improvement_count": self.no_improvement_count
        }
        
        # Add best fitness for each objective
        for obj in self.objectives:
            if self.best_fitness_history[obj]:
                stats[f"best_{obj}"] = self.best_fitness_history[obj][-1]
        
        self.logger.info(f"Generation stats: {stats}")
    
    def optimize(self, 
                 initial_candidates: List[str],
                 context_generator: Optional[Callable[[List[Candidate]], Dict[str, Any]]] = None) -> List[Candidate]:
        """Run the full optimization process.
        
        Args:
            initial_candidates: List of initial candidate content
            context_generator: Function to generate context for mutations based on population
            
        Returns:
            List of best candidates (Pareto front)
        """
        self.logger.info("Starting GEPA optimization")
        
        # Initialize population
        self.initialize_population(initial_candidates)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            # Generate context for mutations
            context = None
            if context_generator:
                context = context_generator(self.population)
            
            # Evolve one generation
            should_continue = self.evolve_generation(context)
            
            if not should_continue:
                self.logger.info(f"Optimization converged at generation {self.current_generation}")
                break
        
        self.logger.info(f"Optimization completed after {self.current_generation} generations")
        
        # Return best candidates
        return self.best_candidates
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            "total_generations": self.current_generation,
            "final_population_size": len(self.population),
            "final_pareto_front_size": len(self.best_candidates),
            "convergence_generations": self.no_improvement_count,
            "fitness_history": self.best_fitness_history,
            "best_candidates": [c.to_dict() for c in self.best_candidates]
        }
    
    def save_state(self, filepath: str) -> None:
        """Save optimizer state to file.
        
        Args:
            filepath: Path to save state
        """
        import json
        
        state = {
            "config": self.config.__dict__,
            "objectives": self.objectives,
            "current_generation": self.current_generation,
            "population": [c.to_dict() for c in self.population],
            "best_candidates": [c.to_dict() for c in self.best_candidates],
            "fitness_history": self.best_fitness_history,
            "no_improvement_count": self.no_improvement_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"State saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Simple fitness function for demonstration
    def simple_fitness(candidate: Candidate) -> Dict[str, float]:
        content = candidate.content.lower()
        
        # Simple heuristics
        length_score = min(len(content) / 100, 1.0)  # Prefer moderate length
        complexity_score = min(content.count(' ') / 20, 1.0)  # Prefer some complexity
        
        return {
            "quality": (length_score + complexity_score) / 2,
            "efficiency": 1.0 - (len(content) / 1000)  # Prefer shorter content
        }
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        objectives=["quality", "efficiency"],
        fitness_function=simple_fitness,
        config=OptimizationConfig(
            population_size=10,
            max_generations=5
        ),
        maximize_objectives={"quality": True, "efficiency": True}
    )
    
    # Initial candidates
    initial_candidates = [
        "Write a simple function",
        "Create a complex algorithm with multiple steps",
        "Implement a basic solution",
        "Design an advanced system"
    ]
    
    # Run optimization
    best_candidates = optimizer.optimize(initial_candidates)
    
    print(f"Optimization completed. Best candidates: {len(best_candidates)}")
    for i, candidate in enumerate(best_candidates):
        print(f"\nBest candidate {i+1}:")
        print(f"Content: {candidate.content}")
        print(f"Fitness: {candidate.fitness_scores}")