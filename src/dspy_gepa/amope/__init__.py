"""AMOPE (Adaptive Multi-Objective Prompt Evolution) Algorithm.

This package implements advanced features that extend beyond the core GEPA algorithm:
- Adaptive mutation strategies based on performance gradients
- Dynamic objective balancing for multi-objective optimization  
- Hierarchical co-evolution of multiple DSPY components
- Advanced reflection engine with LLM guidance
- **NEW: Seamless GEPA integration with bridging layer (Phase 2)**

AMOPE provides a unified interface for sophisticated prompt optimization
with automatic strategy selection and dynamic adaptation.

Phase 2 Features:
- GEPA-compatible population initializer with diverse initial candidates
- Integrated AMOPE Objective Balancer with GEPA evolution process
- Comprehensive strategy tracking bridging AMOPE and GEPA analytics
- Real-time dynamic weight adjustment during GEPA optimization
- Advanced convergence analysis and optimization insights

Quick Start:
    ```python
    from dspy_gepa.amope import AMOPEOptimizer
    
    # Initialize optimizer with your objectives
    optimizer = AMOPEOptimizer(
        objectives={"accuracy": 0.7, "efficiency": 0.3},
        mutation_config={"use_llm_guidance": True},
        balancing_config={"strategy": "adaptive_harmonic"}
    )
    
    # Optimize your prompt (now with enhanced AMOPE-GEPA integration)
    result = optimizer.optimize(
        initial_prompt="Your initial prompt here",
        evaluation_fn=your_evaluation_function,
        generations=50
    )
    
    # Access comprehensive analytics
    print(f"Best prompt: {result.best_prompt}")
    print(f"Strategy usage: {result.strategy_usage}")
    
    if result.comprehensive_analytics:
        analytics = result.comprehensive_analytics
        print(f"Objective effectiveness: {analytics['bridged_metrics']['objective_effectiveness']}")
        print(f"Strategy success rates: {analytics['bridged_metrics']['strategy_success_rates']}")
    
    # Get optimization insights
    insights = optimizer.get_optimization_insights()
    print(f"Recommendations: {insights['recommendations']}")
    ```

Phase 2 Integration Details:

1. **Population Initialization**: The `_initialize_gepa_population()` method creates diverse
   initial candidates with AMOPE configuration metadata, ensuring GEPA starts with a
   well-balanced and diverse population.

2. **Objective Balancer Integration**: The `_setup_amope_gepa_integration()` method hooks
   AMOPE's dynamic objective weighting into GEPA's fitness evaluation process,
   enabling real-time weight adjustments during evolution.

3. **Strategy Tracking Bridge**: Enhanced `_extract_gepa_strategy_usage()` and new analytics
   methods bridge GEPA's mutation tracking with AMOPE's strategy analytics, providing
   comprehensive insights into optimization effectiveness.

4. **Comprehensive Analytics**: The `_bridge_amope_gepa_analytics()` method combines
   metrics from both systems to provide detailed analysis of convergence patterns,
   strategy effectiveness, and objective dynamics.

The integration maintains backward compatibility while adding powerful new capabilities
for analyzing and understanding the optimization process.
"""

import random
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field

# Import GEPA types for type hints
try:
    from gepa.core.candidate import Candidate
    from gepa.core.optimizer import OptimizationConfig
except ImportError:
    # Fallback for development
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from gepa.core.candidate import Candidate
        from gepa.core.optimizer import OptimizationConfig
    except ImportError:
        # Define fallback types for development
        Candidate = Any
        OptimizationConfig = Any

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
    """Enhanced results from AMOPE-GEPA hybrid optimization.
    
    This class combines comprehensive analytics from both AMOPE and GEPA systems,
    providing detailed insights into the optimization process, convergence patterns,
    strategy effectiveness, and Pareto front analysis.
    """
    
    # Core optimization results
    best_prompt: str
    best_score: float
    best_objectives: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    generations_completed: int
    convergence_achieved: bool
    total_evaluation_time: float
    strategy_usage: Dict[str, int]
    
    # GEPA-specific results (Pareto front analysis)
    pareto_front: Optional[List[Dict[str, Any]]] = None
    pareto_front_size: int = 0
    population_diversity: float = 0.0
    convergence_rate: float = 0.0
    
    # AMOPE-specific results
    final_objective_weights: Optional[Dict[str, float]] = None
    stagnation_counter: int = 0
    objective_effectiveness: Optional[Dict[str, float]] = None
    mutation_insights: Optional[Dict[str, Any]] = None
    
    # Hybrid AMOPE-GEPA analytics
    comprehensive_analytics: Optional[Dict[str, Any]] = None
    
    # Performance and convergence analytics
    convergence_analysis: Optional[Dict[str, Any]] = None
    performance_trajectory: Optional[List[Dict[str, float]]] = None
    strategy_effectiveness: Optional[Dict[str, float]] = None
    
    # Optimization insights and recommendations
    optimization_insights: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    
    # Metadata
    optimization_method: str = "AMOPE-GEPA_HYBRID"
    hybrid_integration_successful: bool = True
    total_candidates_evaluated: int = 0
    
    def get_detailed_summary(self) -> str:
        """Generate a comprehensive summary of hybrid AMOPE-GEPA optimization results."""
        summary_lines = [
            f"=== {self.optimization_method} Optimization Results ===",
            f"\n🎯 Core Results:",
            f"  Best Score: {self.best_score:.4f}",
            f"  Best Prompt: '{self.best_prompt[:100]}{'...' if len(self.best_prompt) > 100 else ''}'",
            f"  Final Objectives: {self.best_objectives}",
            f"  Generations Completed: {self.generations_completed}",
            f"  Convergence: {'✅ Achieved' if self.convergence_achieved else '❌ Not Achieved'}",
            f"  Total Evaluation Time: {self.total_evaluation_time:.2f}s",
            f"  Total Candidates Evaluated: {self.total_candidates_evaluated}",
            f"  Hybrid Integration: {'✅ Successful' if self.hybrid_integration_successful else '❌ Failed'}"
        ]
        
        # GEPA-specific results
        if self.pareto_front_size > 0:
            summary_lines.extend([
                f"\n🔬 GEPA Pareto Front Analysis:",
                f"  Pareto Front Size: {self.pareto_front_size}",
                f"  Population Diversity: {self.population_diversity:.3f}",
                f"  Convergence Rate: {self.convergence_rate:.3f}"
            ])
        
        # AMOPE-specific results
        if self.final_objective_weights:
            summary_lines.extend([
                f"\n🧠 AMOPE Adaptive Analysis:",
                f"  Final Objective Weights: {self.final_objective_weights}",
                f"  Stagnation Counter: {self.stagnation_counter}",
                f"  Objective Effectiveness: {self.objective_effectiveness or {}}"
            ])
        
        # Strategy analysis
        summary_lines.extend([
            f"\n📊 Strategy Usage: {self.strategy_usage}",
            f"  Strategy Effectiveness: {self.strategy_effectiveness or {}}"
        ])
        
        # Comprehensive analytics
        if self.comprehensive_analytics:
            analytics = self.comprehensive_analytics
            
            if 'bridged_metrics' in analytics:
                bridged = analytics['bridged_metrics']
                summary_lines.extend([
                    f"\n🔗 Hybrid AMOPE-GEPA Metrics:",
                    f"  Objective Effectiveness: {bridged.get('objective_effectiveness', {})}",
                    f"  Strategy Success Rates: {bridged.get('strategy_success_rates', {})}",
                    f"  Diversity Metrics: {bridged.get('diversity_metrics', {})}"
                ])
            
            if 'gepa_stats' in analytics:
                gepa_stats = analytics['gepa_stats']
                summary_lines.extend([
                    f"\n🧬 GEPA Optimization Stats:",
                    f"  Total Generations: {gepa_stats.get('total_generations', 'N/A')}",
                    f"  Final Population Size: {gepa_stats.get('final_population_size', 'N/A')}",
                    f"  Convergence Generations: {gepa_stats.get('convergence_generations', 'N/A')}"
                ])
        
        # Recommendations
        if self.recommendations:
            summary_lines.extend([
                f"\n💡 Recommendations:"
            ])
            for i, rec in enumerate(self.recommendations[:5], 1):  # Limit to top 5
                summary_lines.append(f"  {i}. {rec}")
        
        summary_lines.append("\n" + "="*60)
        return "\n".join(summary_lines)
    

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
        
        # Runtime state for GEPA integration
        self._current_evaluation_fn: Optional[Callable] = None
    
    def _initialize_gepa_population(self, initial_prompt: str) -> List[Candidate]:
        """Create a diverse initial population for GEPA optimization.
        
        This method converts the initial prompt into proper GEPA Candidates
        with metadata, preserves AMOPE configuration, and creates diversity
        in the initial population.
        
        Args:
            initial_prompt: Starting prompt for optimization
            
        Returns:
            List of GEPA Candidate objects with AMOPE metadata
        """
        if not self._GEPA_AVAILABLE:
            # Fallback - just return the initial prompt
            return [initial_prompt]
        
        candidates = []
        
        # Create the primary candidate from initial prompt
        primary_candidate = Candidate(
            content=initial_prompt,
            generation=0,
            parent_ids=[],
            fitness_scores={},
            metadata={
                "amope_config": {
                    "objectives": dict(self.config.objectives),
                    "mutation_config": dict(self.config.mutation_config),
                    "balancing_config": dict(self.config.balancing_config)
                },
                "initial_prompt": True,
                "generation_type": "initial",
                "diversity_seed": 0
            }
        )
        candidates.append(primary_candidate)
        
        # Generate diverse variations for better GEPA performance
        variations = self._create_prompt_variations(initial_prompt)
        
        for i, variation in enumerate(variations):
            if len(candidates) >= self.config.population_size:
                break
                
            # Create candidate with AMOPE metadata
            candidate = Candidate(
                content=variation,
                generation=0,
                parent_ids=[],
                fitness_scores={},
                metadata={
                    "amope_config": {
                        "objectives": dict(self.config.objectives),
                        "mutation_config": dict(self.config.mutation_config),
                        "balancing_config": dict(self.config.balancing_config)
                    },
                    "initial_prompt": False,
                    "generation_type": "variation",
                    "variation_index": i,
                    "diversity_seed": i + 1,
                    "base_prompt": initial_prompt[:100] + "..." if len(initial_prompt) > 100 else initial_prompt
                }
            )
            candidates.append(candidate)
        
        # If we still need more candidates to reach population_size, create random variations
        while len(candidates) < self.config.population_size:
            extra_candidate = self._create_diverse_candidate(initial_prompt, len(candidates))
            candidates.append(extra_candidate)
        
        if self.config.verbose:
            print(f"Created {len(candidates)} candidates for initial GEPA population")
            print(f"Primary candidate: '{initial_prompt[:50]}...'")
            print(f"Diversity variations: {len(candidates) - 1}")
        
        return candidates
    
    def _create_prompt_variations(self, initial_prompt: str) -> List[str]:
        """Create diverse variations of the initial prompt.
        
        These variations help GEPA explore different directions from the start.
        
        Args:
            initial_prompt: Base prompt to vary
            
        Returns:
            List of prompt variations
        """
        variations = []
        
        # Semantic variations
        base_lower = initial_prompt.lower()
        
        variations.extend([
            f"{initial_prompt}. Please provide a comprehensive response.",
            f"Carefully {base_lower} with attention to detail.",
            f"Consider all aspects when {base_lower}.",
            f"Simplify: {base_lower}",
            f"Be thorough and accurate: {base_lower}.",
            f"Take your time to {base_lower} properly.",
            f"Focus on quality when {base_lower}.",
            f"Think step by step to {base_lower}.",
            f"Be systematic in {base_lower}.",
            f"Ensure accuracy when {base_lower}."
        ])
        
        # Style variations
        style_prefixes = [
            "Expert approach: ",
            "Methodical: ",
            "Systematic: ",
            "Careful analysis: ",
            "Thorough consideration: "
        ]
        
        for prefix in style_prefixes:
            variations.append(prefix + base_lower)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
        
        return unique_variations
    
    def _create_diverse_candidate(self, initial_prompt: str, seed_index: int) -> Candidate:
        """Create a diverse candidate with random variations.
        
        Args:
            initial_prompt: Base prompt
            seed_index: Seed for randomization
            
        Returns:
            GEPA Candidate with diverse content
        """
        import random
        random.seed(seed_index)  # Ensure reproducible diversity
        
        # Random modifications
        modifiers = [
            "",  # No modification
            " Think carefully.",
            " Be methodical.", 
            " Stay focused.",
            " Use structured approach.",
            " Consider multiple perspectives."
        ]
        
        prefixes = [
            "",
            "Carefully: ",
            "Systematically: ",
            "Thoroughly: ",
            "Expertly: "
        ]
        
        suffixes = [
            "",
            " Be detailed.",
            " Stay accurate.",
            " Think step-by-step.",
            " Use proper reasoning."
        ]
        
        # Apply random modifications
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        modifier = random.choice(modifiers)
        
        modified_prompt = prefix + initial_prompt.lower() + modifier + suffix
        
        return Candidate(
            content=modified_prompt,
            generation=0,
            parent_ids=[],
            fitness_scores={},
            metadata={
                "amope_config": {
                    "objectives": dict(self.config.objectives),
                    "mutation_config": dict(self.config.mutation_config),
                    "balancing_config": dict(self.config.balancing_config)
                },
                "initial_prompt": False,
                "generation_type": "diverse",
                "diversity_seed": seed_index,
                "random_modification": True,
                "base_prompt": initial_prompt[:100] + "..." if len(initial_prompt) > 100 else initial_prompt
            }
        )
    
    def _initialize_components(self):
        """Initialize AMOPE components based on configuration."""
        
        # Import GEPA components
        try:
            from gepa.core.optimizer import GeneticOptimizer
            from gepa.core.mutator import TextMutator
            from gepa.core.candidate import Candidate
            self._GEPA_AVAILABLE = True
        except ImportError:
            # Fallback for development
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from gepa.core.optimizer import GeneticOptimizer
                from gepa.core.mutator import TextMutator
                from gepa.core.candidate import Candidate
                self._GEPA_AVAILABLE = True
            except ImportError:
                self._GEPA_AVAILABLE = False
                print("Warning: GEPA not available, using fallback optimization")
        
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
        
        # Initialize GEPA optimizer if available
        if self._GEPA_AVAILABLE:
            from gepa.core.optimizer import OptimizationConfig
            
            # Create GEPA configuration that matches AMOPE settings
            gepa_config = OptimizationConfig(
                population_size=self.config.population_size,
                max_generations=self.config.max_generations,
                mutation_rate=0.8,
                crossover_rate=0.6,
                tournament_size=2,
                elite_size=5,
                early_stop_generations=self.config.stagnation_generations,
                fitness_threshold=0.95
            )
            
            self.gepa_optimizer = GeneticOptimizer(
                objectives=list(self.config.objectives.keys()),
                fitness_function=self._gepa_fitness_function,
                config=gepa_config,
                llm_client=None,  # Will use default LLM client
                maximize_objectives={obj: True for obj in self.config.objectives.keys()}
            )
    
    def optimize(self, 
                 initial_prompt: str,
                 evaluation_fn: Callable[[str], Dict[str, float]],
                 generations: Optional[int] = None) -> OptimizationResult:
        """Run AMOPE optimization using GEPA genetic optimizer.
        
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
        
        # Store evaluation function for use in GEPA fitness function
        self._current_evaluation_fn = evaluation_fn
        
        # Get initial score for comparison
        initial_objectives = evaluation_fn(initial_prompt)
        initial_score = self._evaluate_prompt(initial_prompt, evaluation_fn)
        
        if self.config.verbose:
            print(f"Starting AMOPE optimization with {len(self.config.objectives)} objectives")
            print(f"Using GEPA genetic optimizer with {self.config.population_size} population size")
            print(f"Initial score: {initial_score:.4f}")
        
        # Initialize AMOPE optimization state
        self.current_generation = 0
        self.optimization_history = []
        self.stagnation_counter = 0
        self.best_candidate = None
        self.strategy_usage = {}
        
        # Create diverse initial population using AMOPE's population initializer
        if self._GEPA_AVAILABLE:
            initial_candidates = self._initialize_gepa_population(initial_prompt)
            # Convert GEPA Candidates to strings for GEPA optimizer
            initial_prompt_strings = [candidate.content for candidate in initial_candidates]
        else:
            # Fallback to simple list
            initial_prompt_strings = [initial_prompt]
        
        # Run optimization using integrated AMOPE-GEPA approach
        if self._GEPA_AVAILABLE:
            try:
                # Update GEPA config with specified generations
                self.gepa_optimizer.config.max_generations = generations
                self.gepa_optimizer.config.population_size = self.config.population_size
                
                # Hook AMOPE's objective balancer into GEPA's evolution process
                self._setup_amope_gepa_integration()
                
                if self.config.verbose:
                    print("Starting integrated AMOPE-GEPA optimization...")
                    print(f"Initial objective weights: {dict(self.balancer.current_objectives)}")
                
                # Run GEPA optimization with AMOPE integration
                best_candidates = self.gepa_optimizer.optimize(initial_prompt_strings)
                
                # Extract the best candidate from Pareto front
                if best_candidates:
                    best_candidate = self._select_best_candidate_from_pareto(best_candidates, evaluation_fn)
                    final_prompt = best_candidate.content
                    final_objectives = evaluation_fn(final_prompt)
                    final_score = self._evaluate_prompt(final_prompt, evaluation_fn)
                    
                    # Get optimization statistics from GEPA
                    gepa_stats = self.gepa_optimizer.get_optimization_stats()
                    generations_completed = gepa_stats["total_generations"]
                    
                    # Extract strategy usage from GEPA mutation history
                    strategy_usage = self._extract_gepa_strategy_usage(best_candidates)
                    
                    # Update AMOPE metrics with final results
                    self._update_amope_metrics_final(best_candidates, final_objectives, generations_completed)
                    
                    if self.config.verbose:
                        print(f"AMOPE-GEPA optimization completed in {generations_completed} generations")
                        print(f"Final score: {final_score:.4f}")
                        print(f"Pareto front size: {len(best_candidates)}")
                        print(f"Final objective weights: {dict(self.balancer.current_objectives)}")
                        print(f"Strategy usage: {strategy_usage}")
                else:
                    # Fallback to initial prompt if GEPA fails
                    final_prompt = initial_prompt
                    final_objectives = initial_objectives
                    final_score = initial_score
                    generations_completed = 0
                    strategy_usage = {}
                    
                    if self.config.verbose:
                        print("GEPA optimization failed, using initial prompt")
                        
            except Exception as e:
                if self.config.verbose:
                    print(f"GEPA optimization failed: {e}")
                    print("Falling back to simple evaluation")
                
                # Fallback to initial prompt
                final_prompt = initial_prompt
                final_objectives = initial_objectives
                final_score = initial_score
                generations_completed = 0
                strategy_usage = {}
        else:
            # GEPA not available, use initial prompt
            final_prompt = initial_prompt
            final_objectives = initial_objectives
            final_score = initial_score
            generations_completed = 0
            strategy_usage = {}
            
            if self.config.verbose:
                print("GEPA not available, using initial prompt")
        
        # Create optimization history (simplified for GEPA integration)
        optimization_history = []
        if self.config.save_history and self._GEPA_AVAILABLE and hasattr(self.gepa_optimizer, 'best_fitness_history'):
            # Convert GEPA fitness history to AMOPE format
            for gen in range(len(self.gepa_optimizer.best_fitness_history.get(list(self.config.objectives.keys())[0], []))):
                gen_data = {
                    "generation": gen,
                    "best_score": final_score,  # Simplified - would need more complex tracking
                    "best_objectives": final_objectives,
                    "improvement": 0.0,  # Simplified
                    "objective_weights": dict(self.balancer.current_objectives)
                }
                optimization_history.append(gen_data)
        
        # Calculate final statistics and comprehensive analytics
        total_time = time.time() - start_time
        
        # Generate comprehensive bridging analytics if GEPA was used
        comprehensive_analytics = None
        if self._GEPA_AVAILABLE and 'best_candidates' in locals():
            comprehensive_analytics = self._bridge_amope_gepa_analytics(best_candidates)
        
        # Enhanced result creation with comprehensive AMOPE-GEPA analytics
        result = OptimizationResult(
            # Core optimization results
            best_prompt=final_prompt,
            best_score=final_score,
            best_objectives=final_objectives,
            optimization_history=optimization_history,
            generations_completed=generations_completed,
            convergence_achieved=generations_completed > 0 and self.stagnation_counter < self.config.stagnation_generations,
            total_evaluation_time=total_time,
            strategy_usage=strategy_usage,
            
            # GEPA-specific results
            pareto_front=[c.to_dict() for c in best_candidates] if 'best_candidates' in locals() and best_candidates else None,
            pareto_front_size=len(best_candidates) if 'best_candidates' in locals() and best_candidates else 0,
            population_diversity=self._calculate_population_diversity(self.gepa_optimizer.population) if hasattr(self, 'gepa_optimizer') and self.gepa_optimizer.population else 0.0,
            
            # AMOPE-specific results with error handling
            final_objective_weights=dict(self.balancer.current_objectives) if hasattr(self.balancer, 'current_objectives') else None,
            stagnation_counter=self.stagnation_counter,
            
            # Calculate metrics with error handling
            convergence_rate=self._safe_calculate_convergence_rate(),
            objective_effectiveness=self._safe_calculate_final_objective_effectiveness(),
            mutation_insights=self._safe_generate_mutation_insights(),
            
            # Comprehensive analytics
            comprehensive_analytics=comprehensive_analytics,
            
            # Performance analytics with error handling
            convergence_analysis=self._analyze_convergence_patterns(),
            performance_trajectory=self._safe_extract_performance_trajectory(),
            strategy_effectiveness=self._safe_calculate_strategy_effectiveness(),
            
            # Optimization insights
            optimization_insights=self.get_optimization_insights(),
            recommendations=[],
            
            # Metadata
            optimization_method="AMOPE-GEPA_HYBRID",
            hybrid_integration_successful=self._GEPA_AVAILABLE and comprehensive_analytics is not None,
            total_candidates_evaluated=len(self.optimization_history) * self.config.population_size
        )
        
        if self.config.verbose:
            print(f"\nOptimization completed in {total_time:.2f}s")
            print(f"Final score: {result.best_score:.4f}")
            print(f"Generations completed: {result.generations_completed}")
            print(f"Strategy usage: {result.strategy_usage}")
        
        return result
    
    def _gepa_fitness_function(self, candidate) -> Dict[str, float]:
        """Bridge AMOPE evaluation function with GEPA Candidate system.
        
        This function converts GEPA Candidate objects to the format expected
        by AMOPE's evaluation functions, then converts results back to GEPA
        fitness scores.
        
        Args:
            candidate: GEPA Candidate object with text content
            
        Returns:
            Dictionary of fitness scores for each objective
        """
        try:
            # Store the current evaluation function for use in fitness evaluation
            if not hasattr(self, '_current_evaluation_fn'):
                raise ValueError("No evaluation function available. Call optimize() first.")
            
            # Evaluate the candidate's content using AMOPE's evaluation function
            objectives = self._current_evaluation_fn(candidate.content)
            
            # Ensure all expected objectives are present
            fitness_scores = {}
            for obj_name in self.config.objectives.keys():
                fitness_scores[obj_name] = objectives.get(obj_name, 0.0)
            
            return fitness_scores
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error evaluating candidate {candidate.id}: {e}")
            # Return default scores if evaluation fails
            return {obj: 0.0 for obj in self.config.objectives.keys()}
    
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
    
    # Note: Custom mutation methods removed in Phase 1 of AMOPE-GEPA integration
    # GEPA's TextMutator now handles all mutations automatically through the genetic optimizer
    
    def _select_best_candidate_from_pareto(self, candidates: List, evaluation_fn: Callable):
        """Select the best candidate from Pareto front using AMOPE's weighted scoring.
        
        Args:
            candidates: List of GEPA Candidate objects from Pareto front
            evaluation_fn: AMOPE evaluation function
            
        Returns:
            Best candidate according to AMOPE's weighted scoring
        """
        if not candidates:
            raise ValueError("No candidates provided")
        
        # Score each candidate using AMOPE's weighted evaluation
        best_candidate = None
        best_score = float('-inf')
        
        for candidate in candidates:
            score = self._evaluate_prompt(candidate.content, evaluation_fn)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate or candidates[0]
    
    def _extract_gepa_strategy_usage(self, candidates: List) -> Dict[str, int]:
        """Extract strategy usage information from GEPA candidates.
        
        This method bridges AMOPE's strategy tracking with GEPA's built-in
        mutation tracking system, providing comprehensive analytics on which
        mutation strategies were most effective during optimization.
        
        Args:
            candidates: List of GEPA Candidate objects
            
        Returns:
            Dictionary of strategy usage counts with AMOPE-compatible naming
        """
        strategy_usage = {}
        
        try:
            for candidate in candidates:
                # Extract mutation types from GEPA's mutation history
                if hasattr(candidate, 'mutation_history') and candidate.mutation_history:
                    for mutation in candidate.mutation_history:
                        # Handle different GEPA mutation record formats
                        if hasattr(mutation, 'mutation_type'):
                            gepa_strategy = mutation.mutation_type
                        elif hasattr(mutation, 'type'):
                            gepa_strategy = mutation.type
                        elif isinstance(mutation, str):
                            gepa_strategy = mutation
                        else:
                            # Fallback: use string representation
                            gepa_strategy = str(type(mutation).__name__)
                        
                        # Convert GEPA strategy names to AMOPE-compatible names
                        amope_strategy = self._convert_gepa_to_amope_strategy(gepa_strategy)
                        
                        # Count usage
                        strategy_usage[amope_strategy] = strategy_usage.get(amope_strategy, 0) + 1
                        
                        # Additional tracking: record success rates if available
                        if hasattr(mutation, 'success_rate'):
                            success_key = f"{amope_strategy}_success_rate"
                            if success_key not in strategy_usage:
                                strategy_usage[success_key] = []
                            strategy_usage[success_key].append(mutation.success_rate)
                
                # Also check candidate metadata for strategy information
                if hasattr(candidate, 'metadata') and candidate.metadata:
                    if 'mutation_strategy' in candidate.metadata:
                        strategy = candidate.metadata['mutation_strategy']
                        amope_strategy = self._convert_gepa_to_amope_strategy(strategy)
                        strategy_usage[amope_strategy] = strategy_usage.get(amope_strategy, 0) + 1
                    
                    if 'generation_type' in candidate.metadata:
                        gen_type = candidate.metadata['generation_type']
                        strategy_usage[f"generated_{gen_type}"] = strategy_usage.get(f"generated_{gen_type}", 0) + 1
        
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not extract full strategy usage from GEPA candidates: {e}")
            # Fallback: basic counting
            strategy_usage['gepa_optimization'] = len(candidates)
        
        # Process success rate averages
        final_strategy_usage = {}
        for key, value in strategy_usage.items():
            if key.endswith('_success_rate') and isinstance(value, list):
                # Calculate average success rate
                final_strategy_usage[key] = sum(value) / len(value) if value else 0.0
            else:
                final_strategy_usage[key] = value
        
        return final_strategy_usage
    
    def _convert_gepa_to_amope_strategy(self, gepa_strategy: str) -> str:
        """Convert GEPA strategy names to AMOPE-compatible naming.
        
        This ensures consistency between AMOPE's strategy tracking
        and GEPA's mutation strategy names.
        
        Args:
            gepa_strategy: Strategy name from GEPA
            
        Returns:
            AMOPE-compatible strategy name
        """
        # Define mapping between GEPA and AMOPE strategy names
        strategy_mapping = {
            # GEPA text mutations -> AMOPE strategies
            'text_substitution': 'pattern_based',
            'word_replacement': 'pattern_based',
            'sentence_rearrangement': 'structural',
            'semantic_mutation': 'llm_guided',
            'contextual_mutation': 'llm_guided',
            'llm_mutation': 'llm_guided',
            'random_mutation': 'gradient_based',
            'crossover': 'crossover',
            'elitist_selection': 'selection',
            'tournament_selection': 'selection',
            
            # Generic mappings
            'mutation': 'gradient_based',
            'variation': 'pattern_based',
            'evolution': 'genetic_algorithm'
        }
        
        # Convert to lowercase for case-insensitive matching
        gepa_lower = gepa_strategy.lower()
        
        # Try exact matches first
        if gepa_lower in strategy_mapping:
            return strategy_mapping[gepa_lower]
        
        # Try partial matches
        for gepa_name, amope_name in strategy_mapping.items():
            if gepa_name in gepa_lower or gepa_lower in gepa_name:
                return amope_name
        
        # Try to infer from common patterns
        if 'llm' in gepa_lower or 'gpt' in gepa_lower:
            return 'llm_guided'
        elif 'pattern' in gepa_lower or 'template' in gepa_lower:
            return 'pattern_based'
        elif 'gradient' in gepa_lower or 'random' in gepa_lower:
            return 'gradient_based'
        elif 'crossover' in gepa_lower or 'combine' in gepa_lower:
            return 'crossover'
        elif 'selection' in gepa_lower or 'choose' in gepa_lower:
            return 'selection'
        
        # Fallback: use the original name but ensure it's valid
        return gepa_strategy.replace(' ', '_').replace('-', '_').lower()
    
    def _bridge_amope_gepa_analytics(self, candidates: List) -> Dict[str, Any]:
        """Create comprehensive analytics bridging AMOPE and GEPA optimization data.
        
        This method extracts and combines analytics from both AMOPE's adaptive
        features and GEPA's genetic optimization process.
        
        Args:
            candidates: List of GEPA Candidate objects from optimization
            
        Returns:
            Comprehensive analytics dictionary
        """
        analytics = {
            'amope_metrics': {},
            'gepa_metrics': {},
            'bridged_metrics': {}
        }
        
        try:
            # AMOPE metrics
            analytics['amope_metrics'] = {
                'objective_weights': dict(self.balancer.current_objectives),
                'stagnation_counter': self.stagnation_counter,
                'generations_completed': self.current_generation,
                'strategy_usage': self.strategy_usage,
                'optimization_history_length': len(self.optimization_history)
            }
            
            # GEPA metrics
            if hasattr(self, 'gepa_optimizer'):
                gepa_stats = self.gepa_optimizer.get_optimization_stats()
                analytics['gepa_metrics'] = {
                    'total_generations': gepa_stats.get('total_generations', 0),
                    'total_evaluations': gepa_stats.get('total_evaluations', 0),
                    'best_fitness': gepa_stats.get('best_fitness', {}),
                    'convergence_generation': gepa_stats.get('convergence_generation', None),
                    'population_size': gepa_stats.get('population_size', self.config.population_size)
                }
            
            # Bridged metrics (combined insights)
            analytics['bridged_metrics'] = {
                'objective_effectiveness': self._calculate_objective_effectiveness(candidates),
                'strategy_success_rates': self._calculate_strategy_success_rates(candidates),
                'diversity_metrics': self._calculate_diversity_metrics(candidates),
                'convergence_analysis': self._analyze_convergence_patterns()
            }
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not generate full analytics: {e}")
            analytics['error'] = str(e)
        
        return analytics
    
    def _calculate_objective_effectiveness(self, candidates: List) -> Dict[str, float]:
        """Calculate effectiveness scores for each objective based on final results.
        
        Args:
            candidates: List of GEPA Candidate objects
            
        Returns:
            Dictionary mapping objectives to effectiveness scores
        """
        effectiveness = {}
        
        if not candidates or not hasattr(self, '_current_evaluation_fn') or self._current_evaluation_fn is None:
            return effectiveness
        
        try:
            # Calculate average and best scores for each objective
            objective_scores = {obj: [] for obj in self.config.objectives.keys()}
            
            for candidate in candidates[:10]:  # Sample top candidates
                if hasattr(candidate, 'content'):
                    scores = self._current_evaluation_fn(candidate.content)
                    for obj in self.config.objectives.keys():
                        objective_scores[obj].append(scores.get(obj, 0.0))
            
            # Calculate effectiveness as normalized improvement
            for obj, scores in objective_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    # Effectiveness: combination of average and best performance
                    effectiveness[obj] = (avg_score + max_score) / 2.0
                else:
                    effectiveness[obj] = 0.0
        
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate objective effectiveness: {e}")
        
        return effectiveness
    
    def _calculate_strategy_success_rates(self, candidates: List) -> Dict[str, float]:
        """Calculate success rates for different mutation strategies.
        
        Args:
            candidates: List of GEPA Candidate objects
            
        Returns:
            Dictionary mapping strategies to success rates
        """
        success_rates = {}
        
        try:
            strategy_results = {}
            
            for candidate in candidates:
                if hasattr(candidate, 'mutation_history'):
                    for mutation in candidate.mutation_history:
                        strategy = self._convert_gepa_to_amope_strategy(
                            getattr(mutation, 'mutation_type', 'unknown')
                        )
                        
                        if strategy not in strategy_results:
                            strategy_results[strategy] = {'successes': 0, 'attempts': 0}
                        
                        strategy_results[strategy]['attempts'] += 1
                        
                        # Consider candidate successful if it's in Pareto front
                        if hasattr(candidate, 'fitness_scores'):
                            total_fitness = sum(candidate.fitness_scores.values())
                            if total_fitness > 0.5:  # Simple success threshold
                                strategy_results[strategy]['successes'] += 1
            
            # Calculate success rates
            for strategy, results in strategy_results.items():
                if results['attempts'] > 0:
                    success_rates[strategy] = results['successes'] / results['attempts']
                else:
                    success_rates[strategy] = 0.0
        
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate strategy success rates: {e}")
        
        return success_rates
    
    def _calculate_diversity_metrics(self, candidates: List) -> Dict[str, float]:
        """Calculate various diversity metrics for the candidate population.
        
        Args:
            candidates: List of GEPA Candidate objects
            
        Returns:
            Dictionary of diversity metrics
        """
        diversity_metrics = {}
        
        if not candidates:
            return diversity_metrics
        
        try:
            # Length diversity
            lengths = [len(candidate.content) for candidate in candidates if hasattr(candidate, 'content')]
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
                diversity_metrics['length_diversity'] = min(length_variance / (avg_length ** 2), 1.0)
            
            # Strategy diversity
            strategies_used = set()
            for candidate in candidates:
                if hasattr(candidate, 'mutation_history'):
                    for mutation in candidate.mutation_history:
                        strategy = self._convert_gepa_to_amope_strategy(
                            getattr(mutation, 'mutation_type', 'unknown')
                        )
                        strategies_used.add(strategy)
            
            diversity_metrics['strategy_diversity'] = len(strategies_used) / max(len(candidates), 1)
            
            # Fitness diversity
            if hasattr(candidates[0], 'fitness_scores'):
                fitness_values = []
                for candidate in candidates:
                    if hasattr(candidate, 'fitness_scores'):
                        total_fitness = sum(candidate.fitness_scores.values())
                        fitness_values.append(total_fitness)
                
                if fitness_values:
                    avg_fitness = sum(fitness_values) / len(fitness_values)
                    fitness_variance = sum((f - avg_fitness) ** 2 for f in fitness_values) / len(fitness_values)
                    diversity_metrics['fitness_diversity'] = min(fitness_variance / (avg_fitness ** 2 + 0.001), 1.0)
        
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate diversity metrics: {e}")
        
        return diversity_metrics
    
    def _analyze_convergence_patterns(self) -> Dict[str, Any]:
        """Analyze convergence patterns from optimization history.
        
        Returns:
            Dictionary with convergence analysis
        """
        convergence_analysis = {}
        
        if len(self.optimization_history) < 3:
            return convergence_analysis
        
        try:
            scores = [entry['best_score'] for entry in self.optimization_history]
            
            # Calculate improvement rates
            improvements = []
            for i in range(1, len(scores)):
                if scores[i-1] > 0:
                    improvement = (scores[i] - scores[i-1]) / scores[i-1]
                    improvements.append(improvement)
            
            if improvements:
                convergence_analysis['avg_improvement_rate'] = sum(improvements) / len(improvements)
                convergence_analysis['max_improvement'] = max(improvements)
                convergence_analysis['min_improvement'] = min(improvements)
                convergence_analysis['improvement_stability'] = 1.0 - (max(improvements) - min(improvements))
            
            # Stagnation analysis
            convergence_analysis['stagnation_periods'] = self.stagnation_counter
            convergence_analysis['convergence_generation'] = self.current_generation
            
            # Objective weight evolution
            if len(self.optimization_history) > 1:
                initial_weights = self.optimization_history[0]['objective_weights']
                final_weights = self.optimization_history[-1]['objective_weights']
                
                weight_changes = {}
                for obj in initial_weights:
                    if obj in final_weights:
                        change = abs(final_weights[obj] - initial_weights[obj])
                        weight_changes[obj] = change
                
                convergence_analysis['objective_weight_changes'] = weight_changes
                convergence_analysis['max_weight_change'] = max(weight_changes.values()) if weight_changes else 0.0
        
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not analyze convergence patterns: {e}")
        
        return convergence_analysis
    
    def _setup_amope_gepa_integration(self):
        """Set up the integration between AMOPE's objective balancer and GEPA's evolution.
        
        This method hooks AMOPE's adaptive features into GEPA's genetic optimizer
        to create a synergistic optimization process.
        """
        # Override GEPA's fitness evaluation to include AMOPE's dynamic weighting
        original_fitness_function = self.gepa_optimizer.fitness_function
        
        def amope_gepa_fitness_function(candidate):
            """Integrated fitness function that combines GEPA evaluation with AMOPE's dynamic weighting."""
            # Get raw fitness scores from original evaluation
            raw_fitness = original_fitness_function(candidate)
            
            # Apply AMOPE's current dynamic weights
            weighted_fitness = {}
            for objective, score in raw_fitness.items():
                if objective in self.balancer.current_objectives:
                    weight = self.balancer.current_objectives[objective]
                    weighted_fitness[objective] = score * weight
                else:
                    weighted_fitness[objective] = score
            
            return weighted_fitness
        
        # Replace the fitness function
        self.gepa_optimizer.fitness_function = amope_gepa_fitness_function
        
        # Hook into generation end for objective balancing updates
        if hasattr(self.gepa_optimizer, 'post_generation_hook'):
            original_hook = self.gepa_optimizer.post_generation_hook
            
            def amope_post_generation_hook(population, generation):
                """Hook to update AMOPE's objective balancer after each GEPA generation."""
                # Call original hook if exists
                if original_hook:
                    original_hook(population, generation)
                
                # Update AMOPE's state
                self.current_generation = generation
                
                # Extract current best objectives for balancer update
                best_candidate = max(population, key=lambda c: sum(c.fitness_scores.values()))
                current_objectives = {}
                
                # Convert back to raw scores (without AMOPE weights) for balancer
                for objective in self.config.objectives.keys():
                    if objective in best_candidate.fitness_scores:
                        # Divide by AMOPE weight to get raw score
                        amope_weight = self.balancer.current_objectives.get(objective, 1.0)
                        raw_score = best_candidate.fitness_scores[objective] / amope_weight
                        current_objectives[objective] = raw_score
                
                # Update AMOPE's objective balancer
                self.balancer.update_fitness(current_objectives)
                
                # Record generation data
                generation_data = {
                    "generation": generation,
                    "best_score": sum(best_candidate.fitness_scores.values()),
                    "best_objectives": current_objectives,
                    "objective_weights": dict(self.balancer.current_objectives),
                    "population_diversity": self._calculate_population_diversity(population)
                }
                self.optimization_history.append(generation_data)
                
                # Check for stagnation and update counter
                if len(self.optimization_history) >= 2:
                    prev_score = self.optimization_history[-2]["best_score"]
                    curr_score = generation_data["best_score"]
                    improvement = (curr_score - prev_score) / max(prev_score, 0.001)
                    
                    if improvement < self.config.convergence_threshold:
                        self.stagnation_counter += 1
                    else:
                        self.stagnation_counter = 0
                
                if self.config.verbose and generation % 10 == 0:
                    print(f"Generation {generation}: Score={generation_data['best_score']:.4f}, "
                          f"Weights={self.balancer.current_objectives}")
            
            self.gepa_optimizer.post_generation_hook = amope_post_generation_hook
        
        if self.config.verbose:
            print("AMOPE-GEPA integration hooks established")
    
    def _calculate_population_diversity(self, population) -> float:
        """Calculate diversity measure for the current population.
        
        Args:
            population: List of GEPA Candidate objects
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(population) < 2:
            return 0.0
        
        # Simple diversity based on content length and character distribution
        contents = [c.content for c in population]
        avg_length = sum(len(c) for c in contents) / len(contents)
        
        # Calculate variance in lengths
        length_variance = sum((len(c) - avg_length) ** 2 for c in contents) / len(contents)
        normalized_variance = min(length_variance / (avg_length ** 2), 1.0)
        
        return normalized_variance
    
    def _update_amope_metrics_final(self, best_candidates: List, final_objectives: Dict, generations_completed: int):
        """Update final AMOPE metrics after GEPA optimization completes.
        
        Args:
            best_candidates: List of GEPA Candidate objects from Pareto front
            final_objectives: Final objective scores
            generations_completed: Total number of generations completed
        """
        # Update best candidate tracking
        if best_candidates:
            self.best_candidate = best_candidates[0]
        
        # Finalize strategy usage tracking
        self.strategy_usage = self._extract_gepa_strategy_usage(best_candidates)
        
        # Update final state
        self.current_generation = generations_completed
        
        # Add final optimization history entry
        if self.optimization_history:
            final_entry = {
                "generation": generations_completed,
                "best_score": self._evaluate_prompt(self.best_candidate.content if self.best_candidate else "", self._current_evaluation_fn),
                "best_objectives": final_objectives,
                "objective_weights": dict(self.balancer.current_objectives),
                "stagnation_counter": self.stagnation_counter,
                "pareto_front_size": len(best_candidates),
                "strategy_usage": dict(self.strategy_usage)
            }
            self.optimization_history.append(final_entry)
    
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
    
    def get_amope_gepa_status(self) -> Dict[str, Any]:
        """Get current status of AMOPE-GEPA integration.
        
        Returns:
            Dictionary containing current integration status and metrics
        """
        status = {
            'gepa_available': self._GEPA_AVAILABLE,
            'current_generation': self.current_generation,
            'stagnation_counter': self.stagnation_counter,
            'objective_weights': dict(self.balancer.current_objectives),
            'strategy_usage': dict(self.strategy_usage),
            'optimization_history_length': len(self.optimization_history)
        }
        
        if self._GEPA_AVAILABLE and hasattr(self, 'gepa_optimizer'):
            try:
                gepa_stats = self.gepa_optimizer.get_optimization_stats()
                status['gepa_stats'] = gepa_stats
            except Exception as e:
                status['gepa_stats_error'] = str(e)
        
        return status
    
    def force_objective_weight_update(self, new_weights: Dict[str, float]):
        """Manually update objective weights.
        
        This method allows external control over objective weights,
        useful for adaptive strategies or human guidance.
        
        Args:
            new_weights: Dictionary of objective names to new weights
        """
        for objective, weight in new_weights.items():
            if objective in self.balancer.current_objectives:
                old_weight = self.balancer.current_objectives[objective]
                self.balancer.current_objectives[objective] = weight
                
                if self.config.verbose:
                    print(f"Updated {objective} weight: {old_weight:.3f} -> {weight:.3f}")
        
        # If GEPA is running, update its fitness function to use new weights
        if self._GEPA_AVAILABLE and hasattr(self, 'gepa_optimizer'):
            self._setup_amope_gepa_integration()
    
    def _generate_gepa_context(self, population: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Generate GEPA context using AMOPE's PerformanceAnalyzer insights.
        
        This method provides AMOPE intelligence to GEPA's mutation process,
        including objective effectiveness, stagnation patterns, and strategy success.
        
        Args:
            population: Current population for analysis (optional)
            
        Returns:
            Dictionary containing AMOPE insights formatted for GEPA mutations
        """
        try:
            # Initialize PerformanceAnalyzer if not available
            if not hasattr(self, 'performance_analyzer'):
                from .adaptive_mutator import PerformanceAnalyzer
                self.performance_analyzer = PerformanceAnalyzer()
            
            # Build fitness history from optimization history
            fitness_history = []
            if self.optimization_history:
                for gen_data in self.optimization_history[-10:]:  # Last 10 generations
                    fitness_history.append(gen_data.get('best_objectives', {}))
            
            # Analyze gradient characteristics
            gradient_analysis = self.performance_analyzer.analyze_gradient(fitness_history)
            
            # Analyze objective effectiveness using balancer data
            objective_effectiveness = {}
            if hasattr(self.balancer, 'current_objectives'):
                current_weights = dict(self.balancer.current_objectives)
                
                # Calculate effectiveness based on weight changes and performance
                for obj_name, weight in current_weights.items():
                    # Get recent performance for this objective
                    recent_scores = []
                    for gen_data in self.optimization_history[-5:]:  # Last 5 generations
                        if obj_name in gen_data.get('best_objectives', {}):
                            recent_scores.append(gen_data['best_objectives'][obj_name])
                    
                    # Calculate effectiveness score
                    improvement = 0.0  # Initialize improvement
                    if recent_scores:
                        improvement = (recent_scores[-1] - recent_scores[0]) if len(recent_scores) > 1 else 0
                        effectiveness = min(1.0, max(0.0, improvement + weight * 0.1))
                    else:
                        effectiveness = weight * 0.5  # Default based on weight
                    
                    objective_effectiveness[obj_name] = {
                        "effectiveness_score": effectiveness,
                        "current_weight": weight,
                        "trend": "improving" if improvement > 0.01 else "stable" if improvement > -0.01 else "declining",
                        "recent_improvement": improvement
                    }
            
            # Analyze stagnation patterns
            stagnation_patterns = {
                "stagnation_counter": self.stagnation_counter,
                "stagnation_threshold": self.config.stagnation_generations,
                "is_stagnant": self.stagnation_counter >= self.config.stagnation_generations * 0.7,
                "diversity_metrics": self._calculate_population_diversity_for_context(population),
                "convergence_stage": gradient_analysis.get("trend", "stable")
            }
            
            # Analyze strategy success rates
            strategy_success_rates = {}
            if self.strategy_usage:
                total_mutations = sum(self.strategy_usage.values())
                for strategy, count in self.strategy_usage.items():
                    success_rate = count / total_mutations if total_mutations > 0 else 0
                    
                    # Estimate effectiveness based on recent performance
                    recent_success = self._estimate_strategy_success(strategy)
                    
                    strategy_success_rates[strategy] = {
                        "usage_rate": success_rate,
                        "estimated_effectiveness": recent_success,
                        "recommendation": self._get_strategy_recommendation(strategy, success_rate, recent_success)
                    }
            
            # Create mutation guidance
            mutation_guidance = {
                "focus_areas": self._identify_focus_areas(objective_effectiveness),
                "mutation_intensity": self._determine_mutation_intensity(stagnation_patterns),
                "recommended_strategies": self._get_recommended_strategies(strategy_success_rates),
                "avoid_patterns": self._identify_patterns_to_avoid(stagnation_patterns)
            }
            
            # Compile complete context
            gepa_context = {
                "amope_insights": {
                    "gradient_analysis": gradient_analysis,
                    "objective_effectiveness": objective_effectiveness,
                    "stagnation_patterns": stagnation_patterns,
                    "strategy_success_rates": strategy_success_rates,
                    "mutation_guidance": mutation_guidance,
                    "current_weights": dict(self.balancer.current_objectives) if hasattr(self.balancer, 'current_objectives') else {},
                    "optimization_phase": self._determine_optimization_phase()
                },
                "mutation_context": {
                    "generation": self.current_generation,
                    "total_generations": self.config.max_generations,
                    "progress_ratio": min(1.0, self.current_generation / max(1, self.config.max_generations)),
                    "best_score": self.optimization_history[-1]['best_score'] if self.optimization_history else 0.0,
                    "target_score": 0.95
                },
                "timestamp": time.time()
            }
            
            if self.config.verbose:
                print(f"Generated GEPA context with {len(objective_effectiveness)} objectives analyzed")
                print(f"Stagnation level: {stagnation_patterns['stagnation_counter']}/{stagnation_patterns['stagnation_threshold']}")
                print(f"Recommended strategies: {mutation_guidance['recommended_strategies']}")
            
            return gepa_context
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error generating GEPA context: {e}")
            
            # Return minimal context on error
            return {
                "amope_insights": {
                    "gradient_analysis": {"trend": "stable"},
                    "objective_effectiveness": {},
                    "stagnation_patterns": {"stagnation_counter": self.stagnation_counter},
                    "strategy_success_rates": {},
                    "mutation_guidance": {"recommended_strategies": ["adaptive"]}
                },
                "mutation_context": {
                    "generation": self.current_generation,
                    "progress_ratio": 0.0
                }
            }
    
    def _calculate_population_diversity_for_context(self, population: Optional[List[Any]]) -> Dict[str, float]:
        """Calculate diversity metrics for the current population."""
        if not population or len(population) < 2:
            return {"avg_distance": 0.0, "diversity_score": 0.0}
        
        try:
            # Simple diversity calculation based on content length differences
            contents = []
            for candidate in population:
                if hasattr(candidate, 'content'):
                    contents.append(candidate.content)
                elif isinstance(candidate, str):
                    contents.append(candidate)
            
            if len(contents) < 2:
                return {"avg_distance": 0.0, "diversity_score": 0.0}
            
            # Calculate pairwise distances (simplified)
            distances = []
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    # Simple distance based on length difference
                    len_diff = abs(len(contents[i]) - len(contents[j]))
                    max_len = max(len(contents[i]), len(contents[j]))
                    distance = len_diff / max_len if max_len > 0 else 0
                    distances.append(distance)
            
            avg_distance = sum(distances) / len(distances) if distances else 0
            diversity_score = min(1.0, avg_distance * 2)  # Normalize to [0, 1]
            
            return {
                "avg_distance": avg_distance,
                "diversity_score": diversity_score,
                "population_size": len(contents)
            }
            
        except Exception:
            return {"avg_distance": 0.0, "diversity_score": 0.0}
    
    def _identify_focus_areas(self, objective_effectiveness: Dict[str, Any]) -> List[str]:
        """Identify objectives that need focus based on effectiveness."""
        focus_areas = []
        
        for obj_name, effectiveness_data in objective_effectiveness.items():
            score = effectiveness_data.get("effectiveness_score", 0.5)
            trend = effectiveness_data.get("trend", "stable")
            
            # Focus on low effectiveness or declining objectives
            if score < 0.3 or trend == "declining":
                focus_areas.append(obj_name)
        
        return focus_areas
    
    def _determine_mutation_intensity(self, stagnation_patterns: Dict[str, Any]) -> str:
        """Determine appropriate mutation intensity based on stagnation."""
        stagnation_ratio = stagnation_patterns.get("stagnation_counter", 0) / max(1, stagnation_patterns.get("stagnation_threshold", 15))
        diversity_score = stagnation_patterns.get("diversity_metrics", {}).get("diversity_score", 0.5)
        
        if stagnation_ratio > 0.8 or diversity_score < 0.2:
            return "high"  # Need aggressive mutations
        elif stagnation_ratio > 0.5 or diversity_score < 0.4:
            return "medium"  # Moderate mutations needed
        else:
            return "low"  # Gentle mutations sufficient
    
    def _get_recommended_strategies(self, strategy_success_rates: Dict[str, Any]) -> List[str]:
        """Get recommended mutation strategies based on success rates."""
        recommendations = []
        
        for strategy, data in strategy_success_rates.items():
            effectiveness = data.get("estimated_effectiveness", 0.5)
            usage_rate = data.get("usage_rate", 0.0)
            
            # Recommend strategies with good effectiveness or underutilized ones
            if effectiveness > 0.6 or usage_rate < 0.1:
                recommendations.append(strategy)
        
        # Ensure at least one recommendation
        if not recommendations:
            recommendations = ["adaptive", "llm_guided"]
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _identify_patterns_to_avoid(self, stagnation_patterns: Dict[str, Any]) -> List[str]:
        """Identify patterns to avoid based on stagnation analysis."""
        patterns_to_avoid = []
        
        if stagnation_patterns.get("is_stagnant", False):
            patterns_to_avoid.extend(["minimal_changes", "conservative_mutations"])
        
        diversity_score = stagnation_patterns.get("diversity_metrics", {}).get("diversity_score", 0.5)
        if diversity_score < 0.3:
            patterns_to_avoid.extend(["repetitive_patterns", "similar_mutations"])
        
        return patterns_to_avoid
    
    def _estimate_strategy_success(self, strategy: str) -> float:
        """Estimate strategy success based on recent performance."""
        # Simple heuristic based on strategy type and current state
        if self.stagnation_counter > self.config.stagnation_generations * 0.7:
            # Stagnated - prefer aggressive strategies
            if strategy in ["llm_guided", "gradient_based"]:
                return 0.7
            elif strategy in ["pattern_based"]:
                return 0.4
            else:
                return 0.5
        else:
            # Normal progress - prefer balanced strategies
            if strategy in ["adaptive", "statistical"]:
                return 0.6
            elif strategy in ["llm_guided"]:
                return 0.5
            else:
                return 0.4
    
    def _get_strategy_recommendation(self, strategy: str, usage_rate: float, effectiveness: float) -> str:
        """Get recommendation for a specific strategy."""
        if effectiveness > 0.7 and usage_rate < 0.2:
            return "underutilized_high_potential"
        elif effectiveness > 0.6:
            return "effective"
        elif usage_rate > 0.5 and effectiveness < 0.3:
            return "overused_ineffective"
        else:
            return "experimental"
    
    def _determine_optimization_phase(self) -> str:
        """Determine current optimization phase."""
        progress_ratio = self.current_generation / max(1, self.config.max_generations)
        
        if progress_ratio < 0.2:
            return "exploration"
        elif progress_ratio < 0.7:
            return "exploitation"
        elif self.stagnation_counter > self.config.stagnation_generations * 0.7:
            return "diversification"
        else:
            return "convergence"
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get detailed insights from the optimization process.
        
        Returns:
            Dictionary containing optimization insights and recommendations
        """
        insights = {
            'performance_analysis': {},
            'recommendations': [],
            'bottlenecks': [],
            'success_factors': []
        }
        
        if not self.optimization_history:
            insights['recommendations'].append("Run optimization to generate insights")
            return insights
        
        try:
            # Analyze performance trends
            if len(self.optimization_history) >= 5:
                recent_scores = [entry['best_score'] for entry in self.optimization_history[-5:]]
                early_scores = [entry['best_score'] for entry in self.optimization_history[:5]]
                
                recent_avg = sum(recent_scores) / len(recent_scores)
                early_avg = sum(early_scores) / len(early_scores)
                
                improvement = (recent_avg - early_avg) / max(early_avg, 0.001)
                insights['performance_analysis']['overall_improvement'] = improvement
                
                if improvement < 0.1:
                    insights['bottlenecks'].append("Low overall improvement - consider adjusting objectives or mutation strategies")
                elif improvement > 0.5:
                    insights['success_factors'].append("Strong improvement trend - current configuration is effective")
            
            # Analyze strategy effectiveness
            if self.strategy_usage:
                total_usage = sum(self.strategy_usage.values())
                if total_usage > 0:
                    # Find most and least effective strategies
                    strategy_effectiveness = {}
                    for strategy, count in self.strategy_usage.items():
                        if isinstance(count, int) and count > 0:
                            strategy_effectiveness[strategy] = count / total_usage
                    
                    if strategy_effectiveness:
                        most_used = max(strategy_effectiveness, key=strategy_effectiveness.get)
                        insights['success_factors'].append(f"Most effective strategy: {most_used}")
                        
                        # Identify underutilized strategies
                        for strategy, usage in strategy_effectiveness.items():
                            if usage < 0.1:  # Less than 10% usage
                                insights['recommendations'].append(f"Consider increasing {strategy} strategy usage")
            
            # Analyze objective weight dynamics
            if len(self.optimization_history) >= 2:
                initial_weights = self.optimization_history[0]['objective_weights']
                final_weights = self.optimization_history[-1]['objective_weights']
                
                weight_changes = {}
                for obj in initial_weights:
                    if obj in final_weights:
                        change = abs(final_weights[obj] - initial_weights[obj])
                        weight_changes[obj] = change
                
                if weight_changes:
                    most_dynamic = max(weight_changes, key=weight_changes.get)
                    insights['success_factors'].append(f"Most dynamic objective: {most_dynamic}")
                    
                    # Check for stagnant objectives
                    for obj, change in weight_changes.items():
                        if change < 0.01:
                            insights['bottlenecks'].append(f"Objective {obj} shows little weight change - may need adjustment")
            
            # Stagnation analysis
            if self.stagnation_counter > self.config.stagnation_generations // 2:
                insights['bottlenecks'].append(f"High stagnation detected ({self.stagnation_counter} generations)")
                insights['recommendations'].append("Consider increasing mutation rate or adjusting objectives")
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not generate full insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def _safe_calculate_convergence_rate(self) -> float:
        """Safely calculate convergence rate with fallback error handling."""
        try:
            return self._calculate_convergence_rate()
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Error calculating convergence rate: {e}")
            return 0.0
    
    def _safe_calculate_final_objective_effectiveness(self) -> Dict[str, float]:
        """Safely calculate final objective effectiveness with fallback error handling."""
        try:
            return self._calculate_final_objective_effectiveness()
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Error calculating final objective effectiveness: {e}")
            return {}
    
    def _safe_generate_mutation_insights(self) -> Dict[str, Any]:
        """Safely generate mutation insights with fallback error handling."""
        try:
            return self._generate_mutation_insights()
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Error generating mutation insights: {e}")
            return {
                'strategy_performance': {},
                'usage_patterns': {},
                'recommendations': [f"Error generating insights: {str(e)}"],
                'overall_effectiveness': 0.0
            }
    
    def _safe_extract_performance_trajectory(self) -> List[Dict[str, Any]]:
        """Safely extract performance trajectory with fallback error handling."""
        try:
            return self._extract_performance_trajectory()
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Error extracting performance trajectory: {e}")
            return []
    
    def _safe_calculate_strategy_effectiveness(self) -> Dict[str, float]:
        """Safely calculate strategy effectiveness with fallback error handling."""
        try:
            return self._calculate_strategy_effectiveness()
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Error calculating strategy effectiveness: {e}")
            return {}
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate optimization convergence rate based on fitness history.
        
        Returns:
            Convergence rate as a float between 0 and 1
        """
        try:
            if len(self.optimization_history) < 2:
                return 0.0
            
            # Extract best scores from history
            scores = [entry.get('best_score', 0.0) for entry in self.optimization_history if 'best_score' in entry]
            
            if len(scores) < 2:
                return 0.0
            
            # Calculate initial and final performance
            initial_score = max(scores[0], 0.001)  # Avoid division by zero
            final_score = scores[-1]
            
            # Calculate convergence rate based on improvement trajectory
            if len(scores) >= 4:
                # Use slope of improvement over recent generations
                recent_scores = scores[-4:]
                improvements = []
                for i in range(1, len(recent_scores)):
                    if recent_scores[i-1] > 0:
                        improvement = (recent_scores[i] - recent_scores[i-1]) / recent_scores[i-1]
                        improvements.append(improvement)
                
                if improvements:
                    avg_improvement = sum(improvements) / len(improvements)
                    # Normalize convergence rate (negative improvement = lower convergence)
                    convergence_rate = max(0.0, min(1.0, 0.5 + avg_improvement * 10))
                else:
                    convergence_rate = 0.5
            else:
                # Simple improvement ratio for short histories
                total_improvement = (final_score - initial_score) / initial_score
                convergence_rate = max(0.0, min(1.0, 0.5 + total_improvement))
            
            # Factor in stagnation
            stagnation_penalty = min(0.3, self.stagnation_counter / max(1, self.config.stagnation_generations) * 0.3)
            convergence_rate = max(0.0, convergence_rate - stagnation_penalty)
            
            return convergence_rate
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate convergence rate: {e}")
            return 0.0
    
    def _calculate_final_objective_effectiveness(self) -> Dict[str, float]:
        """Calculate final effectiveness scores for objectives.
        
        Returns:
            Dictionary mapping objectives to their effectiveness scores
        """
        try:
            effectiveness = {}
            
            # Get current objective weights from balancer
            if hasattr(self.balancer, 'current_objectives') and self.balancer.current_objectives:
                current_weights = dict(self.balancer.current_objectives)
            else:
                return effectiveness
            
            # If we have optimization history, analyze objective performance
            if self.optimization_history:
                # Calculate effectiveness based on weight evolution and performance
                for objective in current_weights.keys():
                    obj_effectiveness = 0.0
                    
                    # Analyze weight stability (consistent weights suggest effectiveness)
                    weight_values = []
                    for entry in self.optimization_history:
                        if 'objective_weights' in entry and objective in entry['objective_weights']:
                            weight_values.append(entry['objective_weights'][objective])
                    
                    if weight_values:
                        # Calculate weight stability (less variation = more stable = more effective)
                        avg_weight = sum(weight_values) / len(weight_values)
                        if avg_weight > 0:
                            weight_variance = sum((w - avg_weight) ** 2 for w in weight_values) / len(weight_values)
                            stability_score = max(0.0, 1.0 - weight_variance)
                            obj_effectiveness += stability_score * 0.4
                    
                    # Factor in final weight (higher weight suggests more importance/effectiveness)
                    final_weight = current_weights.get(objective, 0.0)
                    obj_effectiveness += final_weight * 0.6
                    
                    effectiveness[objective] = min(1.0, obj_effectiveness)
            else:
                # No history - use current weights as effectiveness indicators
                total_weight = sum(current_weights.values())
                if total_weight > 0:
                    for objective, weight in current_weights.items():
                        effectiveness[objective] = weight / total_weight
            
            return effectiveness
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate final objective effectiveness: {e}")
            return {}
    
    def _generate_mutation_insights(self) -> Dict[str, Any]:
        """Generate insights about mutation strategy performance.
        
        Returns:
            Dictionary containing mutation strategy insights
        """
        try:
            insights = {
                'strategy_performance': {},
                'usage_patterns': {},
                'recommendations': [],
                'overall_effectiveness': 0.0
            }
            
            if not self.strategy_usage:
                insights['recommendations'].append("No mutation strategy usage data available")
                return insights
            
            # Analyze strategy usage patterns
            total_usage = sum(self.strategy_usage.values())
            if total_usage == 0:
                return insights
            
            # Calculate usage percentages
            for strategy, usage_count in self.strategy_usage.items():
                if isinstance(usage_count, int):
                    usage_percentage = usage_count / total_usage
                    insights['usage_patterns'][strategy] = usage_percentage
            
            # Analyze strategy effectiveness based on optimization progress
            if self.optimization_history and len(self.optimization_history) >= 2:
                initial_score = self.optimization_history[0].get('best_score', 0.0)
                final_score = self.optimization_history[-1].get('best_score', 0.0)
                overall_improvement = max(0.0, (final_score - initial_score) / max(initial_score, 0.001))
                insights['overall_effectiveness'] = min(1.0, overall_improvement)
                
                # Generate strategy-specific insights
                for strategy in self.strategy_usage.keys():
                    strategy_insights = {
                        'usage_rate': insights['usage_patterns'].get(strategy, 0.0),
                        'effectiveness_score': 0.0,
                        'recommendation': 'unknown'
                    }
                    
                    # Estimate effectiveness based on usage and overall improvement
                    usage_rate = strategy_insights['usage_rate']
                    if usage_rate > 0.3:  # Heavily used
                        if overall_improvement > 0.2:
                            strategy_insights['effectiveness_score'] = 0.8
                            strategy_insights['recommendation'] = 'highly_effective'
                        else:
                            strategy_insights['effectiveness_score'] = 0.3
                            strategy_insights['recommendation'] = 'overused_ineffective'
                    elif usage_rate > 0.1:  # Moderately used
                        strategy_insights['effectiveness_score'] = 0.6
                        strategy_insights['recommendation'] = 'moderately_effective'
                    else:  # Lightly used
                        strategy_insights['effectiveness_score'] = 0.4
                        strategy_insights['recommendation'] = 'underutilized'
                    
                    insights['strategy_performance'][strategy] = strategy_insights
            
            # Generate overall recommendations
            if insights['overall_effectiveness'] < 0.1:
                insights['recommendations'].append("Low overall effectiveness - consider adjusting mutation parameters")
            elif insights['overall_effectiveness'] > 0.5:
                insights['recommendations'].append("Good overall effectiveness - current strategies working well")
            
            # Check for strategy imbalances
            max_usage = max(insights['usage_patterns'].values()) if insights['usage_patterns'] else 0.0
            if max_usage > 0.7:
                dominant_strategy = max(insights['usage_patterns'], key=insights['usage_patterns'].get)
                insights['recommendations'].append(f"Strategy imbalance detected - {dominant_strategy} dominates usage")
            
            return insights
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not generate mutation insights: {e}")
            return {
                'strategy_performance': {},
                'usage_patterns': {},
                'recommendations': [f"Error generating insights: {str(e)}"],
                'overall_effectiveness': 0.0
            }
    
    def _extract_performance_trajectory(self) -> List[Dict[str, Any]]:
        """Extract performance trajectory from optimization history.
        
        Returns:
            List of performance data points for trajectory analysis
        """
        try:
            trajectory = []
            
            if not self.optimization_history:
                return trajectory
            
            for i, entry in enumerate(self.optimization_history):
                trajectory_point = {
                    'generation': i,
                    'best_score': entry.get('best_score', 0.0),
                    'avg_score': entry.get('avg_score', 0.0),
                    'stagnation_counter': entry.get('stagnation_counter', 0),
                    'objective_weights': entry.get('objective_weights', {}),
                    'timestamp': entry.get('timestamp', 0.0)
                }
                
                # Add additional metrics if available
                if 'strategy_usage' in entry:
                    trajectory_point['strategy_usage'] = entry['strategy_usage']
                
                if 'mutation_rate' in entry:
                    trajectory_point['mutation_rate'] = entry['mutation_rate']
                
                # Calculate improvement rate
                if i > 0 and trajectory:
                    prev_score = trajectory[-1]['best_score']
                    curr_score = trajectory_point['best_score']
                    if prev_score > 0:
                        improvement_rate = (curr_score - prev_score) / prev_score
                        trajectory_point['improvement_rate'] = improvement_rate
                    else:
                        trajectory_point['improvement_rate'] = 0.0
                else:
                    trajectory_point['improvement_rate'] = 0.0
                
                trajectory.append(trajectory_point)
            
            return trajectory
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not extract performance trajectory: {e}")
            return []
    
    def _calculate_strategy_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness scores for different mutation strategies.
        
        Returns:
            Dictionary mapping strategy names to effectiveness scores
        """
        try:
            effectiveness = {}
            
            if not self.strategy_usage:
                return effectiveness
            
            # Calculate basic usage-based effectiveness
            total_usage = sum(self.strategy_usage.values())
            if total_usage == 0:
                return effectiveness
            
            # Base effectiveness on usage patterns and optimization progress
            if self.optimization_history and len(self.optimization_history) >= 2:
                initial_score = self.optimization_history[0].get('best_score', 0.0)
                final_score = self.optimization_history[-1].get('best_score', 0.0)
                overall_improvement = max(0.0, (final_score - initial_score) / max(initial_score, 0.001))
                
                for strategy, usage_count in self.strategy_usage.items():
                    if isinstance(usage_count, int) and usage_count > 0:
                        usage_rate = usage_count / total_usage
                        
                        # Calculate effectiveness based on usage rate and overall improvement
                        if usage_rate > 0.5:  # Heavily used strategies
                            if overall_improvement > 0.3:
                                base_effectiveness = 0.8
                            elif overall_improvement > 0.1:
                                base_effectiveness = 0.6
                            else:
                                base_effectiveness = 0.3  # Overused but ineffective
                        elif usage_rate > 0.2:  # Moderately used
                            base_effectiveness = 0.7
                        elif usage_rate > 0.05:  # Lightly used
                            base_effectiveness = 0.5
                        else:  # Rarely used
                            base_effectiveness = 0.3
                        
                        # Adjust based on stagnation
                        stagnation_factor = max(0.0, 1.0 - (self.stagnation_counter / max(1, self.config.stagnation_generations)))
                        
                        effectiveness[strategy] = base_effectiveness * stagnation_factor
                    else:
                        effectiveness[strategy] = 0.0
            else:
                # No history - use usage-based effectiveness
                for strategy, usage_count in self.strategy_usage.items():
                    if isinstance(usage_count, int) and total_usage > 0:
                        effectiveness[strategy] = min(1.0, usage_count / total_usage)
            
            return effectiveness
            
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not calculate strategy effectiveness: {e}")
            return {}


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