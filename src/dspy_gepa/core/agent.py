"""Simplified GEPAAgent using the real GEPA library.

Provides a high-level interface for prompt optimization
using the real GEPA.optimize() function directly.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import the real GEPA library
try:
    from gepa import optimize
    from gepa.core.result import GEPAResult
    _GEPA_AVAILABLE = True
except ImportError as e:
    _GEPA_AVAILABLE = False
    optimize = None
    GEPAResult = None
    
    import warnings
    warnings.warn(
        f"GEPA package not available: {e}. Install with: pip install gepa",
        stacklevel=2
    )

# Try to import LiteLLM for error handling
try:
    import litellm
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    litellm = None

from ..utils.logging import get_logger
from ..utils.config import (
    configure_litellm,
    get_default_llm_provider,
    is_llm_configured,
    get_provider_config,
    validate_and_suggest_model
)


_logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration in GEPAAgent."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"  # Updated to gpt-4o-mini
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    enabled: bool = True
    
    # Auto-detected information
    is_available: bool = field(init=False, default=False)
    configuration_source: str = field(init=False, default="unknown")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from dictionary."""
        model = config_dict.get("model", "gpt-4o-mini")
        # Validate and suggest compatible model
        validated_model = validate_and_suggest_model(model)
        
        return cls(
            provider=config_dict.get("provider", "openai"),
            model=validated_model,
            api_base=config_dict.get("api_base"),
            api_key=config_dict.get("api_key") or os.getenv(f"{config_dict.get('provider', 'openai').upper()}_API_KEY"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 2048),
            enabled=config_dict.get("enabled", True)
        )
    
    @classmethod
    def auto_detect(cls, config_path: Optional[str] = None) -> "LLMConfig":
        """Auto-detect LLM configuration from config.yaml and environment."""
        try:
            llm_config = load_llm_config(config_path)
            default_provider = llm_config.get("default_provider", "openai")
            provider_config = get_provider_config(default_provider)
            
            model = provider_config.get("model", "gpt-4o-mini")
            # Validate and suggest compatible model
            validated_model = validate_and_suggest_model(model)
            
            config = cls(
                provider=default_provider,
                model=validated_model,
                api_base=provider_config.get("api_base"),
                api_key=provider_config.get("api_key") or os.getenv(f"{default_provider.upper()}_API_KEY"),
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens", 2048),
                enabled=True
            )
            
            config.is_available = is_llm_configured(default_provider)
            config.configuration_source = "config.yaml" if config.is_available else "config.yaml (incomplete)"
            
            return config
            
        except Exception as e:
            _logger.warning(f"Failed to auto-detect LLM config: {e}")
            config = cls(enabled=False)
            config.is_available = False
            config.configuration_source = "failed_detection"
            return config


@dataclass
class AgentConfig:
    """Configuration for GEPAAgent."""
    max_generations: int = 25
    population_size: int = 6
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.2
    verbose: bool = True
    random_seed: Optional[int] = None
    llm_config: Optional[LLMConfig] = None
    use_llm_when_available: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create AgentConfig from dictionary."""
        llm_config_dict = config_dict.get("llm_config")
        llm_config = None
        if llm_config_dict:
            if isinstance(llm_config_dict, dict):
                llm_config = LLMConfig.from_dict(llm_config_dict)
            else:
                llm_config = llm_config_dict
        
        return cls(
            max_generations=config_dict.get("max_generations", 25),
            population_size=config_dict.get("population_size", 6),
            mutation_rate=config_dict.get("mutation_rate", 0.1),
            crossover_rate=config_dict.get("crossover_rate", 0.7),
            elitism_rate=config_dict.get("elitism_rate", 0.2),
            verbose=config_dict.get("verbose", True),
            random_seed=config_dict.get("random_seed"),
            llm_config=llm_config,
            use_llm_when_available=config_dict.get("use_llm_when_available", True)
        )


@dataclass
class OptimizationSummary:
    """Summary of optimization results."""
    best_prompt: str
    best_score: float
    initial_score: float
    generations_completed: int
    total_evaluations: int
    optimization_time: float
    improvement: float
    
    @property
    def improvement_percentage(self) -> float:
        """Improvement as percentage."""
        if self.initial_score == 0:
            return 0.0
        return ((self.best_score - self.initial_score) / self.initial_score) * 100


class GEPAAgent:
    """Simplified GEPAAgent using the real GEPA library.
    
    The GEPAAgent provides a clean, user-friendly interface for
    prompt optimization using the real GEPA.optimize() function.
    It removes the complexity of AMOPE and provides direct access
    to GEPA's genetic programming capabilities.
    
    Example:
        ```python
        from dspy_gepa import GEPAAgent
        
        # Define evaluation function
        def evaluate_prompt(prompt: str) -> float:
            # Your evaluation logic here
            return 0.8  # Score between 0.0 and 1.0
        
        # Create agent
        agent = GEPAAgent(max_generations=20)
        
        # Optimize prompt
        result = agent.optimize_prompt("Initial prompt", evaluate_prompt)
        print(f"Best prompt: {result.best_prompt}")
        print(f"Improvement: {result.improvement_percentage:.1f}%")
        ```
    """
    
    def __init__(
        self,
        max_generations: Optional[int] = None,
        population_size: Optional[int] = None,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
        load_from_file: Optional[str] = None,
        llm_config: Optional[Union[Dict[str, Any], LLMConfig]] = None,
        auto_detect_llm: bool = True,
        **kwargs: Any
    ):
        """Initialize GEPAAgent with real GEPA support.
        
        Args:
            max_generations: Maximum number of optimization generations
            population_size: Population size for optimization
            config: Configuration object or dictionary
            load_from_file: Path to configuration file
            llm_config: LLM configuration object or dictionary
            auto_detect_llm: Whether to auto-detect LLM configuration
            **kwargs: Additional configuration parameters
        """
        if not _GEPA_AVAILABLE:
            raise ImportError(
                "GEPA package is required. Install with: pip install gepa"
            )
        
        _logger.info("Initializing GEPAAgent with simplified configuration")
        
        # Configure LiteLLM first
        configure_litellm()
        
        # Extract verbose from kwargs
        verbose = kwargs.get('verbose', True)
        
        # Create default configuration
        self.config = AgentConfig(
            max_generations=max_generations or 10,
            population_size=population_size or 6,
            mutation_rate=0.3,  # Fixed default values
            crossover_rate=0.7,
            elitism_rate=0.2,
            verbose=verbose
        )
        
        # Handle LLM configuration
        if llm_config:
            if isinstance(llm_config, dict):
                self.config.llm_config = LLMConfig.from_dict(llm_config)
            else:
                self.config.llm_config = llm_config
        elif auto_detect_llm and not self.config.llm_config:
            self.config.llm_config = LLMConfig.auto_detect(load_from_file)
        
        # Initialize LLM client if available
        self._llm_client = None
        self._init_llm_client()
        
        # Optimization history
        self.optimization_history: List[OptimizationSummary] = []
        
        # Log status
        if self.config.verbose:
            print(f"🚀 GEPAAgent initialized with real GEPA library")
            print(f"📊 Max generations: {self.config.max_generations}")
            print(f"👥 Population size: {self.config.population_size}")
            
            llm_status = self.get_llm_status()
            if llm_status["available"]:
                print(f"🤖 LLM: {llm_status['provider']} - {llm_status['model']}")
            else:
                print(f"🔧 LLM not available - using evaluation functions only")
        
        _logger.info(f"GEPAAgent initialized successfully")
    
    def _init_llm_client(self) -> None:
        """Initialize the LLM client if available."""
        if not self.config.llm_config or not self.config.llm_config.enabled:
            return
        
        provider = self.config.llm_config.provider.lower()
        
        try:
            if provider == "openai":
                import openai
                api_key = self.config.llm_config.api_key or os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._llm_client = openai.OpenAI(api_key=api_key)
                    self.config.llm_config.is_available = True
                else:
                    self.config.llm_config.is_available = False
            
            elif provider == "anthropic":
                import anthropic
                api_key = self.config.llm_config.api_key or os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self._llm_client = anthropic.Anthropic(api_key=api_key)
                    self.config.llm_config.is_available = True
                else:
                    self.config.llm_config.is_available = False
            
            else:
                self.config.llm_config.is_available = False
                
        except ImportError:
            self.config.llm_config.is_available = False
    
    def _generate_llm_response(self, prompt: str) -> Optional[str]:
        """Generate response from LLM if available."""
        if not self._llm_client or not self.config.llm_config.is_available:
            return None
        
        try:
            provider = self.config.llm_config.provider.lower()
            
            if provider == "openai":
                response = self._llm_client.chat.completions.create(
                    model=self.config.llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.llm_config.temperature,
                    max_tokens=self.config.llm_config.max_tokens
                )
                return response.choices[0].message.content
            
            elif provider == "anthropic":
                response = self._llm_client.messages.create(
                    model=self.config.llm_config.model,
                    max_tokens=self.config.llm_config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.llm_config.temperature
                )
                return response.content[0].text
        
        except Exception as e:
            # Handle LiteLLM UnsupportedParamsError specifically
            if _LITELLM_AVAILABLE and hasattr(litellm, 'UnsupportedParamsError'):
                if isinstance(e, litellm.UnsupportedParamsError):
                    _logger.warning(f"LLM parameter error: {e}. Try using a model that supports structured outputs like gpt-4o-mini.")
                    return None
            
            _logger.warning(f"LLM generation failed: {e}")
            return None
    
    def optimize_prompt(
        self,
        initial_prompt: str,
        evaluation_fn: Callable[[str], float],
        generations: Optional[int] = None,
        return_summary: bool = True
    ) -> Union[str, OptimizationSummary]:
        """Optimize a prompt using the real GEPA library.
        
        Args:
            initial_prompt: Starting prompt for optimization
            evaluation_fn: Function that evaluates a prompt and returns a score (0.0 to 1.0)
            generations: Number of generations to run (overrides config)
            return_summary: Whether to return full summary or just the best prompt
            
        Returns:
            Either the best prompt (str) or OptimizationSummary with detailed results
        """
        _logger.info(f"Starting GEPA optimization: {initial_prompt[:50]}...")
        
        start_time = time.time()
        
        # Get initial score
        try:
            initial_score = evaluation_fn(initial_prompt)
        except Exception as e:
            _logger.error(f"Failed to evaluate initial prompt: {e}")
            raise ValueError(f"Initial prompt evaluation failed: {e}")
        
        if self.config.verbose:
            print(f"🚀 Starting GEPA optimization with real GEPA library")
            print(f"📝 Initial prompt: {initial_prompt}")
            print(f"📊 Initial score: {initial_score:.4f}")
            
            # Show LLM status
            llm_status = self.get_llm_status()
            if llm_status["available"]:
                print(f"🤖 LLM available: {llm_status['provider']} - {llm_status['model']}")
            else:
                print(f"🔧 Using evaluation function only (no LLM)")
        
        # Define evaluation function for GEPA
        def gepa_evaluate_fn(candidate):
            """Evaluation function for GEPA."""
            # Extract prompt from candidate
            if hasattr(candidate, 'content'):
                prompt = candidate.content
            elif hasattr(candidate, 'prompt'):
                prompt = candidate.prompt
            elif isinstance(candidate, dict):
                prompt = candidate.get('prompt', candidate.get('content', str(candidate)))
            else:
                prompt = str(candidate)
            
            # Evaluate using the provided function
            try:
                return evaluation_fn(prompt)
            except Exception as e:
                _logger.warning(f"Evaluation failed for candidate: {e}")
                return 0.0
        
        # Prepare seed candidate for GEPA
        seed_candidate = {
            "prompt": initial_prompt,
            "content": initial_prompt
        }
        
        # Configure GEPA parameters
        gepa_config = {
            "max_generations": generations or self.config.max_generations,
            "population_size": self.config.population_size,
            "mutation_rate": self.config.mutation_rate,
            "crossover_rate": self.config.crossover_rate,
            "elitism_rate": self.config.elitism_rate,
            "random_seed": self.config.random_seed
        }
        
        try:
            # Create a simple dataset for GEPA
            # We'll use a single dummy data point since we have a simple evaluation function
            from gepa.core.adapter import GEPAAdapter
            from typing import Any, List
            
            from gepa.core.adapter import EvaluationBatch
            
            class SimpleEvalAdapter:
                """Simple adapter that uses our evaluation function."""
                
                def __init__(self, eval_fn):
                    self.eval_fn = eval_fn
                
                def evaluate(self, batch: List[Any], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
                    """Evaluate candidate using our evaluation function."""
                    # Extract prompt from candidate
                    prompt = candidate.get('prompt', candidate.get('content', str(candidate)))
                    
                    # Evaluate for each item in batch (all will get same score)
                    outputs = []
                    scores = []
                    trajectories = [] if capture_traces else None
                    
                    for item in batch:
                        try:
                            score = self.eval_fn(prompt)
                            scores.append(float(score))
                            outputs.append(f"Evaluated: {prompt[:50]}...")  # Simple output
                            if capture_traces:
                                trajectories.append({"prompt": prompt, "score": score, "item": item})
                        except Exception as e:
                            _logger.warning(f"Evaluation failed: {e}")
                            scores.append(0.0)
                            outputs.append(f"Error: {str(e)}")
                            if capture_traces:
                                trajectories.append({"prompt": prompt, "error": str(e), "item": item})
                    
                    return EvaluationBatch(
                        outputs=outputs,
                        scores=scores,
                        trajectories=trajectories
                    )
                
                def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: List[str]) -> dict[str, List[dict]]:
                    """Build reflective dataset for component updates."""
                    datasets = {}
                    for component in components_to_update:
                        dataset = []
                        for i, (score, trajectory) in enumerate(zip(eval_batch.scores, eval_batch.trajectories or [])):
                            dataset.append({
                                "Inputs": {"prompt": candidate.get(component, "")},
                                "Generated Outputs": eval_batch.outputs[i] if i < len(eval_batch.outputs) else "",
                                "Feedback": f"Score: {score:.3f}"
                            })
                        datasets[component] = dataset
                    return datasets
            
            # Create adapter
            adapter = SimpleEvalAdapter(evaluation_fn)
            
            # Create simple dataset - just dummy data points
            trainset = [{'dummy': i} for i in range(3)]  # 3 dummy examples
            valset = [{'dummy': i} for i in range(1)]   # 1 validation example
            
            # Run GEPA optimization with proper API
            # Use a simple reflection LM - we'll use a dummy one for testing
            result = optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                adapter=adapter,
                reflection_lm="gpt-4o-mini",  # Use a simple model identifier
                max_metric_calls=(generations or self.config.max_generations) * 3  # Rough estimate
            )
            
            optimization_time = time.time() - start_time
            
            # Extract results from GEPA result
            if hasattr(result, 'best_candidate'):
                best_prompt = result.best_candidate.content if hasattr(result.best_candidate, 'content') else str(result.best_candidate)
                best_score = getattr(result, 'best_fitness', evaluation_fn(best_prompt))
                generations_completed = getattr(result, 'generation', gepa_config["max_generations"])
                total_evaluations = getattr(result, 'evaluations_count', generations_completed * self.config.population_size)
            else:
                # Fallback if result structure is different
                best_prompt = initial_prompt
                best_score = initial_score
                generations_completed = gepa_config["max_generations"]
                total_evaluations = generations_completed * self.config.population_size
            
            # Re-evaluate best prompt to confirm
            final_score = evaluation_fn(best_prompt)
            
            improvement = final_score - initial_score
            
            # Create summary
            summary = OptimizationSummary(
                best_prompt=best_prompt,
                best_score=final_score,
                initial_score=initial_score,
                generations_completed=generations_completed,
                total_evaluations=total_evaluations,
                optimization_time=optimization_time,
                improvement=improvement
            )
            
            # Store in history
            self.optimization_history.append(summary)
            
            if self.config.verbose:
                print(f"✅ GEPA optimization completed in {optimization_time:.1f}s")
                print(f"🎉 Final score: {final_score:.4f} (+{summary.improvement_percentage:.1f}%)")
                print(f"📝 Best prompt: {best_prompt}")
                print(f"🔬 Generations: {generations_completed}, Evaluations: {total_evaluations}")
            
            _logger.info(f"GEPA optimization completed. Score: {final_score:.4f}, Improvement: {summary.improvement_percentage:.1f}%")
            
            return summary if return_summary else best_prompt
            
        except Exception as e:
            _logger.error(f"GEPA optimization failed: {e}")
            raise RuntimeError(f"Prompt optimization failed: {e}")
    
    def evaluate_current_best(
        self,
        evaluation_fn: Callable[[str], float]
    ) -> Optional[OptimizationSummary]:
        """Evaluate the current best prompt from history.
        
        Args:
            evaluation_fn: Function to evaluate the prompt
            
        Returns:
            OptimizationSummary with current evaluation or None if no history
        """
        if not self.optimization_history:
            return None
        
        last_summary = self.optimization_history[-1]
        
        # Re-evaluate the best prompt
        current_score = evaluation_fn(last_summary.best_prompt)
        
        return OptimizationSummary(
            best_prompt=last_summary.best_prompt,
            best_score=current_score,
            initial_score=current_score,
            generations_completed=0,
            total_evaluations=1,
            optimization_time=0.0,
            improvement=0.0
        )
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history.
        
        Returns:
            Dictionary with optimization insights and recommendations
        """
        if not self.optimization_history:
            return {"status": "No optimization history available"}
        
        # Calculate statistics
        improvements = [h.improvement_percentage for h in self.optimization_history]
        
        insights = {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "best_improvement": max(improvements) if improvements else 0,
            "optimization_config": {
                "max_generations": self.config.max_generations,
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
                "elitism_rate": self.config.elitism_rate
            },
            "llm_status": self.get_llm_status()
        }
        
        # Add best overall score if we have history
        if self.optimization_history:
            best_summary = max(self.optimization_history, key=lambda x: x.best_score)
            insights["best_overall_score"] = best_summary.best_score
        
        return insights
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available for optimization.
        
        Returns:
            True if LLM is configured and available, False otherwise
        """
        return (
            self.config.llm_config is not None and
            self.config.llm_config.enabled and
            self.config.llm_config.is_available and
            self.config.use_llm_when_available
        )
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get comprehensive LLM status information.
        
        Returns:
            Dictionary with LLM status details
        """
        if not self.config.llm_config:
            return {
                "status": "not_configured",
                "message": "LLM not configured",
                "available": False,
                "provider": None,
                "model": None
            }
        
        llm_available = self.is_llm_available()
        
        return {
            "status": "available" if llm_available else "unavailable",
            "message": (
                f"LLM configured and available via {self.config.llm_config.configuration_source}"
                if llm_available else
                f"LLM configured but not available: {self.config.llm_config.configuration_source}"
            ),
            "available": llm_available,
            "provider": self.config.llm_config.provider,
            "model": self.config.llm_config.model,
            "api_base": self.config.llm_config.api_base,
            "temperature": self.config.llm_config.temperature,
            "max_tokens": self.config.llm_config.max_tokens,
            "configuration_source": self.config.llm_config.configuration_source
        }
    
    def configure_llm(self, provider: str, **kwargs) -> None:
        """Configure LLM settings manually.
        
        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic')
            **kwargs: Additional LLM configuration parameters
        """
        _logger.info(f"Configuring LLM: {provider}")
        
        # Create new LLM config
        config_dict = {"provider": provider, **kwargs}
        self.config.llm_config = LLMConfig.from_dict(config_dict)
        
        # Reinitialize LLM client
        self._init_llm_client()
        
        status = "available" if self.config.llm_config.is_available else "not available"
        _logger.info(f"LLM configured: {provider} - {status}")
    
    def reset_history(self) -> None:
        """Reset optimization history."""
        self.optimization_history.clear()
        _logger.info("Optimization history reset")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        llm_indicator = "🤖" if self.is_llm_available() else "🔧"
        return (
            f"GEPAAgent{llm_indicator}(max_generations={self.config.max_generations}, "
            f"population_size={self.config.population_size})"
        )


_logger.debug("GEPAAgent module initialized with real GEPA library")