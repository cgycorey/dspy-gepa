"""High-level agent interface for dspy-gepa.

Provides a simplified interface for prompt optimization
using the AMOPE algorithm internally.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..amope import AMOPEOptimizer, AMOPEConfig, OptimizationResult
from ..utils.logging import get_logger
from ..utils.config import (
    load_config, 
    get_config_value,
    load_llm_config,
    get_default_llm_provider,
    get_provider_config,
    is_llm_configured
)


_logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration in GEPAAgent.
    
    Provides comprehensive LLM configuration with auto-detection
    and fallback support for different providers.
    """
    provider: str = "openai"
    model: str = "gpt-4"
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
        return cls(
            provider=config_dict.get("provider", "openai"),
            model=config_dict.get("model", "gpt-4"),
            api_base=config_dict.get("api_base"),
            api_key=config_dict.get("api_key") or os.getenv(f"{config_dict.get('provider', 'openai').upper()}_API_KEY"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 2048),
            enabled=config_dict.get("enabled", True)
        )
    
    @classmethod
    def auto_detect(cls, config_path: Optional[str] = None) -> "LLMConfig":
        """Auto-detect LLM configuration from config.yaml and environment.
        
        Args:
            config_path: Path to config file for auto-detection
            
        Returns:
            LLMConfig with auto-detected settings
        """
        try:
            llm_config = load_llm_config(config_path)
            default_provider = llm_config.get("default_provider", "openai")
            provider_config = get_provider_config(default_provider)
            
            # Create config with detected settings
            config = cls(
                provider=default_provider,
                model=provider_config.get("model", "gpt-4"),
                api_base=provider_config.get("api_base"),
                api_key=provider_config.get("api_key") or os.getenv(f"{default_provider.upper()}_API_KEY"),
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens", 2048),
                enabled=True
            )
            
            # Check if LLM is actually available
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
    objectives: Dict[str, float]
    max_generations: int = 25
    population_size: int = 6
    mutation_rate: float = 0.1
    verbose: bool = True
    random_seed: Optional[int] = None
    llm_config: Optional[LLMConfig] = None
    use_llm_when_available: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create AgentConfig from dictionary."""
        objectives = config_dict.get("objectives", {"performance": 1.0})
        
        # Handle LLM configuration
        llm_config_dict = config_dict.get("llm_config")
        llm_config = None
        if llm_config_dict:
            if isinstance(llm_config_dict, dict):
                llm_config = LLMConfig.from_dict(llm_config_dict)
            else:
                llm_config = llm_config_dict  # Assume it's already an LLMConfig
        
        return cls(
            objectives=objectives,
            max_generations=config_dict.get("max_generations", 25),
            population_size=config_dict.get("population_size", 6),
            mutation_rate=config_dict.get("mutation_rate", 0.1),
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
    objectives_score: Dict[str, float]
    generations_completed: int
    total_evaluations: int
    optimization_time: float
    initial_score: float
    improvement: float
    
    @property
    def improvement_percentage(self) -> float:
        """Improvement as percentage."""
        if self.initial_score == 0:
            return 0.0
        return ((self.best_score - self.initial_score) / self.initial_score) * 100


class GEPAAgent:
    """High-level agent interface for GEPA optimization.
    
    The GEPAAgent provides a simplified, user-friendly interface for
    prompt optimization using the powerful AMOPE algorithm internally.
    It aims to make prompt optimization accessible while still providing
    access to advanced features when needed.
    
    Example:
        ```python
        from dspy_gepa import GEPAAgent
        
        # Define evaluation function
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            # Your evaluation logic here
            return {"accuracy": 0.8, "clarity": 0.7}
        
        # Create agent
        agent = GEPAAgent(
            objectives={"accuracy": 0.6, "clarity": 0.4},
            max_generations=20
        )
        
        # Optimize prompt
        result = agent.optimize_prompt("Initial prompt", evaluate_prompt)
        print(f"Best prompt: {result.best_prompt}")
        print(f"Improvement: {result.improvement_percentage:.1f}%")
        ```
    """
    
    def __init__(
        self,
        signature: Optional[Any] = None,
        name: Optional[str] = None,
        objectives: Optional[Dict[str, float]] = None,
        max_generations: Optional[int] = None,
        population_size: Optional[int] = None,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
        load_from_file: Optional[str] = None,
        llm_config: Optional[Union[Dict[str, Any], LLMConfig]] = None,
        auto_detect_llm: bool = True,
        **kwargs: Any
    ):
        """Initialize GEPAAgent.
        
        Args:
            signature: Optional signature for compatibility with test expectations
            name: Optional name for the agent
            objectives: Dictionary mapping objective names to weights
            max_generations: Maximum number of optimization generations
            population_size: Population size for optimization
            config: Configuration object or dictionary
            load_from_file: Path to configuration file
            llm_config: LLM configuration object or dictionary
            auto_detect_llm: Whether to auto-detect LLM configuration
            **kwargs: Additional configuration parameters
        """
        _logger.info("Initializing GEPAAgent with LLM support")
        
        # Load configuration
        if load_from_file:
            file_config = load_config(load_from_file)
            self.config = AgentConfig.from_dict(file_config)
        elif isinstance(config, AgentConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = AgentConfig.from_dict(config)
        else:
            # Load default config and override with parameters
            default_config = load_config()
            agent_config = AgentConfig.from_dict(default_config)
            
            # Override with explicit parameters
            if objectives:
                agent_config.objectives = objectives
            if max_generations:
                agent_config.max_generations = max_generations
            if population_size:
                agent_config.population_size = population_size
            
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(agent_config, key):
                    setattr(agent_config, key, value)
            
            self.config = agent_config
        
        # Handle LLM configuration
        if llm_config:
            if isinstance(llm_config, dict):
                self.config.llm_config = LLMConfig.from_dict(llm_config)
            else:
                self.config.llm_config = llm_config
        elif auto_detect_llm and not self.config.llm_config:
            # Auto-detect LLM configuration
            self.config.llm_config = LLMConfig.auto_detect(load_from_file)
        
        # Store additional parameters
        self.signature = signature
        self.name = name or f"gepa_agent_{id(self)}"
        
        # Initialize AMOPE optimizer with LLM support
        self._initialize_optimizer()
        
        # Optimization history
        self.optimization_history: List[OptimizationSummary] = []
        
        # Log LLM status
        if self.config.llm_config and self.config.llm_config.is_available:
            _logger.info(f"LLM available: {self.config.llm_config.provider} - {self.config.llm_config.model}")
        else:
            _logger.info("LLM not available - using handcrafted mutations only")
        
        _logger.info(f"GEPAAgent initialized with {len(self.config.objectives)} objectives")
    
    def _initialize_optimizer(self) -> None:
        """Initialize the AMOPE optimizer with LLM support."""
        # Prepare LLM configuration for AMOPE if available
        amope_llm_config = None
        if (self.config.llm_config and 
            self.config.llm_config.enabled and 
            self.config.llm_config.is_available and
            self.config.use_llm_when_available):
            
            amope_llm_config = {
                "provider": self.config.llm_config.provider,
                "model": self.config.llm_config.model,
                "api_base": self.config.llm_config.api_base,
                "api_key": self.config.llm_config.api_key,
                "temperature": self.config.llm_config.temperature,
                "max_tokens": self.config.llm_config.max_tokens
            }
        
        self.optimizer = AMOPEOptimizer(
            objectives=self.config.objectives,
            max_generations=self.config.max_generations,
            population_size=self.config.population_size,
            verbose=self.config.verbose,
            random_seed=self.config.random_seed,
            llm_config=amope_llm_config or {}
        )
    
    def optimize_prompt(
        self,
        initial_prompt: str,
        evaluation_fn: Callable[[str], Dict[str, float]],
        generations: Optional[int] = None,
        return_summary: bool = True
    ) -> Union[str, OptimizationSummary]:
        """Optimize a prompt using the AMOPE algorithm.
        
        Args:
            initial_prompt: Starting prompt for optimization
            evaluation_fn: Function that evaluates a prompt and returns objective scores
            generations: Number of generations to run (overrides config)
            return_summary: Whether to return full summary or just the best prompt
            
        Returns:
            Either the best prompt (str) or OptimizationSummary with detailed results
        """
        _logger.info(f"Starting prompt optimization: {initial_prompt[:50]}...")
        
        start_time = time.time()
        
        # Get initial score
        try:
            initial_objectives = evaluation_fn(initial_prompt)
            initial_score = self.optimizer._evaluate_prompt(initial_prompt, evaluation_fn)
        except Exception as e:
            _logger.error(f"Failed to evaluate initial prompt: {e}")
            raise ValueError(f"Initial prompt evaluation failed: {e}")
        
        if self.config.verbose:
            print(f"🚀 Starting GEPA optimization")
            print(f"📝 Initial prompt: {initial_prompt}")
            print(f"📊 Initial score: {initial_score:.4f}")
            print(f"🎯 Objectives: {self.config.objectives}")
            
            # Show LLM status
            llm_status = self.get_llm_status()
            if llm_status["will_use_llm"]:
                print(f"🤖 LLM: {llm_status['provider']} - {llm_status['model']} ({llm_status['mutation_type']})")
            else:
                print(f"🔧 Mutations: {llm_status['mutation_type']}")
                if not llm_status["available"]:
                    print(f"   Status: {llm_status['message']}")
        
        # Run optimization
        try:
            result = self.optimizer.optimize(
                initial_prompt=initial_prompt,
                evaluation_fn=evaluation_fn,
                generations=generations or self.config.max_generations
            )
        except Exception as e:
            _logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Prompt optimization failed: {e}")
        
        optimization_time = time.time() - start_time
        
        # Create summary
        summary = OptimizationSummary(
            best_prompt=result.best_prompt,
            best_score=result.best_score,
            objectives_score=result.best_objectives,
            generations_completed=result.generations_completed,
            total_evaluations=result.total_candidates_evaluated,
            optimization_time=optimization_time,
            initial_score=initial_score,
            improvement=result.best_score - initial_score
        )
        
        # Store in history
        self.optimization_history.append(summary)
        
        if self.config.verbose:
            print(f"✅ Optimization completed in {optimization_time:.1f}s")
            print(f"🎉 Best score: {result.best_score:.4f} (+{summary.improvement_percentage:.1f}%)")
            print(f"📝 Best prompt: {result.best_prompt}")
            if result.best_objectives:
                print(f"📊 Objectives: {result.best_objectives}")
            
            # Show what mutation type was actually used
            llm_status = self.get_llm_status()
            print(f"🔬 Mutations used: {llm_status['mutation_type']}")
        
        _logger.info(f"Optimization completed. Score: {result.best_score:.4f}, Improvement: {summary.improvement_percentage:.1f}%")
        
        return summary if return_summary else result.best_prompt
    
    def evaluate_current_best(
        self,
        evaluation_fn: Callable[[str], Dict[str, float]]
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
        current_objectives = evaluation_fn(last_summary.best_prompt)
        current_score = self.optimizer._evaluate_prompt(last_summary.best_prompt, evaluation_fn)
        
        return OptimizationSummary(
            best_prompt=last_summary.best_prompt,
            best_score=current_score,
            objectives_score=current_objectives,
            generations_completed=0,
            total_evaluations=1,
            optimization_time=0.0,
            initial_score=current_score,
            improvement=0.0
        )
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history.
        
        Returns:
            Dictionary with optimization insights and recommendations
        """
        if not self.optimization_history:
            return {"status": "No optimization history available"}
        
        # Get AMOPE insights if available
        try:
            amope_insights = self.optimizer.get_optimization_insights()
        except Exception:
            amope_insights = {}
        
        # Calculate statistics
        improvements = [h.improvement_percentage for h in self.optimization_history]
        
        insights = {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "best_improvement": max(improvements) if improvements else 0,
            "current_objectives": self.config.objectives,
            "optimization_config": {
                "max_generations": self.config.max_generations,
                "population_size": self.config.population_size
            },
            "llm_status": self.get_llm_status()
        }
        
        # Add best overall score if we have history
        if self.optimization_history:
            best_summary = max(self.optimization_history, key=lambda x: x.best_score)
            insights["best_overall_score"] = best_summary.best_score
        
        # Merge with AMOPE insights
        if amope_insights:
            insights["amope_insights"] = amope_insights
        
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
                "model": None,
                "will_use_llm": False
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
            "configuration_source": self.config.llm_config.configuration_source,
            "will_use_llm": llm_available and self.config.use_llm_when_available,
            "mutation_type": (
                "LLM-guided + handcrafted" if llm_available and self.config.use_llm_when_available
                else "handcrafted only"
            )
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
        
        # Test availability
        self.config.llm_config.is_available = is_llm_configured(provider)
        self.config.llm_config.configuration_source = "manual_configuration"
        
        # Reinitialize optimizer with new LLM config
        self._initialize_optimizer()
        
        status = "available" if self.config.llm_config.is_available else "not available"
        _logger.info(f"LLM configured: {provider} - {status}")
    
    def update_objectives(self, objectives: Dict[str, float]) -> None:
        """Update optimization objectives and reinitialize optimizer.
        
        Args:
            objectives: New objectives dictionary
        """
        _logger.info(f"Updating objectives: {objectives}")
        self.config.objectives = objectives
        self._initialize_optimizer()
    
    def reset_history(self) -> None:
        """Reset optimization history."""
        self.optimization_history.clear()
        _logger.info("Optimization history reset")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        llm_indicator = "🤖" if self.is_llm_available() else "🔧"
        return (
            f"GEPAAgent{llm_indicator}(objectives={len(self.config.objectives)}, "
            f"max_generations={self.config.max_generations}, "
            f"population_size={self.config.population_size})"
        )


_logger.debug("GEPAAgent module initialized")