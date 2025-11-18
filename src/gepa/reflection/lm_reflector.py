"""Main LLM reflection class for GEPA.

This module provides the LMReflector class that handles LLM-based reflection
on execution traces and performance metrics, generating actionable suggestions
for improvement.

The LMReflector supports multiple LLM providers and is designed to work with
different types of text components (prompts, code, specifications).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.candidate import Candidate, ExecutionTrace

# Import dependency handler for graceful optional dependency management
try:
    from dspy_gepa.utils.dependency_handler import (
        DependencyError, 
        require_openai, 
        require_anthropic,
        is_openai_available,
        is_anthropic_available,
        create_mock_provider
    )
except ImportError:
    # Fallback for development environment
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dspy_gepa.utils.dependency_handler import (
        DependencyError, 
        require_openai, 
        require_anthropic,
        is_openai_available,
        is_anthropic_available,
        create_mock_provider
    )


class ReflectionConfig(BaseModel):
    """Configuration for LLM reflection."""
    
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    
    # Reflection-specific settings
    focus_areas: List[str] = Field(default_factory=lambda: ["performance", "efficiency", "robustness"])
    verbosity: int = Field(default=2, ge=1, le=5)  # 1=concise, 5=verbose
    include_suggestions: bool = Field(default=True)
    include_code_examples: bool = Field(default=False)
    
    class Config:
        validate_assignment = True


class ReflectionResult(BaseModel):
    """Result of LLM reflection on a candidate."""
    
    candidate_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    reflection_summary: str
    detailed_analysis: str
    improvement_suggestions: List[str] = Field(default_factory=list)
    priority_areas: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Metadata about the reflection
    llm_provider: str
    llm_model: str
    processing_time: float
    token_usage: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_reflection(self, prompt: str, config: ReflectionConfig) -> str:
        """Generate reflection using the LLM.
        
        Args:
            prompt: The reflection prompt
            config: Configuration for generation
            
        Returns:
            Generated reflection text
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider for LLM reflection."""
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            use_mock: If True, use mock provider instead of real OpenAI
        """
        # Always check if we should use mock
        if use_mock or not is_openai_available():
            if not is_openai_available():
                print("⚠️ OpenAI library not available. Using mock provider.")
                print("   Install with: pip install 'dspy-gepa[openai]' or pip install openai")
            self.client = create_mock_provider("openai")
            self.provider_name = "openai-mock"
            return
        
        # Try to initialize real OpenAI client
        try:
            openai = require_openai()
            # Check if API key is available
            import os
            if not api_key and not os.getenv('OPENAI_API_KEY'):
                print("⚠️ No OpenAI API key found. Using mock provider.")
                print("   Set OPENAI_API_KEY environment variable or pass api_key parameter")
                self.client = create_mock_provider("openai")
                self.provider_name = "openai-mock"
                return
                
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.provider_name = "openai"
        except DependencyError as e:
            print(f"⚠️ {e}")
            self.client = create_mock_provider("openai")
            self.provider_name = "openai-mock"
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI client: {e}. Using mock provider.")
            self.client = create_mock_provider("openai")
            self.provider_name = "openai-mock"
    
    async def generate_reflection(self, prompt: str, config: ReflectionConfig) -> str:
        """Generate reflection using OpenAI."""
        if self.provider_name == "openai-mock":
            # Use mock provider
            return await self.client.generate_reflection(prompt, config)
        
        try:
            response = await self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing algorithm performance and providing actionable improvement suggestions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def get_provider_name(self) -> str:
        return self.provider_name


class AnthropicProvider(LLMProvider):
    """Anthropic provider for LLM reflection."""
    
    def __init__(self, api_key: Optional[str] = None, use_mock: bool = False):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (if None, uses environment variable)
            use_mock: If True, use mock provider instead of real Anthropic
        """
        # Always check if we should use mock
        if use_mock or not is_anthropic_available():
            if not is_anthropic_available():
                print("⚠️ Anthropic library not available. Using mock provider.")
                print("   Install with: pip install 'dspy-gepa[anthropic]' or pip install anthropic")
            self.client = create_mock_provider("anthropic")
            self.provider_name = "anthropic-mock"
            return
        
        # Try to initialize real Anthropic client
        try:
            anthropic = require_anthropic()
            # Check if API key is available
            import os
            if not api_key and not os.getenv('ANTHROPIC_API_KEY'):
                print("⚠️ No Anthropic API key found. Using mock provider.")
                print("   Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
                self.client = create_mock_provider("anthropic")
                self.provider_name = "anthropic-mock"
                return
                
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.provider_name = "anthropic"
        except DependencyError as e:
            print(f"⚠️ {e}")
            self.client = create_mock_provider("anthropic")
            self.provider_name = "anthropic-mock"
        except Exception as e:
            print(f"⚠️ Failed to initialize Anthropic client: {e}. Using mock provider.")
            self.client = create_mock_provider("anthropic")
            self.provider_name = "anthropic-mock"
    
    async def generate_reflection(self, prompt: str, config: ReflectionConfig) -> str:
        """Generate reflection using Anthropic."""
        if self.provider_name == "anthropic-mock":
            # Use mock provider
            return await self.client.generate_reflection(prompt, config)
        
        try:
            response = await self.client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[
                    {"role": "user", "content": f"You are an expert at analyzing algorithm performance and providing actionable improvement suggestions.\n\n{prompt}"}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def get_provider_name(self) -> str:
        return self.provider_name


class LocalProvider(LLMProvider):
    """Local LLM provider for offline reflection."""
    
    def __init__(self, model_path: str = "llama2"):
        """Initialize local provider.
        
        Args:
            model_path: Path or name of the local model
        """
        self.model_path = model_path
        self.provider_name = "local"
        # Note: Implement actual local model loading based on your preferred framework
    
    async def generate_reflection(self, prompt: str, config: ReflectionConfig) -> str:
        """Generate reflection using local model."""
        # Placeholder implementation
        # In practice, you would integrate with transformers, llama.cpp, etc.
        return f"Local model reflection for: {prompt[:100]}..."
    
    def get_provider_name(self) -> str:
        return self.provider_name


class LMReflector:
    """Main LLM reflection class for GEPA.
    
    The LMReflector takes execution traces and performance metrics as input,
    generates detailed reflection feedback using LLMs, and provides actionable
    suggestions for improvement.
    
    Attributes:
        config: Configuration for reflection
        provider: LLM provider instance
        reflection_history: List of past reflections
    """
    
    def __init__(self, config: Optional[ReflectionConfig] = None,
                 provider: Optional[LLMProvider] = None):
        """Initialize the LMReflector.
        
        Args:
            config: Reflection configuration (if None, uses defaults)
            provider: LLM provider (if None, creates based on config)
        """
        self.config = config or ReflectionConfig()
        self.provider = provider or self._create_default_provider()
        self.reflection_history: List[ReflectionResult] = []
    
    def _create_default_provider(self) -> LLMProvider:
        """Create default LLM provider based on configuration."""
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "local": LocalProvider
        }
        
        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            available_providers = list(provider_map.keys())
            raise ValueError(
                f"Unsupported provider: {self.config.provider}. "
                f"Available providers: {available_providers}"
            )
        
        try:
            return provider_class()
        except ImportError as e:
            # Provide helpful guidance for missing dependencies
            if self.config.provider == "openai":
                raise ImportError(
                    f"Cannot initialize OpenAI provider: {e}. "
                    "Install with: pip install 'dspy-gepa[openai]' or pip install openai"
                ) from e
            elif self.config.provider == "anthropic":
                raise ImportError(
                    f"Cannot initialize Anthropic provider: {e}. "
                    "Install with: pip install 'dspy-gepa[anthropic]' or pip install anthropic"
                ) from e
            else:
                raise e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.config.provider} provider: {e}") from e
    
    async def reflect_on_candidate(self, candidate: Candidate,
                                 context: Optional[Dict[str, Any]] = None) -> ReflectionResult:
        """Generate reflection for a candidate.
        
        Args:
            candidate: Candidate to reflect on
            context: Additional context for reflection
            
        Returns:
            Reflection result with analysis and suggestions
        """
        start_time = time.time()
        
        # Generate reflection prompt
        prompt = self._create_reflection_prompt(candidate, context)
        
        # Generate reflection using LLM
        raw_reflection = await self._generate_with_retry(prompt)
        
        # Parse and structure the reflection
        reflection_result = self._parse_reflection_response(
            raw_reflection, candidate, time.time() - start_time
        )
        
        # Store in history
        self.reflection_history.append(reflection_result)
        
        return reflection_result
    
    def _create_reflection_prompt(self, candidate: Candidate,
                                context: Optional[Dict[str, Any]] = None) -> str:
        """Create reflection prompt for the candidate.
        
        Args:
            candidate: Candidate to analyze
            context: Additional context
            
        Returns:
            Formatted reflection prompt
        """
        prompt_parts = [
            "Please analyze the following candidate and provide detailed reflection and improvement suggestions.",
            "",
            "CANDIDATE INFORMATION:",
            f"Content: {candidate.content[:1000]}..." if len(candidate.content) > 1000 else f"Content: {candidate.content}",
            f"Generation: {candidate.generation}",
            f"Fitness Scores: {candidate.fitness_scores}",
        ]
        
        # Add execution trace analysis
        if candidate.execution_traces:
            recent_traces = candidate.execution_traces[-5:]  # Last 5 traces
            prompt_parts.extend([
                "",
                "RECENT EXECUTION TRACES:",
                *[f"- Trace {i+1}: Success={trace.success}, Time={trace.execution_time:.3f}s, "
                  f"Error={trace.error[:100] if trace.error else 'None'}"
                  for i, trace in enumerate(recent_traces)]
            ])
        
        # Add mutation history
        if candidate.mutation_history:
            recent_mutations = candidate.mutation_history[-3:]  # Last 3 mutations
            prompt_parts.extend([
                "",
                "RECENT MUTATIONS:",
                *[f"- {mut.mutation_type}: {mut.description}"
                  for mut in recent_mutations]
            ])
        
        # Add context information
        if context:
            prompt_parts.extend([
                "",
                "ADDITIONAL CONTEXT:",
                *[f"- {k}: {v}" for k, v in context.items()]
            ])
        
        # Add focus areas
        prompt_parts.extend([
            "",
            f"FOCUS AREAS: {', '.join(self.config.focus_areas)}",
            "",
            "Please provide:",
            "1. A concise reflection summary",
            "2. Detailed analysis of performance patterns",
            "3. Specific improvement suggestions",
            "4. Priority areas for optimization",
            "5. Confidence score in your analysis (0-1)",
            "",
            "Format your response as JSON with keys: summary, analysis, suggestions, priorities, confidence"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _generate_with_retry(self, prompt: str) -> str:
        """Generate reflection with retry logic."""
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.provider.generate_reflection(prompt, self.config)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise RuntimeError(f"Failed to generate reflection after {self.config.max_retries + 1} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_reflection_response(self, raw_response: str,
                                 candidate: Candidate,
                                 processing_time: float) -> ReflectionResult:
        """Parse LLM response into structured reflection result.
        
        Args:
            raw_response: Raw LLM response
            candidate: Candidate that was analyzed
            processing_time: Time taken for processing
            
        Returns:
            Structured reflection result
        """
        try:
            # Try to parse as JSON first
            if raw_response.strip().startswith('{'):
                data = json.loads(raw_response)
                return ReflectionResult(
                    candidate_id=candidate.id,
                    reflection_summary=data.get("summary", raw_response[:200]),
                    detailed_analysis=data.get("analysis", raw_response),
                    improvement_suggestions=data.get("suggestions", []),
                    priority_areas=data.get("priorities", []),
                    confidence_score=float(data.get("confidence", 0.5)),
                    llm_provider=self.provider.get_provider_name(),
                    llm_model=self.config.model,
                    processing_time=processing_time
                )
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: treat as plain text
        return ReflectionResult(
            candidate_id=candidate.id,
            reflection_summary=raw_response[:200],
            detailed_analysis=raw_response,
            improvement_suggestions=[],
            priority_areas=[],
            confidence_score=0.5,
            llm_provider=self.provider.get_provider_name(),
            llm_model=self.config.model,
            processing_time=processing_time
        )
    
    def get_reflection_history(self, candidate_id: Optional[str] = None) -> List[ReflectionResult]:
        """Get reflection history.
        
        Args:
            candidate_id: If provided, filter by candidate ID
            
        Returns:
            List of reflection results
        """
        if candidate_id:
            return [r for r in self.reflection_history if r.candidate_id == candidate_id]
        return self.reflection_history.copy()
    
    def clear_history(self) -> None:
        """Clear reflection history."""
        self.reflection_history.clear()
    
    def update_config(self, **kwargs) -> None:
        """Update reflection configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        config_dict = self.config.dict()
        config_dict.update(kwargs)
        self.config = ReflectionConfig.parse_obj(config_dict)
        
        # Update provider if provider changed
        if "provider" in kwargs:
            self.provider = self._create_default_provider()


# Utility function for async compatibility
import asyncio


def sync_reflect(reflector: LMReflector, candidate: Candidate,
                context: Optional[Dict[str, Any]] = None) -> ReflectionResult:
    """Synchronous wrapper for reflection.
    
    Args:
        reflector: LMReflector instance
        candidate: Candidate to reflect on
        context: Additional context
        
    Returns:
        Reflection result
    """
    return asyncio.run(reflector.reflect_on_candidate(candidate, context))


# Example usage and testing
if __name__ == "__main__":
    # Create a sample candidate for testing
    sample_candidate = Candidate(
        content="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        fitness_scores={"performance": 0.7, "efficiency": 0.3, "robustness": 0.8}
    )
    
    # Add sample execution trace
    sample_trace = ExecutionTrace(
        execution_time=1.5,
        success=True,
        output="55",
        metrics={"memory_mb": 128, "calls": 15}
    )
    sample_candidate.add_execution_trace(sample_trace)
    
    # Create reflector
    reflector = LMReflector()
    
    print("✓ LMReflector implementation ready for testing")
    print(f"Sample candidate: {sample_candidate}")
    print(f"Reflector config: {reflector.config}")