"""Text mutator for GEPA algorithm.

This module implements the TextMutator class which performs LLM-driven
mutations on text components (prompts, code, specifications) in the GEPA
(Genetic-Pareto Algorithm) framework.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .candidate import Candidate


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, candidate: Candidate, context: Optional[Dict[str, Any]] = None) -> str:
        """Apply mutation to a candidate.
        
        Args:
            candidate: Candidate to mutate
            context: Additional context for mutation
            
        Returns:
            Mutated content
        """
        pass


class LLMReflectionMutator(MutationStrategy):
    """LLM-based mutation using reflection feedback.
    
    This strategy uses an LLM to generate mutations based on reflection
    feedback and performance metrics. It's the core mutation strategy
    in the GEPA algorithm.
    """
    
    def __init__(self, llm_client: Optional[Any] = None, 
                 reflection_template: Optional[str] = None):
        """Initialize the LLM reflection mutator.
        
        Args:
            llm_client: LLM client for generating mutations
            reflection_template: Template for reflection prompts
        """
        self.llm_client = llm_client
        self.reflection_template = reflection_template or self._default_reflection_template()
    
    def _default_reflection_template(self) -> str:
        """Default reflection prompt template."""
        return """
Based on the performance feedback and current content, generate an improved version:

Current content:
{content}

Performance metrics:
{metrics}

Reflection feedback:
{reflection}

Please generate an improved version that addresses the feedback and improves performance.
Focus on maintaining the core functionality while making targeted improvements.

Generated content:
"""
    
    def mutate(self, candidate: Candidate, context: Optional[Dict[str, Any]] = None) -> str:
        """Apply LLM-based reflection mutation.
        
        Args:
            candidate: Candidate to mutate
            context: Context containing reflection feedback and metrics
            
        Returns:
            Mutated content
        """
        if not self.llm_client:
            # Fallback to simple text-based mutation if no LLM available
            return self._fallback_mutation(candidate.content)
        
        # Extract reflection and metrics from context
        reflection = context.get("reflection", "") if context else ""
        metrics = context.get("metrics", {}) if context else {}
        
        # Format reflection prompt
        prompt = self.reflection_template.format(
            content=candidate.content,
            metrics=self._format_metrics(metrics),
            reflection=reflection
        )
        
        try:
            # Generate mutation using LLM
            response = self.llm_client.generate(prompt)
            mutated_content = response.strip()
            
            # Ensure we got meaningful content
            if not mutated_content or len(mutated_content) < 10:
                return self._fallback_mutation(candidate.content)
            
            return mutated_content
        
        except Exception as e:
            # Fallback to simple mutation on error
            return self._fallback_mutation(candidate.content)
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for the reflection prompt."""
        if not metrics:
            return "No metrics available"
        
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{key}: {value:.3f}")
            else:
                formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)
    
    def _fallback_mutation(self, content: str) -> str:
        """Fallback mutation when LLM is not available."""
        # Simple text-based mutations
        mutations = [
            self._add_variation(content),
            self._rephrase_sentence(content),
            self._add_examples(content),
            self._remove_redundancy(content)
        ]
        
        return random.choice(mutations)
    
    def _add_variation(self, content: str) -> str:
        """Add variation to the content."""
        variations = [
            "Consider this approach: ",
            "Alternatively, you could: ",
            "For better results: ",
            "Try this method: "
        ]
        
        lines = content.split('\n')
        if len(lines) > 1:
            insert_pos = random.randint(1, len(lines) - 1)
            lines.insert(insert_pos, random.choice(variations))
            return '\n'.join(lines)
        
        return content + " " + random.choice(variations)
    
    def _rephrase_sentence(self, content: str) -> str:
        """Rephrase a random sentence in the content."""
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 1:
            # Pick a random sentence to rephrase
            target_idx = random.randint(0, len(sentences) - 2)
            original = sentences[target_idx].strip()
            
            if original:
                # Simple rephrasing patterns
                patterns = [
                    f"Please {original.lower()}",
                    f"Make sure to {original.lower()}",
                    f"You should {original.lower()}",
                    f"Consider {original.lower()}"
                ]
                
                sentences[target_idx] = random.choice(patterns)
                
        return '. '.join(filter(None, sentences))
    
    def _add_examples(self, content: str) -> str:
        """Add examples to the content."""
        example_templates = [
            "\n\nExample: {example}",
            "\n\nFor instance: {example}",
            "\n\nHere's an example: {example}",
            "\n\nConsider this case: {example}"
        ]
        
        # Generate a simple example based on content
        words = content.split()[:5]  # Use first 5 words as basis
        example = "...".join(words) + " [example output]"
        
        template = random.choice(example_templates)
        return content + template.format(example=example)
    
    def _remove_redundancy(self, content: str) -> str:
        """Remove redundant phrases from content."""
        redundant_phrases = [
            "please note that",
            "it is important to",
            "make sure that",
            "it should be noted that",
            "in order to"
        ]
        
        result = content
        for phrase in redundant_phrases:
            result = result.replace(phrase, "")
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        return result


class TextMutator:
    """Main text mutator class for GEPA algorithm.
    
    This class coordinates different mutation strategies and manages
    the mutation process for text components.
    
    Attributes:
        strategies: List of available mutation strategies
        mutation_rates: Dictionary of mutation rates for different strategies
        default_strategy: Default mutation strategy to use
    """
    
    def __init__(self, 
                 llm_client: Optional[Any] = None,
                 strategies: Optional[List[MutationStrategy]] = None,
                 mutation_rates: Optional[Dict[str, float]] = None):
        """Initialize the text mutator.
        
        Args:
            llm_client: LLM client for intelligent mutations
            strategies: List of mutation strategies to use
            mutation_rates: Mutation rates for different strategies
        """
        self.strategies = strategies or self._default_strategies(llm_client)
        self.mutation_rates = mutation_rates or self._default_mutation_rates()
        
        # Strategy name mapping
        self.strategy_map = {type(s).__name__: s for s in self.strategies}
        self.default_strategy = "LLMReflectionMutator"
    
    def _default_strategies(self, llm_client: Optional[Any]) -> List[MutationStrategy]:
        """Create default mutation strategies."""
        strategies = []
        
        # Add LLM-based mutation if client is available
        if llm_client:
            strategies.append(LLMReflectionMutator(llm_client))
        else:
            # Add fallback LLM mutator
            strategies.append(LLMReflectionMutator())
        
        return strategies
    
    def _default_mutation_rates(self) -> Dict[str, float]:
        """Default mutation rates for different strategies."""
        return {
            "LLMReflectionMutator": 0.7,  # High rate for intelligent mutations
        }
    
    def mutate_candidate(self, 
                        candidate: Candidate, 
                        strategy_name: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> Candidate:
        """Apply mutation to a candidate.
        
        Args:
            candidate: Candidate to mutate
            strategy_name: Name of mutation strategy to use
            context: Additional context for mutation (reflection, metrics, etc.)
            
        Returns:
            New mutated candidate
        """
        # Choose mutation strategy
        if strategy_name is None:
            strategy_name = self._choose_strategy()
        
        if strategy_name not in self.strategy_map:
            strategy_name = self.default_strategy
        
        strategy = self.strategy_map[strategy_name]
        
        # Apply mutation
        old_content = candidate.content
        mutated_content = strategy.mutate(candidate, context)
        
        # Create new candidate with mutated content
        mutated_candidate = candidate.copy()
        mutated_candidate.update_content(
            new_content=mutated_content,
            mutation_type=strategy_name,
            description=f"Applied {strategy_name} mutation",
            changes_made={
                "strategy": strategy_name,
                "old_length": len(old_content),
                "new_length": len(mutated_content),
                "context_provided": context is not None
            }
        )
        
        # Increment generation
        mutated_candidate.generation = candidate.generation + 1
        
        return mutated_candidate
    
    def _choose_strategy(self) -> str:
        """Choose a mutation strategy based on rates."""
        if not self.strategies:
            return self.default_strategy
        
        # Weighted random selection based on mutation rates
        strategies = list(self.strategy_map.keys())
        weights = [self.mutation_rates.get(s, 0.1) for s in strategies]
        
        return random.choices(strategies, weights=weights)[0]
    
    def add_strategy(self, strategy: MutationStrategy, rate: float = 0.1) -> None:
        """Add a new mutation strategy.
        
        Args:
            strategy: Mutation strategy to add
            rate: Mutation rate for this strategy
        """
        strategy_name = type(strategy).__name__
        self.strategy_map[strategy_name] = strategy
        self.mutation_rates[strategy_name] = rate
        
        if strategy not in self.strategies:
            self.strategies.append(strategy)
    
    def set_mutation_rate(self, strategy_name: str, rate: float) -> None:
        """Set mutation rate for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            rate: Mutation rate (0.0 to 1.0)
        """
        if 0.0 <= rate <= 1.0:
            self.mutation_rates[strategy_name] = rate
        else:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
    
    def batch_mutate(self, 
                    candidates: List[Candidate],
                    mutation_prob: float = 0.8,
                    context: Optional[Dict[str, Any]] = None) -> List[Candidate]:
        """Apply mutations to a batch of candidates.
        
        Args:
            candidates: List of candidates to mutate
            mutation_prob: Probability of mutating each candidate
            context: Context to apply to all mutations
            
        Returns:
            List of mutated candidates
        """
        mutated_candidates = []
        
        for candidate in candidates:
            if random.random() < mutation_prob:
                mutated_candidate = self.mutate_candidate(candidate, context=context)
                mutated_candidates.append(mutated_candidate)
            else:
                # Keep original candidate
                mutated_candidates.append(candidate)
        
        return mutated_candidates
    
    def adaptive_mutation(self, 
                         candidate: Candidate,
                         population_fitness: Dict[str, List[float]],
                         context: Optional[Dict[str, Any]] = None) -> Candidate:
        """Apply adaptive mutation based on population fitness distribution.
        
        Args:
            candidate: Candidate to mutate
            population_fitness: Fitness distribution of the current population
            context: Additional context for mutation
            
        Returns:
            Mutated candidate
        """
        # Analyze population fitness to determine mutation intensity
        strategy_name = self._select_adaptive_strategy(candidate, population_fitness)
        
        return self.mutate_candidate(candidate, strategy_name, context)
    
    def _select_adaptive_strategy(self, 
                                 candidate: Candidate,
                                 population_fitness: Dict[str, List[float]]) -> str:
        """Select mutation strategy based on population fitness."""
        # Simple adaptive strategy: if candidate is performing poorly,
        # use more aggressive mutations
        
        if not population_fitness or not candidate.fitness_scores:
            return self.default_strategy
        
        # Check if candidate is below median in any objective
        for obj, score in candidate.fitness_scores.items():
            if obj in population_fitness:
                scores = population_fitness[obj]
                if scores and score < sum(scores) / len(scores):
                    # Below average, use more aggressive strategy
                    return self.default_strategy
        
        # Candidate is performing well, use conservative mutation
        return self.default_strategy


# Example usage
if __name__ == "__main__":
    # Create a test candidate
    candidate = Candidate(
        content="Write a function that calculates the factorial of a number.",
        fitness_scores={"accuracy": 0.7, "efficiency": 0.6}
    )
    
    # Create mutator
    mutator = TextMutator()
    
    # Apply mutation
    mutated = mutator.mutate_candidate(
        candidate,
        context={
            "reflection": "The function should handle edge cases like negative numbers.",
            "metrics": {"accuracy": 0.7, "efficiency": 0.6}
        }
    )
    
    print(f"Original: {candidate.content}")
    print(f"Mutated: {mutated.content}")
    print(f"Generation: {mutated.generation}")
    print(f"Mutation history: {len(mutated.mutation_history)} mutations")
    
    # Test batch mutation
    candidates = [
        Candidate(content=f"Prompt {i}", fitness_scores={"quality": 0.5 + i * 0.1})
        for i in range(5)
    ]
    
    print(f"\nBatch mutation test:")
    mutated_batch = mutator.batch_mutate(candidates, mutation_prob=1.0)
    
    for i, (orig, mut) in enumerate(zip(candidates, mutated_batch)):
        print(f"  Candidate {i}: {orig.content} -> {mut.content}")
        print(f"    Generation: {mut.generation}")
    
    # Test adaptive mutation
    print(f"\nAdaptive mutation test:")
    population_fitness = {"accuracy": [0.5, 0.6, 0.7, 0.8, 0.9]}
    
    adaptive_mutated = mutator.adaptive_mutation(
        candidate,
        population_fitness,
        context={"metrics": {"accuracy": 0.7}}
    )
    
    print(f"Adaptive mutation result: {adaptive_mutated.content}")
    print(f"Generation: {adaptive_mutated.generation}")