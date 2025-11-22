"""Enhanced mutation engine for multi-objective GEPA optimization.

This module provides sophisticated mutation operators that are
task-aware, semantically intelligent, and dynamically adaptive
based on optimization state and convergence metrics.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .interfaces import (
    MutationOperator, TaskType, OptimizationDirection,
    EvaluationResult, ObjectiveEvaluation
)
from ..utils.logging import get_logger


_logger = get_logger(__name__)


class SemanticMutator(MutationOperator):
    """Mutation operator that applies semantically meaningful transformations."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__("semantic_mutator", weight)
        self.mutation_patterns = [
            self._paraphrase_instruction,
            self._enhance_clarity,
            self._adjust_style,
            self._add_domain_specific_terminology,
            self._modify_instruction_format,
            self._add_constraints,
            self._simplify_language
        ]
    
    def mutate(self, solution: Any, **kwargs) -> Any:
        """Apply semantic mutation to a solution prompt."""
        prompt = self._extract_prompt(solution)
        if not prompt:
            return solution
        
        # Select a semantic mutation pattern
        mutation_fn = random.choice(self.mutation_patterns)
        mutated_prompt = mutation_fn(prompt)
        
        return self._create_solution(solution, mutated_prompt)
    
    def _extract_prompt(self, solution: Any) -> Optional[str]:
        """Extract prompt text from solution."""
        if isinstance(solution, str):
            return solution
        elif hasattr(solution, 'prompt'):
            return solution.prompt
        elif hasattr(solution, 'content'):
            return solution.content
        elif hasattr(solution, 'instructions'):
            return solution.instructions
        return None
    
    def _create_solution(self, original_solution: Any, mutated_prompt: str) -> Any:
        """Create a new solution with the mutated prompt."""
        if isinstance(original_solution, str):
            return mutated_prompt
        
        # For objects, try to update the prompt attribute
        try:
            if hasattr(original_solution, 'prompt'):
                new_solution = type(original_solution)()
                new_solution.prompt = mutated_prompt
                # Copy other attributes
                for attr, value in original_solution.__dict__.items():
                    if attr != 'prompt':
                        setattr(new_solution, attr, value)
                return new_solution
            elif hasattr(original_solution, 'content'):
                new_solution = type(original_solution)()
                new_solution.content = mutated_prompt
                for attr, value in original_solution.__dict__.items():
                    if attr != 'content':
                        setattr(new_solution, attr, value)
                return new_solution
        except Exception as e:
            _logger.warning(f"Failed to create mutated solution: {e}")
        
        return original_solution
    
    def _paraphrase_instruction(self, prompt: str) -> str:
        """Paraphrase the instruction while preserving intent."""
        paraphrases = [
            lambda p: p.replace("Please", "Could you please"),
            lambda p: p.replace("You should", "It would be helpful if you could"),
            lambda p: p.replace("Complete", "Finish completing"),
            lambda p: p.replace("Generate", "Create"),
            lambda p: p.replace("Provide", "Give"),
            lambda p: p.replace("Explain", "Describe"),
            lambda p: p.replace("Analyze", "Examine"),
            lambda p: p.replace("Translate", "Convert"),
            lambda p: p.replace("Write", "Compose"),
        ]
        
        for paraphrase in paraphrases:
            if random.random() < 0.3:  # 30% chance for each paraphrase
                try:
                    return paraphrase(prompt)
                except:
                    continue
        
        return prompt
    
    def _enhance_clarity(self, prompt: str) -> str:
        """Enhance instruction clarity."""
        clarity_enhancements = [
            "Be specific and detailed in your response.",
            "Provide a clear, step-by-step explanation.",
            "Ensure your answer is well-structured and easy to follow.",
            "Include relevant details and context.",
            "Be thorough and comprehensive."
        ]
        
        if random.random() < 0.4 and not any(enh in prompt for enh in clarity_enhancements):
            enhancement = random.choice(clarity_enhancements)
            return f"{prompt} {enhancement}"
        
        return prompt
    
    def _adjust_style(self, prompt: str) -> str:
        """Adjust the style and tone of the prompt."""
        style_adjustments = [
            ("formal", "Please provide a formal, professional response."),
            ("casual", "Feel free to use a conversational tone."),
            ("technical", "Use appropriate technical terminology."),
            ("simple", "Explain in simple, accessible language."),
            ("detailed", "Provide comprehensive details and examples.")
        ]
        
        if random.random() < 0.3:
            style, instruction = random.choice(style_adjustments)
            if style not in prompt.lower():
                return f"{prompt} {instruction}"
        
        return prompt
    
    def _add_domain_specific_terminology(self, prompt: str) -> str:
        """Add domain-specific terminology based on context."""
        domain_terms = {
            "programming": ["function", "algorithm", "code", "implementation"],
            "writing": ["paragraph", "narrative", "style", "composition"],
            "analysis": ["analyze", "evaluate", "assess", "examine"],
            "business": ["strategy", "outcome", "objective", "metric"],
            "science": ["hypothesis", "experiment", "data", "methodology"]
        }
        
        # Detect domain from keywords
        detected_domains = []
        for domain, keywords in domain_terms.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                detected_domains.append(domain)
        
        if detected_domains and random.random() < 0.3:
            domain = detected_domains[0]
            additional_term = random.choice(domain_terms[domain])
            if additional_term not in prompt.lower():
                return f"{prompt} Consider the {additional_term} aspect."
        
        return prompt
    
    def _modify_instruction_format(self, prompt: str) -> str:
        """Modify the format of instructions."""
        format_modifications = [
            lambda p: f"**Task:** {p}",
            lambda p: f"**Instructions:** {p}",
            lambda p: f"**Objective:** {p}",
            lambda p: f"[TASK] {p} [/TASK]",
            lambda p: p.replace(".", ". **Format:** Provide a structured response."),
            lambda p: f"Step 1: {p}"
        ]
        
        if random.random() < 0.3:
            modification = random.choice(format_modifications)
            try:
                return modification(prompt)
            except:
                return prompt
        
        return prompt
    
    def _add_constraints(self, prompt: str) -> str:
        """Add constraints to the prompt."""
        constraints = [
            "Limit your response to 2-3 sentences.",
            "Keep your answer under 100 words.",
            "Focus on the most important aspects.",
            "Avoid unnecessary details.",
            "Be concise but complete.",
            "Use bullet points for clarity."
        ]
        
        if random.random() < 0.4 and not any(cons in prompt for cons in constraints[:4]):
            constraint = random.choice(constraints)
            return f"{prompt} {constraint}"
        
        return prompt
    
    def _simplify_language(self, prompt: str) -> str:
        """Simplify complex language in the prompt."""
        simplifications = {
            r'\belucidate\b': 'explain',
            r'\butilize\b': 'use',
            r'\bcommence\b': 'start',
            r'\bterminate\b': 'end',
            r'\bfacilitate\b': 'help',
            r'\bsubsequently\b': 'then',
            r'\bconsequently\b': 'so',
            r'\bnevertheless\b': 'however'
        }
        
        for complex_word, simple_word in simplifications.items():
            if re.search(complex_word, prompt, re.IGNORECASE) and random.random() < 0.5:
                prompt = re.sub(complex_word, simple_word, prompt, flags=re.IGNORECASE)
        
        return prompt


class TaskSpecificMutator(MutationOperator):
    """Task-aware mutation operator that applies domain-specific mutations."""
    
    def __init__(self, task_types: Optional[List[TaskType]] = None):
        super().__init__("task_specific_mutator", 1.0, task_types or list(TaskType))
        
        self.task_mutations = {
            TaskType.TRANSLATION: [
                self._adjust_language_pair,
                self._modify_formality_level,
                self._add_context_instructions,
                self._specify_target_audience,
                self._add_cultural_notes
            ],
            TaskType.CODE_GENERATION: [
                self._change_programming_language,
                self._modify_algorithm_approach,
                self._add_code_style_requirements,
                self._specify_performance_constraints,
                self._add_documentation_requirements
            ],
            TaskType.SUMMARIZATION: [
                self._adjust_summary_length,
                self._change_summary_focus,
                self._specify_target_audience,
                self._modify_summary_style,
                self._add_key_points_emphasis
            ],
            TaskType.QUESTION_ANSWERING: [
                self._modify_answer_format,
                self._add_source_requirements,
                self._adjust_complexity_level,
                self._specify_answer_length,
                self._add_reasoning_requirements
            ],
            TaskType.CLASSIFICATION: [
                self._modify_classification_criteria,
                self._add_confidence_requirements,
                self._adjust_decision_threshold,
                self._add_explanation_requirements,
                self._specify_output_format
            ]
        }
    
    def mutate(self, solution: Any, task_type: TaskType = TaskType.CUSTOM, **kwargs) -> Any:
        """Apply task-specific mutation."""
        prompt = self._extract_prompt(solution)
        if not prompt:
            return solution
        
        # Get task-specific mutations
        mutations = self.task_mutations.get(task_type, [])
        if not mutations:
            # Apply generic mutations if no task-specific ones
            mutations = [
                self._add_general_instructions,
                self._modify_output_requirements,
                self._adjust_complexity
            ]
        
        # Apply a random mutation
        mutation_fn = random.choice(mutations)
        mutated_prompt = mutation_fn(prompt, **kwargs)
        
        return self._create_solution(solution, mutated_prompt)
    
    def _extract_prompt(self, solution: Any) -> Optional[str]:
        """Extract prompt text from solution."""
        if isinstance(solution, str):
            return solution
        elif hasattr(solution, 'prompt'):
            return solution.prompt
        elif hasattr(solution, 'content'):
            return solution.content
        return None
    
    def _create_solution(self, original_solution: Any, mutated_prompt: str) -> Any:
        """Create a new solution with the mutated prompt."""
        if isinstance(original_solution, str):
            return mutated_prompt
        
        try:
            if hasattr(original_solution, 'prompt'):
                new_solution = type(original_solution)()
                new_solution.prompt = mutated_prompt
                for attr, value in original_solution.__dict__.items():
                    if attr != 'prompt':
                        setattr(new_solution, attr, value)
                return new_solution
        except Exception as e:
            _logger.warning(f"Failed to create mutated solution: {e}")
        
        return original_solution
    
    # Translation-specific mutations
    def _adjust_language_pair(self, prompt: str, **kwargs) -> str:
        """Adjust language pair specifications."""
        language_pairs = [
            ("English", "Spanish"),
            ("English", "French"),
            ("English", "German"),
            ("English", "Chinese"),
            ("English", "Japanese")
        ]
        
        source, target = random.choice(language_pairs)
        return f"Translate the following text from {source} to {target}: {prompt}"
    
    def _modify_formality_level(self, prompt: str, **kwargs) -> str:
        """Modify formality level for translation."""
        formality_levels = [
            "Use formal language appropriate for business contexts.",
            "Use informal, conversational language.",
            "Use academic, scholarly language.",
            "Use casual, everyday language."
        ]
        
        level = random.choice(formality_levels)
        return f"{prompt} {level}"
    
    def _add_context_instructions(self, prompt: str, **kwargs) -> str:
        """Add context-specific instructions."""
        contexts = [
            "This is for a professional business setting.",
            "This is for casual social media.",
            "This is for academic purposes.",
            "This is for technical documentation."
        ]
        
        context = random.choice(contexts)
        return f"{prompt} Context: {context}"
    
    def _specify_target_audience(self, prompt: str, **kwargs) -> str:
        """Specify target audience for translation."""
        audiences = [
            "Target audience: technical experts.",
            "Target audience: general public.",
            "Target audience: children.",
            "Target audience: business professionals."
        ]
        
        audience = random.choice(audiences)
        return f"{prompt} {audience}"
    
    def _add_cultural_notes(self, prompt: str, **kwargs) -> str:
        """Add cultural adaptation notes."""
        return f"{prompt} Ensure cultural appropriateness and idiomatic expressions."
    
    # Code generation-specific mutations
    def _change_programming_language(self, prompt: str, **kwargs) -> str:
        """Change programming language specification."""
        languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
        language = random.choice(languages)
        return f"Write {language.lower()} code to: {prompt}"
    
    def _modify_algorithm_approach(self, prompt: str, **kwargs) -> str:
        """Modify algorithmic approach."""
        approaches = [
            "Use a recursive approach.",
            "Use an iterative approach.",
            "Use dynamic programming.",
            "Use a greedy algorithm.",
            "Use divide and conquer."
        ]
        
        approach = random.choice(approaches)
        return f"{prompt} {approach}"
    
    def _add_code_style_requirements(self, prompt: str, **kwargs) -> str:
        """Add code style requirements."""
        styles = [
            "Follow PEP 8 style guidelines.",
            "Include comprehensive comments.",
            "Use descriptive variable names.",
            "Write clean, readable code.",
            "Include type hints."
        ]
        
        style = random.choice(styles)
        return f"{prompt} {style}"
    
    def _specify_performance_constraints(self, prompt: str, **kwargs) -> str:
        """Add performance constraints."""
        constraints = [
            "Optimize for time complexity.",
            "Optimize for space complexity.",
            "Prioritize readability over performance.",
            "Ensure O(n) time complexity.",
            "Minimize memory usage."
        ]
        
        constraint = random.choice(constraints)
        return f"{prompt} {constraint}"
    
    def _add_documentation_requirements(self, prompt: str, **kwargs) -> str:
        """Add documentation requirements."""
        return f"{prompt} Include docstrings and usage examples."
    
    # Summarization-specific mutations
    def _adjust_summary_length(self, prompt: str, **kwargs) -> str:
        """Adjust summary length requirements."""
        lengths = [
            "Provide a one-sentence summary.",
            "Summarize in 2-3 sentences.",
            "Create a paragraph-length summary.",
            "Provide a detailed summary (several paragraphs).",
            "Create a bulleted summary."
        ]
        
        length = random.choice(lengths)
        return f"{prompt} {length}"
    
    def _change_summary_focus(self, prompt: str, **kwargs) -> str:
        """Change summary focus."""
        focuses = [
            "Focus on the main conclusions.",
            "Focus on key findings and insights.",
            "Focus on methodology and approach.",
            "Focus on implications and consequences.",
            "Focus on quantitative results."
        ]
        
        focus = random.choice(focuses)
        return f"{prompt} {focus}"
    
    def _modify_summary_style(self, prompt: str, **kwargs) -> str:
        """Modify summary style."""
        styles = [
            "Write in an academic tone.",
            "Write in a journalistic style.",
            "Write in plain, accessible language.",
            "Write in a formal business style.",
            "Write in a conversational tone."
        ]
        
        style = random.choice(styles)
        return f"{prompt} {style}"
    
    # Question answering-specific mutations
    def _modify_answer_format(self, prompt: str, **kwargs) -> str:
        """Modify answer format requirements."""
        formats = [
            "Answer in a single paragraph.",
            "Use bullet points for your answer.",
            "Structure your answer with headings.",
            "Provide a step-by-step answer.",
            "Answer in a numbered list format."
        ]
        
        format_req = random.choice(formats)
        return f"{prompt} {format_req}"
    
    def _add_source_requirements(self, prompt: str, **kwargs) -> str:
        """Add source citation requirements."""
        return f"{prompt} Cite your sources and provide references."
    
    def _adjust_complexity_level(self, prompt: str, **kwargs) -> str:
        """Adjust complexity level of expected answer."""
        complexities = [
            "Provide a beginner-level explanation.",
            "Provide an intermediate-level answer.",
            "Provide an expert-level detailed response.",
            "Make your answer accessible to non-experts.",
            "Include advanced technical details."
        ]
        
        complexity = random.choice(complexities)
        return f"{prompt} {complexity}"
    
    # Generic mutations for custom tasks
    def _add_general_instructions(self, prompt: str, **kwargs) -> str:
        """Add general improvement instructions."""
        instructions = [
            "Be thorough and comprehensive.",
            "Provide clear and structured answers.",
            "Include relevant examples.",
            "Address all aspects of the question.",
            "Consider multiple perspectives."
        ]
        
        instruction = random.choice(instructions)
        return f"{prompt} {instruction}"
    
    def _modify_output_requirements(self, prompt: str, **kwargs) -> str:
        """Modify output requirements."""
        requirements = [
            "Ensure output is well-formatted.",
            "Provide concrete examples.",
            "Include practical applications.",
            "Make it actionable and useful.",
            "Ensure clarity and precision."
        ]
        
        requirement = random.choice(requirements)
        return f"{prompt} {requirement}"
    
    def _adjust_complexity(self, prompt: str, **kwargs) -> str:
        """Adjust overall complexity."""
        adjustments = [
            "Simplify the task for better understanding.",
            "Add more detail and complexity.",
            "Focus on the essential elements only.",
            "Provide additional context and depth.",
            "Streamline for efficiency."
        ]
        
        adjustment = random.choice(adjustments)
        return f"{prompt} {adjustment}"
    
    # Missing method implementations
    def _add_key_points_emphasis(self, prompt: str, **kwargs) -> str:
        """Add emphasis on key points for summarization."""
        return f"{prompt} Emphasize the most important points and key takeaways."
    
    def _specify_answer_length(self, prompt: str, **kwargs) -> str:
        """Specify answer length requirements."""
        lengths = [
            "Keep your answer concise (under 100 words).",
            "Provide a detailed answer (200-300 words).",
            "Answer briefly (2-3 sentences).",
            "Give a comprehensive response."
        ]
        length = random.choice(lengths)
        return f"{prompt} {length}"
    
    def _add_reasoning_requirements(self, prompt: str, **kwargs) -> str:
        """Add reasoning requirements."""
        return f"{prompt} Show your reasoning and explain your thought process."
    
    def _modify_classification_criteria(self, prompt: str, **kwargs) -> str:
        """Modify classification criteria."""
        criteria = [
            "Classify based on primary characteristics.",
            "Consider multiple categories for classification.",
            "Use binary classification (yes/no).",
            "Apply multi-label classification if applicable."
        ]
        criterion = random.choice(criteria)
        return f"{prompt} {criterion}"
    
    def _add_confidence_requirements(self, prompt: str, **kwargs) -> str:
        """Add confidence scoring requirements."""
        return f"{prompt} Include a confidence score (0-100) for your classification."
    
    def _adjust_decision_threshold(self, prompt: str, **kwargs) -> str:
        """Adjust decision threshold."""
        thresholds = [
            "Use 0.5 as decision threshold.",
            "Set higher threshold for positive classification (0.7).",
            "Use conservative threshold (0.8).",
            "Apply lower threshold for sensitive cases (0.3)."
        ]
        threshold = random.choice(thresholds)
        return f"{prompt} {threshold}"
    
    def _add_explanation_requirements(self, prompt: str, **kwargs) -> str:
        """Add explanation requirements."""
        return f"{prompt} Provide brief explanation for your classification decision."
    
    def _specify_output_format(self, prompt: str, **kwargs) -> str:
        """Specify output format for classification."""
        formats = [
            "Output as 'Category: <category_name>'.",
            "Use JSON format: {'category': '<category>'}.",
            "Return just the category name.",
            "Output as 'Category: <category>, Confidence: <score>'."
        ]
        format_spec = random.choice(formats)
        return f"{prompt} {format_spec}"


class AdaptiveRateMutator(MutationOperator):
    """Mutation operator with adaptive rate adjustment based on optimization state."""
    
    def __init__(self, base_mutation_rate: float = 0.1):
        super().__init__("adaptive_rate_mutator", 1.0)
        self.base_mutation_rate = base_mutation_rate
        self.performance_history: List[float] = []
        self.diversity_history: List[float] = []
    
    def mutate(self, solution: Any, generation: int = 0, convergence_metrics: Optional[Dict[str, float]] = None, **kwargs) -> Any:
        """Apply mutation with adaptive rate."""
        # Calculate adaptive mutation rate
        mutation_rate = self.get_mutation_rate(generation, convergence_metrics or {})
        
        if random.random() < mutation_rate:
            # Apply semantic mutation
            semantic_mutator = SemanticMutator()
            return semantic_mutator.mutate(solution, **kwargs)
        
        return solution
    
    def get_mutation_rate(self, generation: int, convergence_metrics: Dict[str, float]) -> float:
        """Calculate adaptive mutation rate based on optimization state."""
        base_rate = self.base_mutation_rate
        
        # Adjust based on generation (increase mutation in early generations)
        generation_factor = max(0.1, 1.0 - (generation / 100.0))
        
        # Adjust based on convergence (increase mutation if converging)
        convergence_score = convergence_metrics.get("convergence_score", 0.0)
        convergence_factor = 1.0 + (convergence_score * 0.5)
        
        # Adjust based on diversity (increase mutation if diversity is low)
        diversity_score = convergence_metrics.get("diversity_score", 0.5)
        diversity_factor = 2.0 - diversity_score  # Higher factor for lower diversity
        
        # Adjust based on recent performance improvement
        performance_factor = 1.0
        if len(self.performance_history) >= 3:
            recent_improvement = (
                self.performance_history[-1] - self.performance_history[-3]
            )
            if recent_improvement < 0.01:  # Low improvement
                performance_factor = 1.5
        
        # Combine all factors
        adaptive_rate = base_rate * generation_factor * convergence_factor * diversity_factor * performance_factor
        
        # Clamp to reasonable bounds
        return max(0.01, min(0.5, adaptive_rate))
    
    def update_performance(self, performance_score: float) -> None:
        """Update performance history for rate adaptation."""
        self.performance_history.append(performance_score)
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
    
    def update_diversity(self, diversity_score: float) -> None:
        """Update diversity history."""
        self.diversity_history.append(diversity_score)
        if len(self.diversity_history) > 20:
            self.diversity_history = self.diversity_history[-20:]


class CompositeMutator(MutationOperator):
    """Composite mutation operator that combines multiple mutation strategies."""
    
    def __init__(
        self,
        mutators: Optional[List[MutationOperator]] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__("composite_mutator", 1.0)
        
        self.mutators = mutators or [
            SemanticMutator(),
            TaskSpecificMutator(),
            AdaptiveRateMutator()
        ]
        
        self.weights = weights or [mutator.weight for mutator in self.mutators]
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
    
    def mutate(self, solution: Any, **kwargs) -> Any:
        """Apply composite mutation."""
        # Select mutator based on weights
        mutator = random.choices(self.mutators, weights=self.weights)[0]
        return mutator.mutate(solution, **kwargs)
    
    def add_mutator(self, mutator: MutationOperator, weight: float = 1.0) -> None:
        """Add a new mutator to the composite."""
        self.mutators.append(mutator)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
    
    def get_mutation_rate(self, generation: int, convergence_metrics: Dict[str, float]) -> float:
        """Get weighted average mutation rate."""
        rates = []
        weights = []
        
        for mutator, weight in zip(self.mutators, self.weights):
            rate = mutator.get_mutation_rate(generation, convergence_metrics)
            rates.append(rate)
            weights.append(weight)
        
        # Calculate weighted average
        if sum(weights) > 0:
            return sum(r * w for r, w in zip(rates, weights)) / sum(weights)
        return 0.1


class MutationStrategy:
    """High-level mutation strategy that orchestrates multiple mutators."""
    
    def __init__(
        self,
        primary_mutator: MutationOperator,
        secondary_mutators: Optional[List[MutationOperator]] = None,
        selection_strategy: str = "adaptive"  # "adaptive", "random", "轮换"
    ):
        """Initialize mutation strategy.
        
        Args:
            primary_mutator: Main mutation operator
            secondary_mutators: Additional mutation operators
            selection_strategy: How to select mutators
        """
        self.primary_mutator = primary_mutator
        self.secondary_mutators = secondary_mutators or []
        self.selection_strategy = selection_strategy
        self.usage_history: Dict[str, int] = {}
        self.performance_history: Dict[str, List[float]] = {}
    
    def apply_mutation(
        self,
        solution: Any,
        task_type: TaskType = TaskType.CUSTOM,
        generation: int = 0,
        convergence_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Any:
        """Apply mutation using the configured strategy."""
        mutator = self._select_mutator(task_type, generation, convergence_metrics or {})
        
        # Track usage
        mutator_name = mutator.name
        self.usage_history[mutator_name] = self.usage_history.get(mutator_name, 0) + 1
        
        return mutator.mutate(
            solution,
            task_type=task_type,
            generation=generation,
            convergence_metrics=convergence_metrics,
            **kwargs
        )
    
    def _select_mutator(
        self,
        task_type: TaskType,
        generation: int,
        convergence_metrics: Dict[str, float]
    ) -> MutationOperator:
        """Select mutator based on strategy."""
        if self.selection_strategy == "adaptive":
            return self._adaptive_selection(task_type, convergence_metrics)
        elif self.selection_strategy == "random":
            return self._random_selection(task_type)
        elif self.selection_strategy == "轮换":
            return self._轮换_selection(task_type, generation)
        else:
            return self.primary_mutator
    
    def _adaptive_selection(self, task_type: TaskType, convergence_metrics: Dict[str, float]) -> MutationOperator:
        """Select mutator based on performance history and convergence state."""
        # Find task-specific mutators
        task_mutators = [
            mutator for mutator in [self.primary_mutator] + self.secondary_mutators
            if mutator.can_handle_task(task_type)
        ]
        
        if not task_mutators:
            task_mutators = [self.primary_mutator]
        
        # If no performance history, use primary mutator
        if not self.performance_history:
            return self.primary_mutator
        
        # Select based on recent performance
        best_mutator = task_mutators[0]
        best_performance = float('-inf')
        
        for mutator in task_mutators:
            if mutator.name in self.performance_history:
                recent_performance = self.performance_history[mutator.name][-5:]  # Last 5 uses
                avg_performance = sum(recent_performance) / len(recent_performance)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_mutator = mutator
        
        return best_mutator
    
    def _random_selection(self, task_type: TaskType) -> MutationOperator:
        """Randomly select a task-compatible mutator."""
        task_mutators = [
            mutator for mutator in [self.primary_mutator] + self.secondary_mutators
            if mutator.can_handle_task(task_type)
        ]
        
        if not task_mutators:
            return self.primary_mutator
        
        return random.choice(task_mutators)
    
    def _轮换_selection(self, task_type: TaskType, generation: int) -> MutationOperator:
        """轮换 mutators in sequence."""
        task_mutators = [
            mutator for mutator in [self.primary_mutator] + self.secondary_mutators
            if mutator.can_handle_task(task_type)
        ]
        
        if not task_mutators:
            return self.primary_mutator
        
        # 轮换 based on generation
        index = generation % len(task_mutators)
        return task_mutators[index]
    
    def update_performance(self, mutator_name: str, performance_score: float) -> None:
        """Update performance tracking for a mutator."""
        if mutator_name not in self.performance_history:
            self.performance_history[mutator_name] = []
        
        self.performance_history[mutator_name].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[mutator_name]) > 20:
            self.performance_history[mutator_name] = self.performance_history[mutator_name][-20:]
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for all mutators."""
        return self.usage_history.copy()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get average performance statistics for all mutators."""
        stats = {}
        for mutator_name, performances in self.performance_history.items():
            if performances:
                stats[mutator_name] = sum(performances) / len(performances)
        return stats