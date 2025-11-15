"""GEPA Reflection System

This module provides intelligent reflection capabilities for the GEPA algorithm,
enabling LLM-driven analysis of execution traces and feedback generation for
evolutionary mutations.

Components:
- LMReflector: Main LLM reflection interface with multi-provider support
- TraceAnalyzer: Extracts patterns and insights from execution traces
- FeedbackGenerator: Synthesizes analysis with LLM feedback for mutation guidance

The reflection system enables GEPA to:
- Analyze execution performance and failure modes
- Generate actionable improvement suggestions
- Prioritize mutations based on learned patterns
- Adapt strategies based on historical performance

Example usage:
    from gepa.reflection import LMReflector, TraceAnalyzer, FeedbackGenerator
    
    # Initialize reflection components
    reflector = LMReflector(provider="openai", model="gpt-4")
    analyzer = TraceAnalyzer()
    generator = FeedbackGenerator(reflector)
    
    # Analyze candidate performance
    trace_analysis = analyzer.analyze_traces(candidate.execution_traces)
    feedback = generator.generate_feedback(candidate, trace_analysis)
"""

from .lm_reflector import LMReflector
from .trace_analyzer import TraceAnalyzer
from .feedback_generator import FeedbackGenerator

__all__ = [
    "LMReflector",
    "TraceAnalyzer", 
    "FeedbackGenerator"
]