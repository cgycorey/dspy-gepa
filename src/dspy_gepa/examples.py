"""Quick reference examples for DSPY-GEPA.

This module provides ready-to-use examples for common optimization scenarios,
making it easy for beginners to understand how to use the framework.
"""

from typing import List, Dict, Any


def quick_start_example():
    """Quick start example for basic prompt optimization."""
    print("=== Quick Start Example ===")
    print(
        """
from dspy_gepa import SimpleGEPA

# Create optimizer with default settings
optimizer = SimpleGEPA()

# Define your evaluation data
evaluation_data = [
    {"text": "Hello", "expected": "Hola"},
    {"text": "Goodbye", "expected": "Adiós"},
    {"text": "Thank you", "expected": "Gracias"}
]

# Optimize your prompt
result = optimizer.optimize(
    initial_prompt="Translate {text} to Spanish:",
    evaluation_data=evaluation_data
)

print(f"Best prompt: {result.best_prompt}")
print(f"Improvement: {result.improvement_percentage:.1f}%")
        """
    )


def preset_configuration_example():
    """Example using preset configurations."""
    print("=== Preset Configuration Example ===")
    print(
        """
from dspy_gepa import SimpleGEPA, get_quality_preset

# Create optimizer with quality-focused preset
optimizer = SimpleGEPA()
quality_preset = get_quality_preset()

evaluation_data = [
    {"article": "Long article about AI...", "summary": "AI is changing the world..."},
    # More examples...
]

result = optimizer.optimize(
    initial_prompt="Summarize this article: {article}",
    evaluation_data=evaluation_data,
    **quality_preset.to_dict()
)
        """
    )


def multi_objective_example():
    """Example of multi-objective optimization."""
    print("=== Multi-Objective Optimization Example ===")
    print(
        """
from dspy_gepa import (
    MultiObjectiveGEPAAgent,
    AccuracyMetric, TokenUsageMetric, FluencyMetric,
    create_multi_objective_agent
)

# Define multiple objectives
objectives = [
    AccuracyMetric(weight=0.6),  # 60% weight on accuracy
    TokenUsageMetric(weight=0.2),  # 20% weight on efficiency
    FluencyMetric(weight=0.2)  # 20% weight on fluency
]

# Create multi-objective agent
agent = create_multi_objective_agent(
    objectives=objectives,
    lm=your_lm_instance
)

evaluation_data = [
    {"input": "Hello", "expected": "Hola"},
    # More examples...
]

result = agent.optimize(
    initial_prompt="Translate {input} to Spanish:",
    evaluation_data=evaluation_data
)

print(f"Best prompt: {result.best_prompt}")
print(f"Scores: {result.objective_scores}")
        """
    )


def custom_objectives_example():
    """Example of creating custom objectives."""
    print("=== Custom Objectives Example ===")
    print(
        """
from dspy_gepa import (
    SimpleGEPA, Objective, TaskType, OptimizationDirection
)

# Create a custom objective
class CustomMetric:
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def evaluate(self, prediction: str, reference: str) -> float:
        # Your custom evaluation logic here
        # For example, check for specific keywords
        keywords = ["please", "thank you", "polite"]
        score = sum(1 for kw in keywords if kw in prediction.lower())
        return score / len(keywords)

# Use the custom objective
custom_objective = Objective(
    name="politeness",
    metric=CustomMetric(weight=0.3),
    direction=OptimizationDirection.MAXIMIZE,
    description="Measures politeness of the response"
)

optimizer = SimpleGEPA()
result = optimizer.optimize(
    initial_prompt="Respond politely to: {input}",
    evaluation_data=evaluation_data,
    objectives=[custom_objective]
)
        """
    )


def translation_task_example():
    """Example specifically for translation tasks."""
    print("=== Translation Task Example ===")
    print(
        """
from dspy_gepa import SimpleGEPA, get_preset

# Use translation-specific preset
optimizer = SimpleGEPA()
translation_preset = get_preset('translation')

# Translation examples
translation_data = [
    {"source": "Hello world", "target": "Hola mundo"},
    {"source": "How are you?", "target": "¿Cómo estás?"},
    {"source": "Good morning", "target": "Buenos días"}
]

result = optimizer.optimize(
    initial_prompt="Translate the following text to Spanish: {source}",
    evaluation_data=translation_data,
    **translation_preset.to_dict()
)

print(f"Optimized translation prompt: {result.best_prompt}")
print(f"Translation accuracy improvement: {result.improvement_percentage:.1f}%")
        """
    )


def code_generation_example():
    """Example for code generation tasks."""
    print("=== Code Generation Example ===")
    print(
        """
from dspy_gepa import SimpleGEPA, get_preset

# Use code generation preset
optimizer = SimpleGEPA()
code_preset = get_preset('code_generation')

code_data = [
    {
        "description": "function that adds two numbers",
        "expected_code": "def add(a, b):\n    return a + b"
    },
    {
        "description": "function that checks if a number is even",
        "expected_code": "def is_even(n):\n    return n % 2 == 0"
    }
]

result = optimizer.optimize(
    initial_prompt="Write Python code to: {description}",
    evaluation_data=code_data,
    **code_preset.to_dict()
)

print(f"Optimized code generation prompt: {result.best_prompt}")
        """
    )


def monitoring_example():
    """Example of using monitoring features."""
    print("=== Monitoring Example ===")
    print(
        """
from dspy_gepa import SimpleGEPA
from dspy_gepa.core.monitoring import ResourceMonitor

# Create optimizer with monitoring
optimizer = SimpleGEPA()
monitor = ResourceMonitor()

# Start monitoring
monitor.start_monitoring()

# Run optimization
result = optimizer.optimize(
    initial_prompt="Your prompt here",
    evaluation_data=evaluation_data
)

# Get monitoring stats
stats = monitor.get_stats()
print(f"Memory used: {stats['memory_mb']:.1f} MB")
print(f"Execution time: {stats['execution_time']:.1f} seconds")
print(f"API calls made: {stats['api_calls']}")
        """
    )


def error_handling_example():
    """Example of proper error handling."""
    print("=== Error Handling Example ===")
    print(
        """
from dspy_gepa import SimpleGEPA, is_llm_configured, print_llm_status
import dspy

# Check LLM configuration before starting
if not is_llm_configured():
    print("LLM not configured. Please set up your API key.")
    print_llm_status()
    exit(1)

try:
    optimizer = SimpleGEPA()
    result = optimizer.optimize(
        initial_prompt="Your prompt here",
        evaluation_data=evaluation_data
    )
except Exception as e:
    print(f"Optimization failed: {e}")
    # Handle error appropriately
    
# Alternative: Use try-except with specific error types
from dspy_gepa.core.error_handling import (
    OptimizationError, ConfigurationError, ValidationError
)

try:
    optimizer = SimpleGEPA()
    result = optimizer.optimize(...)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print("Please check your LLM configuration")
except ValidationError as e:
    print(f"Validation error: {e}")
    print("Please check your evaluation data format")
except OptimizationError as e:
    print(f"Optimization error: {e}")
    print("The optimization process encountered an issue")
        """
    )


def list_all_examples():
    """List all available examples."""
    examples = [
        "quick_start",
        "preset_configuration", 
        "multi_objective",
        "custom_objectives",
        "translation_task",
        "code_generation",
        "monitoring",
        "error_handling"
    ]
    
    print("Available examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nUse show_example('example_name') to see the code.")


def show_example(example_name: str):
    """Show a specific example."""
    examples = {
        "quick_start": quick_start_example,
        "preset_configuration": preset_configuration_example,
        "multi_objective": multi_objective_example,
        "custom_objectives": custom_objectives_example,
        "translation_task": translation_task_example,
        "code_generation": code_generation_example,
        "monitoring": monitoring_example,
        "error_handling": error_handling_example
    }
    
    if example_name in examples:
        examples[example_name]()
    else:
        print(f"Unknown example '{example_name}'")
        list_all_examples()


# Export functions for easy access
__all__ = [
    "quick_start_example",
    "preset_configuration_example",
    "multi_objective_example", 
    "custom_objectives_example",
    "translation_task_example",
    "code_generation_example",
    "monitoring_example",
    "error_handling_example",
    "list_all_examples",
    "show_example"
]
