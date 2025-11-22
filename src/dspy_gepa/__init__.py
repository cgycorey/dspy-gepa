"""🐶 DSPY-GEPA: User-Friendly Prompt Optimization

A powerful yet simple framework for optimizing prompts using evolutionary algorithms
and real LLM providers. Perfect for beginners and experts alike!

✨ Quick Start:
    ```python
    from dspy_gepa import quick_optimize
    
    result = quick_optimize(
        "Translate {text} to Spanish:",
        [{"text": "Hello", "expected": "Hola"}]
    )
    print(f"Best prompt: {result.best_prompt}")
    ```

🎯 Key Features:
• Easy-to-use interface with presets
• Real LLM provider support (OpenAI, Anthropic)
• Multi-objective optimization
• Built-in monitoring and error handling
• Comprehensive examples and documentation

📚 Get Help:
• show_help() - View comprehensive help
• print_welcome_message() - Getting started guide
• list_available_functions() - See all functions
• interactive_setup() - Guided setup wizard
"""

# Apply compatibility patches first
from .utils.patch_litellm import apply_all_patches
apply_all_patches()

# === CONVENIENCE FUNCTIONS (Most Common Use Cases) ===
from .convenience import (
    quick_optimize,
    optimize_translation,
    optimize_summarization,
    optimize_code_generation,
    multi_objective_optimize,
    batch_optimize,
    compare_prompts,
    setup_openai,
    setup_anthropic,
    check_setup
)

# === PRESETS (Ready-to-use Configurations) ===
from .presets import (
    get_preset,
    list_presets,
    get_quick_preset,
    get_balanced_preset,
    get_quality_preset,
    get_efficiency_preset,
    preset_registry
)

# === CORE CLASSES (For Advanced Users) ===
from .simple_gepa import SimpleGEPA
from .core.agent import GEPAAgent
from .gepa_agent import GEPAAgent as DSPYGEPAAgent
from .core.agent import GEPAAgent as Agent
from .core.multi_objective_agent import MultiObjectiveGEPAAgent, create_multi_objective_agent

# === INTERFACES AND OBJECTIVES ===
from .core.interfaces import (
    Objective, TaskType, OptimizationDirection, PreferenceVector,
    MetricConverter, ResourceMonitor, CheckpointManager
)
from .core.objectives import (
    AccuracyMetric, FluencyMetric, RelevanceMetric, TokenUsageMetric,
    ExecutionTimeMetric, CompositeMetric,
    create_default_task_objectives, create_efficiency_focused_objectives,
    create_quality_focused_objectives
)

# === ADVANCED COMPONENTS ===
from .core.mutation_engine import (
    SemanticMutator, TaskSpecificMutator, AdaptiveRateMutator, CompositeMutator
)
from .core.parameter_tuner import (
    ConvergenceBasedTuner, ResourceAwareTuner, CompositeParameterTuner
)

# === ERROR HANDLING ===
from .core.error_handling import (
    MultiObjectiveGEPAError, ConfigurationError, ValidationError,
    ExecutionError, ResourceError, OptimizationError, LLMError,
    handle_common_errors, set_debug_mode
)

# === USER GUIDE AND HELP ===
from .user_guide import (
    print_welcome_message,
    show_help,
    print_best_practices,
    interactive_setup,
    create_evaluation_data_template,
    list_available_functions
)

# === EXAMPLES ===
from . import examples

# === UTILITIES ===
from .utils.config import get_default_llm_provider, is_llm_configured, print_llm_status

__version__ = "0.3.0"

# === MOST COMMON FUNCTIONS (Start Here!) ===
__all__ = [
    # Quick start functions
    "quick_optimize",
    "optimize_translation",
    "optimize_summarization",
    "optimize_code_generation",
    "multi_objective_optimize",
    "batch_optimize",
    "compare_prompts",
    
    # Setup functions
    "setup_openai",
    "setup_anthropic",
    "check_setup",
    
    # Presets
    "get_preset",
    "list_presets",
    "get_quick_preset",
    "get_balanced_preset",
    "get_quality_preset",
    "get_efficiency_preset",
    
    # Help and guidance
    "print_welcome_message",
    "show_help",
    "print_best_practices",
    "interactive_setup",
    "create_evaluation_data_template",
    "list_available_functions",
    
    # Core classes (for advanced users)
    "SimpleGEPA",
    "GEPAAgent",
    "DSPYGEPAAgent",
    "Agent",
    "MultiObjectiveGEPAAgent",
    "create_multi_objective_agent",
    
    # Interfaces and objectives
    "Objective",
    "TaskType",
    "OptimizationDirection",
    "PreferenceVector",
    "MetricConverter",
    "ResourceMonitor",
    "CheckpointManager",
    "AccuracyMetric",
    "FluencyMetric",
    "RelevanceMetric",
    "TokenUsageMetric",
    "ExecutionTimeMetric",
    "CompositeMetric",
    "create_default_task_objectives",
    "create_efficiency_focused_objectives",
    "create_quality_focused_objectives",
    
    # Advanced components
    "SemanticMutator",
    "TaskSpecificMutator",
    "AdaptiveRateMutator",
    "CompositeMutator",
    "ConvergenceBasedTuner",
    "ResourceAwareTuner",
    "CompositeParameterTuner",
    
    # Error handling
    "MultiObjectiveGEPAError",
    "ConfigurationError",
    "ValidationError",
    "ExecutionError",
    "ResourceError",
    "OptimizationError",
    "LLMError",
    "handle_common_errors",
    "set_debug_mode",
    
    # Utilities
    "get_default_llm_provider",
    "is_llm_configured",
    "print_llm_status",
    
    # Examples module
    "examples",
    
    # Preset registry
    "preset_registry"
]


# === WELCOME MESSAGE ===
# Print welcome message when imported for the first time
import os
if os.environ.get('DSPY_GEPA_WELCOME', '1') == '1':
    try:
        print_welcome_message()
        print("\n💡 Tip: Set DSPY_GEPA_WELCOME=0 to disable this message")
        print("\n🚀 Quick start: quick_optimize('Your prompt', evaluation_data)\n")
    except:
        # Silently fail if there are any import issues
        pass