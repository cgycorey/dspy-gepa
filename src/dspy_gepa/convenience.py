"""Convenience functions for common operations.

This module provides high-level convenience functions that make it easy
to perform common optimization tasks with minimal setup.
"""

from typing import List, Dict, Any, Optional, Union
import dspy

from .simple_gepa import SimpleGEPA
from .presets import (
    get_preset, get_quick_preset, get_balanced_preset, 
    get_quality_preset, get_efficiency_preset
)
from .core.multi_objective_agent import create_multi_objective_agent
from .core.objectives import (
    AccuracyMetric, FluencyMetric, RelevanceMetric, 
    TokenUsageMetric, ExecutionTimeMetric
)
from .utils.config import is_llm_configured, print_llm_status


def quick_optimize(initial_prompt: str, 
                  evaluation_data: List[Dict[str, Any]],
                  preset: str = "quick") -> Any:
    """Quick optimization with minimal setup.
    
    Args:
        initial_prompt: The prompt to optimize
        evaluation_data: List of evaluation examples
        preset: Preset name ('quick', 'balanced', 'quality', 'efficiency')
        
    Returns:
        Optimization result
        
    Example:
        result = quick_optimize(
            "Translate {text} to Spanish:",
            [{"text": "Hello", "expected": "Hola"}]
        )
    """
    if not is_llm_configured():
        raise RuntimeError(
            "LLM not configured. Please set up your API key first. "
            "Run print_llm_status() for help."
        )
    
    optimizer = SimpleGEPA()
    preset_config = get_preset(preset)
    
    return optimizer.optimize(
        initial_prompt=initial_prompt,
        evaluation_data=evaluation_data,
        **preset_config.to_dict()
    )


def optimize_translation(initial_prompt: str,
                        translation_data: List[Dict[str, str]],
                        focus: str = "balanced") -> Any:
    """Optimize prompts specifically for translation tasks.
    
    Args:
        initial_prompt: Translation prompt template
        translation_data: List with 'source' and 'target' keys
        focus: Focus area ('quick', 'balanced', 'quality', 'efficiency')
        
    Example:
        result = optimize_translation(
            "Translate {source} to Spanish:",
            [{"source": "Hello", "target": "Hola"}]
        )
    """
    optimizer = SimpleGEPA()
    preset = get_preset('translation' if focus == 'balanced' else focus)
    
    return optimizer.optimize(
        initial_prompt=initial_prompt,
        evaluation_data=translation_data,
        **preset.to_dict()
    )


def optimize_summarization(initial_prompt: str,
                          summarization_data: List[Dict[str, str]],
                          focus: str = "balanced") -> Any:
    """Optimize prompts for text summarization.
    
    Args:
        initial_prompt: Summarization prompt template
        summarization_data: List with 'article' and 'summary' keys
        focus: Focus area ('quick', 'balanced', 'quality', 'efficiency')
        
    Example:
        result = optimize_summarization(
            "Summarize this article: {article}",
            [{"article": "Long text...", "summary": "Short summary..."}]
        )
    """
    optimizer = SimpleGEPA()
    preset = get_preset('summarization' if focus == 'balanced' else focus)
    
    return optimizer.optimize(
        initial_prompt=initial_prompt,
        evaluation_data=summarization_data,
        **preset.to_dict()
    )


def optimize_code_generation(initial_prompt: str,
                           code_data: List[Dict[str, str]],
                           focus: str = "balanced") -> Any:
    """Optimize prompts for code generation.
    
    Args:
        initial_prompt: Code generation prompt template
        code_data: List with 'description' and 'expected_code' keys
        focus: Focus area ('quick', 'balanced', 'quality', 'efficiency')
        
    Example:
        result = optimize_code_generation(
            "Write Python code to: {description}",
            [{"description": "add two numbers", "expected_code": "def add(a,b): return a+b"}]
        )
    """
    optimizer = SimpleGEPA()
    preset = get_preset('code_generation' if focus == 'balanced' else focus)
    
    return optimizer.optimize(
        initial_prompt=initial_prompt,
        evaluation_data=code_data,
        **preset.to_dict()
    )


def multi_objective_optimize(initial_prompt: str,
                           evaluation_data: List[Dict[str, Any]],
                           objectives: Optional[List[str]] = None,
                           weights: Optional[Dict[str, float]] = None) -> Any:
    """Multi-objective optimization with simple interface.
    
    Args:
        initial_prompt: The prompt to optimize
        evaluation_data: List of evaluation examples
        objectives: List of objective names (default: ['accuracy', 'token_usage'])
        weights: Dictionary of objective weights
        
    Example:
        result = multi_objective_optimize(
            "Translate {text} to Spanish:",
            [{"text": "Hello", "expected": "Hola"}],
            objectives=['accuracy', 'token_usage', 'fluency'],
            weights={'accuracy': 0.6, 'token_usage': 0.2, 'fluency': 0.2}
        )
    """
    if objectives is None:
        objectives = ['accuracy', 'token_usage']
    
    if weights is None:
        weights = {obj: 1.0/len(objectives) for obj in objectives}
    
    # Map objective names to metric instances
    objective_map = {
        'accuracy': AccuracyMetric,
        'fluency': FluencyMetric,
        'relevance': RelevanceMetric,
        'token_usage': TokenUsageMetric,
        'execution_time': ExecutionTimeMetric
    }
    
    metric_objectives = []
    for obj_name in objectives:
        if obj_name not in objective_map:
            available = ', '.join(objective_map.keys())
            raise ValueError(
                f"Unknown objective '{obj_name}'. Available: {available}"
            )
        
        weight = weights.get(obj_name, 1.0/len(objectives))
        metric_objectives.append(objective_map[obj_name](weight=weight))
    
    agent = create_multi_objective_agent(objectives=metric_objectives)
    
    return agent.optimize(
        initial_prompt=initial_prompt,
        evaluation_data=evaluation_data
    )


def batch_optimize(prompts: List[str],
                  evaluation_data: List[Dict[str, Any]],
                  preset: str = "quick") -> List[Any]:
    """Optimize multiple prompts with the same data.
    
    Args:
        prompts: List of prompts to optimize
        evaluation_data: Evaluation data (same for all prompts)
        preset: Preset configuration to use
        
    Returns:
        List of optimization results
        
    Example:
        results = batch_optimize(
            ["Translate {text} to Spanish:", "Convert {text} to Spanish:"],
            [{"text": "Hello", "expected": "Hola"}]
        )
    """
    results = []
    preset_config = get_preset(preset)
    
    for i, prompt in enumerate(prompts):
        print(f"Optimizing prompt {i+1}/{len(prompts)}...")
        
        result = quick_optimize(
            initial_prompt=prompt,
            evaluation_data=evaluation_data,
            preset=preset
        )
        
        results.append(result)
    
    return results


def compare_prompts(prompts: List[str],
                   evaluation_data: List[Dict[str, Any]],
                   preset: str = "quick") -> Dict[str, Any]:
    """Compare multiple prompts and return the best one.
    
    Args:
        prompts: List of prompts to compare
        evaluation_data: Evaluation data
        preset: Preset configuration
        
    Returns:
        Dictionary with comparison results
        
    Example:
        comparison = compare_prompts(
            ["Translate {text} to Spanish:", "Convert {text} to Spanish:"],
            [{"text": "Hello", "expected": "Hola"}]
        )
        print(f"Best prompt: {comparison['best_prompt']}")
    """
    results = batch_optimize(prompts, evaluation_data, preset)
    
    # Find the best result
    best_result = max(results, key=lambda r: r.best_score)
    best_index = results.index(best_result)
    
    return {
        'best_prompt': prompts[best_index],
        'best_score': best_result.best_score,
        'best_result': best_result,
        'all_results': list(zip(prompts, results)),
        'ranking': sorted(
            zip(prompts, results), 
            key=lambda x: x[1].best_score, 
            reverse=True
        )
    }


def setup_openai(api_key: str, model: str = "gpt-3.5-turbo") -> bool:
    """Quick setup for OpenAI API.
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        
    Returns:
        True if setup successful
    """
    try:
        import os
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Configure DSPY
        lm = dspy.OpenAI(model=model, api_key=api_key)
        dspy.settings.configure(lm=lm)
        
        return True
    except Exception as e:
        print(f"Failed to setup OpenAI: {e}")
        return False


def setup_anthropic(api_key: str, model: str = "claude-3-sonnet-20240229") -> bool:
    """Quick setup for Anthropic API.
    
    Args:
        api_key: Anthropic API key
        model: Model to use
        
    Returns:
        True if setup successful
    """
    try:
        import os
        os.environ['ANTHROPIC_API_KEY'] = api_key
        
        # Configure DSPY
        lm = dspy.Anthropic(model=model, api_key=api_key)
        dspy.settings.configure(lm=lm)
        
        return True
    except Exception as e:
        print(f"Failed to setup Anthropic: {e}")
        return False


def check_setup() -> Dict[str, Any]:
    """Check if the environment is properly set up.
    
    Returns:
        Dictionary with setup status
    """
    status = {
        'llm_configured': is_llm_configured(),
        'dspy_available': False,
        'openai_available': False,
        'anthropic_available': False,
        'recommendations': []
    }
    
    try:
        import dspy
        status['dspy_available'] = True
    except ImportError:
        status['recommendations'].append("Install DSPY: pip install dspy-ai")
    
    try:
        import openai
        status['openai_available'] = True
    except ImportError:
        status['recommendations'].append("Install OpenAI: pip install openai")
    
    try:
        import anthropic
        status['anthropic_available'] = True
    except ImportError:
        status['recommendations'].append("Install Anthropic: pip install anthropic")
    
    if not status['llm_configured']:
        status['recommendations'].append(
            "Configure your LLM API key. Use setup_openai() or setup_anthropic()"
        )
    
    return status


# Export all convenience functions
__all__ = [
    "quick_optimize",
    "optimize_translation",
    "optimize_summarization", 
    "optimize_code_generation",
    "multi_objective_optimize",
    "batch_optimize",
    "compare_prompts",
    "setup_openai",
    "setup_anthropic",
    "check_setup"
]
