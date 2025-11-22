"""User guide and helper functions.

This module provides comprehensive guidance for users, including
setup instructions, troubleshooting, and best practices.
"""

from typing import Dict, Any, List


def print_welcome_message():
    """Print welcome message and getting started info."""
    print(
        """
🐶 Welcome to DSPY-GEPA!

DSPY-GEPA is a powerful framework for prompt optimization that makes it easy
to improve your prompts using evolutionary algorithms and real LLM providers.

Quick Start:
1. Set up your LLM API key
2. Use quick_optimize() for instant results
3. Explore presets for specific use cases

Need help? Call show_help() for detailed guidance.
        """
    )


def show_help(topic: str = "all"):
    """Show help for specific topics.
    
    Args:
        topic: Help topic ('setup', 'examples', 'presets', 'troubleshooting', 'all')
    """
    if topic == "all" or topic == "setup":
        print_setup_help()
    
    if topic == "all" or topic == "examples":
        print_examples_help()
    
    if topic == "all" or topic == "presets":
        print_presets_help()
    
    if topic == "all" or topic == "troubleshooting":
        print_troubleshooting_help()


def print_setup_help():
    """Print setup help."""
    print(
        """
=== SETUP HELP ===

1. Install Dependencies:
   pip install dspy-ai openai anthropic

2. Configure API Key:
   from dspy_gepa import setup_openai, setup_anthropic
   
   # For OpenAI
   setup_openai("your-api-key-here")
   
   # For Anthropic  
   setup_anthropic("your-api-key-here")

3. Verify Setup:
   from dspy_gepa import check_setup, print_llm_status
   
   status = check_setup()
   print(status)
   
   print_llm_status()

4. Start Optimizing:
   from dspy_gepa import quick_optimize
   
   result = quick_optimize(
       "Your prompt template here",
       evaluation_data
   )
        """
    )


def print_examples_help():
    """Print examples help."""
    print(
        """
=== EXAMPLES HELP ===

Quick Examples:

1. Basic Translation:
   from dspy_gepa import optimize_translation
   
   result = optimize_translation(
       "Translate {source} to Spanish:",
       [{"source": "Hello", "target": "Hola"}]
   )

2. Code Generation:
   from dspy_gepa import optimize_code_generation
   
   result = optimize_code_generation(
       "Write Python code to: {description}",
       [{"description": "add two numbers", "expected_code": "def add(a,b): return a+b"}]
   )

3. Multi-Objective:
   from dspy_gepa import multi_objective_optimize
   
   result = multi_objective_optimize(
       "Your prompt here",
       evaluation_data,
       objectives=['accuracy', 'token_usage', 'fluency']
   )

4. Compare Prompts:
   from dspy_gepa import compare_prompts
   
   comparison = compare_prompts(
       ["Prompt 1", "Prompt 2", "Prompt 3"],
       evaluation_data
   )
   print(f"Best: {comparison['best_prompt']}")

For more examples, use:
   from dspy_gepa.examples import show_example
   show_example('multi_objective')
        """
    )


def print_presets_help():
    """Print presets help."""
    print(
        """
=== PRESETS HELP ===

Available Presets:

1. 'quick' - Fast optimization for experiments
   - 5 generations, 3 population size
   - 2 minute time limit
   - Good for quick testing

2. 'balanced' - General purpose optimization
   - 10 generations, 5 population size  
   - 5 minute time limit
   - Recommended for most use cases

3. 'quality' - Focus on output quality
   - 15 generations, 7 population size
   - 10 minute time limit
   - Best for production prompts

4. 'efficiency' - Optimize for speed/resources
   - 8 generations, 4 population size
   - 3 minute time limit
   - Good for cost-sensitive apps

5. Task-Specific:
   - 'translation' - Optimized for translation
   - 'summarization' - Optimized for summarization
   - 'code_generation' - Optimized for code generation

Usage:
   from dspy_gepa import quick_optimize, get_preset
   
   # Use preset directly
   result = quick_optimize(prompt, data, preset='quality')
   
   # Get preset details
   preset = get_preset('balanced')
   print(preset.description)
        """
    )


def print_troubleshooting_help():
    """Print troubleshooting help."""
    print(
        """
=== TROUBLESHOOTING HELP ===

Common Issues:

1. "LLM not configured" Error:
   - Check your API key is set
   - Run print_llm_status() for details
   - Use setup_openai() or setup_anthropic()

2. "No improvement in results":
   - Try the 'quality' preset for more iterations
   - Check your evaluation data quality
   - Increase population size

3. "Optimization takes too long":
   - Use the 'quick' or 'efficiency' presets
   - Reduce max_generations in preset
   - Use smaller evaluation dataset

4. "API rate limits":
   - Use the 'efficiency' preset
   - Reduce population_size and max_generations
   - Add delays between API calls

5. "Poor evaluation scores":
   - Check evaluation data format
   - Ensure reference answers are correct
   - Try different objectives

Debug Mode:
   from dspy_gepa.core.error_handling import set_debug_mode
   set_debug_mode(True)
   
   # This will provide detailed error messages

Get Help:
   - Check status: check_setup()
   - View examples: show_example('quick_start')
   - See presets: list_presets()
        """
    )


def print_best_practices():
    """Print best practices guidance."""
    print(
        """
=== BEST PRACTICES ===

1. Evaluation Data Quality:
   - Use 5-20 diverse examples
   - Ensure reference answers are correct
   - Cover edge cases and variations

2. Prompt Design:
   - Use clear variable names: {input}, {text}, {source}
   - Be specific about desired output format
   - Include examples in the prompt when helpful

3. Optimization Strategy:
   - Start with 'quick' preset for testing
   - Use 'balanced' for final optimization
   - Consider 'quality' for critical applications

4. Multi-Objective Optimization:
   - Balance accuracy with efficiency
   - Use token_usage for cost control
   - Add fluency for user-facing text

5. Resource Management:
   - Monitor API usage and costs
   - Use appropriate model (GPT-3.5 vs GPT-4)
   - Cache results when possible

6. Testing:
   - Test on held-out data
   - Validate with real user scenarios
   - Monitor performance over time
        """
    )


def interactive_setup():
    """Interactive setup wizard."""
    print("🐶 Let's set up DSPY-GEPA together!")
    print()
    
    # Check current status
    from .convenience import check_setup
    status = check_setup()
    
    print("Current setup status:")
    for key, value in status.items():
        if key != 'recommendations':
            print(f"  {key}: {'✅' if value else '❌'}")
    
    if status['recommendations']:
        print("\nRecommendations:")
        for rec in status['recommendations']:
            print(f"  • {rec}")
    
    if not status['llm_configured']:
        print("\nLet's configure your LLM provider:")
        provider = input("Which provider? (openai/anthropic): ").lower().strip()
        
        if provider == 'openai':
            api_key = input("Enter your OpenAI API key: ").strip()
            model = input("Model (default: gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
            
            from .convenience import setup_openai
            if setup_openai(api_key, model):
                print("✅ OpenAI configured successfully!")
            else:
                print("❌ Failed to configure OpenAI")
                
        elif provider == 'anthropic':
            api_key = input("Enter your Anthropic API key: ").strip()
            model = input("Model (default: claude-3-sonnet-20240229): ").strip() or "claude-3-sonnet-20240229"
            
            from .convenience import setup_anthropic
            if setup_anthropic(api_key, model):
                print("✅ Anthropic configured successfully!")
            else:
                print("❌ Failed to configure Anthropic")
        else:
            print("Unknown provider. Please use 'openai' or 'anthropic'")
    
    print("\nSetup complete! 🎉")
    print("Try: quick_optimize('Your prompt here', evaluation_data)")


def create_evaluation_data_template(task_type: str = "general") -> str:
    """Create a template for evaluation data.
    
    Args:
        task_type: Type of task ('general', 'translation', 'summarization', 'code')
        
    Returns:
        Template string
    """
    templates = {
        "general": """
# General task evaluation data
evaluation_data = [
    {
        "input": "Example input 1",
        "expected": "Expected output 1"
    },
    {
        "input": "Example input 2", 
        "expected": "Expected output 2"
    },
    # Add 3-10 more examples...
]
        """,
        
        "translation": """
# Translation evaluation data
evaluation_data = [
    {
        "source": "Hello world",
        "target": "Hola mundo"
    },
    {
        "source": "How are you?",
        "target": "¿Cómo estás?"
    },
    # Add more translation pairs...
]
        """,
        
        "summarization": """
# Summarization evaluation data
evaluation_data = [
    {
        "article": "Long article text here...",
        "summary": "Concise summary of the article..."
    },
    {
        "article": "Another article...",
        "summary": "Another summary..."
    },
    # Add more article-summary pairs...
]
        """,
        
        "code": """
# Code generation evaluation data
evaluation_data = [
    {
        "description": "function that adds two numbers",
        "expected_code": "def add(a, b):\n    return a + b"
    },
    {
        "description": "function that checks if a number is even",
        "expected_code": "def is_even(n):\n    return n % 2 == 0"
    },
    # Add more description-code pairs...
]
        """
    }
    
    return templates.get(task_type, templates["general"])


def list_available_functions():
    """List all available convenience functions."""
    print(
        """
=== AVAILABLE FUNCTIONS ===

Optimization Functions:
• quick_optimize() - Fast optimization with presets
• optimize_translation() - Translation-specific optimization
• optimize_summarization() - Summarization-specific optimization
• optimize_code_generation() - Code generation optimization
• multi_objective_optimize() - Multi-objective optimization
• batch_optimize() - Optimize multiple prompts
• compare_prompts() - Compare and rank prompts

Setup Functions:
• setup_openai() - Configure OpenAI API
• setup_anthropic() - Configure Anthropic API
• check_setup() - Verify environment setup
• interactive_setup() - Guided setup wizard

Preset Functions:
• get_preset() - Get optimization preset
• list_presets() - List all presets
• get_quick_preset() - Quick optimization preset
• get_balanced_preset() - Balanced preset
• get_quality_preset() - Quality-focused preset
• get_efficiency_preset() - Efficiency-focused preset

Help Functions:
• show_help() - Show help topics
• print_best_practices() - Show best practices
• create_evaluation_data_template() - Create data template
• list_available_functions() - This function

Example Functions:
• from dspy_gepa.examples import show_example
• show_example('quick_start') - Show quick start example
• show_example('multi_objective') - Show multi-objective example
        """
    )


# Export guide functions
__all__ = [
    "print_welcome_message",
    "show_help",
    "print_setup_help",
    "print_examples_help", 
    "print_presets_help",
    "print_troubleshooting_help",
    "print_best_practices",
    "interactive_setup",
    "create_evaluation_data_template",
    "list_available_functions"
]
