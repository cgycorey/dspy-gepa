#!/usr/bin/env python3
"""Demo of the new DSPY-GEPA user interface.

This script showcases the improved user interface and demonstrates
how easy it is to use the framework for prompt optimization.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_welcome_and_help():
    """Demonstrate the welcome message and help system."""
    print("🐶 === WELCOME AND HELP DEMO ===")
    print()
    
    # Mock the welcome message since we can't import dspy
    welcome_message = '''
🐶 Welcome to DSPY-GEPA!

DSPY-GEPA is a powerful framework for prompt optimization that makes it easy
to improve your prompts using evolutionary algorithms and real LLM providers.

Quick Start:
1. Set up your LLM API key
2. Use quick_optimize() for instant results
3. Explore presets for specific use cases

Need help? Call show_help() for detailed guidance.
    '''
    print(welcome_message)
    
    # Show available functions (mock version)
    print("📋 === AVAILABLE FUNCTIONS ===")
    print("""
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
    """)


def demo_presets():
    """Demonstrate the preset system."""
    print("\n🎯 === PRESETS DEMO ===")
    print()
    
    # Mock preset data
    presets_info = {
        'quick': {
            'description': 'Fast optimization for quick experiments and prototyping',
            'max_generations': 5,
            'population_size': 3,
            'time_limit': '2 minutes'
        },
        'balanced': {
            'description': 'Balanced optimization for general use cases',
            'max_generations': 10,
            'population_size': 5,
            'time_limit': '5 minutes'
        },
        'quality': {
            'description': 'Focus on output quality and accuracy',
            'max_generations': 15,
            'population_size': 7,
            'time_limit': '10 minutes'
        },
        'efficiency': {
            'description': 'Optimize for speed and resource usage',
            'max_generations': 8,
            'population_size': 4,
            'time_limit': '3 minutes'
        },
        'translation': {
            'description': 'Optimized for translation tasks',
            'max_generations': 12,
            'population_size': 6,
            'time_limit': '8 minutes'
        }
    }
    
    print("Available presets:")
    for name, info in presets_info.items():
        print(f"\n📦 {name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Generations: {info['max_generations']}")
        print(f"   Population: {info['population_size']}")
        print(f"   Time limit: {info['time_limit']}")
    
    print("\n💡 Usage examples:")
    print("   # Use preset directly")
    print("   result = quick_optimize(prompt, data, preset='quality')")
    print("   ")
    print("   # Get preset details")
    print("   preset = get_preset('balanced')")
    print("   print(preset.description)")


def demo_convenience_functions():
    """Demonstrate the convenience functions."""
    print("\n🚀 === CONVENIENCE FUNCTIONS DEMO ===")
    print()
    
    print("1. QUICK OPTIMIZATION:")
    print("   result = quick_optimize(")
    print("       initial_prompt='Translate {text} to Spanish:',")
    print("       evaluation_data=[")
    print("           {'text': 'Hello', 'expected': 'Hola'},")
    print("           {'text': 'Goodbye', 'expected': 'Adiós'}")
    print("       ],")
    print("       preset='quick'  # optional")
    print("   )")
    print()
    
    print("2. TASK-SPECIFIC OPTIMIZATION:")
    print("   # Translation")
    print("   result = optimize_translation(")
    print("       'Translate {source} to Spanish:',")
    print("       [{'source': 'Hello', 'target': 'Hola'}]")
    print("   )")
    print()
    print("   # Code Generation")
    print("   result = optimize_code_generation(")
    print("       'Write Python code to: {description}',")
    print("       [{'description': 'add two numbers', 'expected_code': 'def add(a,b): return a+b'}]")
    print("   )")
    print()
    
    print("3. MULTI-OBJECTIVE OPTIMIZATION:")
    print("   result = multi_objective_optimize(")
    print("       'Your prompt here',")
    print("       evaluation_data,")
    print("       objectives=['accuracy', 'token_usage', 'fluency'],")
    print("       weights={'accuracy': 0.6, 'token_usage': 0.2, 'fluency': 0.2}")
    print("   )")
    print()
    
    print("4. BATCH AND COMPARE:")
    print("   # Optimize multiple prompts")
    print("   results = batch_optimize([prompt1, prompt2, prompt3], data)")
    print("   ")
    print("   # Compare and rank prompts")
    print("   comparison = compare_prompts([prompt1, prompt2], data)")
    print("   print(f'Best: {comparison[\"best_prompt\"]}')")


def demo_setup_and_configuration():
    """Demonstrate setup and configuration."""
    print("\n⚙️ === SETUP AND CONFIGURATION DEMO ===")
    print()
    
    print("1. QUICK SETUP:")
    print("   # OpenAI")
    print("   setup_openai('your-api-key-here', model='gpt-3.5-turbo')")
    print("   ")
    print("   # Anthropic")
    print("   setup_anthropic('your-api-key-here', model='claude-3-sonnet')")
    print()
    
    print("2. ENVIRONMENT CHECK:")
    print("   status = check_setup()")
    print("   print(status)")
    print("   # Output:")
    print("   # {'llm_configured': True, 'dspy_available': True, ...}")
    print()
    
    print("3. INTERACTIVE SETUP:")
    print("   interactive_setup()")
    print("   # Guides you through the setup process step by step")
    print()
    
    print("4. DEBUG MODE:")
    print("   set_debug_mode(True)  # Enable detailed error messages")
    print("   set_debug_mode(False) # Disable debug mode")


def demo_error_handling():
    """Demonstrate improved error handling."""
    print("\n🛡️ === ERROR HANDLING DEMO ===")
    print()
    
    print("The new interface provides helpful error messages with suggestions:")
    print()
    
    print("❌ Configuration Error:")
    print("   Configuration Error: LLM not configured")
    print("   💡 Suggestions:")
    print("      • Check your API key is set correctly")
    print("      • Verify your LLM provider configuration")
    print("      • Run print_llm_status() for detailed help")
    print()
    
    print("❌ Validation Error:")
    print("   Validation Error: Invalid evaluation data format")
    print("   💡 Suggestions:")
    print("      • Check your evaluation data format")
    print("      • Ensure all required fields are present")
    print("      • Verify data types match expected format")
    print()
    
    print("❌ LLM Error:")
    print("   LLM Error: API rate limit exceeded")
    print("   💡 Suggestions:")
    print("      • Check your API key and quota")
    print("      • Verify internet connection")
    print("      • Try a different model or provider")
    print()
    
    print("You can also use error handling decorators:")
    print("   @handle_common_errors")
    print("   def my_function():")
    print("       # Your code here")
    print("       pass")


def demo_examples():
    """Demonstrate the examples system."""
    print("\n📚 === EXAMPLES DEMO ===")
    print()
    
    print("Built-in examples for common use cases:")
    print()
    
    examples = [
        ('quick_start', 'Basic prompt optimization'),
        ('preset_configuration', 'Using preset configurations'),
        ('multi_objective', 'Multi-objective optimization'),
        ('custom_objectives', 'Creating custom objectives'),
        ('translation_task', 'Translation-specific optimization'),
        ('code_generation', 'Code generation optimization'),
        ('monitoring', 'Using monitoring features'),
        ('error_handling', 'Proper error handling')
    ]
    
    for name, description in examples:
        print(f"📖 {name}: {description}")
    
    print("\n💡 Usage:")
    print("   from dspy_gepa.examples import show_example")
    print("   show_example('quick_start')      # Shows code example")
    print("   show_example('multi_objective')  # Shows multi-objective example")
    print()
    
    print("📝 Create evaluation data templates:")
    print("   template = create_evaluation_data_template('translation')")
    print("   print(template)  # Shows template structure")


def demo_best_practices():
    """Demonstrate best practices guidance."""
    print("\n✨ === BEST PRACTICES DEMO ===")
    print()
    
    print("💡 Key Best Practices:")
    print()
    
    print("1. EVALUATION DATA:")
    print("   • Use 5-20 diverse examples")
    print("   • Ensure reference answers are correct")
    print("   • Cover edge cases and variations")
    print()
    
    print("2. PROMPT DESIGN:")
    print("   • Use clear variable names: {input}, {text}, {source}")
    print("   • Be specific about desired output format")
    print("   • Include examples in the prompt when helpful")
    print()
    
    print("3. OPTIMIZATION STRATEGY:")
    print("   • Start with 'quick' preset for testing")
    print("   • Use 'balanced' for final optimization")
    print("   • Consider 'quality' for critical applications")
    print()
    
    print("4. RESOURCE MANAGEMENT:")
    print("   • Monitor API usage and costs")
    print("   • Use appropriate model (GPT-3.5 vs GPT-4)")
    print("   • Cache results when possible")
    print()
    
    print("📖 For complete guidance:")
    print("   print_best_practices()  # Shows comprehensive best practices")


def run_demo():
    """Run the complete demo."""
    print("🐶 DSPY-GEPA New Interface Demo")
    print("="*50)
    
    demos = [
        demo_welcome_and_help,
        demo_presets,
        demo_convenience_functions,
        demo_setup_and_configuration,
        demo_error_handling,
        demo_examples,
        demo_best_practices
    ]
    
    for demo_func in demos:
        demo_func()
        print("\n" + "="*50)
    
    print("\n🎉 Demo completed!")
    print("\n🚀 Ready to get started?")
    print("1. Install dependencies: pip install dspy-ai openai anthropic")
    print("2. Set up your API key: setup_openai('your-key')")
    print("3. Start optimizing: quick_optimize('Your prompt', data)")
    print("\n💡 Need help? Call show_help() for detailed guidance!")


if __name__ == "__main__":
    run_demo()
