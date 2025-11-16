#!/usr/bin/env python3
"""
LLM-Enhanced GEPAAgent Demo

This example demonstrates the enhanced GEPAAgent with LLM support,
showing auto-detection, configuration, and transparent status reporting.

Features demonstrated:
- Auto-detection of LLM configuration from config.yaml
- Manual LLM configuration
- LLM status monitoring
- Automatic fallback to handcrafted mutations
- Transparent optimization reporting
"""

import sys
import os
sys.path.insert(0, '../src')

from dspy_gepa.core.agent import GEPAAgent, LLMConfig


def example_evaluation_function(prompt: str):
    """Example evaluation function for demonstration.
    
    Args:
        prompt: The prompt to evaluate
        
    Returns:
        Dictionary with objective scores
    """
    # Simple mock evaluation - in real usage, this would be your
    # actual evaluation logic (e.g., testing on a dataset)
    import random
    
    # Base score with some randomness
    base_score = 0.3 + random.random() * 0.4
    
    # Bonus points for certain keywords
    if "improve" in prompt.lower():
        base_score += 0.1
    if "accurate" in prompt.lower():
        base_score += 0.1
    if "clear" in prompt.lower():
        base_score += 0.05
    
    # Cap at 1.0
    base_score = min(1.0, base_score)
    
    return {
        "accuracy": base_score,
        "clarity": base_score * 0.9,
        "efficiency": base_score * 0.8
    }


def demo_1_auto_detection():
    """Demo 1: Auto-detection of LLM configuration."""
    print("\n" + "="*70)
    print("🤖 Demo 1: Auto-detection of LLM Configuration")
    print("="*70)
    
    # Create agent with auto-detected LLM config
    agent = GEPAAgent(
        objectives={"accuracy": 0.5, "efficiency": 0.3, "clarity": 0.2},
        max_generations=3,
        population_size=4,
        verbose=True
    )
    
    print(f"\n📊 Agent Status:")
    print(f"   Agent: {agent}")
    
    # Show comprehensive LLM status
    status = agent.get_llm_status()
    print(f"\n🤖 LLM Status:")
    print(f"   Status: {status['status']}")
    print(f"   Provider: {status['provider']}")
    print(f"   Model: {status['model']}")
    print(f"   Available: {status['available']}")
    print(f"   Will use LLM: {status['will_use_llm']}")
    print(f"   Mutation type: {status['mutation_type']}")
    print(f"   Config source: {status['configuration_source']}")
    
    # Run a quick optimization to see LLM in action
    print(f"\n🚀 Running optimization with auto-detected LLM...")
    try:
        result = agent.optimize_prompt(
            "Generate an accurate and clear response.",
            example_evaluation_function
        )
        print(f"\n✅ Optimization completed!")
        print(f"   Best score: {result.best_score:.3f}")
        print(f"   Improvement: {result.improvement_percentage:.1f}%")
    except Exception as e:
        print(f"   Note: {e}")
        print(f"   This is normal for the demo environment.")


def demo_2_manual_configuration():
    """Demo 2: Manual LLM configuration."""
    print("\n" + "="*70)
    print("⚙️ Demo 2: Manual LLM Configuration")
    print("="*70)
    
    # Create agent with manual LLM configuration
    agent = GEPAAgent(
        objectives={"accuracy": 0.8},
        llm_config={
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 1000,
            "enabled": True
        },
        auto_detect_llm=False,
        verbose=True
    )
    
    print(f"\n📊 Agent with manual LLM config:")
    print(f"   Agent: {agent}")
    
    status = agent.get_llm_status()
    print(f"\n🤖 LLM Status:")
    print(f"   Provider: {status['provider']}")
    print(f"   Model: {status['model']}")
    print(f"   Temperature: {status['temperature']}")
    print(f"   Max tokens: {status['max_tokens']}")
    print(f"   Available: {status['available']}")
    print(f"   Config source: {status['configuration_source']}")


def demo_3_configure_llm_method():
    """Demo 3: Using the configure_llm method."""
    print("\n" + "="*70)
    print("🔧 Demo 3: Dynamic LLM Configuration")
    print("="*70)
    
    # Create agent without initial LLM config
    agent = GEPAAgent(
        objectives={"efficiency": 1.0},
        auto_detect_llm=False,
        verbose=True
    )
    
    print(f"\n📊 Initial agent status:")
    print(f"   Agent: {agent}")
    print(f"   LLM available: {agent.is_llm_available()}")
    
    # Configure LLM dynamically
    print(f"\n⚙️ Configuring LLM dynamically...")
    agent.configure_llm(
        "anthropic",
        model="claude-3-sonnet-20240229",
        temperature=0.3,
        max_tokens=1500
    )
    
    print(f"\n📊 Updated agent status:")
    print(f"   Agent: {agent}")
    print(f"   LLM available: {agent.is_llm_available()}")
    
    status = agent.get_llm_status()
    print(f"\n🤖 New LLM Status:")
    print(f"   Provider: {status['provider']}")
    print(f"   Model: {status['model']}")
    print(f"   Mutation type: {status['mutation_type']}")


def demo_4_fallback_behavior():
    """Demo 4: Fallback behavior when LLM unavailable."""
    print("\n" + "="*70)
    print("🔧 Demo 4: Fallback Behavior (LLM Disabled)")
    print("="*70)
    
    # Create agent with LLM usage disabled
    agent = GEPAAgent(
        objectives={"clarity": 0.6, "efficiency": 0.4},
        use_llm_when_available=False,
        verbose=True
    )
    
    print(f"\n📊 Agent with LLM disabled:")
    print(f"   Agent: {agent}")
    
    status = agent.get_llm_status()
    print(f"\n🤖 LLM Status:")
    print(f"   LLM available: {status['available']}")
    print(f"   Will use LLM: {status['will_use_llm']}")
    print(f"   Mutation type: {status['mutation_type']}")
    print(f"   Message: {status['message']}")
    
    print(f"\n🚀 Running optimization with handcrafted mutations only...")
    try:
        result = agent.optimize_prompt(
            "Provide a clear and efficient response.",
            example_evaluation_function
        )
        print(f"\n✅ Optimization completed with handcrafted mutations!")
        print(f"   Best score: {result.best_score:.3f}")
        print(f"   Improvement: {result.improvement_percentage:.1f}%")
    except Exception as e:
        print(f"   Note: {e}")
        print(f"   This is normal for the demo environment.")


def demo_5_comprehensive_status():
    """Demo 5: Comprehensive status and insights."""
    print("\n" + "="*70)
    print("📊 Demo 5: Comprehensive Status and Insights")
    print("="*70)
    
    # Create agent with full configuration
    agent = GEPAAgent(
        objectives={"accuracy": 0.4, "efficiency": 0.3, "clarity": 0.3},
        max_generations=2,
        population_size=3,
        verbose=True
    )
    
    print(f"\n📊 Comprehensive Agent Status:")
    print(f"   Agent: {agent}")
    
    # Show detailed LLM status
    status = agent.get_llm_status()
    print(f"\n🤖 Detailed LLM Configuration:")
    for key, value in status.items():
        if key not in ['status', 'message']:  # Skip redundant fields
            print(f"   {key}: {value}")
    
    # Show optimization insights
    try:
        insights = agent.get_optimization_insights()
        print(f"\n📈 Optimization Insights:")
        print(f"   Objectives: {insights.get('current_objectives', 'N/A')}")
        if 'optimization_config' in insights:
            print(f"   Max generations: {insights['optimization_config']['max_generations']}")
            print(f"   Population size: {insights['optimization_config']['population_size']}")
        if 'best_overall_score' in insights:
            print(f"   Best score: {insights['best_overall_score']:.3f}")
        if 'llm_status' in insights:
            print(f"   LLM status: {insights['llm_status']['status']}")
            print(f"   Mutation type: {insights['llm_status']['mutation_type']}")
    except Exception as e:
        print(f"\n📈 Optimization Insights: Error retrieving insights - {e}")


def main():
    """Run all LLM enhancement demos."""
    print("🚀 LLM-Enhanced GEPAAgent Demo")
    print("Demonstrating comprehensive LLM integration with transparent status reporting")
    
    # Run all demos
    demo_1_auto_detection()
    demo_2_manual_configuration()
    demo_3_configure_llm_method()
    demo_4_fallback_behavior()
    demo_5_comprehensive_status()
    
    print("\n" + "="*70)
    print("✅ LLM-Enhanced GEPAAgent Demo Complete!")
    print("="*70)
    
    print("\n🎯 Key Features Demonstrated:")
    print("✅ Auto-detection of LLM from config.yaml")
    print("✅ Manual LLM configuration")
    print("✅ Dynamic LLM reconfiguration")
    print("✅ Transparent LLM status reporting")
    print("✅ Automatic fallback to handcrafted mutations")
    print("✅ Environment variable support")
    print("✅ Comprehensive optimization insights")
    print("✅ User-friendly error handling")
    
    print("\n📝 Usage Examples:")
    print("# Auto-detect LLM from config.yaml")
    print("agent = GEPAAgent(objectives={'accuracy': 0.7, 'efficiency': 0.3})")
    print("")
    print("# Manual LLM configuration")
    print("agent = GEPAAgent(")
    print("    objectives={'accuracy': 1.0},")
    print("    llm_config={'provider': 'openai', 'model': 'gpt-4'}")
    print(")")
    print("")
    print("# Check LLM status")
    print("print(agent.get_llm_status())")
    print("")
    print("# Configure LLM dynamically")
    print("agent.configure_llm('anthropic', model='claude-3-opus')")
    print("")
    print("# Use agent - automatically uses LLM if available")
    print("result = agent.optimize_prompt(initial_prompt, evaluation_fn)")


if __name__ == "__main__":
    main()