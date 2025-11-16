#!/usr/bin/env python3
"""
Enhanced GEPAAgent Interface Demonstration

This script demonstrates the enhanced agent interface with LLM detection,
configuration options, and fallback behavior.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    """Main demonstration of enhanced agent interface."""
    
    print("🎯 Enhanced GEPAAgent Interface Demonstration")
    print("=" * 60)
    
    from dspy_gepa import GEPAAgent
    
    # Demo 1: Auto-detection from config.yaml
    print("\n📋 Demo 1: Auto-detection from config.yaml")
    print("-" * 40)
    
    agent = GEPAAgent(objectives={"accuracy": 0.7, "efficiency": 0.3})
    status = agent.get_llm_status()
    print(f"✅ LLM Status: {status['status']}")
    if status['status'] == 'available':
        print(f"✅ Provider: {status.get('provider', 'N/A')}")
        print(f"✅ Model: {status.get('model', 'N/A')}")
        print(f"✅ Mutation Type: {status.get('mutation_type', 'N/A')}")
        print(f"✅ Config Source: {status.get('configuration_source', 'N/A')}")
    else:
        print("ℹ️  LLM not configured - will use handcrafted mutations")
        print(f"ℹ️  Message: {status.get('message', 'No details available')}")
    
    # Demo 2: Manual LLM configuration
    print("\n📋 Demo 2: Manual LLM configuration")
    print("-" * 40)
    
    agent_manual = GEPAAgent(
        objectives={"accuracy": 1.0},
        llm_config={
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.5,
            "enabled": True
        },
        auto_detect_llm=False
    )
    manual_status = agent_manual.get_llm_status()
    print(f"✅ Manual Config - Provider: {manual_status.get('provider')}")
    print(f"✅ Manual Config - Model: {manual_status.get('model')}")
    print(f"✅ Manual Config - Temperature: {manual_status.get('temperature')}")
    print(f"✅ Manual Config - Available: {manual_status.get('available')}")
    
    # Demo 3: Dynamic configuration
    print("\n📋 Demo 3: Dynamic LLM reconfiguration")
    print("-" * 40)
    
    agent.configure_llm("anthropic", model="claude-3-opus", temperature=0.3)
    dynamic_status = agent.get_llm_status()
    print(f"✅ Dynamic Config - Provider: {dynamic_status.get('provider')}")
    print(f"✅ Dynamic Config - Model: {dynamic_status.get('model')}")
    print(f"✅ Dynamic Config - Temperature: {dynamic_status.get('temperature')}")
    print(f"✅ Dynamic Config - Mutation Type: {dynamic_status.get('mutation_type')}")
    
    # Demo 4: Simple optimization workflow
    print("\n📋 Demo 4: Simple optimization workflow")
    print("-" * 40)
    
    def simple_evaluate(prompt: str) -> dict:
        """Simple evaluation function for demonstration."""
        length_score = min(len(prompt) / 50, 1.0)
        clarity_score = 0.8 if '.' in prompt else 0.5
        completeness_score = 0.7 if any(word in prompt.lower() for word in ['analyze', 'evaluate', 'process']) else 0.4
        
        return {
            "accuracy": (length_score + clarity_score + completeness_score) / 3,
            "efficiency": 0.7 - length_score * 0.2,
            "clarity": clarity_score
        }
    
    simple_agent = GEPAAgent(
        objectives={"accuracy": 0.6, "efficiency": 0.4},
        max_generations=2
    )
    
    print(f"✅ Using: {simple_agent.get_llm_status()['mutation_type']}")
    
    result = simple_agent.optimize_prompt(
        "Analyze the data and provide insights.",
        simple_evaluate,
        generations=2
    )
    
    print(f"✅ Best prompt: {result.best_prompt[:60]}...")
    print(f"✅ Best score: {result.best_score:.4f}")
    print(f"✅ Improvement: {result.improvement_percentage:.1f}%")
    print(f"✅ Generations: {result.generations_completed}")
    print(f"✅ Optimization time: {result.optimization_time:.1f}s")
    print(f"✅ Objectives scores: {result.objectives_score}")
    
    # Demo 5: Environment variable support
    print("\n📋 Demo 5: Environment variable support")
    print("-" * 40)
    
    # Set environment variables (example)
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key-here'
    
    print("✅ Environment variables set (replace with actual keys for production)")
    print("✅ Agent will automatically detect and use these when available")
    
    env_agent = GEPAAgent(objectives={"accuracy": 1.0})
    env_status = env_agent.get_llm_status()
    
    if env_status['status'] == 'available':
        print(f"✅ LLM detected from environment: {env_status.get('provider')}")
    else:
        print("ℹ️  Valid API keys needed for LLM detection")
        print(f"ℹ️  Current status: {env_status.get('message')}")
    
    # Demo 6: Fallback behavior demonstration
    print("\n📋 Demo 6: Fallback behavior (LLM unavailable)")
    print("-" * 40)
    
    fallback_agent = GEPAAgent(
        objectives={"accuracy": 0.8, "efficiency": 0.2},
        llm_config={
            "provider": "openai",
            "api_key": "invalid-key-demo",
            "enabled": True
        },
        auto_detect_llm=False
    )
    
    fallback_status = fallback_agent.get_llm_status()
    print(f"✅ Fallback - Will use LLM: {fallback_status['will_use_llm']}")
    print(f"✅ Fallback - Mutation type: {fallback_status['mutation_type']}")
    print(f"✅ Fallback - Status: {fallback_status['message']}")
    
    # Demo 7: Advanced usage patterns
    print("\n📋 Demo 7: Advanced usage patterns")
    print("-" * 40)
    
    # Multi-objective optimization
    multi_agent = GEPAAgent(
        objectives={
            "accuracy": 0.4,
            "efficiency": 0.3,
            "clarity": 0.2,
            "creativity": 0.1
        },
        max_generations=3,
        population_size=4
    )
    
    def multi_evaluate(prompt: str) -> dict:
        """Multi-objective evaluation function."""
        return {
            "accuracy": 0.6 + (hash(prompt) % 100) / 200,
            "efficiency": 0.7 - len(prompt) / 500,
            "clarity": 0.8 if len(prompt.split()) < 20 else 0.6,
            "creativity": 0.5 + (len(set(prompt.lower())) / 1000)
        }
    
    print(f"✅ Multi-objective agent: {multi_agent}")
    print(f"✅ Objectives: {list(multi_agent.config.objectives.keys())}")
    print(f"✅ LLM Status: {multi_agent.get_llm_status()['mutation_type']}")
    
    # Demo 8: Configuration inspection
    print("\n📋 Demo 8: Configuration inspection and insights")
    print("-" * 40)
    
    insights = agent.get_optimization_insights()
    print("✅ Available insights:")
    for key, value in insights.items():
        if key != 'amope_insights':  # Skip nested for clarity
            print(f"   {key}: {value}")
    
    # Demo 9: Agent representation and debugging
    print("\n📋 Demo 9: Agent representation and debugging")
    print("-" * 40)
    
    print(f"✅ Agent repr: {agent}")
    print(f"✅ Agent name: {agent.name}")
    print(f"✅ Config objectives: {agent.config.objectives}")
    print(f"✅ Config max_generations: {agent.config.max_generations}")
    print(f"✅ Config population_size: {agent.config.population_size}")
    print(f"✅ LLM available: {agent.is_llm_available()}")
    
    # Demo 10: Error handling and robust configuration
    print("\n📋 Demo 10: Error handling and robust configuration")
    print("-" * 40)
    
    try:
        # This should handle gracefully
        robust_agent = GEPAAgent(
            objectives={},  # Empty objectives - should use default
            llm_config={
                "provider": "nonexistent_provider",
                "api_key": None
            },
            auto_detect_llm=False
        )
        print("✅ Robust agent created successfully despite invalid config")
        print(f"✅ Robust status: {robust_agent.get_llm_status()['message']}")
    except Exception as e:
        print(f"ℹ️  Error handling: {e}")
    
    print("\n🎉 Demonstration completed successfully!")
    print("=" * 60)
    
    print("\n💡 Usage Tips:")
    print("1. Set up config.yaml with your LLM provider details")
    print("2. Or use environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY)")
    print("3. Agent automatically falls back to handcrafted mutations if LLM unavailable")
    print("4. Use get_llm_status() to check what's being used")
    print("5. Use configure_llm() to change providers dynamically")
    print("6. Use get_optimization_insights() for performance analysis")
    print("7. Agent works with both single and multi-objective optimization")
    print("8. All configurations are robust with proper fallback handling")

if __name__ == "__main__":
    main()