#!/usr/bin/env python3
"""
Advanced Features Demo
======================

Demonstrates advanced GEPAAgent features including:
- Configuration inspection and debugging
- Error handling and robust configuration
- Performance insights and monitoring
- Advanced usage patterns

Run with:
  uv run python examples/04_advanced_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_gepa import GEPAAgent

def robust_evaluation(prompt: str) -> dict:
    """Robust evaluation function that handles edge cases."""
    if not prompt or not prompt.strip():
        return {"quality": 0.0, "robustness": 0.0}
    
    prompt = prompt.strip()
    
    # Quality score
    quality = 0.0
    if len(prompt.split()) >= 3:
        quality += 0.3
    if any(word in prompt.lower() for word in ["analyze", "create", "generate"]):
        quality += 0.4
    if prompt.endswith(('.', '!', '?')):
        quality += 0.3
    
    # Robustness score (handles malformed input gracefully)
    robustness = 0.8  # Default good score
    try:
        # Test various edge cases
        if len(prompt) > 1000:
            robustness -= 0.2  # Too long
        if len(prompt.split()) > 100:
            robustness -= 0.2  # Too many words
        if prompt.count('?') > 3:
            robustness -= 0.1  # Too many questions
    except Exception:
        robustness = 0.0  # Something went wrong
    
    return {
        "quality": min(1.0, quality),
        "robustness": max(0.0, robustness)
    }

def demo_configuration_inspection():
    """Demonstrate configuration inspection and debugging."""
    print("\n🔍 Configuration Inspection")
    print("-" * 30)
    
    agent = GEPAAgent(
        objectives={
            "quality": 0.6,
            "robustness": 0.4
        },
        max_generations=10,
        population_size=8
    )
    
    print(f"✅ Agent: {agent}")
    print(f"✅ Name: {agent.name}")
    print(f"✅ Objectives: {list(agent.config.objectives.keys())}")
    print(f"✅ Weights: {agent.config.objectives}")
    print(f"✅ Max Generations: {agent.config.max_generations}")
    print(f"✅ Population Size: {agent.config.population_size}")
    
    # Get detailed insights
    insights = agent.get_optimization_insights()
    print("\n📊 Optimization Insights:")
    for key, value in insights.items():
        if key != 'amope_insights':  # Skip nested for clarity
            print(f"   {key}: {value}")

def demo_error_handling():
    """Demonstrate robust error handling."""
    print("\n🛡️  Error Handling & Robustness")
    print("-" * 35)
    
    # Test 1: Empty objectives
    print("\n📋 Test 1: Empty objectives (should use defaults)")
    try:
        agent1 = GEPAAgent(objectives={})
        print(f"✅ Handled gracefully: {len(agent1.config.objectives)} objectives")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 2: Invalid LLM provider
    print("\n📋 Test 2: Invalid LLM provider")
    try:
        agent2 = GEPAAgent(
            objectives={"quality": 1.0},
            llm_config={
                "provider": "nonexistent_provider",
                "enabled": True
            },
            auto_detect_llm=False
        )
        status = agent2.get_llm_status()
        print(f"✅ Handled gracefully: {status['message']}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Malformed evaluation function
    print("\n📋 Test 3: Malformed prompts")
    agent3 = GEPAAgent(objectives={"quality": 1.0}, max_generations=2)
    
    malformed_prompts = [
        "",  # Empty
        "   ",  # Whitespace only
        "x" * 2000,  # Too long
        "???" * 50,  # Too many questions
        "analyze data"  # Normal (for comparison)
    ]
    
    for i, prompt in enumerate(malformed_prompts):
        try:
            scores = robust_evaluation(prompt)
            print(f"   Test {i+1}: Quality={scores['quality']:.2f}, Robustness={scores['robustness']:.2f}")
        except Exception as e:
            print(f"   Test {i+1}: Error = {e}")

def demo_advanced_patterns():
    """Demonstrate advanced usage patterns."""
    print("\n🎯 Advanced Usage Patterns")
    print("-" * 30)
    
    # Pattern 1: Progressive optimization
    print("\n📋 Pattern 1: Progressive Optimization")
    
    initial_prompt = "analyze data"
    agent = GEPAAgent(
        objectives={"quality": 1.0},
        max_generations=3,
        population_size=4
    )
    
    current_prompt = initial_prompt
    for stage in range(3):
        print(f"\n   Stage {stage + 1}: '{current_prompt}'")
        
        result = agent.optimize_prompt(
            current_prompt,
            robust_evaluation,
            generations=2
        )
        
        current_prompt = result.best_prompt
        print(f"   Result: '{current_prompt}' (Score: {result.best_score:.3f})")
    
    # Pattern 2: Multi-agent comparison
    print("\n📋 Pattern 2: Multi-Strategy Comparison")
    
    strategies = [
        ("Conservative", {"max_generations": 5, "population_size": 4}),
        ("Balanced", {"max_generations": 8, "population_size": 6}),
        ("Aggressive", {"max_generations": 12, "population_size": 8})
    ]
    
    test_prompt = "analyze the dataset"
    
    for name, config in strategies:
        agent = GEPAAgent(
            objectives={"quality": 0.7, "robustness": 0.3},
            **config
        )
        
        result = agent.optimize_prompt(
            test_prompt,
            robust_evaluation,
            generations=config["max_generations"]
        )
        
        print(f"   {name}: Score={result.best_score:.3f}, Time={result.optimization_time:.1f}s")

def demo_performance_monitoring():
    """Demonstrate performance monitoring and insights."""
    print("\n📈 Performance Monitoring")
    print("-" * 25)
    
    agent = GEPAAgent(
        objectives={"quality": 1.0},
        max_generations=5,
        population_size=6
    )
    
    # Run optimization with monitoring
    result = agent.optimize_prompt(
        "analyze data patterns",
        robust_evaluation,
        generations=5
    )
    
    # Detailed results analysis
    print(f"✅ Optimization completed")
    print(f"   Generations: {result.generations_completed}/{agent.config.max_generations}")
    print(f"   Total Time: {result.optimization_time:.2f}s")
    print(f"   Avg Time/Generation: {result.optimization_time/max(1, result.generations_completed):.2f}s")
    print(f"   Improvement: {result.improvement_percentage:.1f}%")
    
    # Get performance insights
    insights = agent.get_optimization_insights()
    if 'performance_metrics' in insights:
        metrics = insights['performance_metrics']
        print(f"\n📊 Performance Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")

def main():
    """Run advanced features demonstration."""
    print("🚀 Advanced Features Demo")
    print("=" * 40)
    
    # Run all advanced demos
    demo_configuration_inspection()
    demo_error_handling()
    demo_advanced_patterns()
    demo_performance_monitoring()
    
    print("\n💡 Advanced Tips:")
    print("1. Use get_optimization_insights() for performance analysis")
    print("2. Implement robust evaluation functions that handle edge cases")
    print("3. Test with progressive optimization for complex problems")
    print("4. Compare different strategies to find optimal settings")
    print("5. Monitor performance metrics to fine-tune your configuration")
    print("6. Always handle malformed input gracefully in evaluation functions")
    
    print("\n🎉 Advanced features demo completed!")

if __name__ == "__main__":
    main()