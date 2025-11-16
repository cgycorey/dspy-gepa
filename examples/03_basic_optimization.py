#!/usr/bin/env python3
"""
Basic Optimization Demo
=======================

Demonstrates fundamental prompt optimization with GEPAAgent.
Shows simple single-objective and multi-objective optimization.

Run with:
  uv run python examples/03_basic_optimization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_gepa import GEPAAgent

def simple_evaluation(prompt: str) -> dict:
    """Simple evaluation function that rewards clear, actionable prompts."""
    score = 0.0
    prompt_lower = prompt.lower().strip()
    
    # Reward action words (40% weight)
    action_words = ["analyze", "create", "generate", "write", "provide", "evaluate"]
    if any(word in prompt_lower for word in action_words):
        score += 0.4
    
    # Reward reasonable length (30% weight)
    word_count = len(prompt.split())
    if 5 <= word_count <= 20:
        score += 0.3
    elif word_count > 20:
        score += 0.1  # Penalty for being too wordy
    
    # Reward complete sentence (30% weight)
    if prompt.strip().endswith(('.', '!', '?')):
        score += 0.3
    
    return {"quality": score}

def multi_objective_evaluation(prompt: str) -> dict:
    """Multi-objective evaluation for balancing different qualities."""
    prompt_lower = prompt.lower().strip()
    word_count = len(prompt.split())
    
    # Accuracy: rewards clear, specific instructions
    accuracy_score = 0.0
    if any(word in prompt_lower for word in ["analyze", "evaluate", "assess"]):
        accuracy_score += 0.5
    if any(word in prompt_lower for word in ["specific", "detailed", "comprehensive"]):
        accuracy_score += 0.3
    if word_count >= 8:  # Enough detail
        accuracy_score += 0.2
    accuracy_score = min(1.0, accuracy_score)
    
    # Efficiency: rewards conciseness
    efficiency_score = 1.0
    if word_count > 25:
        efficiency_score = max(0.3, 1.0 - (word_count - 25) * 0.05)
    elif word_count < 5:
        efficiency_score = 0.6  # Too brief
    
    # Clarity: rewards proper grammar and structure
    clarity_score = 0.8 if prompt.strip().endswith(('.', '!', '?')) else 0.4
    if '.' in prompt and prompt.count('.') <= 2:  # Complete sentences, not too many
        clarity_score += 0.2
    clarity_score = min(1.0, clarity_score)
    
    return {
        "accuracy": accuracy_score,
        "efficiency": efficiency_score,
        "clarity": clarity_score
    }

def demo_single_objective():
    """Demonstrate single-objective optimization."""
    print("\n🎯 Single-Objective Optimization")
    print("-" * 35)
    
    agent = GEPAAgent(
        objectives={"quality": 1.0},
        max_generations=5,
        population_size=6
    )
    
    initial_prompt = "analyze data"
    print(f"📝 Initial: '{initial_prompt}'")
    
    result = agent.optimize_prompt(
        initial_prompt,
        simple_evaluation,
        generations=5
    )
    
    print(f"✅ Optimized: '{result.best_prompt}'")
    print(f"📈 Improvement: {result.improvement_percentage:.1f}%")
    print(f"🎯 Final Score: {result.best_score:.3f}")
    print(f"⚡ Generations: {result.generations_completed}")
    print(f"⏱️  Time: {result.optimization_time:.1f}s")
    
    return result

def demo_multi_objective():
    """Demonstrate multi-objective optimization."""
    print("\n🎯 Multi-Objective Optimization")
    print("-" * 35)
    
    agent = GEPAAgent(
        objectives={
            "accuracy": 0.5,
            "efficiency": 0.3,
            "clarity": 0.2
        },
        max_generations=5,
        population_size=6
    )
    
    initial_prompt = "analyze data"
    print(f"📝 Initial: '{initial_prompt}'")
    print(f"🎯 Objectives: {list(agent.config.objectives.keys())}")
    
    result = agent.optimize_prompt(
        initial_prompt,
        multi_objective_evaluation,
        generations=5
    )
    
    print(f"✅ Optimized: '{result.best_prompt}'")
    print(f"📈 Improvement: {result.improvement_percentage:.1f}%")
    print(f"🎯 Overall Score: {result.best_score:.3f}")
    print(f"📊 Individual Scores:")
    for obj, score in result.objectives_score.items():
        print(f"   {obj}: {score:.3f}")
    print(f"⚡ Generations: {result.generations_completed}")
    
    return result

def demo_comparison():
    """Compare different approaches side by side."""
    print("\n🔍 Approach Comparison")
    print("-" * 25)
    
    approaches = [
        ("Handcrafted Only", {"enabled": False}),
        ("LLM-Enhanced", {"enabled": True, "provider": "auto"})
    ]
    
    initial_prompt = "analyze the data"
    
    for name, llm_config in approaches:
        print(f"\n📋 {name}:")
        try:
            agent = GEPAAgent(
                objectives={"quality": 1.0},
                llm_config=llm_config,
                max_generations=3,
                population_size=4
            )
            
            status = agent.get_llm_status()
            print(f"   Using: {status['mutation_type']}")
            
            result = agent.optimize_prompt(
                initial_prompt,
                simple_evaluation,
                generations=3
            )
            
            print(f"   Result: '{result.best_prompt}'")
            print(f"   Score: {result.best_score:.3f}")
            
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Run basic optimization demonstrations."""
    print("🚀 Basic Optimization Demo")
    print("=" * 40)
    
    # Check what mutation type we'll be using
    test_agent = GEPAAgent(objectives={"quality": 1.0})
    status = test_agent.get_llm_status()
    print(f"✅ Using: {status['mutation_type']}")
    
    # Run demonstrations
    demo_single_objective()
    demo_multi_objective()
    demo_comparison()
    
    print("\n💡 Optimization Tips:")
    print("1. Start with clear evaluation functions")
    print("2. Balance objectives based on your priorities")
    print("3. Use fewer generations for quick tests")
    print("4. Increase population size for diverse solutions")
    print("5. Multi-objective helps balance trade-offs")
    
    print("\n🎉 Basic optimization demo completed!")

if __name__ == "__main__":
    main()