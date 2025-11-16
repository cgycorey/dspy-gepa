#!/usr/bin/env python3
"""
Quick Start Demo - GEPAAgent for Beginners
==========================================

This example shows the simplest possible way to use GEPAAgent.
Perfect for absolute beginners to see immediate results.

Run with:
  uv run python examples/01_quick_start.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dspy_gepa import GEPAAgent

def simple_evaluation(prompt):
    """Simple scoring function - rewards clear, actionable prompts."""
    score = 0.0
    prompt = prompt.lower().strip()
    
    # Reward action words
    action_words = ["analyze", "create", "generate", "write", "provide"]
    if any(word in prompt for word in action_words):
        score += 0.4
    
    # Reward reasonable length
    if 5 <= len(prompt.split()) <= 20:
        score += 0.3
    
    # Reward complete sentence
    if prompt.endswith(('.', '!', '?')):
        score += 0.3
    
    return {"quality": score}

def main():
    """Run a quick demonstration of GEPAAgent."""
    print("🚀 GEPAAgent Quick Start Demo")
    print("=" * 40)
    
    # Step 1: Create the agent
    agent = GEPAAgent(objectives={"quality": 1.0})
    print("✅ Agent created")
    
    # Step 2: Check what's being used
    status = agent.get_llm_status()
    print(f"✅ Using: {status['mutation_type']}")
    
    # Step 3: Run optimization
    initial_prompt = "analyze data"
    print(f"\n📝 Initial prompt: '{initial_prompt}'")
    
    result = agent.optimize_prompt(initial_prompt, simple_evaluation, generations=3)
    
    # Step 4: Show results
    print(f"🎉 Optimized prompt: '{result.best_prompt}'")
    print(f"📈 Improvement: {result.improvement_percentage:.1f}%")
    print(f"🎯 Final score: {result.best_score:.3f}")
    print(f"⚡ Generations: {result.generations_completed}")
    
    print("\n🎊 Success! Your prompt has been optimized!")
    print("📚 Check README.md for more advanced examples")

if __name__ == "__main__":
    main()