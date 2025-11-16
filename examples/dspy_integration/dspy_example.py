#!/usr/bin/env python3
"""
DSPY Integration Demo
====================

This example demonstrates how to use DSPY-GEPA integration for optimizing
DSPY programs using genetic evolutionary programming.

Note: DSPY is optional. This demo includes mock DSPY classes for demonstration
when DSPY is not installed.

Run with:
  uv run python examples/dspy_integration/dspy_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Try to import DSPY, fall back to mock if not available
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    print("⚠️ DSPY not installed. Using mock DSPY for demonstration.")
    DSPY_AVAILABLE = False
    
    # Create mock DSPY classes for demonstration
    class MockPredictor:
        def __init__(self, signature=None):
            self.signature = signature
        
        def dump_state(self):
            return {"signature": str(self.signature) if self.signature else "MockSignature"}
        
        def load_state(self, state):
            pass
    
    class MockModule:
        def __init__(self):
            # Simulate SimpleQA structure
            self.generate_answer = MockPredictor()
        
        def dump_state(self):
            return {
                "generate_answer": self.generate_answer.dump_state(),
                "program_type": "MockSimpleQA"
            }
        
        def load_state(self, state):
            self.generate_answer = MockPredictor()
    
    # Create mock dspy module
    class MockDSPY:
        Module = MockModule
        Predict = MockPredictor
        
        class Signature:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def __str__(self):
                return f"MockSignature({self.kwargs})"
    
    dspy = MockDSPY()

from dspy_gepa import GEPAAgent

def create_dspy_program():
    """Create a simple DSPY program for optimization."""
    
    if DSPY_AVAILABLE:
        # Real DSPY program
        class SimpleQA(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate_answer = dspy.Predict(
                    dspy.Signature("question -> answer")
                )
            
            def forward(self, question):
                return self.generate_answer(question=question)
        
        return SimpleQA()
    else:
        # Mock DSPY program
        return dspy.Module()

def evaluate_dspy_program(program_state, question="What is the capital of France?"):
    """Evaluate a DSPY program based on its state configuration.
    
    Args:
        program_state: Serialized DSPY program state
        question: Test question for evaluation
        
    Returns:
        dict: Performance scores for different objectives
    """
    # Simulate evaluation based on program characteristics
    state_str = str(program_state)
    
    # Accuracy: rewards clear, specific signatures
    accuracy_score = 0.5
    if "question" in state_str.lower() and "answer" in state_str.lower():
        accuracy_score += 0.3
    if "signature" in state_str.lower():
        accuracy_score += 0.2
    
    # Efficiency: rewards simple, lightweight programs
    efficiency_score = 0.8
    if len(state_str) > 1000:  # Too complex
        efficiency_score -= 0.3
    if len(state_str) > 2000:  # Very complex
        efficiency_score -= 0.4
    
    # Clarity: rewards well-structured programs
    clarity_score = 0.7
    if "generate_answer" in state_str:
        clarity_score += 0.2
    if "SimpleQA" in state_str or "MockSimpleQA" in state_str:
        clarity_score += 0.1
    
    return {
        "accuracy": min(1.0, accuracy_score),
        "efficiency": max(0.0, efficiency_score),
        "clarity": min(1.0, clarity_score)
    }

def program_state_to_prompt(program_state):
    """Convert DSPY program state to a prompt representation for optimization."""
    state_str = str(program_state)
    
    # Extract key components from the program state
    if "signature" in state_str.lower():
        return f"DSPY program with question->answer signature: {state_str[:100]}"
    elif "generate_answer" in state_str:
        return f"DSPY program with answer generation: {state_str[:100]}"
    else:
        return f"DSPY program configuration: {state_str[:100]}"

def prompt_to_program_state(prompt):
    """Convert optimized prompt back to DSPY program state."""
    # In a real implementation, this would involve parsing the prompt
    # and updating the DSPY program accordingly. For demo purposes,
    # we'll return a modified state string.
    
    if DSPY_AVAILABLE:
        # Create a new program with optimized characteristics
        optimized_program = create_dspy_program()
        state = optimized_program.dump_state()
        
        # Simulate optimization by updating the state
        if "better" in prompt.lower():
            state["optimized"] = True
        if "efficient" in prompt.lower():
            state["efficient"] = True
            
        return state
    else:
        # Mock program state
        return {
            "generate_answer": {
                "signature": "question -> answer (optimized)",
                "optimized": True,
                "efficient": "efficient" in prompt.lower()
            },
            "program_type": "OptimizedMockSimpleQA"
        }

def demo_basic_dspy_optimization():
    """Demonstrate basic DSPY program optimization."""
    print("\n🧠 Basic DSPY Program Optimization")
    print("-" * 40)
    
    # Create initial DSPY program
    initial_program = create_dspy_program()
    initial_state = initial_program.dump_state()
    
    print(f"✅ Created {'real' if DSPY_AVAILABLE else 'mock'} DSPY program")
    print(f"📝 Initial state: {str(initial_state)[:100]}...")
    
    # Convert to prompt for optimization
    initial_prompt = program_state_to_prompt(initial_state)
    print(f"🔄 Converted to prompt: {initial_prompt}")
    
    # Set up GEPAAgent for DSPY optimization
    agent = GEPAAgent(
        objectives={
            "accuracy": 0.5,
            "efficiency": 0.3,
            "clarity": 0.2
        },
        max_generations=5,
        population_size=6
    )
    
    print(f"✅ GEPAAgent configured for multi-objective optimization")
    
    # Create evaluation function that simulates DSPY program performance
    def dspy_evaluation(prompt):
        # Convert prompt back to program state
        program_state = prompt_to_program_state(prompt)
        # Evaluate the program
        return evaluate_dspy_program(program_state)
    
    # Run optimization
    result = agent.optimize_prompt(
        initial_prompt,
        dspy_evaluation,
        generations=5
    )
    
    print(f"✅ Optimization completed")
    print(f"📈 Improvement: {result.improvement_percentage:.1f}%")
    print(f"🎯 Final score: {result.best_score:.3f}")
    print(f"📝 Optimized prompt: {result.best_prompt}")
    
    # Convert back to program state
    optimized_state = prompt_to_program_state(result.best_prompt)
    print(f"🔄 Optimized state: {str(optimized_state)[:100]}...")
    
    # Show individual objective scores
    print(f"📊 Objective scores:")
    for obj, score in result.objectives_score.items():
        print(f"   {obj}: {score:.3f}")
    
    return result

def demo_dspy_vs_traditional():
    """Compare DSPY-optimized prompts vs traditional prompts."""
    print("\n🔍 DSPY vs Traditional Optimization")
    print("-" * 40)
    
    # Traditional prompt optimization
    traditional_agent = GEPAAgent(
        objectives={"quality": 1.0},
        max_generations=3
    )
    
    def traditional_eval(prompt):
        # Simple evaluation for traditional prompts
        score = 0.0
        if any(word in prompt.lower() for word in ["answer", "question", "generate"]):
            score += 0.5
        if len(prompt.split()) >= 5:
            score += 0.3
        if prompt.endswith(('.', '!', '?')):
            score += 0.2
        return {"quality": score}
    
    traditional_result = traditional_agent.optimize_prompt(
        "Generate an answer for the question",
        traditional_eval,
        generations=3
    )
    
    # DSPY-optimized prompt
    dspy_program = create_dspy_program()
    dspy_prompt = program_state_to_prompt(dspy_program.dump_state())
    
    dspy_agent = GEPAAgent(
        objectives={"accuracy": 0.6, "efficiency": 0.4},
        max_generations=3
    )
    
    def dspy_eval(prompt):
        program_state = prompt_to_program_state(prompt)
        return evaluate_dspy_program(program_state)
    
    dspy_result = dspy_agent.optimize_prompt(
        dspy_prompt,
        dspy_eval,
        generations=3
    )
    
    print(f"📋 Traditional: Score={traditional_result.best_score:.3f}")
    print(f"   Prompt: {traditional_result.best_prompt}")
    
    print(f"📋 DSPY: Score={dspy_result.best_score:.3f}")
    print(f"   Prompt: {dspy_result.best_prompt}")
    
    print(f"\n💡 DSPY optimization provides:")
    print(f"   ✅ Structured program representation")
    print(f"   ✅ Multi-objective evaluation")
    print(f"   ✅ Domain-specific optimization")

def main():
    """Run DSPY integration demonstration."""
    print("🧠 DSPY Integration Demo")
    print("=" * 40)
    
    print(f"✅ DSPY available: {DSPY_AVAILABLE}")
    
    if not DSPY_AVAILABLE:
        print("💡 Install DSPY with: uv add dspy")
        print("   (This demo works with mock DSPY classes)")
    
    # Run demonstrations
    demo_basic_dspy_optimization()
    demo_dspy_vs_traditional()
    
    print("\n💡 DSPY Integration Benefits:")
    print("1. 🧠 Structured program optimization")
    print("2. 🎯 Multi-objective evaluation")
    print("3. 🔄 State-to-prompt conversion")
    print("4. 📊 Domain-specific metrics")
    print("5. 🚀 Genetic algorithm enhancement")
    
    print("\n🎉 DSPY integration demo completed!")
    
    if not DSPY_AVAILABLE:
        print("\n💡 For full DSPY functionality:")
        print("   uv add dspy")
        print("   Then run this demo again!")

if __name__ == "__main__":
    main()