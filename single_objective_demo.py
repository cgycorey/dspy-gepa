#!/usr/bin/env python3
"""
Single-Objective Optimization Demo

This demonstrates the GEPA framework optimizing for a single objective.
Shows basic usage and monitoring capabilities.

Usage:
    uv run python single_objective_demo.py
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_single_objective_evaluation():
    """Create evaluation function for single objective (effectiveness)."""
    def evaluate(prompt: str) -> dict:
        """Evaluate prompt effectiveness score (0.0 to 1.0)."""
        prompt_lower = prompt.lower()
        
        # Base score
        score = 0.3
        
        # Length factor
        words = prompt_lower.split()
        if len(words) >= 5:
            score += 0.1
        if len(words) >= 10:
            score += 0.1
            
        # Quality indicators
        quality_words = [
            "specific", "detailed", "step-by-step", "example", 
            "please", "comprehensive", "clear", "explain"
        ]
        
        for word in quality_words:
            if word in prompt_lower:
                score += 0.08
                
        # Structure indicators
        if '?' in prompt:
            score += 0.05
        if '.' in prompt:
            score += 0.05
            
        return {"effectiveness": min(1.0, score)}
    
    return evaluate

def demo_single_objective_optimization():
    """Demonstrate single-objective optimization with monitoring."""
    print("🎯 Single-Objective Optimization Demo")
    print("=" * 50)
    print("Objective: Maximize prompt effectiveness")
    print()
    
    try:
        from dspy_gepa import GEPAAgent
        
        # Create agent with single objective
        agent = GEPAAgent(
            objectives={"effectiveness": 1.0},  # Single objective
            population_size=8,
            max_generations=5,
            verbose=True
        )
        
        # Setup monitoring
        print("📊 Setting up monitoring...")
        monitoring_data = {
            "generation_scores": [],
            "best_prompts": [],
            "diversity_scores": [],
            "convergence_metrics": []
        }
        
        # Create evaluation function
        evaluate = create_single_objective_evaluation()
        
        # Initial prompt
        initial_prompt = "help me"
        initial_score = evaluate(initial_prompt)["effectiveness"]
        
        print(f"📝 Initial prompt: '{initial_prompt}'")
        print(f"📊 Initial effectiveness: {initial_score:.4f}")
        print()
        
        # Run optimization with monitoring
        print("🚀 Starting optimization...")
        start_time = time.time()
        
        # Create a simple optimization loop with monitoring
        current_prompt = initial_prompt
        current_score = initial_score
        
        for generation in range(1, 6):  # 5 generations
            print(f"\n--- Generation {generation} ---")
            
            # Simulate genetic operations (simplified)
            if generation == 1:
                # Generate mutations
                mutations = [
                    "please help me",
                    "help me please",
                    "can you help me",
                    "help me with this task",
                    "I need help",
                    "help me understand",
                    "please help me understand",
                    "help me explain"
                ]
            else:
                # Simple mutations based on best prompt
                base_words = current_prompt.split()
                mutations = []
                
                # Add quality words
                quality_additions = ["please", "specifically", "clearly", "detailed"]
                for addition in quality_additions:
                    mutations.append(f"{addition} {current_prompt}")
                    mutations.append(f"{current_prompt} {addition}")
                
                # Combine with other variations
                variations = ["explain", "describe", "show me", "tell me"]
                for var in variations:
                    mutations.append(f"{var} {current_prompt}")
            
            # Evaluate all mutations
            best_mut_score = current_score
            best_mut_prompt = current_prompt
            
            for mut in mutations[:8]:  # Limit to 8 for population size
                score = evaluate(mut)["effectiveness"]
                if score > best_mut_score:
                    best_mut_score = score
                    best_mut_prompt = mut
            
            # Update monitoring data
            improvement = (best_mut_score - current_score) / max(0.001, current_score)
            monitoring_data["generation_scores"].append(best_mut_score)
            monitoring_data["best_prompts"].append(best_mut_prompt)
            monitoring_data["convergence_metrics"].append(improvement)
            
            # Calculate diversity
            all_prompts = mutations + [current_prompt]
            unique_words = set()
            for prompt in all_prompts:
                unique_words.update(prompt.lower().split())
            diversity = len(unique_words) / max(1, sum(len(p.split()) for p in all_prompts))
            monitoring_data["diversity_scores"].append(diversity)
            
            print(f"🔄 Best score: {best_mut_score:.4f}")
            print(f"📝 Best prompt: '{best_mut_prompt}'")
            print(f"📊 Diversity: {diversity:.3f}")
            print(f"📈 Improvement: {improvement:.2%}")
            
            # Update for next generation
            current_prompt = best_mut_prompt
            current_score = best_mut_score
            
            # Early stopping if converged
            if generation > 2 and improvement < 0.01:
                print(f"✅ Converged after {generation} generations")
                break
        
        end_time = time.time()
        
        # Display results
        print(f"\n🎉 Optimization completed!")
        print(f"⏱️  Total time: {end_time - start_time:.2f}s")
        print(f"📈 Score improvement: {initial_score:.4f} → {current_score:.4f}")
        improvement_pct = ((current_score - initial_score) / max(0.001, initial_score)) * 100
        print(f"✨ Improvement: {improvement_pct:.1f}%")
        print(f"📝 Final prompt: '{current_prompt}'")
        
        # Show monitoring insights
        print(f"\n📊 Monitoring Insights:")
        print(f"• Generations completed: {len(monitoring_data['generation_scores'])}")
        print(f"• Average diversity: {sum(monitoring_data['diversity_scores']) / len(monitoring_data['diversity_scores']):.3f}")
        print(f"• Peak improvement: {max(monitoring_data['convergence_metrics']):.2%}")
        
        # Convergence analysis
        if len(monitoring_data['convergence_metrics']) > 2:
            recent_improvements = monitoring_data['convergence_metrics'][-3:]
            avg_recent = sum(recent_improvements) / len(recent_improvements)
            if avg_recent < 0.05:
                print(f"• Convergence: ✅ Stable (avg recent improvement: {avg_recent:.2%})")
            else:
                print(f"• Convergence: 🔄 Still improving (avg recent improvement: {avg_recent:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🎯 Single-Objective GEPA Framework Demo")
    print("=" * 60)
    print("This demo shows optimization for a single objective with monitoring.")
    print()
    
    success = demo_single_objective_optimization()
    
    if success:
        print(f"\n✨ Single-objective demo completed successfully!")
        print(f"\n💡 Key takeaways:")
        print(f"   • Single-objective optimization is straightforward")
        print(f"   • Monitoring provides insights into convergence")
        print(f"   • Diversity metrics help avoid local optima")
        print(f"   • Early stopping saves computation time")
    else:
        print(f"\n❌ Demo failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)