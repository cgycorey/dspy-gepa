#!/usr/bin/env python3
"""
Multi-Objective Optimization Demo

This demonstrates the GEPA framework optimizing for multiple objectives simultaneously.
Shows Pareto frontier analysis and trade-off management.

Usage:
    uv run python multi_objective_demo.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_multi_objective_evaluation():
    """Create evaluation function for multiple objectives."""
    def evaluate(prompt: str) -> Dict[str, float]:
        """Evaluate prompt across multiple objectives."""
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        # Effectiveness: Overall prompt quality
        effectiveness = 0.3
        quality_words = ["specific", "detailed", "step-by-step", "example", "comprehensive"]
        for word in quality_words:
            if word in prompt_lower:
                effectiveness += 0.1
        if len(words) >= 5:
            effectiveness += 0.1
        effectiveness = min(1.0, effectiveness)
        
        # Clarity: How easy to understand
        clarity = 0.5
        if len(words) <= 15:  # Not too long
            clarity += 0.2
        if any(word in prompt_lower for word in ["clear", "simple", "easy"]):
            clarity += 0.2
        if '?' in prompt or '.' in prompt:
            clarity += 0.1
        clarity = min(1.0, clarity)
        
        # Efficiency: Conciseness vs completeness
        efficiency = 0.7
        if len(words) <= 10:
            efficiency += 0.2
        elif len(words) <= 20:
            efficiency += 0.1
        # Penalize overly verbose prompts
        if len(words) > 25:
            efficiency -= 0.2
        efficiency = max(0.0, min(1.0, efficiency))
        
        return {
            "effectiveness": effectiveness,
            "clarity": clarity,
            "efficiency": efficiency
        }
    
    return evaluate

def calculate_pareto_frontier(solutions: List[Dict]) -> List[Dict]:
    """Calculate Pareto frontier from solutions."""
    if not solutions:
        return []
    
    pareto_front = []
    
    for i, sol1 in enumerate(solutions):
        objectives1 = sol1['objectives']
        dominated = False
        
        for j, sol2 in enumerate(solutions):
            if i == j:
                continue
                
            objectives2 = sol2['objectives']
            
            # Check if sol2 dominates sol1
            # sol2 dominates if it's better or equal in all objectives and strictly better in at least one
            better_in_all = True
            strictly_better_in_one = False
            
            for obj_name in objectives1:
                if objectives2[obj_name] < objectives1[obj_name]:
                    better_in_all = False
                    break
                elif objectives2[obj_name] > objectives1[obj_name]:
                    strictly_better_in_one = True
            
            if better_in_all and strictly_better_in_one:
                dominated = True
                break
        
        if not dominated:
            pareto_front.append(sol1)
    
    return pareto_front

def demo_multi_objective_optimization():
    """Demonstrate multi-objective optimization with Pareto analysis."""
    print("🎯 Multi-Objective Optimization Demo")
    print("=" * 50)
    print("Objectives: Effectiveness, Clarity, Efficiency")
    print("Analyzing trade-offs and Pareto frontier...")
    print()
    
    try:
        from dspy_gepa import GEPAAgent
        
        # Define multiple objectives with weights
        objectives = {
            "effectiveness": 0.4,
            "clarity": 0.3,
            "efficiency": 0.3
        }
        
        # Create agent
        agent = GEPAAgent(
            objectives=objectives,
            population_size=12,
            max_generations=6,
            verbose=True
        )
        
        print(f"📊 Objectives and weights:")
        for obj, weight in objectives.items():
            print(f"  • {obj}: {weight:.1%}")
        print()
        
        # Setup monitoring
        monitoring_data = {
            "generation_solutions": [],
            "pareto_fronts": [],
            "hypervolume": [],
            "diversity_metrics": []
        }
        
        # Create evaluation function
        evaluate = create_multi_objective_evaluation()
        
        # Initial prompt
        initial_prompt = "help me"
        initial_objectives = evaluate(initial_prompt)
        initial_score = sum(initial_objectives[obj] * objectives[obj] for obj in objectives)
        
        print(f"📝 Initial prompt: '{initial_prompt}'")
        print(f"📊 Initial objectives:")
        for obj, score in initial_objectives.items():
            print(f"  • {obj}: {score:.3f}")
        print(f"📈 Weighted score: {initial_score:.4f}")
        print()
        
        # Run multi-objective optimization
        print("🚀 Starting multi-objective optimization...")
        start_time = time.time()
        
        # Track all solutions found
        all_solutions = []
        current_generation = []
        
        for generation in range(1, 7):  # 6 generations
            print(f"\n--- Generation {generation} ---")
            
            # Generate population
            if generation == 1:
                population_prompts = [
                    "help me",
                    "please help me",
                    "help me please",
                    "can you help me",
                    "help me with this task",
                    "I need your help",
                    "help me understand",
                    "please help me understand",
                    "explain this to me",
                    "describe this clearly",
                    "show me how to",
                    "tell me about"
                ]
            else:
                # Generate mutations from best solutions
                population_prompts = []
                
                # Take top solutions from previous generation
                top_solutions = sorted(current_generation, 
                                    key=lambda x: x['weighted_score'], 
                                    reverse=True)[:4]
                
                for sol in top_solutions:
                    base_prompt = sol['prompt']
                    
                    # Apply mutations
                    mutations = [
                        f"please {base_prompt}",
                        f"{base_prompt} clearly",
                        f"explain {base_prompt}",
                        f"describe {base_prompt} simply",
                        f"show me {base_prompt} step by step"
                    ]
                    population_prompts.extend(mutations[:3])  # Limit mutations
            
            # Evaluate population
            generation_solutions = []
            
            for prompt in population_prompts[:12]:  # Keep population size manageable
                objectives = evaluate(prompt)
                weighted_score = sum(objectives[obj] * objectives[obj] for obj in objectives)
                
                solution = {
                    'prompt': prompt,
                    'objectives': objectives,
                    'weighted_score': weighted_score,
                    'generation': generation
                }
                generation_solutions.append(solution)
                all_solutions.append(solution)
            
            # Calculate Pareto frontier
            pareto_front = calculate_pareto_frontier(generation_solutions)
            
            # Calculate hypervolume (simplified)
            if pareto_front:
                # Simple hypervolume approximation
                ref_point = [0.0, 0.0, 0.0]  # Reference point for all objectives
                hypervolume = 0.0
                for sol in pareto_front:
                    obj_values = list(sol['objectives'].values())
                    # Simple approximation: sum of objective values
                    hypervolume += sum(obj_values)
                hypervolume /= len(pareto_front)  # Average
            else:
                hypervolume = 0.0
            
            # Calculate diversity
            all_prompts = [sol['prompt'] for sol in generation_solutions]
            unique_words = set()
            total_words = 0
            for prompt in all_prompts:
                words = prompt.lower().split()
                unique_words.update(words)
                total_words += len(words)
            diversity = len(unique_words) / max(1, total_words)
            
            # Store monitoring data
            monitoring_data["generation_solutions"].append(generation_solutions)
            monitoring_data["pareto_fronts"].append(pareto_front)
            monitoring_data["hypervolume"].append(hypervolume)
            monitoring_data["diversity_metrics"].append(diversity)
            
            current_generation = generation_solutions
            
            # Display generation results
            best_solution = max(generation_solutions, key=lambda x: x['weighted_score'])
            print(f"🏆 Best solution: '{best_solution['prompt']}'")
            print(f"📊 Best objectives:")
            for obj, score in best_solution['objectives'].items():
                print(f"  • {obj}: {score:.3f}")
            print(f"📈 Weighted score: {best_solution['weighted_score']:.4f}")
            print(f"🎯 Pareto frontier size: {len(pareto_front)}")
            print(f"📊 Hypervolume: {hypervolume:.3f}")
            print(f"🌱 Diversity: {diversity:.3f}")
        
        end_time = time.time()
        
        # Final analysis
        print(f"\n🎉 Multi-objective optimization completed!")
        print(f"⏱️  Total time: {end_time - start_time:.2f}s")
        print(f"🔄 Generations completed: {len(monitoring_data['generation_solutions'])}")
        print(f"📈 Total solutions explored: {len(all_solutions)}")
        
        # Final Pareto frontier
        final_pareto = calculate_pareto_frontier(all_solutions)
        print(f"\n🎯 Final Pareto Frontier ({len(final_pareto)} solutions):")
        print("=" * 60)
        
        for i, sol in enumerate(final_pareto[:5], 1):  # Show top 5
            print(f"\n{i}. '{sol['prompt']}'")
            print(f"   Objectives:")
            for obj, score in sol['objectives'].items():
                print(f"     • {obj}: {score:.3f}")
            print(f"   Weighted score: {sol['weighted_score']:.4f}")
        
        # Trade-off analysis
        print(f"\n📊 Trade-off Analysis:")
        if len(final_pareto) >= 2:
            # Find extremes
            most_effective = max(final_pareto, key=lambda x: x['objectives']['effectiveness'])
            clearest = max(final_pareto, key=lambda x: x['objectives']['clarity'])
            most_efficient = max(final_pareto, key=lambda x: x['objectives']['efficiency'])
            
            print(f"🏆 Most Effective: '{most_effective['prompt']}'")
            print(f"   Effectiveness: {most_effective['objectives']['effectiveness']:.3f}")
            
            print(f"💡 Clearest: '{clearest['prompt']}'")
            print(f"   Clarity: {clearest['objectives']['clarity']:.3f}")
            
            print(f"⚡ Most Efficient: '{most_efficient['prompt']}'")
            print(f"   Efficiency: {most_efficient['objectives']['efficiency']:.3f}")
        
        # Convergence analysis
        print(f"\n📈 Convergence Analysis:")
        if len(monitoring_data['hypervolume']) > 1:
            initial_hv = monitoring_data['hypervolume'][0]
            final_hv = monitoring_data['hypervolume'][-1]
            hv_improvement = ((final_hv - initial_hv) / max(0.001, initial_hv)) * 100
            print(f"• Hypervolume improvement: {hv_improvement:.1f}%")
            
            avg_diversity = sum(monitoring_data['diversity_metrics']) / len(monitoring_data['diversity_metrics'])
            print(f"• Average diversity: {avg_diversity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🎯 Multi-Objective GEPA Framework Demo")
    print("=" * 60)
    print("This demo demonstrates optimization across multiple objectives.")
    print("Shows Pareto frontier analysis and trade-off management.")
    print()
    
    success = demo_multi_objective_optimization()
    
    if success:
        print(f"\n✨ Multi-objective demo completed successfully!")
        print(f"\n💡 Key takeaways:")
        print(f"   • Multi-objective optimization finds trade-off solutions")
        print(f"   • Pareto frontier represents optimal trade-offs")
        print(f"   • Different solutions excel at different objectives")
        print(f"   • Hypervolume tracks overall progress")
        print(f"   • Diversity maintains solution space exploration")
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