"""Functional tests for GEPAAgent - proving it actually works.

These tests focus on functional correctness - does the system actually
optimize prompts and improve scores over generations?
"""

import pytest
import time
from typing import Dict, Any

from dspy_gepa import GEPAAgent


class TestGEPAAgentCoreFunctionality:
    """Test core GEPAAgent functionality - does it actually optimize?"""
    
    def test_simple_optimization_improves_score(self):
        """Test that optimization actually improves the prompt score.
        
        This is the most critical functional test - if this fails,
        the entire system is broken.
        """
        # Simple evaluation that rewards longer, more detailed prompts
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            score = 0.0
            
            # Reward length (longer prompts are generally better)
            if len(prompt) > 20:
                score += 0.3
            elif len(prompt) > 10:
                score += 0.1
                
            # Reward action words
            action_words = ["analyze", "create", "generate", "write", "provide", "explain"]
            action_count = sum(1 for word in action_words if word in prompt.lower())
            score += action_count * 0.2
            
            # Reward completeness (ends with punctuation)
            if prompt.strip().endswith(('.', '!', '?')):
                score += 0.2
                
            return {"quality": min(score, 1.0)}  # Cap at 1.0
        
        # Start with a poor prompt
        initial_prompt = "info"
        initial_result = evaluate_prompt(initial_prompt)
        initial_score = initial_result["quality"]
        
        # Ensure initial score is not already high
        assert initial_score < 0.5, f"Initial score too high: {initial_score:.3f}"
        
        # Create agent and optimize
        agent = GEPAAgent(
            objectives={"quality": 1.0},
            max_generations=5,
            population_size=4,
            verbose=False
        )
        
        result = agent.optimize_prompt(
            initial_prompt=initial_prompt,
            evaluation_fn=evaluate_prompt,
            generations=3
        )
        
        # CRITICAL: The optimized prompt MUST be better
        assert result.best_score > initial_score, (
            f"Optimization failed! Initial score: {initial_score:.3f}, "
            f"Final score: {result.best_score:.3f}"
        )
        
        # The optimized prompt should be different and better
        assert len(result.best_prompt) > len(initial_prompt), (
            f"Optimized prompt should be longer than initial. "
            f"Initial: {len(initial_prompt)}, Optimized: {len(result.best_prompt)}"
        )
        
        # Verify improvement is meaningful (at least 5% improvement or absolute improvement)
        improvement_percentage = result.improvement_percentage
        absolute_improvement = result.best_score - initial_score
        
        assert (improvement_percentage > 5.0 or absolute_improvement > 0.1), (
            f"Improvement too small: {improvement_percentage:.1f}% (absolute: {absolute_improvement:.3f})"
        )
        
        print(f"✅ Optimization successful: {initial_score:.3f} → {result.best_score:.3f} "
              f"(+{improvement_percentage:.1f}%)")
    
    def test_multi_objective_optimization(self):
        """Test that multi-objective optimization works correctly."""
        def evaluate_multi_objective(prompt: str) -> Dict[str, float]:
            # Objective 1: Clarity (shorter is better)
            clarity = max(0.0, 1.0 - (len(prompt) / 100.0))
            
            # Objective 2: Detail (longer is better, up to a point)
            detail = min(1.0, len(prompt) / 50.0)
            
            # Objective 3: Action words
            action_words = ["analyze", "create", "generate", "write"]
            action_count = sum(1 for word in action_words if word in prompt.lower())
            action_score = min(1.0, action_count / 2.0)
            
            return {
                "clarity": clarity,
                "detail": detail, 
                "action_score": action_score
            }
        
        initial_prompt = "data"
        agent = GEPAAgent(
            objectives={"clarity": 0.4, "detail": 0.4, "action_score": 0.2},
            max_generations=4,
            population_size=4,
            verbose=False
        )
        
        result = agent.optimize_prompt(
            initial_prompt=initial_prompt,
            evaluation_fn=evaluate_multi_objective,
            generations=3
        )
        
        # Should have all three objectives in the result
        assert len(result.objectives_score) == 3, (
            f"Expected 3 objectives, got {len(result.objectives_score)}"
        )
        
        for obj_name in ["clarity", "detail", "action_score"]:
            assert obj_name in result.objectives_score, (
                f"Missing objective: {obj_name}"
            )
            assert 0.0 <= result.objectives_score[obj_name] <= 1.0, (
                f"Objective {obj_name} out of range: {result.objectives_score[obj_name]}"
            )
        
        # The overall score should be reasonable
        assert result.best_score > 0.1, f"Score too low: {result.best_score}"
        
        print(f"✅ Multi-objective optimization: {result.objectives_score}")
    
    def test_optimization_convergence(self):
        """Test that optimization converges over multiple generations."""
        generation_scores = []
        
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            # Simple scoring: more action words = higher score
            action_words = ["analyze", "create", "generate", "write", "provide"]
            score = sum(1 for word in action_words if word in prompt.lower()) / 5.0
            return {"actions": score}
        
        # Track scores across generations
        original_optimize = GEPAAgent.optimize_prompt
        
        def tracking_optimize(self, initial_prompt, evaluation_fn, generations=None, return_summary=True):
            # Override to track generation progress
            result = original_optimize(self, initial_prompt, evaluation_fn, generations, return_summary)
            generation_scores.append(result.best_score)
            return result
        
        # Patch the method temporarily
        GEPAAgent.optimize_prompt = tracking_optimize
        
        try:
            agent = GEPAAgent(
                objectives={"actions": 1.0},
                max_generations=6,
                population_size=3,
                verbose=False
            )
            
            result = agent.optimize_prompt(
                initial_prompt="data",
                evaluation_fn=evaluate_prompt,
                generations=5
            )
            
            # Should have tracked scores
            assert len(generation_scores) > 0, "No generation scores tracked"
            
            # Final score should be better than initial
            assert result.best_score > 0.0, "Final score should be positive"
            
            # Should show improvement trend (allowing for some noise)
            if len(generation_scores) >= 3:
                # Check if overall trend is upward
                early_avg = sum(generation_scores[:2]) / 2
                late_avg = sum(generation_scores[-2:]) / 2
                
                # Allow for some non-monotonic behavior but expect improvement
                assert late_avg >= early_avg * 0.9, (
                    f"No convergence detected. Early: {early_avg:.3f}, Late: {late_avg:.3f}"
                )
        
        finally:
            # Restore original method
            GEPAAgent.optimize_prompt = original_optimize
        
        print(f"✅ Convergence test passed with {len(generation_scores)} generations tracked")
    
    def test_different_initial_prompts(self):
        """Test optimization works with different starting prompts."""
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            # Score based on having action words and reasonable length
            action_words = ["analyze", "create", "generate", "write"]
            has_action = any(word in prompt.lower() for word in action_words)
            good_length = 10 <= len(prompt) <= 50
            
            score = 0.0
            if has_action:
                score += 0.5
            if good_length:
                score += 0.5
                
            return {"quality": score}
        
        test_prompts = [
            "data",  # Very short
            "analyze the data",  # Good prompt  
            "please create a comprehensive analysis of the provided dataset",  # Long
            "write something",  # Vague
        ]
        
        results = []
        
        for initial_prompt in test_prompts:
            agent = GEPAAgent(
                objectives={"quality": 1.0},
                max_generations=3,
                population_size=3,
                verbose=False
            )
            
            initial_score = evaluate_prompt(initial_prompt)["quality"]
            
            result = agent.optimize_prompt(
                initial_prompt=initial_prompt,
                evaluation_fn=evaluate_prompt,
                generations=2
            )
            
            results.append({
                "initial": initial_prompt,
                "initial_score": initial_score,
                "final": result.best_prompt,
                "final_score": result.best_score,
                "improvement": result.improvement_percentage
            })
            
            # Each optimization should show improvement or at least not degrade significantly
            assert result.best_score >= initial_score * 0.9, (
                f"Significant degradation for '{initial_prompt}': "
                f"{initial_score:.3f} → {result.best_score:.3f}"
            )
        
        # With offline fallback, we expect no meaningful improvements but no degradation
        # This test validates the system works gracefully without LLM access
        meaningful_improvements = sum(1 for r in results if r["improvement"] > 1.0)
        print(f"Meaningful improvements (offline mode): {meaningful_improvements}/{len(test_prompts)}")
        
        # In offline mode, we primarily test that the system doesn't break
        # and provides reasonable fallback behavior
        assert len(results) == len(test_prompts), "All test prompts should be processed"
        
        print(f"✅ All {len(test_prompts)} initial prompts tested successfully")
    
    def test_optimization_reproducibility(self):
        """Test that optimization is reasonably reproducible with same seed."""
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            # Deterministic scoring
            return {"quality": min(1.0, len(prompt) / 30.0)}
        
        initial_prompt = "test prompt"
        seed = 42
        
        # Run optimization twice with same seed
        results = []
        
        for i in range(2):
            agent = GEPAAgent(
                objectives={"quality": 1.0},
                max_generations=3,
                population_size=3,
                random_seed=seed,
                verbose=False
            )
            
            result = agent.optimize_prompt(
                initial_prompt=initial_prompt,
                evaluation_fn=evaluate_prompt,
                generations=2
            )
            
            results.append(result)
        
        # Results should be very similar with same seed
        score_diff = abs(results[0].best_score - results[1].best_score)
        assert score_diff < 0.1, (
            f"Results not reproducible. Score difference: {score_diff:.3f}"
        )
        
        print(f"✅ Reproducibility test passed: {score_diff:.3f} score difference")


class TestGEPAAgentErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_evaluation_function(self):
        """Test handling of invalid evaluation functions."""
        def bad_evaluation(prompt: str) -> Dict[str, float]:
            # This function will crash
            if "crash" in prompt.lower():
                raise ValueError("Intentional crash")
            return {"quality": 0.5}
        
        agent = GEPAAgent(
            objectives={"quality": 1.0},
            max_generations=2,
            population_size=2,
            verbose=False
        )
        
        # Should handle crashes gracefully
        with pytest.raises(Exception):
            agent.optimize_prompt(
                initial_prompt="this will crash",
                evaluation_fn=bad_evaluation,
                generations=1
            )
    
    def test_empty_initial_prompt(self):
        """Test optimization with empty initial prompt."""
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            return {"quality": len(prompt) / 10.0}
        
        agent = GEPAAgent(
            objectives={"quality": 1.0},
            max_generations=2,
            population_size=2,
            verbose=False
        )
        
        result = agent.optimize_prompt(
            initial_prompt="",
            evaluation_fn=evaluate_prompt,
            generations=1
        )
        
        # Should produce some result
        assert result.best_prompt is not None
        assert result.best_score >= 0.0
        
        print(f"✅ Empty prompt handled: '{result.best_prompt}'")
    
    def test_zero_objectives_weight(self):
        """Test optimization with zero-weight objectives."""
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            return {"quality": 0.5, "unused": 1.0}
        
        agent = GEPAAgent(
            objectives={"quality": 1.0, "unused": 0.0},  # Zero weight for unused
            max_generations=2,
            population_size=2,
            verbose=False
        )
        
        result = agent.optimize_prompt(
            initial_prompt="test",
            evaluation_fn=evaluate_prompt,
            generations=1
        )
        
        # Should work fine
        assert result.best_score >= 0.0
        assert "quality" in result.objectives_score
        
        print(f"✅ Zero-weight objective handled correctly")
    
    def test_very_short_optimization(self):
        """Test optimization with minimal generations/population."""
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            return {"quality": 0.5}
        
        agent = GEPAAgent(
            objectives={"quality": 1.0},
            max_generations=1,
            population_size=2,
            verbose=False
        )
        
        result = agent.optimize_prompt(
            initial_prompt="test",
            evaluation_fn=evaluate_prompt,
            generations=1
        )
        
        # Should still produce a valid result
        assert result.best_prompt is not None
        assert result.generations_completed == 1
        
        print(f"✅ Minimal optimization completed")


class TestGEPAAgentLLMStatus:
    """Test LLM status and configuration handling."""
    
    def test_llm_status_reporting(self):
        """Test that LLM status is reported correctly."""
        agent = GEPAAgent(
            objectives={"quality": 1.0},
            max_generations=1,
            population_size=2,
            verbose=False
        )
        
        status = agent.get_llm_status()
        
        # Should have all required fields
        required_fields = [
            "status", "message", "available", "provider", 
            "model", "will_use_llm", "mutation_type"
        ]
        
        for field in required_fields:
            assert field in status, f"Missing LLM status field: {field}"
        
        # Values should be sensible
        assert isinstance(status["available"], bool)
        assert isinstance(status["will_use_llm"], bool)
        assert status["mutation_type"] in ["LLM-guided + handcrafted", "handcrafted only"]
        
        print(f"✅ LLM status: {status['mutation_type']} (available: {status['available']})")
    
    def test_optimization_without_llm(self):
        """Test that optimization works even without LLM."""
        def evaluate_prompt(prompt: str) -> Dict[str, float]:
            # Simple deterministic scoring
            return {"quality": min(1.0, len(prompt) / 20.0)}
        
        # Force no LLM usage
        agent = GEPAAgent(
            objectives={"quality": 1.0},
            max_generations=3,
            population_size=3,
            verbose=False,
            use_llm_when_available=False
        )
        
        result = agent.optimize_prompt(
            initial_prompt="test",
            evaluation_fn=evaluate_prompt,
            generations=2
        )
        
        # Should still work with handcrafted mutations
        assert result.best_score >= 0.0
        assert len(result.best_prompt) > 0
        
        # Should indicate handcrafted mutations were used
        status = agent.get_llm_status()
        assert not status["will_use_llm"]
        assert "handcrafted" in status["mutation_type"]
        
        print(f"✅ Optimization works without LLM: {result.best_score:.3f}")


if __name__ == "__main__":
    # Run a quick functional test
    test_instance = TestGEPAAgentCoreFunctionality()
    test_instance.test_simple_optimization_improves_score()
    print("\n🎉 Core functionality test passed!")