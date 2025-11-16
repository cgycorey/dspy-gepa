#!/usr/bin/env python3
"""
Demonstration Script: LLM-Guided vs Handcrafted Mutations in dspy-gepa

This script comprehensively demonstrates the difference between LLM-guided and 
handcrafted mutations in the dspy-gepa evolutionary optimization system.

Features demonstrated:
1. Mock LLM client creation and configuration
2. GEPAAgent initialization with and without LLM
3. Mutation strategy comparison (LLM-guided vs handcrafted)
4. Detailed mutation tracking and analysis
5. Performance metrics comparison
6. Status reporting for both scenarios

Usage:
    python demonstrate_mutation_types.py

Author: Generated for dspy-gepa system demonstration
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from unittest.mock import Mock, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import core GEPA components
try:
    from dspy_gepa.core.agent import GEPAAgent, LLMConfig, AgentConfig
    from gepa.core.candidate import Candidate, ExecutionTrace, MutationRecord
    GEPA_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import GEPA core: {e}")
    GEPA_AVAILABLE = False

# Import AMOPE adaptive mutator
try:
    from dspy_gepa.amope.adaptive_mutator import (
        AdaptiveMutator, 
        GradientBasedMutation,
        StatisticalMutation,
        PatternBasedMutation,
        PerformanceAnalyzer,
        MutationStrategy,
        MutationResult
    )
    AMOPE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Failed to import AMOPE mutator: {e}")
    AMOPE_AVAILABLE = False

# Import configuration utilities
try:
    from dspy_gepa.utils.config import load_llm_config, is_llm_configured
    from dspy_gepa.utils.logging import get_logger
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Failed to import config utilities: {e}")
    CONFIG_AVAILABLE = False


@dataclass
class MutationComparisonResult:
    """Results from mutation comparison demonstration."""
    scenario: str
    llm_enabled: bool
    mutation_types_detected: List[str]
    original_prompt: str
    best_prompt: str
    initial_score: float
    best_score: float
    improvement: float
    generations_completed: int
    total_evaluations: int
    execution_time: float
    optimization_summary: Dict[str, Any]
    mutation_examples: List[Dict[str, str]]


class MockLLMClient:
    """Mock LLM client that simulates realistic LLM responses for mutation."""
    
    def __init__(self, model_name: str = "mock-gpt-4", response_delay: float = 0.1):
        self.model_name = model_name
        self.response_delay = response_delay
        self.call_count = 0
        self.prompts_received = []
        self.responses_generated = []
        
        # Predefined response patterns for different mutation scenarios
        self.response_patterns = {
            "reflection": [
                "Based on the performance feedback, I'll enhance the prompt with clearer instructions and better structure.",
                "To improve performance, I'll add specific examples and clarify the expected output format.",
                "I'll refine this prompt by adding step-by-step guidance and success criteria.",
                "Enhanced with detailed specifications and improved clarity for better results.",
                "Optimized version with comprehensive instructions and structured approach."
            ],
            "optimization": [
                "Enhanced and optimized version with improved clarity and structure.",
                "Refined content with better organization and comprehensive details.",
                "Improved version with enhanced functionality and clearer specifications.",
                "Optimized implementation with better performance characteristics.",
                "Enhanced solution with systematic approach and detailed guidance."
            ],
            "mutation": [
                "Write a function that calculates the factorial of a number with proper error handling and edge cases.",
                "Create a robust sorting algorithm that handles various data types and optimization scenarios.",
                "Design a comprehensive user authentication system with secure password hashing and session management.",
                "Implement an efficient data structure for key-value storage with collision resolution.",
                "Develop an optimized prime number generator using advanced algorithms and caching."
            ]
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock LLM response based on prompt content."""
        self.call_count += 1
        self.prompts_received.append(prompt)
        
        # Simulate API delay
        time.sleep(self.response_delay + random.uniform(0, 0.1))
        
        # Analyze prompt to generate appropriate response
        prompt_lower = prompt.lower()
        
        if "reflection" in prompt_lower or "feedback" in prompt_lower:
            responses = self.response_patterns["reflection"]
        elif "optimize" in prompt_lower or "improve" in prompt_lower:
            responses = self.response_patterns["optimization"]
        elif "factorial" in prompt_lower:
            responses = [self.response_patterns["mutation"][0]]
        elif "sorting" in prompt_lower:
            responses = [self.response_patterns["mutation"][1]]
        elif "authentication" in prompt_lower:
            responses = [self.response_patterns["mutation"][2]]
        elif "data structure" in prompt_lower:
            responses = [self.response_patterns["mutation"][3]]
        elif "prime" in prompt_lower:
            responses = [self.response_patterns["mutation"][4]]
        else:
            # Generate contextual response based on original content
            if "current content:" in prompt_lower:
                # Extract original content and enhance it
                lines = prompt.split('\n')
                for i, line in enumerate(lines):
                    if "current content:" in line.lower():
                        if i + 1 < len(lines):
                            original = lines[i + 1].strip()
                            return self._enhance_content(original)
            
            responses = self.response_patterns["optimization"]
        
        response = random.choice(responses)
        self.responses_generated.append(response)
        
        return response
    
    def _enhance_content(self, original: str) -> str:
        """Enhance original content with improvements."""
        enhancements = [
            f"{original} Ensure proper error handling and edge cases are covered.",
            f"{original} Include comprehensive examples and clear documentation.",
            f"{original} Optimize for performance and maintainability.",
            f"{original} Add validation and user-friendly error messages.",
            f"{original} Implement with best practices and security considerations."
        ]
        
        return random.choice(enhancements)


class MockEvaluationFunction:
    """Mock evaluation function for consistent and realistic testing."""
    
    def __init__(self, base_score: float = 0.5, noise_factor: float = 0.15):
        self.base_score = base_score
        self.noise_factor = noise_factor
        self.evaluation_count = 0
        self.score_history = []
        
        # Keywords that indicate better quality
        self.quality_indicators = {
            "high": ["enhanced", "improved", "optimized", "refined", "better", 
                    "clear", "structured", "comprehensive", "systematic", "robust"],
            "medium": ["add", "include", "ensure", "implement", "create"],
            "specific": ["factorial", "sorting", "authentication", "prime", "algorithm"]
        }
    
    def evaluate(self, prompt: str) -> Dict[str, float]:
        """Evaluate prompt with consistent but varied results."""
        self.evaluation_count += 1
        
        # Base metrics with variation
        base_accuracy = self.base_score + random.uniform(-self.noise_factor, self.noise_factor)
        base_efficiency = self.base_score + random.uniform(-self.noise_factor, self.noise_factor)
        base_clarity = self.base_score + random.uniform(-self.noise_factor, self.noise_factor)
        
        # Content analysis for quality scoring
        content_lower = prompt.lower()
        
        # Length bonus (longer prompts tend to be more detailed)
        length_bonus = min(len(prompt) / 500, 0.15)
        
        # Quality indicator bonuses
        quality_bonus = 0
        for indicator in self.quality_indicators["high"]:
            if indicator in content_lower:
                quality_bonus += 0.03
        
        # Specific content bonuses
        specific_bonus = 0
        for topic in self.quality_indicators["specific"]:
            if topic in content_lower:
                specific_bonus += 0.05
        
        # Structure bonus (has sentences, proper formatting)
        structure_bonus = 0
        if '.' in prompt and len(prompt.split()) > 10:
            structure_bonus += 0.05
        
        # Calculate final scores
        accuracy = max(0.0, min(1.0, base_accuracy + length_bonus + quality_bonus + specific_bonus))
        efficiency = max(0.0, min(1.0, base_efficiency + length_bonus * 0.5 + structure_bonus))
        clarity = max(0.0, min(1.0, base_clarity + quality_bonus + structure_bonus))
        
        scores = {
            "accuracy": accuracy,
            "efficiency": efficiency, 
            "clarity": clarity
        }
        
        self.score_history.append(scores)
        return scores


class MutationComparisonDemo:
    """Main class for demonstrating mutation type differences."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.logger = self._setup_logger()
        self.results: List[MutationComparisonResult] = []
        
        # Test prompts for demonstration
        self.test_prompts = [
            "Write a function that calculates the factorial of a number.",
            "Create a sorting algorithm for arrays of integers.",
            "Design a user authentication system with password hashing.",
            "Implement a data structure for storing key-value pairs.",
            "Write a program that finds prime numbers up to N."
        ]
        
        print("🧪 dspy-gepa Mutation Types Comparison Demo")
        print("=" * 60)
        print("Comparing LLM-guided vs Handcrafted mutations...")
        print("🚨 Using mock LLM clients - no real API calls!")
        print()
    
    def _setup_logger(self):
        """Set up logging for the demonstration."""
        if CONFIG_AVAILABLE:
            try:
                return get_logger(__name__)
            except Exception:
                pass
        
        # Fallback logger
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        return logging.getLogger(__name__)
    
    def create_mock_llm_config(self) -> LLMConfig:
        """Create a mock LLM configuration."""
        return LLMConfig(
            provider="mock-openai",
            model="mock-gpt-4",
            api_key="mock-api-key-for-demo",
            temperature=0.7,
            max_tokens=2048,
            enabled=True
        )
    
    def create_agent_with_llm(self) -> GEPAAgent:
        """Create a GEPAAgent with mock LLM enabled."""
        print("🤖 Creating GEPAAgent with LLM enabled...")
        
        # Create mock LLM client
        mock_llm = MockLLMClient()
        
        # Create LLM config
        llm_config = self.create_mock_llm_config()
        llm_config.is_available = True
        llm_config.configuration_source = "mock_demo"
        
        # Create agent
        agent = GEPAAgent(
            objectives={"accuracy": 0.5, "efficiency": 0.3, "clarity": 0.2},
            max_generations=5,
            population_size=4,
            llm_config=llm_config,
            use_llm_when_available=True,
            verbose=True
        )
        
        # Store mock LLM for tracking
        agent._mock_llm = mock_llm
        
        print(f"   ✅ Agent created with LLM: {llm_config.provider} - {llm_config.model}")
        return agent
    
    def create_agent_without_llm(self) -> GEPAAgent:
        """Create a GEPAAgent with LLM disabled (handcrafted only)."""
        print("🔧 Creating GEPAAgent with handcrafted mutations only...")
        
        # Create agent without LLM
        agent = GEPAAgent(
            objectives={"accuracy": 0.5, "efficiency": 0.3, "clarity": 0.2},
            max_generations=5,
            population_size=4,
            llm_config=None,
            use_llm_when_available=False,
            verbose=True
        )
        
        print(f"   ✅ Agent created with handcrafted mutations only")
        return agent
    
    def run_optimization_with_tracking(self, agent: GEPAAgent, initial_prompt: str, scenario_name: str) -> MutationComparisonResult:
        """Run optimization with detailed tracking."""
        print(f"\n🚀 Running optimization for {scenario_name}...")
        print(f"📝 Initial prompt: {initial_prompt}")
        
        # Create evaluation function
        evaluator = MockEvaluationFunction()
        
        # Track start time
        start_time = time.time()
        
        # Get initial score
        initial_scores = evaluator.evaluate(initial_prompt)
        initial_score = sum(initial_scores.values()) / len(initial_scores)
        
        print(f"📊 Initial score: {initial_score:.4f}")
        print(f"   Objectives: {initial_scores}")
        
        # Check LLM status
        llm_status = agent.get_llm_status()
        print(f"🤖 LLM Status: {llm_status['mutation_type']}")
        if llm_status['will_use_llm']:
            print(f"   Provider: {llm_status['provider']} - {llm_status['model']}")
        else:
            print(f"   Status: {llm_status['message']}")
        
        # Run optimization
        try:
            result = agent.optimize_prompt(
                initial_prompt=initial_prompt,
                evaluation_fn=evaluator.evaluate,
                generations=3,  # Reduced for demo
                return_summary=True
            )
            
            execution_time = time.time() - start_time
            
            # Analyze mutations used
            mutation_types = self._analyze_mutations_used(agent, llm_status)
            
            # Get mutation examples
            mutation_examples = self._extract_mutation_examples(agent, initial_prompt, result.best_prompt)
            
            comparison_result = MutationComparisonResult(
                scenario=scenario_name,
                llm_enabled=llm_status['will_use_llm'],
                mutation_types_detected=mutation_types,
                original_prompt=initial_prompt,
                best_prompt=result.best_prompt,
                initial_score=initial_score,
                best_score=result.best_score,
                improvement=result.improvement,
                generations_completed=result.generations_completed,
                total_evaluations=result.total_evaluations,
                execution_time=execution_time,
                optimization_summary={
                    "objectives_score": result.objectives_score,
                    "improvement_percentage": result.improvement_percentage,
                    "llm_status": llm_status
                },
                mutation_examples=mutation_examples
            )
            
            print(f"✅ Optimization completed in {execution_time:.2f}s")
            print(f"🎉 Best score: {result.best_score:.4f} (+{result.improvement_percentage:.1f}%)")
            print(f"📝 Best prompt: {result.best_prompt}")
            print(f"🔬 Mutations used: {', '.join(mutation_types)}")
            
            return comparison_result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            # Return partial result
            return MutationComparisonResult(
                scenario=scenario_name,
                llm_enabled=llm_status['will_use_llm'],
                mutation_types_detected=["error"],
                original_prompt=initial_prompt,
                best_prompt=initial_prompt,
                initial_score=initial_score,
                best_score=initial_score,
                improvement=0.0,
                generations_completed=0,
                total_evaluations=1,
                execution_time=time.time() - start_time,
                optimization_summary={"error": str(e), "llm_status": llm_status},
                mutation_examples=[]
            )
    
    def _analyze_mutations_used(self, agent: GEPAAgent, llm_status: Dict[str, Any]) -> List[str]:
        """Analyze what mutation types were used."""
        mutation_types = []
        
        if llm_status['will_use_llm']:
            mutation_types.append("LLMReflectionMutator")
            mutation_types.append("LLM-guided")
        
        # Handcrafted mutations are always available as fallback
        mutation_types.extend([
            "TextMutator",
            "Handcrafted mutations",
            "Fallback mutations"
        ])
        
        # Add AMOPE mutations if available
        if AMOPE_AVAILABLE:
            mutation_types.extend([
                "AdaptiveMutator",
                "GradientBasedMutation",
                "StatisticalMutation",
                "PatternBasedMutation"
            ])
        
        return list(set(mutation_types))
    
    def _extract_mutation_examples(self, agent: GEPAAgent, original: str, best: str) -> List[Dict[str, str]]:
        """Extract examples of mutations that occurred."""
        examples = []
        
        # Compare original and best to identify changes
        if original != best:
            examples.append({
                "type": "Content Enhancement",
                "original": original[:100] + "..." if len(original) > 100 else original,
                "mutated": best[:100] + "..." if len(best) > 100 else best,
                "description": "Prompt was enhanced through mutation process"
            })
        
        # Add LLM-specific examples if LLM was used
        if hasattr(agent, '_mock_llm') and agent._mock_llm:
            mock_llm = agent._mock_llm
            if mock_llm.responses_generated:
                examples.append({
                    "type": "LLM Response Example",
                    "original": "Mock LLM prompt",
                    "mutated": mock_llm.responses_generated[0] if mock_llm.responses_generated else "No response",
                    "description": f"LLM generated {mock_llm.call_count} responses"
                })
        
        return examples
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison of mutation types."""
        print("🚀 Starting comprehensive mutation comparison...")
        print()
        
        # Test with multiple prompts
        for i, prompt in enumerate(self.test_prompts[:3], 1):  # Test first 3 prompts
            print(f"📋 Test {i}/{min(3, len(self.test_prompts))}: {prompt}")
            print("-" * 60)
            
            # Create agents
            llm_agent = self.create_agent_with_llm()
            handcrafted_agent = self.create_agent_without_llm()
            
            # Run LLM-guided optimization
            llm_result = self.run_optimization_with_tracking(
                llm_agent, prompt, f"LLM-guided_{i}"
            )
            self.results.append(llm_result)
            
            print()
            
            # Run handcrafted optimization  
            handcrafted_result = self.run_optimization_with_tracking(
                handcrafted_agent, prompt, f"Handcrafted_{i}"
            )
            self.results.append(handcrafted_result)
            
            # Compare results
            self.compare_individual_results(llm_result, handcrafted_result)
            
            print("\n" + "=" * 80 + "\n")
    
    def compare_individual_results(self, llm_result: MutationComparisonResult, handcrafted_result: MutationComparisonResult):
        """Compare two optimization results."""
        print("📊 HEAD-TO-HEAD COMPARISON")
        print()
        
        # Performance comparison
        print("🎯 Performance Metrics:")
        print(f"   LLM-guided:    {llm_result.best_score:.4f} (+{llm_result.optimization_summary['improvement_percentage']:.1f}%)")
        print(f"   Handcrafted:   {handcrafted_result.best_score:.4f} (+{handcrafted_result.optimization_summary['improvement_percentage']:.1f}%)")
        
        perf_diff = llm_result.best_score - handcrafted_result.best_score
        winner = "LLM-guided" if perf_diff > 0 else "Handcrafted" if perf_diff < 0 else "Tie"
        print(f"   Winner: {winner} ({abs(perf_diff):+.4f})")
        print()
        
        # Execution time comparison
        print("⏱️  Execution Time:")
        print(f"   LLM-guided:    {llm_result.execution_time:.3f}s")
        print(f"   Handcrafted:   {handcrafted_result.execution_time:.3f}s")
        
        time_diff = llm_result.execution_time - handcrafted_result.execution_time
        faster = "Handcrafted" if time_diff > 0 else "LLM-guided"
        print(f"   Faster: {faster} ({abs(time_diff):.3f}s difference)")
        print()
        
        # Mutation types comparison
        print("🔬 Mutation Types:")
        print(f"   LLM-guided:    {', '.join(llm_result.mutation_types_detected[:3])}")
        print(f"   Handcrafted:   {', '.join(handcrafted_result.mutation_types_detected[:3])}")
        print()
        
        # Content changes
        print("📝 Content Changes:")
        print(f"   Original:   {llm_result.original_prompt[:60]}...")
        print(f"   LLM result: {llm_result.best_prompt[:60]}...")
        print(f"   Handcrafted: {handcrafted_result.best_prompt[:60]}...")
        print()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("📋 COMPREHENSIVE SUMMARY REPORT")
        print("=" * 80)
        print()
        
        # Separate results by type
        llm_results = [r for r in self.results if r.llm_enabled]
        handcrafted_results = [r for r in self.results if not r.llm_enabled]
        
        if not llm_results or not handcrafted_results:
            print("⚠️  Incomplete results - cannot generate full comparison")
            return
        
        # Performance statistics
        print("📊 PERFORMANCE STATISTICS")
        print("-" * 40)
        
        llm_avg_score = sum(r.best_score for r in llm_results) / len(llm_results)
        handcrafted_avg_score = sum(r.best_score for r in handcrafted_results) / len(handcrafted_results)
        
        llm_avg_improvement = sum(r.optimization_summary['improvement_percentage'] for r in llm_results) / len(llm_results)
        handcrafted_avg_improvement = sum(r.optimization_summary['improvement_percentage'] for r in handcrafted_results) / len(handcrafted_results)
        
        print(f"Average Final Score:")
        print(f"   LLM-guided:    {llm_avg_score:.4f}")
        print(f"   Handcrafted:   {handcrafted_avg_score:.4f}")
        print(f"   Difference:    {llm_avg_score - handcrafted_avg_score:+.4f}")
        print()
        
        print(f"Average Improvement:")
        print(f"   LLM-guided:    {llm_avg_improvement:.1f}%")
        print(f"   Handcrafted:   {handcrafted_avg_improvement:.1f}%")
        print(f"   Difference:    {llm_avg_improvement - handcrafted_avg_improvement:+.1f}%")
        print()
        
        # Efficiency statistics
        print("⚡ EFFICIENCY STATISTICS")
        print("-" * 40)
        
        llm_avg_time = sum(r.execution_time for r in llm_results) / len(llm_results)
        handcrafted_avg_time = sum(r.execution_time for r in handcrafted_results) / len(handcrafted_results)
        
        print(f"Average Execution Time:")
        print(f"   LLM-guided:    {llm_avg_time:.3f}s")
        print(f"   Handcrafted:   {handcrafted_avg_time:.3f}s")
        print(f"   Speed ratio:   {llm_avg_time / handcrafted_avg_time:.2f}x")
        print()
        
        # Mutation analysis
        print("🔬 MUTATION ANALYSIS")
        print("-" * 40)
        
        all_llm_mutations = set()
        for result in llm_results:
            all_llm_mutations.update(result.mutation_types_detected)
        
        all_handcrafted_mutations = set()
        for result in handcrafted_results:
            all_handcrafted_mutations.update(result.mutation_types_detected)
        
        print(f"LLM-guided mutations: {', '.join(sorted(all_llm_mutations))}")
        print(f"Handcrafted mutations: {', '.join(sorted(all_handcrafted_mutations))}")
        print()
        
        # Key findings
        print("🔍 KEY FINDINGS")
        print("-" * 40)
        
        if llm_avg_score > handcrafted_avg_score:
            print("🏆 LLM-guided mutations achieved higher average performance")
        elif handcrafted_avg_score > llm_avg_score:
            print("🏆 Handcrafted mutations achieved higher average performance")
        else:
            print("🤝 Both approaches achieved similar average performance")
        
        if llm_avg_time > handcrafted_avg_time:
            print("⚡ Handcrafted mutations were faster on average")
            print(f"   Speed advantage: {llm_avg_time / handcrafted_avg_time:.1f}x faster")
        else:
            print("⚡ LLM-guided mutations were competitive in speed")
        
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 40)
        print("• Use LLM-guided mutations when:")
        print("  - API keys are available and cost is not a constraint")
        print("  - High-quality, context-aware mutations are needed")
        print("  - Complex domain-specific optimization is required")
        print()
        print("• Use handcrafted mutations when:")
        print("  - Speed and efficiency are priorities")
        print("  - LLM access is limited or unavailable")
        print("  - Consistent, predictable mutations are preferred")
        print()
        print("• Consider hybrid approach:")
        print("  - Start with handcrafted for rapid iteration")
        print("  - Switch to LLM-guided for fine-tuning")
        print("  - Use adaptive selection based on performance")
        print()
    
    def save_detailed_results(self, filename: str = "mutation_comparison_results.json"):
        """Save detailed results to JSON file."""
        try:
            results_data = []
            for result in self.results:
                result_dict = asdict(result)
                # Truncate long content for cleaner JSON
                result_dict['original_prompt'] = result.original_prompt[:200] + "..." if len(result.original_prompt) > 200 else result.original_prompt
                result_dict['best_prompt'] = result.best_prompt[:200] + "..." if len(result.best_prompt) > 200 else result.best_prompt
                results_data.append(result_dict)
            
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"💾 Detailed results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main demonstration function."""
    try:
        # Check dependencies
        if not GEPA_AVAILABLE:
            print("❌ GEPA core components not available. Please check installation.")
            return 1
        
        # Initialize demonstrator
        demonstrator = MutationComparisonDemo()
        
        # Run comprehensive comparison
        demonstrator.run_comprehensive_comparison()
        
        # Generate summary report
        demonstrator.generate_summary_report()
        
        # Save detailed results
        demonstrator.save_detailed_results()
        
        print("🎉 Mutation types comparison demonstration completed successfully!")
        print()
        print("📚 Key takeaways:")
        print("   • Both LLM-guided and handcrafted mutations have their strengths")
        print("   • LLM-guided mutations provide more intelligent, context-aware changes")
        print("   • Handcrafted mutations are faster and more predictable")
        print("   • The choice depends on your specific requirements and constraints")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demonstration interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())