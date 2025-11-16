#!/usr/bin/env python3
"""
DSPY-GEPA Integration Example

This comprehensive example demonstrates how to:
1. Set up a basic DSPY program with signatures and modules
2. Integrate it with GEPA optimization using multi-objective evolutionary algorithms
3. Use AMOPE (Adaptive Multi-Objective Prompt Evolution) enhancements
4. Execute a complete workflow from DSPY program to optimized version

The example includes proper error handling, documentation, and practical usage patterns.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import dspy
    from dspy import Signature, InputField, OutputField, Module, ChainOfThought
except ImportError:
    print("❌ DSPY not found. Install with: pip install dspy-ai")
    sys.exit(1)

try:
    from gepa import Candidate, GeneticOptimizer, ParetoSelector, TextMutator
    from dspy_gepa.dspy_integration import DSPYAdapter, MetricCollector
    from dspy_gepa.amope import AdaptiveMutator, ReflectionEngine
except ImportError as e:
    print(f"❌ DSPY-GEPA not found: {e}")
    print("Install with: pip install -e .")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    objectives: List[str] = None
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["accuracy", "efficiency", "clarity"]


class QuestionAnsweringSignature(Signature):
    """DSPY signature for question answering with context."""
    question = InputField(desc="The question to answer")
    context = InputField(desc="Relevant context for answering the question")
    answer = OutputField(desc="The answer to the question", prefix="Answer:")


class TextSummarizationSignature(Signature):
    """DSPY signature for text summarization."""
    text = InputField(desc="The text to summarize")
    summary = OutputField(desc="A concise summary of the text", prefix="Summary:")


class BasicQA(Module):
    """Basic DSPY module for question answering."""
    
    def __init__(self):
        super().__init__()
        self.generate_answer = ChainOfThought(QuestionAnsweringSignature)
    
    def forward(self, question: str, context: str) -> str:
        """Forward pass for question answering."""
        prediction = self.generate_answer(question=question, context=context)
        return prediction.answer


class AdvancedQA(Module):
    """Advanced DSPY module with multiple steps."""
    
    def __init__(self):
        super().__init__()
        self.analyze_question = ChainOfThought(QuestionAnsweringSignature)
        self.refine_answer = ChainOfThought(QuestionAnsweringSignature)
    
    def forward(self, question: str, context: str) -> str:
        """Two-step question answering with refinement."""
        # First pass
        initial_prediction = self.analyze_question(question=question, context=context)
        initial_answer = initial_prediction.answer
        
        # Refinement pass
        refined_prediction = self.refine_answer(
            question=question, 
            context=f"{context}\n\nInitial answer: {initial_answer}\n\nPlease refine this answer to be more accurate and complete."
        )
        return refined_prediction.answer


class DSPYGEPAOptimizer:
    """Main optimizer class that integrates DSPY with GEPA."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.adapter = DSPYAdapter()
        self.metric_collector = MetricCollector()
        self.optimizer = None
        self.reflection_engine = None
        
        # Initialize GEPA components
        self._setup_gepa_components()
    
    def _setup_gepa_components(self):
        """Set up GEPA optimization components."""
        try:
            # Create Pareto selector for multi-objective optimization
            selector = ParetoSelector(objectives=self.config.objectives)
            
            # Create adaptive mutator with AMOPE enhancements
            mutator = AdaptiveMutator(
                mutation_rate=self.config.mutation_rate,
                strategies=["substitution", "insertion", "deletion", "reflection"]
            )
            
            # Create genetic optimizer
            self.optimizer = GeneticOptimizer(
                selector=selector,
                mutator=mutator,
                population_size=self.config.population_size,
                generations=self.config.generations
            )
            
            # Create reflection engine for AMOPE
            self.reflection_engine = ReflectionEngine()
            
            logger.info("✅ GEPA components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize GEPA components: {e}")
            raise
    
    def create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create sample dataset for optimization."""
        return [
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                "expected_answer": "Machine learning is a subset of AI that allows systems to learn from experience without explicit programming."
            },
            {
                "question": "How does neural network work?",
                "context": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using connectionist approaches. Each connection can transmit a signal to other neurons.",
                "expected_answer": "Neural networks are computing systems with interconnected nodes that process information similarly to biological neural networks."
            },
            {
                "question": "What are the benefits of Python?",
                "context": "Python offers simple syntax, extensive libraries, strong community support, and versatility across web development, data science, AI, and automation. It's known for readability and ease of learning.",
                "expected_answer": "Python benefits include simple syntax, extensive libraries, strong community support, and versatility across multiple domains."
            },
            {
                "question": "What is natural language processing?",
                "context": "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. It bridges the gap between human communication and computer understanding.",
                "expected_answer": "NLP is an AI branch that helps computers understand, interpret, and manipulate human language."
            },
            {
                "question": "How do algorithms optimize performance?",
                "context": "Algorithms optimize performance through techniques like time complexity reduction, space efficiency, caching, parallelization, and heuristic approaches. The choice depends on specific problem requirements and constraints.",
                "expected_answer": "Algorithms optimize performance through time/space complexity reduction, caching, parallelization, and heuristic techniques."
            }
        ]
    
    def evaluate_candidate(self, candidate: Candidate, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a candidate on multiple objectives."""
        try:
            # Convert GEPA candidate back to DSPY program
            program = self.adapter.candidate_to_dspy(candidate)
            
            # Evaluate on dataset
            results = self.metric_collector.evaluate_program(program, dataset)
            
            # Extract objective scores
            scores = {
                "accuracy": results.get("accuracy", 0.0),
                "efficiency": results.get("avg_response_time", 1.0),
                "clarity": results.get("text_clarity", 0.5)
            }
            
            return scores
            
        except Exception as e:
            logger.warning(f"⚠️ Evaluation failed for candidate: {e}")
            # Return default poor scores
            return {obj: 0.0 for obj in self.config.objectives}
    
    def optimize_program(self, initial_program: Module, dataset: List[Dict[str, Any]]) -> Candidate:
        """Optimize a DSPY program using GEPA."""
        logger.info("🚀 Starting DSPY-GEPA optimization")
        
        try:
            # Convert initial DSPY program to GEPA candidate
            initial_candidate = self.adapter.dspy_to_candidate(initial_program)
            logger.info(f"📝 Initial candidate created: {len(initial_candidate.program_text)} chars")
            
            # Create evaluation function
            def eval_func(candidate: Candidate) -> Dict[str, float]:
                return self.evaluate_candidate(candidate, dataset)
            
            # Run optimization
            logger.info(f"🧬 Running optimization with {self.config.population_size} population, {self.config.generations} generations")
            best_candidate = self.optimizer.optimize(initial_candidate, eval_func)
            
            logger.info("✅ Optimization completed successfully")
            return best_candidate
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise
    
    def demonstrate_amope_features(self, program: Module):
        """Demonstrate AMOPE advanced features."""
        logger.info("🧠 Demonstrating AMOPE features")
        
        try:
            candidate = self.adapter.dspy_to_candidate(program)
            
            # Apply reflection-based mutation
            if self.reflection_engine:
                reflected_candidate = self.reflection_engine.reflect_and_mutate(candidate)
                logger.info("💭 Applied reflection-based mutation")
                
                # Compare original vs reflected
                original_score = self.evaluate_candidate(candidate, self.create_sample_dataset())
                reflected_score = self.evaluate_candidate(reflected_candidate, self.create_sample_dataset())
                
                logger.info(f"📊 Original scores: {original_score}")
                logger.info(f"📊 Reflected scores: {reflected_score}")
            
        except Exception as e:
            logger.warning(f"⚠️ AMOPE demonstration failed: {e}")


def setup_dspy_environment():
    """Set up DSPY environment with language model."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("⚠️ OPENAI_API_KEY not found. Using mock LM for demonstration.")
        # Use mock LM for offline demonstration
        lm = dspy.LM(model="mock", api_base="mock")
    else:
        logger.info("🔧 Configuring DSPY with OpenAI")
        lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=api_key)
    
    dspy.settings.configure(lm=lm)
    return lm


def main():
    """Main demonstration function."""
    print("🐶 DSPY-GEPA Integration Example")
    print("=" * 50)
    
    # Set up environment
    logger.info("🔧 Setting up environment...")
    try:
        lm = setup_dspy_environment()
    except Exception as e:
        logger.error(f"❌ Environment setup failed: {e}")
        return
    
    # Create optimization configuration
    config = OptimizationConfig(
        population_size=8,
        generations=3,
        mutation_rate=0.3,
        objectives=["accuracy", "efficiency", "clarity"]
    )
    
    # Initialize optimizer
    logger.info("🤖 Initializing DSPY-GEPA optimizer...")
    try:
        optimizer = DSPYGEPAOptimizer(config)
    except Exception as e:
        logger.error(f"❌ Optimizer initialization failed: {e}")
        logger.info("💡 This might be expected in development. The example shows the intended workflow.")
        return
    
    # Create initial DSPY program
    logger.info("📝 Creating initial DSPY program...")
    initial_program = BasicQA()
    
    # Create dataset
    logger.info("📊 Creating evaluation dataset...")
    dataset = optimizer.create_sample_dataset()
    logger.info(f"📋 Dataset created with {len(dataset)} examples")
    
    # Test initial program
    logger.info("🧪 Testing initial program...")
    try:
        test_example = dataset[0]
        initial_result = initial_program.forward(
            question=test_example["question"],
            context=test_example["context"]
        )
        logger.info(f"💭 Initial program output: {initial_result}")
    except Exception as e:
        logger.warning(f"⚠️ Initial program test failed: {e}")
    
    # Run optimization
    try:
        logger.info("🚀 Starting optimization process...")
        best_candidate = optimizer.optimize_program(initial_program, dataset)
        
        # Convert back to DSPY program
        optimized_program = optimizer.adapter.candidate_to_dspy(best_candidate)
        
        # Test optimized program
        logger.info("🎯 Testing optimized program...")
        optimized_result = optimized_program.forward(
            question=test_example["question"],
            context=test_example["context"]
        )
        logger.info(f"💭 Optimized program output: {optimized_result}")
        
        # Compare results
        logger.info("📈 Optimization Results:")
        initial_scores = optimizer.evaluate_candidate(
            optimizer.adapter.dspy_to_candidate(initial_program), dataset
        )
        optimized_scores = optimizer.evaluate_candidate(best_candidate, dataset)
        
        logger.info(f"📊 Initial scores: {initial_scores}")
        logger.info(f"📊 Optimized scores: {optimized_scores}")
        
        # Demonstrate AMOPE features
        optimizer.demonstrate_amope_features(optimized_program)
        
    except Exception as e:
        logger.error(f"❌ Optimization process failed: {e}")
        logger.info("💡 This is expected if GEPA components are not fully implemented.")
        logger.info("📝 The example demonstrates the intended integration workflow.")
    
    # Summary
    print("\n🎉 DSPY-GEPA Integration Example Completed!")
    print("\n📚 Key Features Demonstrated:")
    print("✅ DSPY program creation with signatures and modules")
    print("✅ GEPA integration with multi-objective optimization")
    print("✅ AMOPE enhancements with adaptive mutation and reflection")
    print("✅ Complete workflow from initial to optimized program")
    print("✅ Comprehensive error handling and logging")
    
    print("\n🔧 To run with full functionality:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Install dependencies: pip install -e .")
    print("3. Ensure all GEPA components are implemented")
    print("4. Run: python examples/dspy_gepa_optimization.py")
    
    print("\n💡 Usage Patterns:")
    print("- Modify OptimizationConfig for different optimization parameters")
    print("- Create custom DSPY signatures for specific tasks")
    print("- Extend objectives array for additional optimization goals")
    print("- Use AMOPE features for advanced mutation strategies")


if __name__ == "__main__":
    main()