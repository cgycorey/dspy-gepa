"""GEPAAgent - Clean official DSPY GEPA integration.

This module provides a simplified interface for using DSPY's official GEPA
optimizer for prompt optimization with LLM-based reflection.
"""

import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field

try:
    import dspy
    from dspy import GEPA
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False
    dspy = None
    GEPA = None

# Import LiteLLM for error handling
try:
    import litellm
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    litellm = None


@dataclass
class OptimizationResult:
    """Result from GEPA optimization."""
    optimized_program: Any
    best_score: float
    initial_score: float
    improvement_percentage: float
    generations_completed: int
    optimization_time: float
    detailed_results: Optional[Dict[str, Any]] = None
    
    @property
    def best_prompt(self) -> str:
        """Get the best prompt from the optimized program."""
        if hasattr(self.optimized_program, 'named_predictors'):
            predictors = list(self.optimized_program.named_predictors())
            if predictors:
                name, predictor = predictors[0]
                return getattr(predictor, 'instructions', '') or getattr(predictor, 'signature', '').instructions
        return str(self.optimized_program)


class GEPAAgent:
    """Simplified agent for prompt optimization using official DSPY GEPA.
    
    This class provides a clean interface to DSPY's GEPA optimizer with
    automatic LLM detection and configuration.
    """
    
    def __init__(self,
                 metric: Optional[Callable] = None,
                 max_metric_calls: int = 100,
                 max_generations: int = 20,
                 reflection_lm: Optional[Any] = None,
                 verbose: bool = True,
                 track_stats: bool = True,
                 auto_detect_llm: bool = True):
        """Initialize GEPAAgent with official DSPY GEPA.
        
        Args:
            metric: Evaluation function that returns score and optional feedback
            max_metric_calls: Maximum number of metric evaluations
            max_generations: Maximum number of generations
            reflection_lm: Language model for reflection (auto-detected if None)
            verbose: Whether to show progress
            track_stats: Whether to track detailed statistics
            auto_detect_llm: Whether to auto-detect LLM from DSPy settings
        """
        if not _DSPY_AVAILABLE:
            raise ImportError("DSPY is required. Install with: pip install dspy-ai")
        
        self.metric = metric
        self.max_metric_calls = max_metric_calls
        self.max_generations = max_generations
        self.verbose = verbose
        self.track_stats = track_stats
        self.auto_detect_llm = auto_detect_llm
        
        # Auto-detect reflection LM if not provided
        if reflection_lm is None and auto_detect_llm:
            self.reflection_lm = self._detect_reflection_lm()
        else:
            self.reflection_lm = reflection_lm
        
        # Initialize GEPA optimizer
        self.gepa = GEPA(
            metric=self._create_default_metric() if metric is None else metric,
            max_metric_calls=max_metric_calls,
            reflection_lm=self.reflection_lm,
            track_stats=track_stats,
            track_best_outputs=True
        )
        
        if self.verbose:
            self._print_init_info()
    
    def _detect_reflection_lm(self) -> Optional[Any]:
        """Detect reflection LM from DSPy settings."""
        try:
            # Try to get LM from DSPy settings
            lm = getattr(dspy.settings, 'lm', None)
            if lm is not None:
                if self.verbose:
                    print(f"✅ Detected LLM from DSPy settings: {type(lm).__name__}")
                return lm
            
            # Try to create a default OpenAI LM
            try:
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    # Use gpt-4o-mini instead of gpt-4 for better compatibility
                    lm = dspy.LM(model='openai/gpt-4o-mini', api_key=api_key, temperature=1.0)
                    if self.verbose:
                        print("✅ Created default OpenAI LM (gpt-4o-mini) for reflection")
                    return lm
            except Exception as e:
                if _LITELLM_AVAILABLE and hasattr(litellm, 'UnsupportedParamsError'):
                    if isinstance(e, litellm.UnsupportedParamsError):
                        if self.verbose:
                            print("⚠️  Model doesn't support structured outputs. Try gpt-4o-mini instead.")
                pass
            
            if self.verbose:
                print("⚠️  No LLM detected. GEPA will use instruction proposer only.")
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  LLM detection failed: {e}")
            return None
    
    def _create_default_metric(self) -> Callable:
        """Create a default metric function for optimization."""
        def default_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            """Default metric that returns a simple score."""
            try:
                # Simple score based on whether prediction exists and is reasonable
                if hasattr(pred, 'answer') and pred.answer:
                    # Check if prediction matches expected answer if available
                    if hasattr(gold, 'answer') and gold.answer:
                        score = float(str(pred.answer).strip().lower() == str(gold.answer).strip().lower())
                    else:
                        # If no expected answer, give a modest score for any output
                        score = 0.7
                else:
                    score = 0.0
                
                # Provide basic feedback
                feedback = f"Prediction received with score {score:.2f}"
                if hasattr(pred, 'answer') and pred.answer:
                    feedback = f"Generated answer: '{pred.answer}' (score: {score:.2f})"
                
                return dspy.Prediction(score=score, feedback=feedback)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Metric evaluation error: {e}")
                return dspy.Prediction(score=0.0, feedback=f"Evaluation error: {e}")
        
        return default_metric
    
    def _print_init_info(self):
        """Print initialization information."""
        print("🤖 GEPAAgent initialized with official DSPY GEPA")
        print(f"   Max metric calls: {self.max_metric_calls}")
        print(f"   Reflection LM: {'✅ Configured' if self.reflection_lm else '❌ Not available'}")
        print(f"   Statistics tracking: {'✅ Enabled' if self.track_stats else '❌ Disabled'}")
        
        if self.reflection_lm:
            lm_info = getattr(self.reflection_lm, 'model', 'Unknown model')
            print(f"   Model: {lm_info}")
    
    def optimize(self,
                 program: Any,
                 trainset: List[Any],
                 valset: Optional[List[Any]] = None,
                 metric: Optional[Callable] = None) -> OptimizationResult:
        """Optimize a DSPY program using GEPA.
        
        Args:
            program: DSPY program to optimize
            trainset: Training dataset
            valset: Validation dataset (optional)
            metric: Custom metric function (overrides default)
            
        Returns:
            OptimizationResult with optimized program and metrics
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🚀 Starting GEPA optimization...")
            print(f"   Program type: {type(program).__name__}")
            print(f"   Training examples: {len(trainset)}")
            if valset:
                print(f"   Validation examples: {len(valset)}")
        
        # Use provided metric or default
        optimization_metric = metric or self._create_default_metric()
        
        # Create new GEPA instance if metric is different
        if metric is not None and metric != getattr(self, '_last_metric', None):
            self.gepa = GEPA(
                metric=optimization_metric,
                max_metric_calls=self.max_metric_calls,
                reflection_lm=self.reflection_lm,
                track_stats=self.track_stats,
                track_best_outputs=True
            )
            self._last_metric = metric
        
        try:
            # Evaluate initial program
            initial_score = self._evaluate_program(program, trainset[:5], optimization_metric)
            
            if self.verbose:
                print(f"   Initial score: {initial_score:.4f}")
            
            # Run GEPA optimization
            optimized_program = self.gepa.compile(
                program,
                trainset=trainset,
                valset=valset or trainset
            )
            
            # Evaluate optimized program
            final_score = self._evaluate_program(optimized_program, trainset[:5], optimization_metric)
            
            # Calculate improvement
            if initial_score > 0:
                improvement_percentage = ((final_score - initial_score) / initial_score) * 100
            else:
                improvement_percentage = 0.0
            
            # Get detailed results if available
            detailed_results = None
            if hasattr(optimized_program, 'detailed_results'):
                detailed_results = optimized_program.detailed_results
            
            optimization_time = time.time() - start_time
            
            result = OptimizationResult(
                optimized_program=optimized_program,
                best_score=final_score,
                initial_score=initial_score,
                improvement_percentage=improvement_percentage,
                generations_completed=getattr(self.gepa, 'generations_completed', self.max_generations),
                optimization_time=optimization_time,
                detailed_results=detailed_results
            )
            
            if self.verbose:
                print(f"\n✅ GEPA optimization completed!")
                print(f"   Final score: {final_score:.4f}")
                print(f"   Improvement: {improvement_percentage:.1f}%")
                print(f"   Time: {optimization_time:.2f}s")
                if detailed_results:
                    stats = getattr(detailed_results, 'val_aggregate_scores', {})
                    if stats:
                        print(f"   Pareto frontier: {len(stats)} candidates")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Optimization failed: {e}")
            
            # Return fallback result
            return OptimizationResult(
                optimized_program=program,
                best_score=0.0,
                initial_score=0.0,
                improvement_percentage=0.0,
                generations_completed=0,
                optimization_time=time.time() - start_time
            )
    
    def _evaluate_program(self, program: Any, dataset: List[Any], metric: Callable) -> float:
        """Evaluate a program on a dataset."""
        try:
            scores = []
            for example in dataset:
                try:
                    prediction = program(**example.inputs)
                    result = metric(example, prediction)
                    
                    if hasattr(result, 'score'):
                        scores.append(float(result.score))
                    elif isinstance(result, (int, float)):
                        scores.append(float(result))
                    elif isinstance(result, dict) and 'score' in result:
                        scores.append(float(result['score']))
                    else:
                        scores.append(0.0)
                        
                except Exception:
                    scores.append(0.0)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception:
            return 0.0
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM status information."""
        if self.reflection_lm is not None:
            return {
                "status": "configured",
                "available": True,
                "provider": type(self.reflection_lm).__name__,
                "model": getattr(self.reflection_lm, 'model', 'unknown'),
                "will_use_llm": True,
                "mutation_type": "LLM-guided reflection"
            }
        else:
            return {
                "status": "not_configured",
                "available": False,
                "provider": None,
                "model": None,
                "will_use_llm": False,
                "mutation_type": "instruction proposer only"
            }
    
    def configure_reflection_lm(self, lm: Any):
        """Configure the reflection language model."""
        self.reflection_lm = lm
        self.gepa.reflection_lm = lm
        
        if self.verbose:
            print(f"✅ Reflection LM configured: {type(lm).__name__}")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        llm_status = self.get_llm_status()
        status_emoji = "🤖" if llm_status["available"] else "🔧"
        
        return (
            f"{status_emoji} GEPAAgent("
            f"metric_calls={self.max_metric_calls}, "
            f"llm={'✅' if llm_status['available'] else '❌'})"
        )


# Convenience function for quick optimization
def quick_optimize(program: Any,
                  trainset: List[Any],
                  metric: Optional[Callable] = None,
                  max_metric_calls: int = 50,
                  **kwargs) -> OptimizationResult:
    """Quick optimization function for DSPY programs.
    
    Args:
        program: DSPY program to optimize
        trainset: Training dataset
        metric: Optional custom metric function
        max_metric_calls: Maximum metric evaluations
        **kwargs: Additional arguments for GEPAAgent
        
    Returns:
        OptimizationResult with optimized program
    """
    agent = GEPAAgent(
        metric=metric,
        max_metric_calls=max_metric_calls,
        **kwargs
    )
    
    return agent.optimize(program, trainset)