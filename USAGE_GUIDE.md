# DSPy-GEPA Framework Usage Guide

Welcome to the comprehensive usage guide for the DSPy-GEPA (Genetic Evolutionary Prompt Adaptation) framework! This guide will walk you through everything you need to know to get started with advanced prompt optimization using genetic algorithms.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage - Single-Objective Optimization](#basic-usage---single-objective-optimization)
3. [Advanced Usage - Multi-Objective Optimization](#advanced-usage---multi-objective-optimization)
4. [Monitoring and Visualization](#monitoring-and-visualization)
5. [Running Tests](#running-tests)
6. [Best Practices and Workflow](#best-practices-and-workflow)
7. [Troubleshooting](#troubleshooting)

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended) or `pip`
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd dspy-test
```

### Step 2: Install Dependencies with UV (Recommended)

```bash
# Install uv if you don't have it
pip install uv

# Install the project and all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

### Step 3: Alternative Installation with Pip

```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```python
# Test the installation
from dspy_gepa import GEPAAgent, MultiObjectiveGEPAAgent
print("Installation successful!")
```

---

## Basic Usage - Single-Objective Optimization

The simplest way to use DSPy-GEPA is for single-objective prompt optimization.

### Example 1: Basic Prompt Optimization

```python
from dspy_gepa import GEPAAgent

# Define your evaluation function
def evaluate_prompt(prompt: str) -> float:
    """Evaluate prompt quality - return a score between 0 and 1"""
    # Your evaluation logic here
    # For example: test on a dataset, measure accuracy, etc.
    return 0.75  # Example score

# Create and configure the agent
agent = GEPAAgent(
    max_generations=10,
    population_size=8,
    mutation_rate=0.3,
    elite_ratio=0.2,
    verbose=True
)

# Optimize your prompt
result = agent.optimize_prompt(
    initial_prompt="Translate the following text to Spanish:",
    evaluation_function=evaluate_prompt,
    generations=5,
    return_summary=True
)

# Access results
print(f"Best score: {result.best_score}")
print(f"Best prompt: {result.best_prompt}")
print(f"Total evaluations: {result.total_evaluations}")
```

### Example 2: Using SimpleGEPA for Quick Optimization

```python
from dspy_gepa import SimpleGEPA

# Simple interface for quick optimization
optimizer = SimpleGEPA()

best_prompt = optimizer.optimize(
    prompt="Summarize this article:",
    eval_function=lambda p: len(p.split()) / 10,  # Simple metric
    iterations=10
)

print(f"Optimized prompt: {best_prompt}")
```

### Example 3: Custom Configuration

```python
from dspy_gepa import GEPAAgent
from dspy_gepa.core.mutation_engine import SemanticMutator
from dspy_gepa.core.parameter_tuner import ConvergenceBasedTuner

# Custom mutation strategy
mutator = SemanticMutator(
    mutation_strength=0.5,
    semantic_similarity_threshold=0.7
)

# Custom parameter tuning
tuner = ConvergenceBasedTuner(
    convergence_patience=3,
    adaptation_rate=0.1
)

# Advanced agent configuration
agent = GEPAAgent(
    max_generations=20,
    population_size=12,
    mutator=mutator,
    parameter_tuner=tuner,
    verbose=True
)

result = agent.optimize_prompt(
    "Classify this text as positive or negative:",
    evaluate_prompt,
    generations=15
)
```

---

## Advanced Usage - Multi-Objective Optimization

Multi-objective optimization allows you to optimize prompts across multiple criteria simultaneously.

### Example 1: Basic Multi-Objective Setup

```python
from dspy_gepa import (
    MultiObjectiveGEPAAgent,
    AccuracyMetric,
    FluencyMetric,
    TokenUsageMetric,
    TaskType,
    OptimizationDirection
)

# Define multiple objectives
objectives = [
    AccuracyMetric(
        weight=0.5,
        direction=OptimizationDirection.MAXIMIZE,
        name="accuracy"
    ),
    FluencyMetric(
        weight=0.3,
        direction=OptimizationDirection.MAXIMIZE,
        name="fluency"
    ),
    TokenUsageMetric(
        weight=0.2,
        direction=OptimizationDirection.MINIMIZE,
        name="efficiency"
    )
]

# Create multi-objective agent
mo_agent = MultiObjectiveGEPAAgent(
    objectives=objectives,
    task_type=TaskType.GENERATION,
    max_generations=15,
    population_size=10,
    verbose=True
)

# Define evaluation function that returns multiple scores
def multi_evaluate(prompt: str) -> dict:
    """Evaluate prompt across multiple objectives"""
    return {
        "accuracy": your_accuracy_function(prompt),
        "fluency": your_fluency_function(prompt),
        "efficiency": your_efficiency_function(prompt)
    }

# Optimize with multiple objectives
result = mo_agent.optimize_prompt(
    initial_prompt="Generate a creative story about:",
    evaluation_function=multi_evaluate,
    generations=10,
    return_summary=True
)

# Access Pareto frontier
pareto_solutions = result.get_pareto_frontier_solutions()
print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")

# Select best solution based on preferences
best_solution = result.select_best_solution()
print(f"Best solution: {best_solution.prompt}")
```

### Example 2: Custom Objectives

```python
from dspy_gepa import Objective, OptimizationDirection
from dspy_gepa.core.interfaces import ObjectiveEvaluation
from typing import Dict, Any

class CustomMetric(Objective):
    """Custom metric for domain-specific evaluation"""
    
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(
            name="custom_metric",
            weight=weight,
            direction=OptimizationDirection.MAXIMIZE,
            **kwargs
        )
    
    def evaluate(self, prompt: str, context: Dict[str, Any] = None) -> ObjectiveEvaluation:
        """Evaluate prompt using custom logic"""
        score = your_custom_evaluation_logic(prompt)
        
        return ObjectiveEvaluation(
            objective_name=self.name,
            score=score,
            direction=self.direction,
            metadata={"custom_info": "additional_data"}
        )

# Use custom metric
objectives = [
    AccuracyMetric(weight=0.6),
    CustomMetric(weight=0.4)
]

agent = MultiObjectiveGEPAAgent(objectives=objectives)
```

### Example 3: Preference-Based Solution Selection

```python
from dspy_gepa import PreferenceVector, OptimizationDirection

# Define preferences for solution selection
preferences = PreferenceVector({
    "accuracy": 0.7,      # High preference for accuracy
    "fluency": 0.2,       # Medium preference for fluency
    "efficiency": 0.1     # Low preference for efficiency
})

# Optimize with preferences
result = mo_agent.optimize_prompt(
    "Analyze the sentiment of this review:",
    multi_evaluate,
    generations=12,
    preference_vector=preferences
)

# Get solution matching your preferences
preferred_solution = result.select_solution_by_preferences(preferences)
print(f"Preferred solution: {preferred_solution.prompt}")
```

---

## Monitoring and Visualization

DSPy-GEPA includes comprehensive monitoring and visualization capabilities.

### Example 1: Real-time Monitoring

```python
from dspy_gepa.core.monitoring import OptimizationMonitor, MonitoringConfig
from dspy_gepa.core.visualization import OptimizationProgressVisualizer

# Configure monitoring
monitor_config = MonitoringConfig(
    track_convergence=True,
    track_diversity=True,
    save_intermediate_results=True,
    log_level="INFO"
)

# Create monitor
monitor = OptimizationMonitor(config=monitor_config)

# Create agent with monitoring
agent = GEPAAgent(
    max_generations=20,
    monitor=monitor,
    verbose=True
)

# Run optimization with monitoring
result = agent.optimize_prompt(
    "Extract key information from:",
    evaluate_prompt,
    generations=15
)

# Access monitoring data
print(f"Convergence rate: {monitor.get_convergence_rate()}")
print(f"Population diversity: {monitor.get_diversity_metrics()}")
print(f"Optimization efficiency: {monitor.get_efficiency_score()}")
```

### Example 2: Progress Visualization

```python
from dspy_gepa.core.visualization import (
    OptimizationProgressVisualizer,
    VisualizationConfig,
    create_progress_analysis
)

# Configure visualization
viz_config = VisualizationConfig(
    figure_size=(12, 8),
    style="seaborn",
    color_palette="viridis",
    save_plots=True,
    plot_directory="optimization_plots"
)

# Create visualizer
visualizer = OptimizationProgressVisualizer(config=viz_config)

# Generate progress plots
visualizer.plot_progress(
    monitor_data=monitor.get_monitoring_data(),
    save_path="progress_analysis.png"
)

# Create comprehensive analysis
analysis = create_progress_analysis(
    monitoring_data=monitor.get_monitoring_data(),
    objectives=["accuracy", "fluency", "efficiency"]
)

print(analysis.summary)
```

### Example 3: Pareto Frontier Visualization

```python
from dspy_gepa.core.visualization import (
    ParetoFrontierVisualizer,
    create_pareto_analysis
)

# For multi-objective results
if hasattr(result, 'get_pareto_frontier_solutions'):
    pareto_solutions = result.get_pareto_frontier_solutions()
    
    # Create Pareto frontier visualizer
    pareto_viz = ParetoFrontierVisualizer(config=viz_config)
    
    # Plot 2D Pareto frontier
    pareto_viz.plot_pareto_2d(
        solutions=pareto_solutions,
        objective_x="accuracy",
        objective_y="efficiency",
        save_path="pareto_2d.png"
    )
    
    # Plot 3D Pareto frontier
    pareto_viz.plot_pareto_3d(
        solutions=pareto_solutions,
        objectives=["accuracy", "fluency", "efficiency"],
        save_path="pareto_3d.png"
    )
    
    # Generate comprehensive analysis
    pareto_analysis = create_pareto_analysis(pareto_solutions)
    print(pareto_analysis.summary)
```

---

## Running Tests

The framework includes comprehensive test suites to ensure reliability.

### Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   └── test_simple_plot.py
├── integration/             # Integration tests for component interaction
│   ├── test_architecture_compliance.py
│   ├── test_multi_objective_compatibility.py
│   └── test_dspy_gepa_integration.py
├── test_core/              # Core functionality tests
│   ├── test_comprehensive_integration.py
│   ├── test_monitoring_analysis.py
│   └── test_visualization.py
├── end-to-end/             # End-to-end workflow tests
│   └── test_visualization_demo.py
└── fixtures/               # Test data and utilities
    └── test_data.py
```

### Running All Tests with UV

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=dspy_gepa --cov-report=html
```

### Running Specific Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run only end-to-end tests
uv run pytest tests/end-to-end/

# Run core functionality tests
uv run pytest tests/test_core/
```

### Running Specific Test Files

```bash
# Run architecture compliance test
uv run pytest tests/integration/test_architecture_compliance.py -v

# Run multi-objective compatibility test
uv run pytest tests/integration/test_multi_objective_compatibility.py -v

# Run visualization demo
uv run pytest tests/end-to-end/test_visualization_demo.py -v -s
```

### Running Tests with Specific Markers

```bash
# Run fast tests only
uv run pytest -m "not slow"

# Run integration tests only
uv run pytest -m "integration"

# Run visualization tests only
uv run pytest -m "visualization"
```

### Test Configuration

The test configuration is defined in `pyproject.toml` and `tests/conftest.py`. Key features:

- Automatic fixture management
- Mock data generation
- Test isolation and cleanup
- Performance benchmarking
- Integration with coverage tools

---

## Best Practices and Workflow

### 1. Development Workflow

```python
# 1. Start with a simple baseline
baseline_prompt = "Translate to English:"
baseline_score = evaluate_prompt(baseline_prompt)

# 2. Use SimpleGEPA for quick iteration
quick_optimizer = SimpleGEPA()
improved_prompt = quick_optimizer.optimize(
    baseline_prompt,
    evaluate_prompt,
    iterations=5
)

# 3. Use full GEPAAgent for serious optimization
agent = GEPAAgent(max_generations=15, population_size=8)
result = agent.optimize_prompt(
    improved_prompt,
    evaluate_prompt,
    generations=10
)

# 4. Validate results
validation_score = evaluate_prompt(result.best_prompt)
improvement = (validation_score - baseline_score) / baseline_score
print(f"Improvement: {improvement:.2%}")
```

### 2. Multi-Objective Optimization Strategy

```python
# 1. Start with 2-3 key objectives
objectives = [
    AccuracyMetric(weight=0.6),
    EfficiencyMetric(weight=0.4)
]

# 2. Use moderate population size for exploration
mo_agent = MultiObjectiveGEPAAgent(
    objectives=objectives,
    population_size=10,
    max_generations=12
)

# 3. Analyze Pareto frontier
result = mo_agent.optimize_prompt(prompt, multi_evaluate)
pareto_solutions = result.get_pareto_frontier_solutions()

# 4. Select appropriate solution based on use case
if prioritize_accuracy:
    solution = max(pareto_solutions, key=lambda s: s.objectives["accuracy"])
else:
    solution = result.select_best_solution()
```

### 3. Monitoring and Iteration

```python
# 1. Enable comprehensive monitoring
monitor = OptimizationMonitor(
    config=MonitoringConfig(
        track_convergence=True,
        track_diversity=True,
        save_intermediate_results=True
    )
)

# 2. Run optimization with monitoring
agent = GEPAAgent(monitor=monitor, verbose=True)
result = agent.optimize_prompt(prompt, evaluate_prompt)

# 3. Analyze optimization behavior
convergence_data = monitor.get_convergence_data()
if convergence_data.early_convergence:
    print("Consider increasing population size for better exploration")

if convergence_data.stagnation_detected:
    print("Consider adjusting mutation rate or objectives")

# 4. Visualize results
visualizer = OptimizationProgressVisualizer()
visualizer.plot_progress(monitor.get_monitoring_data())
```

### 4. Performance Optimization

```python
# 1. Use caching for expensive evaluations
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_evaluate(prompt: str) -> float:
    return expensive_evaluation_function(prompt)

# 2. Parallel evaluation for large populations
from concurrent.futures import ThreadPoolExecutor

def parallel_evaluate(prompts: list) -> list:
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cached_evaluate, prompts))

# 3. Adaptive parameters
from dspy_gepa.core.parameter_tuner import ConvergenceBasedTuner

tuner = ConvergenceBasedTuner(
    convergence_patience=3,
    adaptation_rate=0.15
)

agent = GEPAAgent(
    parameter_tuner=tuner,
    adaptive_parameters=True
)
```

### 5. Error Handling and Robustness

```python
from dspy_gepa.core.error_handling import (
    OptimizationError,
    EvaluationError,
    ConfigurationError
)

def robust_optimization(prompt: str):
    try:
        agent = GEPAAgent(max_generations=10)
        result = agent.optimize_prompt(prompt, safe_evaluate)
        return result
        
    except ConfigurationError as e:
        print(f"Configuration issue: {e}")
        return None
        
    except EvaluationError as e:
        print(f"Evaluation failed: {e}")
        # Fallback to simpler evaluation
        return fallback_optimization(prompt)
        
    except OptimizationError as e:
        print(f"Optimization failed: {e}")
        return None
```

### 6. Experiment Tracking

```python
import json
from datetime import datetime

def track_experiment(config, result):
    experiment_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config.__dict__,
        "result": {
            "best_score": result.best_score,
            "best_prompt": result.best_prompt,
            "generations": result.total_generations,
            "evaluations": result.total_evaluations
        }
    }
    
    # Save to experiments log
    with open(f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(experiment_data, f, indent=2)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

```bash
# Issue: ModuleNotFoundError
# Solution: Ensure you're in the project directory with activated virtual environment
source .venv/bin/activate
uv sync
```

#### 2. Convergence Issues

```python
# Problem: Optimization converges too quickly
# Solution: Increase population size and adjust mutation rate
agent = GEPAAgent(
    population_size=12,  # Increase from default
    mutation_rate=0.4,   # Increase mutation
    elite_ratio=0.1      # Decrease elite ratio
)
```

#### 3. Memory Issues

```python
# Problem: High memory usage with large populations
# Solution: Enable result caching and limit history
from dspy_gepa.utils.config import Config

Config.enable_result_caching = True
Config.max_history_size = 100
```

#### 4. Slow Evaluation

```python
# Problem: Evaluation functions are slow
# Solution: Use caching and parallel evaluation
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

@lru_cache(maxsize=500)
def fast_evaluate(prompt: str) -> float:
    return your_evaluation_function(prompt)
```

### Getting Help

- Check the test files for working examples
- Review the architecture documentation
- Run the architecture compliance test: `uv run pytest tests/integration/test_architecture_compliance.py`
- Enable verbose logging: `agent = GEPAAgent(verbose=True)`

---

## Next Steps

Now that you have the complete usage guide, you're ready to:

1. **Start with simple examples** in the Basic Usage section
2. **Explore multi-objective optimization** for complex scenarios
3. **Set up monitoring** to understand optimization behavior
4. **Run tests** to verify everything works correctly
5. **Follow best practices** for production usage

Happy optimizing! 🚀

For more advanced examples and specific use cases, check out the test files in the `tests/` directory and the demo scripts.

---

*This guide is part of the DSPy-GEPA framework. For the latest updates and additional documentation, visit the project repository.*