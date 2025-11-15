# DSPY-GEPA: Adaptive Multi-Objective Prompt Evolution

A genetic evolutionary programming framework that extends GEPA with adaptive optimization strategies for prompt and DSPY program evolution.

## ✅ What's Currently Implemented

- **AMOPE Algorithm**: Adaptive Multi-Objective Prompt Evolution with dynamic strategy selection
- **Adaptive Mutator**: Multiple mutation strategies (gradient-based, statistical, LLM-guided)
- **Objective Balancer**: Dynamic weight adjustment to escape local optima
- **Core GEPA Integration**: Genetic optimization with Pareto selection
- **Working Examples**: Ready-to-run demonstration scripts

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd dspy-test

# Install dependencies with UV
uv sync

# Verify installation
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from dspy_gepa.amope import AMOPEOptimizer
print('✅ Installation successful!')
"
```

### Basic Usage Example

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.amope import AMOPEOptimizer

# Define your evaluation function
def evaluate_prompt(prompt_text):
    # Your evaluation logic here
    # Return a dict with performance score
    return {"performance": 0.8}  # Example score

# Initialize AMOPE optimizer
optimizer = AMOPEOptimizer(
    objectives={"performance": 1.0},
    population_size=8,
    max_generations=25
)

# Run optimization
result = optimizer.optimize(
    initial_prompt="Write a concise summary of the main points.",
    evaluation_fn=evaluate_prompt,
    generations=25
)

print(f"Best prompt: {result.best_prompt}")
print(f"Best score: {result.best_score}")
```

## 📋 Step-by-Step Tutorial

### Step 1: Create Your Evaluation Function

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

def my_evaluation_function(prompt_text):
    """
    Evaluate a prompt and return a dict with performance score.
    
    Args:
        prompt_text (str): The prompt to evaluate
        
    Returns:
        dict: Performance score with key 'performance' (0.0 to 1.0)
    """
    # Example: Check if prompt has key characteristics
    score = 0.0
    
    # Reward clarity
    if "." in prompt_text and len(prompt_text.split()) > 5:
        score += 0.3
    
    # Reward specific instructions
    if any(word in prompt_text.lower() for word in ["please", "write", "provide", "create"]):
        score += 0.4
    
    # Reward reasonable length
    if 10 <= len(prompt_text) <= 200:
        score += 0.3
    
    return {"performance": min(1.0, score)}
```

### Step 2: Set Up Multi-Objective Optimization (Optional)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.amope import AMOPEOptimizer

# Evaluate multiple objectives
def multi_objective_evaluation(prompt_text):
    # Example helper functions (implement these based on your needs)
    def evaluate_clarity(text):
        return 0.7 if len(text.split()) > 5 else 0.3
    
    def evaluate_specificity(text):
        return 0.8 if any(word in text.lower() for word in ["please", "write", "provide"]) else 0.4
    
    def evaluate_efficiency(text):
        return 0.9 if 10 <= len(text) <= 100 else 0.5
    
    return {
        "clarity": evaluate_clarity(prompt_text),
        "specificity": evaluate_specificity(prompt_text),
        "efficiency": evaluate_efficiency(prompt_text)
    }

# Initialize with multiple objectives
optimizer = AMOPEOptimizer(
    objectives={"clarity": 0.4, "specificity": 0.4, "efficiency": 0.2},
    balancing_config={"strategy": "adaptive_harmonic"}
)
```

### Step 3: Run the Optimization

```python
result = optimizer.optimize(
    initial_prompt="Your starting prompt here",
    evaluation_fn=multi_objective_evaluation,
    generations=50
)

# Access results
print(f"Best prompt: {result.best_prompt}")
print(f"Best objectives: {result.best_objectives}")
print(f"Generations completed: {result.generations_completed}")
```

## 🧪 Working Examples

### Run the Basic Example
```bash
cd dspy-test
uv run python examples/basic_dspy_gepa.py
```

### Run the AMOPE Demo
```bash
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from dspy_gepa.amope import AMOPEOptimizer

def demo_eval(prompt):
    return {'performance': 0.5 + 0.1 * (hash(prompt) % 10) / 10}

optimizer = AMOPEOptimizer(objectives={'performance': 1.0})
result = optimizer.optimize('Test prompt', demo_eval, generations=10)
print(f'✅ Demo completed! Best score: {result.best_score:.3f}')
"
```

## 🔧 Available Components

### AMOPE Algorithm
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.amope import AMOPEOptimizer, AMOPEConfig

# Advanced configuration
config = AMOPEConfig(
    population_size=12,
    max_generations=100,
    mutation_config={"strategy": "adaptive"},
    balancing_config={"strategy": "stagnation_focus"}
)

optimizer = AMOPEOptimizer(config=config)
```

### Adaptive Mutator
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.amope import AdaptiveMutator, MutationStrategy

mutator = AdaptiveMutator()
# Use different mutation strategies automatically
```

### Objective Balancer
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.amope import ObjectiveBalancer, BalancingStrategy

balancer = ObjectiveBalancer(
    objectives={"accuracy": 0.7, "efficiency": 0.3},
    strategy=BalancingStrategy.ADAPTIVE_HARMONIC
)
```

## 🐛 Troubleshooting

### Import Errors
```bash
# If you get import errors, try:
uv sync --reinstall
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
import dspy_gepa
print('✅ Imports working')
"
```

### DSPY Integration (Optional)
```bash
# DSPY is optional - install if you want DSPY features:
uv add dspy

# Then use DSPY integration components
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from dspy_gepa.dspy_integration import DSPYAdapter
print('✅ DSPY integration working')
"
```

### Performance Issues
- Reduce `population_size` and `max_generations` for faster results
- Use simpler evaluation functions for testing
- Monitor optimization progress with `verbose=True`

## 📊 Current Features

- ✅ **AMOPE Algorithm**: Adaptive multi-objective optimization
- ✅ **Dynamic Strategy Selection**: Automatic mutation strategy adaptation
- ✅ **Multi-Objective Balancing**: Dynamic weight adjustment
- ✅ **Core GEPA Integration**: Genetic evolutionary programming
- ✅ **Working Examples**: Ready-to-run demonstration scripts
- ✅ **Comprehensive Testing**: Validated components

## 📝 Development Status

This is an **alpha release** with core functionality implemented. The AMOPE algorithm is fully functional and tested. Additional features like advanced DSPY integration and LLM reflection are planned for future releases.

## 📄 License

Copyright (c) 2025 cgycorey. All rights reserved.