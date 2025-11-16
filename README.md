# DSPY-GEPA: Adaptive Multi-Objective Prompt Evolution

A genetic evolutionary programming framework that extends [GEPA](https://github.com/gepa-ai/gepa.git) with adaptive optimization strategies for prompt and DSPY program evolution.

## 🙏 Attribution

This project is built upon and extends the [GEPA (Genetic-Pareto Algorithm)](https://github.com/gepa-ai/gepa.git) framework. GEPA provides the core genetic programming and Pareto optimization capabilities that form the foundation of this work. We deeply appreciate the GEPA team's excellent work in creating this powerful optimization framework.

## 🆕 New Features

### Enhanced Agent Interface
- **Smart LLM Auto-Detection**: Automatically detects and configures available LLM providers
- **Multi-Provider Support**: Seamless integration with OpenAI, Anthropic, and other providers
- **Fallback System**: Graceful degradation to handcrafted mutations when LLMs are unavailable
- **Dynamic LLM Configuration**: Runtime reconfiguration of LLM providers and settings
- **Status Monitoring**: Real-time monitoring of LLM availability and performance

### Advanced LLM Integration
- **Intelligent Provider Selection**: Automatic fallback between providers based on availability
- **API Key Management**: Secure environment variable and configuration file support
- **Model Configuration**: Flexible model selection and parameter tuning
- **Error Handling**: Robust error recovery and retry mechanisms

## ✅ What's Currently Implemented

- **Enhanced Agent Interface**: Smart LLM auto-detection and multi-provider support
- **AMOPE Algorithm**: Adaptive Multi-Objective Prompt Evolution with dynamic strategy selection
- **Adaptive Mutator**: Multiple mutation strategies (gradient-based, statistical, LLM-guided)
- **LLM Integration**: OpenAI, Anthropic, and other provider support with fallback system
- **Objective Balancer**: Dynamic weight adjustment to escape local optima
- **Core GEPA Integration**: Genetic optimization with Pareto selection
- **Working Examples**: Ready-to-run demonstration scripts with LLM integration

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd dspy-test

# Install dependencies with UV
uv sync

# Set up LLM API keys (optional but recommended)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Verify installation
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from dspy_gepa.amope import AMOPEOptimizer
from dspy_gepa import GEPAAgent
print('✅ Installation successful!')
"
```

### Enhanced Quick Start with LLM Auto-Detection

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa import GEPAAgent

# Auto-detect and configure LLM
agent = GEPAAgent(objectives={"accuracy": 0.7, "efficiency": 0.3})

# Check LLM status
status = agent.get_llm_status()
print(f"LLM Status: {status['status']}")
print(f"Mutation Type: {status['mutation_type']}")

# Use the agent for optimization
result = agent.optimize_prompt(
    initial_prompt="Write a concise summary of the main points.",
    generations=25
)

print(f"Best prompt: {result.best_prompt}")
print(f"Best score: {result.best_score}")
```

### Basic Usage Example (Enhanced Interface)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa import GEPAAgent

# Define your evaluation function
def evaluate_prompt(prompt_text):
    # Your evaluation logic here
    # Return a dict with performance score
    return {"performance": 0.8}  # Example score

# Initialize enhanced agent with auto-detected LLM
agent = GEPAAgent(
    objectives={"performance": 1.0},
    population_size=8,
    max_generations=25
)

# Run optimization with enhanced capabilities
result = agent.optimize_prompt(
    initial_prompt="Write a concise summary of the main points.",
    evaluation_fn=evaluate_prompt,
    generations=25
)

print(f"Best prompt: {result.best_prompt}")
print(f"Best score: {result.best_score}")
print(f"Mutation type used: {agent.get_llm_status()['mutation_type']}")
```

### Traditional AMOPE Usage (Backward Compatible)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.amope import AMOPEOptimizer

# Initialize traditional AMOPE optimizer
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

## 🔧 LLM Setup and Configuration

### Environment Variables

Set up your LLM API keys as environment variables:

```bash
# OpenAI Configuration
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # optional, uses default

# Anthropic Configuration
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Optional: Default provider
export DSpyGEPA_DEFAULT_PROVIDER="openai"  # or "anthropic"
```

### Configuration File Setup

Create a `config.yaml` file in your project root:

```yaml
llm:
  default_provider: "openai"
  providers:
    openai:
      api_base: "https://api.openai.com/v1"
      model: "gpt-4"
      temperature: 0.7
      max_tokens: 1000
      timeout: 30
    anthropic:
      api_base: "https://api.anthropic.com"
      model: "claude-3-sonnet-20240229"
      temperature: 0.7
      max_tokens: 1000
      timeout: 30
  fallback:
    enabled: true
    handcrafted_weight: 0.3
  monitoring:
    health_check_interval: 300  # 5 minutes
    retry_attempts: 3
```

### LLM Provider Examples

#### OpenAI Setup
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa import GEPAAgent

# Manual OpenAI configuration
agent = GEPAAgent(
    objectives={"accuracy": 1.0},
    llm_config={
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "enabled": True
    }
)
```

#### Anthropic Setup
```python
# Manual Anthropic configuration
agent = GEPAAgent(
    objectives={"accuracy": 1.0},
    llm_config={
        "provider": "anthropic",
        "model": "claude-3-opus-20240229",
        "temperature": 0.7,
        "enabled": True
    }
)
```

#### Fallback Mode (No LLM)
```python
# Use only handcrafted mutations
agent = GEPAAgent(
    objectives={"accuracy": 1.0},
    llm_config={"enabled": False}
)
```

### Dynamic Reconfiguration

```python
# Start with auto-detection
agent = GEPAAgent(objectives={"accuracy": 0.8, "efficiency": 0.2})

# Check current status
status = agent.get_llm_status()
print(f"Current provider: {status['provider']}")

# Dynamically switch providers
agent.configure_llm("anthropic", model="claude-3-sonnet-20240229")

# Disable LLM and use fallback
agent.configure_llm(None, enabled=False)

# Re-enable with different settings
agent.configure_llm("openai", model="gpt-3.5-turbo", temperature=0.5)
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

### Run the Enhanced Agent Example
```bash
cd dspy-test
uv run python examples/basic_dspy_gepa.py
```

### Run the LLM Integration Demo
```bash
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from dspy_gepa import GEPAAgent

def demo_eval(prompt):
    return {'performance': 0.5 + 0.1 * (hash(prompt) % 10) / 10}

# Test auto-detection
agent = GEPAAgent(objectives={'performance': 1.0})
status = agent.get_llm_status()
print(f'LLM Status: {status["status"]}')
print(f'Provider: {status.get("provider", "None")}')

result = agent.optimize_prompt('Test prompt', demo_eval, generations=10)
print(f'✅ Demo completed! Best score: {result.best_score:.3f}')
"
```

### Run Multi-Provider Demo
```bash
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from dspy_gepa import GEPAAgent

def demo_eval(prompt):
    return {'accuracy': 0.6 + 0.1 * (hash(prompt) % 10) / 10, 'efficiency': 0.7}

# Test with different providers
for provider in ['openai', 'anthropic']:
    print(f'\n--- Testing {provider} ---')
    try:
        agent = GEPAAgent(
            objectives={'accuracy': 0.7, 'efficiency': 0.3},
            llm_config={'provider': provider, 'model': None}  # auto-select model
        )
        status = agent.get_llm_status()
        print(f'Status: {status["status"]}')
        if status['status'] == 'available':
            result = agent.optimize_prompt('Test prompt', demo_eval, generations=5)
            print(f'✅ {provider} working! Score: {result.best_score:.3f}')
    except Exception as e:
        print(f'❌ {provider} failed: {e}')
"
```

### Run Fallback Mode Demo
```bash
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from dspy_gepa import GEPAAgent

def demo_eval(prompt):
    return {'performance': 0.5 + 0.1 * (hash(prompt) % 10) / 10}

# Test fallback mode
agent = GEPAAgent(
    objectives={'performance': 1.0},
    llm_config={'enabled': False}
)

status = agent.get_llm_status()
print(f'LLM Status: {status["status"]}')
print(f'Mutation Type: {status["mutation_type"]}')

result = agent.optimize_prompt('Test prompt', demo_eval, generations=10)
print(f'✅ Fallback mode working! Score: {result.best_score:.3f}')
"
```

## 🔧 Available Components

### Enhanced Agent Interface
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa import GEPAAgent

# Auto-detect LLM and start optimizing
agent = GEPAAgent(objectives={"accuracy": 0.8, "efficiency": 0.2})
status = agent.get_llm_status()

# Optimize with intelligent mutation selection
result = agent.optimize_prompt(
    initial_prompt="Your starting prompt here",
    evaluation_fn=your_eval_function,
    generations=50
)
```

### LLM Manager
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.llm_manager import LLMManager

# Manual LLM management
manager = LLMManager()

# Check available providers
providers = manager.get_available_providers()
print(f"Available: {providers}")

# Test provider health
health = manager.check_provider_health("openai")
print(f"OpenAI healthy: {health}")
```

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

## 🔧 Configuration Guide

### Agent Configuration Options

```python
agent = GEPAAgent(
    objectives={
        "accuracy": 0.6,
        "efficiency": 0.3,
        "clarity": 0.1
    },
    population_size=8,
    max_generations=25,
    llm_config={
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "enabled": True
    },
    mutation_config={
        "strategy": "adaptive",
        "llm_weight": 0.7
    },
    balancing_config={
        "strategy": "adaptive_harmonic",
        "rebalance_frequency": 5
    }
)
```

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | None | Yes (for OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key | None | Yes (for Anthropic) |
| `OPENAI_API_BASE` | OpenAI API base URL | `https://api.openai.com/v1` | No |
| `DSpyGEPA_DEFAULT_PROVIDER` | Default LLM provider | `openai` | No |
| `DSpyGEPA_CONFIG_PATH` | Path to config file | `./config.yaml` | No |

### Configuration File Structure

```yaml
# config.yaml
llm:
  default_provider: "openai"
  providers:
    openai:
      api_base: "https://api.openai.com/v1"
      model: "gpt-4"
      temperature: 0.7
      max_tokens: 1000
      timeout: 30
      retry_attempts: 3
    anthropic:
      api_base: "https://api.anthropic.com"
      model: "claude-3-sonnet-20240229"
      temperature: 0.7
      max_tokens: 1000
      timeout: 30
      retry_attempts: 3
  fallback:
    enabled: true
    handcrafted_weight: 0.3
    auto_fallback: true
  monitoring:
    health_check_interval: 300
    log_level: "INFO"

agent:
  default_population_size: 8
  default_max_generations: 25
  mutation_strategy: "adaptive"
  balancing_strategy: "adaptive_harmonic"
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

### LLM Setup Issues

#### API Key Problems
```bash
# Check if API keys are set
env | grep -E "(OPENAI|ANTHROPIC)_API_KEY"

# Test API connectivity
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from dspy_gepa import GEPAAgent

agent = GEPAAgent()
status = agent.get_llm_status()
print(f'LLM Status: {status}')
"
```

#### Provider Connection Issues
```python
# Debug provider connections
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from dspy_gepa.llm_manager import LLMManager

manager = LLMManager()

# Check all providers
for provider in ["openai", "anthropic"]:
    health = manager.check_provider_health(provider)
    print(f"{provider}: {health}")

# Get detailed status
status = manager.get_detailed_status()
print(f"Detailed status: {status}")
```

#### Fallback Mode Activation
```python
# Force fallback mode if LLM issues persist
agent = GEPAAgent(
    objectives={"accuracy": 1.0},
    llm_config={"enabled": False}
)

# Verify fallback is active
status = agent.get_llm_status()
print(f"Mutation type: {status['mutation_type']}")  # Should be 'handcrafted'
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
- Enable fallback mode if LLM calls are slow: `llm_config={"enabled": False}`
- Use smaller models for faster LLM responses: `model="gpt-3.5-turbo"`

### Common Error Messages

#### "No LLM providers available"
- **Cause**: No API keys set or all providers are unreachable
- **Solution**: Set environment variables or use fallback mode

#### "API key not found"
- **Cause**: Missing environment variable for the selected provider
- **Solution**: Set the appropriate `*_API_KEY` environment variable

#### "Rate limit exceeded"
- **Cause**: Too many API calls to the LLM provider
- **Solution**: Wait and retry, or switch to a different provider

#### "Mutation failed: LLM error"
- **Cause**: LLM provider returned an error during mutation
- **Solution**: System will automatically retry with fallback mutations

## 📊 Current Features

### Enhanced Agent Interface
- ✅ **Smart LLM Auto-Detection**: Automatically detects and configures available LLM providers
- ✅ **Multi-Provider Support**: Seamless integration with OpenAI, Anthropic, and other providers
- ✅ **Fallback System**: Graceful degradation to handcrafted mutations when LLMs are unavailable
- ✅ **Dynamic LLM Configuration**: Runtime reconfiguration of LLM providers and settings
- ✅ **Status Monitoring**: Real-time monitoring of LLM availability and performance

### Core Optimization Features
- ✅ **AMOPE Algorithm**: Adaptive multi-objective optimization with LLM integration
- ✅ **Dynamic Strategy Selection**: Automatic mutation strategy adaptation
- ✅ **Multi-Objective Balancing**: Dynamic weight adjustment for complex objectives
- ✅ **Core GEPA Integration**: Genetic evolutionary programming with Pareto optimization
- ✅ **LLM-Enhanced Mutations**: Intelligent prompt mutations powered by language models
- ✅ **Handcrafted Mutations**: Rule-based mutations for fallback mode

### Developer Experience
- ✅ **Working Examples**: Ready-to-run demonstration scripts with LLM integration
- ✅ **Comprehensive Testing**: Validated components and integration tests
- ✅ **Configuration Management**: Flexible environment variable and YAML configuration
- ✅ **Error Handling**: Robust error recovery and retry mechanisms
- ✅ **Backward Compatibility**: Existing AMOPE code continues to work unchanged

## 📝 Development Status

This is an **alpha release** with core functionality implemented. The AMOPE algorithm is fully functional and tested. Additional features like advanced DSPY integration and LLM reflection are planned for future releases.

## 🔗 Dependencies & Acknowledgments

This project extends and builds upon:
- **[GEPA (Genetic-Pareto Algorithm)](https://github.com/gepa-ai/gepa.git)** - Core genetic programming and Pareto optimization framework
- **[DSPy](https://github.com/stanfordnlp/dspy)** - Programming with foundation models (optional dependency)
- **Open source community** - Various packages for machine learning and optimization

We deeply appreciate the GEPA team's excellent work in creating this powerful optimization framework that forms the foundation of this project.

## 📄 License

Copyright (c) 2025 cgycorey. All rights reserved.