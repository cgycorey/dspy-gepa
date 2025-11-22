# DSPY-GEPA

A simplified interface for DSPY GEPA (Genetic Prompt Engineering Assistant) integration.

## Overview

DSPY-GEPA provides a clean, streamlined interface for prompt optimization using genetic evolutionary algorithms. This project focuses on core functionality with real LLM provider support.

## Features

- ✅ **Real LLM Support**: OpenAI and Anthropic providers
- ✅ **Genetic Optimization**: Evolutionary prompt improvement
- ✅ **Simple API**: Clean, easy-to-use interface
- ✅ **Smart Detection**: Automatic provider selection
- ✅ **Mock Fallback**: Works without API keys for testing

## Quick Start

### Installation

```bash
# Install dependencies
pip install dspy gepa

# Set your API key (optional for testing)
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

### Basic Usage

```python
from dspy_gepa import SimpleGEPA

# Create optimizer
optimizer = SimpleGEPA()

# Optimize a prompt
result = optimizer.optimize(
    initial_prompt="Translate {text} to Spanish:",
    evaluation_data=[
        {"text": "Hello", "expected": "Hola"},
        {"text": "Goodbye", "expected": "Adiós"}
    ],
    generations=5,
    population_size=4
)

print(f"Best prompt: {result['best_prompt']}")
print(f"Improvement: {result['improvement']:.2%}")
```

### Check Provider Status

```python
from dspy_gepa import print_llm_status

# Display current LLM provider configuration
print_llm_status()
```

## Project Structure

```
dspy-gepa/
├── src/dspy_gepa/
│   ├── __init__.py          # Main exports
│   ├── simple_gepa.py       # Simple GEPA interface
│   ├── gepa_agent.py        # Core agent implementation
│   ├── core/
│   │   └── agent.py          # Core agent logic
│   └── utils/
│       ├── config.py         # Configuration utilities
│       └── logging.py        # Logging utilities
├── simple_demo.py            # Working demo
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## LLM Provider Support

The system automatically detects and uses available providers in this priority:

1. **OpenAI** (highest priority)
   - Set `OPENAI_API_KEY` environment variable
2. **Anthropic** 
   - Set `ANTHROPIC_API_KEY` environment variable
3. **Mock** (fallback for testing)
   - Works without API keys

## Configuration

No configuration files needed. The system uses environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-3.5-turbo"  # optional

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-3-haiku-20240307"  # optional
```

## Example: Running the Demo

```bash
# Clone and run
python simple_demo.py
```

The will show the optimization process with real or mock LLM providers.

## Core Components

- **SimpleGEPA**: Main interface for optimization
- **GEPAAgent**: Core optimization engine
- **Agent**: Base agent functionality
- **Config**: Simple provider detection
- **Logging**: Basic logging utilities

## License

See LICENSE file for details.