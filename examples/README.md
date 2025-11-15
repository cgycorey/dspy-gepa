# DSPY-GEPA Examples

This directory contains examples showcasing the integration between DSPY and GEPA for evolutionary optimization of prompt programs.

## Files Overview

### `basic_dspy_gepa.py`
The main demonstration file (583 lines) that shows:
- Basic DSPY-GEPA workflow
- Evolutionary optimization setup
- Enhanced sample data creation
- Comprehensive metrics collection
- Robust error handling and demonstration mode

### `dspy_modules.py`
DSPY module classes and signatures (83 lines):
- `QuestionAnswering` signature for QA tasks
- `SentimentClassification` signature for sentiment analysis
- `SimpleQA` enhanced DSPY module with ChainOfThought
- `SentimentAnalysis` enhanced module with confidence scoring

### `language_model_setup.py`
Robust language model configuration (270 lines):
- `LMConfig` dataclass for model configuration
- `setup_language_model()` function with multi-provider support
- Enhanced mock LM for demonstration purposes
- Automatic API key detection and fallback mechanisms

## Phase 2 Implementation Features

### Real Language Model Support
- **OpenAI**: Automatic setup with `OPENAI_API_KEY`
- **Anthropic**: Support for Claude models with `ANTHROPIC_API_KEY`
- **Local Models**: Ollama/local LLM support via `LOCAL_MODEL_PATH`
- **Enhanced MockLM**: Sophisticated fallback for demo environments

### Environment Variables
```bash
# OpenAI Configuration
export OPENAI_API_KEY="your-openai-key"
export OPENAI_MODEL="gpt-3.5-turbo"
export OPENAI_TEMPERATURE="0.7"
export OPENAI_MAX_TOKENS="1024"

# Anthropic Configuration
export ANTHROPIC_API_KEY="your-anthropic-key"
export ANTHROPIC_MODEL="claude-3-sonnet-20240229"
export ANTHROPIC_TEMPERATURE="0.7"
export ANTHROPIC_MAX_TOKENS="1024"

# Local Model Configuration
export LOCAL_MODEL_PATH="/path/to/model"
export LOCAL_MODEL_NAME="llama2"
export LOCAL_API_BASE="http://localhost:11434"
```

### Enhanced DSPY Modules
- **Proper Signatures**: Type-safe input/output fields
- **ChainOfThought Integration**: Built-in reasoning and explanation
- **Confidence Scoring**: Sentiment analysis with confidence intervals
- **Error Handling**: Graceful fallbacks and validation

### Improved Sample Data
- **Realistic Examples**: 5 diverse QA questions with context
- **Varied Difficulty**: Easy, medium, and hard complexity levels
- **Rich Sentiment Data**: 8 sentiment examples with different text types
- **Comprehensive Ground Truth**: Keywords, confidence thresholds, and expected outputs

### Enhanced Evaluation Metrics
- **Accuracy Analysis**: Keyword and semantic matching
- **Performance Metrics**: Execution time, consistency, diversity
- **Quality Scores**: Confidence quality and complexity assessment
- **Detailed Logging**: Comprehensive result tracking

## Running the Examples

### With Real APIs
```bash
# Set your API key(s)
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"

# Run the demonstration
python3 examples/basic_dspy_gepa.py
```

### In Demonstration Mode
```bash
# No API keys needed - runs with enhanced mock LM
python3 examples/basic_dspy_gepa.py
```

### Dependencies
- DSPY (for full functionality)
- GEPA framework components
- Python 3.8+

Note: The examples will work without DSPY installed, but with limited functionality using mock models.

## Architecture

The Phase 2 implementation follows these design principles:

1. **Modular Design**: Separate concerns into focused modules
2. **Robust Fallbacks**: Work with or without external dependencies
3. **Type Safety**: Comprehensive type hints and validation
4. **Production Ready**: Error handling, logging, and configuration
5. **Demo Friendly**: Enhanced behavior even in demonstration environments

## Next Steps (Phase 3)

The next phase will focus on:
- Advanced evolutionary optimization algorithms
- Multi-objective optimization improvements
- Performance profiling and analysis
- Advanced mutation strategies
- Real-world use case examples