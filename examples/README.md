# DSPY-GEPA Examples

Welcome to the DSPY-GEPA examples! These are organized from beginner to advanced to help you get started quickly.

## 🚀 Quick Start Path

### 1. Absolute Beginners
Start here if you're new to prompt optimization:

```bash
uv run python examples/01_quick_start.py
```

**What you'll learn:**
- Create a GEPAAgent in 1 line
- Run basic optimization in 2 lines
- See immediate results

### 2. LLM Setup & Configuration
Learn how to configure different LLM providers:

```bash
uv run python examples/02_llm_setup.py
```

**What you'll learn:**
- Auto-detect LLM providers
- Manual configuration
- Environment variable setup
- Fallback mode

### 3. Basic Optimization
Master fundamental optimization techniques:

```bash
uv run python examples/03_basic_optimization.py
```

**What you'll learn:**
- Single-objective optimization
- Multi-objective optimization
- Writing evaluation functions
- Comparing different approaches

### 4. Advanced Features
Explore advanced capabilities and patterns:

```bash
uv run python examples/04_advanced_features.py
```

**What you'll learn:**
- Configuration inspection
- Error handling
- Performance monitoring
- Advanced usage patterns

## 🔧 DSPY Integration

For users working with DSPY programs:

```bash
# Install DSPY first (optional)
uv add dspy

# Run DSPY integration example
uv run python examples/dspy_integration/dspy_example.py
```

## 📁 Example Structure

```
examples/
├── 01_quick_start.py          # 🚀 4-line usage example
├── 02_llm_setup.py            # 🔧 LLM configuration demo
├── 03_basic_optimization.py   # 📈 Optimization fundamentals
├── 04_advanced_features.py    # 🎯 Advanced patterns & features
├── dspy_integration/          # 🧠 DSPY program optimization
│   ├── README.md
│   └── dspy_example.py
└── README.md                  # 📚 This file
```

## 💡 Choosing the Right Example

| Your Goal | Start With | Why |
|-----------|------------|-----|
| "I just want to see it work" | `01_quick_start.py` | Instant results in 2 seconds |
| "I need to set up my LLM" | `02_llm_setup.py` | Complete LLM configuration guide |
| "I want to understand optimization" | `03_basic_optimization.py` | Clear fundamentals explained |
| "I'm building a real application" | `04_advanced_features.py` | Production-ready patterns |
| "I work with DSPY programs" | `dspy_integration/dspy_example.py` | DSPY-specific optimization |

## 🎯 Learning Path

1. **Start** with `01_quick_start.py` to see immediate results
2. **Configure** your LLM with `02_llm_setup.py`
3. **Learn** optimization concepts with `03_basic_optimization.py`
4. **Master** advanced features with `04_advanced_features.py`
5. **Specialize** with DSPY integration if needed

## 🛠️ Prerequisites

```bash
# Install dependencies
uv sync

# Optional: Set up LLM API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## 🐛 Troubleshooting

**Import errors?**
```bash
uv sync --reinstall
```

**LLM not working?**
- Check `02_llm_setup.py` for configuration help
- All examples work in fallback mode without LLM

**DSPY errors?**
- DSPY is optional - examples include mock classes
- Install with `uv add dspy` for full functionality

## 🎉 Next Steps

After running the examples:

1. **Read the main README.md** for detailed documentation
2. **Check the configuration guide** for production setup
3. **Explore the source code** in `src/dspy_gepa/`
4. **Run the tests** with `uv run pytest`

Happy optimizing! 🚀