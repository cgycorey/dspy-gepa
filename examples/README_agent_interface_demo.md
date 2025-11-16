# GEPAAgent Interface Demonstration

This directory contains demonstration scripts for the enhanced GEPAAgent interface with LLM support.

## agent_interface_demo.py

A comprehensive demonstration script that showcases all the enhanced features of the GEPAAgent interface:

### Features Demonstrated

1. **LLM Auto-Detection** - Automatic configuration detection from `config.yaml` and environment variables
2. **Manual Configuration** - Setting LLM providers manually via configuration parameters
3. **Dynamic Configuration** - Runtime reconfiguration using `configure_llm()` method
4. **Optimization Workflow** - Complete prompt optimization with and without LLM support
5. **Fallback Behavior** - Graceful degradation when LLM is not available
6. **User-Friendly Interface** - Simple and advanced usage patterns
7. **Multi-Objective Optimization** - Support for multiple optimization objectives
8. **Configuration Inspection** - Getting insights and debugging information
9. **Error Handling** - Robust configuration with proper fallback handling
10. **Performance Monitoring** - Real-time status and optimization metrics

### Usage

```bash
# Run the demonstration
python examples/agent_interface_demo.py

# Or make it executable and run directly
chmod +x examples/agent_interface_demo.py
./examples/agent_interface_demo.py
```

### Prerequisites

- Python 3.8+
- dspy-gepa package installed
- config.yaml file (optional, for LLM configuration)
- Environment variables (optional): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

### What You'll See

The demonstration script will show:

- ✅ Automatic LLM detection from configuration files
- ✅ Manual LLM configuration setup
- ✅ Dynamic provider switching during runtime
- ✅ Complete optimization workflow with real metrics
- ✅ Fallback behavior when LLM is unavailable
- ✅ Multi-objective optimization capabilities
- ✅ Configuration inspection and debugging tools
- ✅ Error handling and robust configuration

### Key Features Highlighted

1. **Auto-Detection**: The agent automatically detects LLM configuration from:
   - `config.yaml` file in the project root
   - Environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
   - Default settings if no configuration is found

2. **Graceful Fallback**: When LLM is not available, the agent automatically falls back to handcrafted mutations:
   - Uses "LLM-guided + handcrafted" when LLM is available
   - Uses "handcrafted only" when LLM is not available
   - Provides clear status messages about what's being used

3. **Dynamic Configuration**: Change LLM providers at runtime:
   ```python
   agent.configure_llm("anthropic", model="claude-3-opus", temperature=0.3)
   ```

4. **Comprehensive Status**: Get detailed information about LLM status:
   ```python
   status = agent.get_llm_status()
   print(f"Provider: {status['provider']}")
   print(f"Available: {status['available']}")
   print(f"Mutation type: {status['mutation_type']}")
   ```

5. **Optimization Insights**: Analyze optimization performance:
   ```python
   insights = agent.get_optimization_insights()
   print(f"Average improvement: {insights['average_improvement']:.1f}%")
   ```

### Production Usage Tips

1. **Configuration Setup**:
   - Set up `config.yaml` with your LLM provider details
   - Or use environment variables for API keys
   - The agent handles configuration automatically

2. **Monitoring**: Use `get_llm_status()` to check what mutation type is being used

3. **Performance**: Use `get_optimization_insights()` for performance analysis

4. **Reliability**: The agent works with or without LLM - no configuration needed for basic functionality

### Integration Examples

The demonstration shows real-world usage patterns that you can adapt for your own projects:

- Simple prompt optimization with single objective
- Multi-objective optimization for complex requirements
- Environment-based configuration for deployment
- Error handling and robust configuration patterns

For more detailed examples and integration patterns, see the other demo files in this directory.

---

**Copyright (c) 2025 cgycorey. All rights reserved.**