"""Basic tests for dspy-gepa package."""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os


def test_package_import():
    """Test that the main package can be imported."""
    import dspy_gepa
    
    # Check version info
    assert hasattr(dspy_gepa, "__version__")
    assert dspy_gepa.__version__ == "0.1.0"
    
    # Check main components are importable
    assert hasattr(dspy_gepa, "GEPAAgent")
    assert hasattr(dspy_gepa, "GEPADataset")
    assert hasattr(dspy_gepa, "ExperimentTracker")


@pytest.mark.unit
def test_logger_import():
    """Test that logging utilities can be imported."""
    from dspy_gepa.utils.logging import get_logger, setup_logging
    
    # Test basic functionality
    logger = get_logger("test")
    assert logger is not None
    
    # Test setup logging (should not raise exceptions)
    setup_logging()  # Should be idempotent


@pytest.mark.unit
def test_config_import():
    """Test that configuration utilities can be imported."""
    from dspy_gepa.utils.config import load_config, save_config
    
    # Test basic functionality
    # Note: This will create/load a default config if file doesn't exist
    config = load_config()
    assert config is not None
    assert isinstance(config, dict)


@pytest.mark.unit
def test_agent_creation_basic():
    """Test that GEPAAgent can be created with basic configuration."""
    from dspy_gepa import GEPAAgent
    
    # Test basic agent creation with minimal config
    agent = GEPAAgent(
        objectives={"performance": 1.0},
        max_generations=5,
        population_size=3,
        verbose=False  # Reduce noise in tests
    )
    
    assert agent is not None
    assert agent.config.objectives == {"performance": 1.0}
    assert agent.config.max_generations == 5
    assert agent.config.population_size == 3
    assert agent.config.verbose == False
    assert hasattr(agent, 'optimizer')
    assert hasattr(agent, 'optimization_history')


@pytest.mark.unit
def test_agent_creation_with_mock_signature():
    """Test that GEPAAgent can be created with signature parameter for compatibility."""
    from dspy_gepa import GEPAAgent
    
    # Create a mock signature
    mock_signature = Mock()
    mock_signature.__str__ = Mock(return_value="MockSignature")
    
    # Test agent creation with signature
    agent = GEPAAgent(
        signature=mock_signature,
        name="test_agent",
        objectives={"accuracy": 0.7, "clarity": 0.3},
        verbose=False
    )
    
    assert agent is not None
    assert agent.signature == mock_signature
    assert agent.name == "test_agent"
    assert agent.config.objectives == {"accuracy": 0.7, "clarity": 0.3}


@pytest.mark.unit
def test_agent_llm_status():
    """Test that GEPAAgent can report LLM status correctly."""
    from dspy_gepa import GEPAAgent
    
    # Create agent without LLM (should work fine)
    agent = GEPAAgent(
        objectives={"performance": 1.0},
        auto_detect_llm=False,
        verbose=False
    )
    
    # Check LLM status
    llm_status = agent.get_llm_status()
    assert isinstance(llm_status, dict)
    assert "status" in llm_status
    assert "available" in llm_status
    assert "will_use_llm" in llm_status
    assert llm_status["will_use_llm"] == False
    
    # Test is_llm_available method
    assert agent.is_llm_available() == False


@pytest.mark.unit
def test_agent_optimization_methods():
    """Test that GEPAAgent optimization methods exist and are callable."""
    from dspy_gepa import GEPAAgent
    
    agent = GEPAAgent(
        objectives={"performance": 1.0},
        verbose=False
    )
    
    # Check that optimization methods exist
    assert hasattr(agent, 'optimize_prompt')
    assert hasattr(agent, 'evaluate_current_best')
    assert hasattr(agent, 'get_optimization_insights')
    assert callable(agent.optimize_prompt)
    assert callable(agent.evaluate_current_best)
    assert callable(agent.get_optimization_insights)
    
    # Test insights with empty history
    insights = agent.get_optimization_insights()
    assert isinstance(insights, dict)
    assert "status" in insights
    assert insights["status"] == "No optimization history available"


@pytest.mark.unit
def test_agent_configuration_methods():
    """Test that GEPAAgent configuration methods work correctly."""
    from dspy_gepa import GEPAAgent
    
    agent = GEPAAgent(
        objectives={"performance": 1.0},
        verbose=False
    )
    
    # Test objective update
    new_objectives = {"accuracy": 0.6, "speed": 0.4}
    agent.update_objectives(new_objectives)
    assert agent.config.objectives == new_objectives
    
    # Test history reset
    agent.optimization_history = ["dummy_entry"]  # Add dummy entry
    agent.reset_history()
    assert len(agent.optimization_history) == 0
    
    # Test LLM configuration
    agent.configure_llm("openai", model="gpt-4", api_key="test_key")
    llm_status = agent.get_llm_status()
    assert llm_status["provider"] == "openai"
    assert llm_status["model"] == "gpt-4"


@pytest.mark.unit
def test_agent_repr():
    """Test that GEPAAgent has a meaningful string representation."""
    from dspy_gepa import GEPAAgent
    
    # Test with LLM disabled
    agent = GEPAAgent(
        objectives={"performance": 1.0},
        auto_detect_llm=False,
        verbose=False
    )
    
    repr_str = repr(agent)
    assert "GEPAAgent" in repr_str
    assert "🔧" in repr_str  # Should show wrench when LLM not available
    assert "objectives=1" in repr_str
    
    # Test repr method exists
    assert hasattr(agent, '__repr__')
    assert callable(agent.__repr__)


@pytest.mark.integration
@pytest.mark.skip(reason="Integration tests require external API keys")
def test_real_llm_connection():
    """Test real LLM connection (requires API keys)."""
    # This test should only run when API keys are available
    # and is marked to be skipped by default
    pass


if __name__ == "__main__":
    pytest.main([__file__])