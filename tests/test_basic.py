"""Basic tests for dspy-gepa package."""

import pytest
from unittest.mock import Mock, patch


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
@patch('dspy_gepa.core.agent.GEPAAgent')
def test_agent_creation(mock_agent_class):
    """Test that GEPAAgent can be created (mocked for now)."""
    from dspy_gepa import GEPAAgent
    
    # This will be mocked when we implement the actual class
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent
    
    # Test basic agent creation
    agent = GEPAAgent(
        signature=Mock(),
        name="test_agent"
    )
    
    assert agent is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Integration tests require external API keys")
def test_real_llm_connection():
    """Test real LLM connection (requires API keys)."""
    # This test should only run when API keys are available
    # and is marked to be skipped by default
    pass


if __name__ == "__main__":
    pytest.main([__file__])