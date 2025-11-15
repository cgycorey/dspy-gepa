"""Shared pytest fixtures and configuration for DSPY-GEPA tests."""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock dependencies that might not be available
MOCK_DSPY = Mock()
MOCK_DSPY.Module = Mock
MOCK_DSPY.ChainOfThought = Mock
MOCK_DSPY.Prediction = Mock
MOCK_DSPY.settings = Mock()
MOCK_DSPY.configure = Mock()

# Test data fixtures
@pytest.fixture
def sample_qa_data():
    """Sample question-answering data for testing."""
    return [
        {"question": "What is machine learning?", "expected_answer": "AI systems that learn from data"},
        {"question": "What is Python?", "expected_answer": "A programming language"},
        {"question": "What is genetic programming?", "expected_answer": "Evolutionary algorithm technique"},
        {"question": "What is neural network?", "expected_answer": "Computational model inspired by brain"},
        {"question": "What is deep learning?", "expected_answer": "Subset of machine learning"},
    ]

@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment analysis data for testing."""
    return [
        {"text": "I love this product!", "expected_sentiment": "positive"},
        {"text": "This is terrible.", "expected_sentiment": "negative"},
        {"text": "It's okay, nothing special.", "expected_sentiment": "neutral"},
        {"text": "Amazing service!", "expected_sentiment": "positive"},
        {"text": "Worst experience ever.", "expected_sentiment": "negative"},
    ]

@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for testing."""
    return {
        "question": [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "Python is a high-level programming language known for its simplicity and versatility.",
            "Genetic programming is an evolutionary algorithm-based methodology inspired by biological evolution.",
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative"],
        "default": "This is a mock response for testing purposes."
    }

@pytest.fixture
def mock_dspy_module():
    """Create a mock DSPY module for testing."""
    class MockDSPYModule:
        def __init__(self, module_type="qa"):
            self.module_type = module_type
            self.name = f"Mock{module_type.title()}Module"
            
        def forward(self, **kwargs):
            if self.module_type == "qa":
                return MOCK_DSPY.Prediction(answer="Mock answer to the question")
            elif self.module_type == "sentiment":
                return MOCK_DSPY.Prediction(sentiment="positive")
            else:
                return MOCK_DSPY.Prediction(output="Mock output")
                
        def __call__(self, **kwargs):
            return self.forward(**kwargs)
    
    return MockDSPYModule

@pytest.fixture
def mock_candidate():
    """Create a mock GEPA candidate for testing."""
    from gepa.core.candidate import Candidate
    
    return Candidate(
        content='{"class_name": "MockModule", "module": "test.mock", "parameters": {}}',
        fitness_scores={"accuracy": 0.8, "efficiency": 0.7, "cost": 0.6},
        generation=0,
        metadata={"program_type": "dspy", "adapter_version": "0.1.0"}
    )

@pytest.fixture
def optimization_config():
    """Configuration for optimization tests."""
    return {
        "population_size": 4,
        "max_generations": 3,
        "mutation_rate": 0.3,
        "crossover_rate": 0.7,
        "objectives": ["accuracy", "efficiency", "cost"],
        "selection_pressure": 2.0,
        "elitism_count": 1
    }

@pytest.fixture
def performance_baseline():
    """Baseline performance metrics for regression testing."""
    return {
        "max_execution_time": 10.0,  # seconds
        "max_memory_usage": 100.0,   # MB
        "min_accuracy": 0.6,
        "min_efficiency": 0.5,
        "max_cost_per_eval": 0.01
    }

@pytest.fixture
def temp_dir(tmp_path_factory):
    """Create a temporary directory for test files."""
    return tmp_path_factory.mktemp("dspy_gepa_test")

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for API calls."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock LLM response"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for API calls."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Mock Claude response")]
    mock_response.usage = Mock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def fitness_history():
    """Sample fitness history for testing adaptive strategies."""
    return [
        {"accuracy": 0.5, "efficiency": 0.6, "cost": 0.7},
        {"accuracy": 0.6, "efficiency": 0.5, "cost": 0.8},
        {"accuracy": 0.7, "efficiency": 0.7, "cost": 0.6},
        {"accuracy": 0.8, "efficiency": 0.8, "cost": 0.5},
        {"accuracy": 0.75, "efficiency": 0.9, "cost": 0.4},
    ]

@pytest.fixture
def population_metrics():
    """Sample population metrics for testing."""
    return {
        "diversity": 0.7,
        "improvement_rate": 0.05,
        "convergence_rate": 0.3,
        "avg_fitness": 0.65,
        "best_fitness": 0.85
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up common test environment patches."""
    # Mock environment variables
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DSPY_GEPA_TEST_MODE", "true")
    
    # Mock external dependencies
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

@pytest.fixture
def benchmark_data():
    """Data for performance benchmarking."""
    return {
        "small_dataset": list(range(10)),
        "medium_dataset": list(range(100)),
        "large_dataset": list(range(1000)),
        "complex_prompts": [
            "Analyze the following text and extract key insights: {text}",
            "Translate the following content while preserving meaning: {content}",
            "Summarize the main points of the following document: {document}",
            "Classify the sentiment of the following text: {text}",
            "Generate a creative response to: {prompt}"
        ]
    }

# Performance monitoring fixture
@pytest.fixture
def performance_monitor():
    """Fixture to monitor performance during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.memory_start = None
            self.metrics = {}
            
        def start(self):
            self.start_time = time.time()
            try:
                import psutil
                process = psutil.Process()
                self.memory_start = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                self.memory_start = 0
                
        def stop(self):
            if self.start_time:
                execution_time = time.time() - self.start_time
                self.metrics["execution_time"] = execution_time
                
                try:
                    import psutil
                    process = psutil.Process()
                    memory_end = process.memory_info().rss / 1024 / 1024  # MB
                    self.metrics["memory_used"] = memory_end - (self.memory_start or 0)
                except ImportError:
                    self.metrics["memory_used"] = 0
                    
            return self.metrics
            
        def __enter__(self):
            self.start()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.stop()
    
    return PerformanceMonitor()

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )

# Helper functions for tests
def create_test_candidate(content: str = "test content", fitness: Dict = None) -> "Candidate":
    """Helper to create test candidates."""
    from gepa.core.candidate import Candidate
    
    return Candidate(
        content=content,
        fitness_scores=fitness or {"accuracy": 0.5, "efficiency": 0.5},
        generation=0,
        metadata={"test": True}
    )

def assert_performance_within_baseline(metrics: Dict, baseline: Dict):
    """Assert that performance metrics are within baseline limits."""
    for metric, value in metrics.items():
        if metric in baseline:
            limit = baseline[metric]
            if metric.startswith("max"):
                assert value <= limit, f"{metric} {value} exceeds limit {limit}"
            elif metric.startswith("min"):
                assert value >= limit, f"{metric} {value} below minimum {limit}"

# Mock DSPY imports patch
@pytest.fixture(autouse=True)
def mock_dspy_imports(monkeypatch):
    """Mock DSPY imports for tests where DSPY is not available."""
    mock_modules = {
        'dspy': MOCK_DSPY,
        'dspy.predict': Mock(),
        'dspy.teleprompt': Mock(),
        'dspy.evaluate': Mock(),
    }
    
    for module_name, mock_module in mock_modules.items():
        monkeypatch.setitem(sys.modules, module_name, mock_module)
