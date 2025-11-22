"""Shared pytest fixtures and configuration for dspy-gepa tests."""

from __future__ import annotations

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Any, Dict
from unittest.mock import Mock, patch

from tests.fixtures.test_data import (
    create_sample_evaluation_results,
    create_sample_optimization_state,
    create_progress_data,
    SAMPLE_CHECKPOINT_STATE,
    RESOURCE_LIMITS,
    CONVERGENCE_CONFIGS,
    STOPPING_CONFIGS,
    MockProcess
)


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_evaluation_results():
    """Provide sample evaluation results for testing."""
    return create_sample_evaluation_results(10)


@pytest.fixture
def sample_optimization_state():
    """Provide sample optimization state for testing."""
    return create_sample_optimization_state()


@pytest.fixture
def sample_progress_data():
    """Provide sample progress data for testing."""
    return create_progress_data(20)


@pytest.fixture
def checkpoint_state():
    """Provide sample checkpoint state for testing."""
    return SAMPLE_CHECKPOINT_STATE.copy()


@pytest.fixture(params=["conservative", "aggressive", "minimal"])
def resource_limits(request):
    """Parametrized resource limits fixture."""
    return RESOURCE_LIMITS[request.param]


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """Create a temporary checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def temp_log_dir(temp_dir):
    """Create a temporary log directory."""
    log_dir = temp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture
def mock_psutil_process():
    """Mock psutil.Process for testing without actual process monitoring."""
    return MockProcess()


@pytest.fixture
def mock_psutil():
    """Mock psutil module for testing."""
    with patch('src.dspy_gepa.core.monitoring.psutil') as mock_psutil:
        mock_psutil.Process.return_value = MockProcess()
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception
        yield mock_psutil


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib for testing without actual plotting."""
    with patch('src.dspy_gepa.core.visualization.MATPLOTLIB_AVAILABLE', False):
        with patch('src.dspy_gepa.core.visualization.plt') as mock_plt:
            mock_plt.subplots.return_value = (Mock(), Mock())
            yield mock_plt


@pytest.fixture
def mock_plotly():
    """Mock plotly for testing without actual interactive plots."""
    with patch('src.dspy_gepa.core.visualization.PLOTLY_AVAILABLE', False):
        with patch('src.dspy_gepa.core.visualization.go') as mock_go:
            yield mock_go


@pytest.fixture
def mock_numpy():
    """Mock numpy for testing without actual numerical computations."""
    with patch('src.dspy_gepa.core.analysis.np') as mock_np:
        # Setup numpy mock with common methods
        mock_np.arange = lambda x: list(range(x))
        mock_np.array = lambda x: x
        mock_np.polyfit = lambda x, y, deg: [0.001, 0.5]  # Return slope, intercept
        yield mock_np


@pytest.fixture
def convergence_configs():
    """Provide convergence detector configurations."""
    return CONVERGENCE_CONFIGS


@pytest.fixture
def stopping_configs():
    """Provide optimal stopping estimator configurations."""
    return STOPPING_CONFIGS


@pytest.fixture(params=CONVERGENCE_CONFIGS.keys())
def convergence_config(request):
    """Parametrized convergence configuration fixture."""
    return CONVERGENCE_CONFIGS[request.param]


@pytest.fixture(params=STOPPING_CONFIGS.keys())
def stopping_config(request):
    """Parametrized stopping configuration fixture."""
    return STOPPING_CONFIGS[request.param]


@pytest.fixture
def mock_evaluation_result():
    """Create a single mock evaluation result."""
    from src.dspy_gepa.core.interfaces import EvaluationResult, ObjectiveEvaluation, SolutionMetadata, OptimizationDirection
    
    return EvaluationResult(
        solution_id="test_solution_001",
        objectives={
            "accuracy": ObjectiveEvaluation(
                objective_name="accuracy",
                score=0.85,
                direction=OptimizationDirection.MAXIMIZE,
                evaluation_time=0.1
            ),
            "efficiency": ObjectiveEvaluation(
                objective_name="efficiency",
                score=0.75,
                direction=OptimizationDirection.MINIMIZE,
                evaluation_time=0.05
            )
        },
        overall_score=0.8,
        evaluation_time=0.15,
        metadata=SolutionMetadata(
            generation=1,
            parent_ids=["parent_001"],
            mutation_type="test_mutation"
        )
    )


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def monitor():
        start_time = time.time()
        yield
        duration = time.time() - start_time
        # Could add assertions about duration here
        print(f"Test duration: {duration:.3f}s")
    
    return monitor


# Thread safety testing fixture
@pytest.fixture
def thread_safety_test():
    """Test thread safety of components."""
    import threading
    import time
    from typing import Callable
    
    def run_concurrently(func: Callable, num_threads: int = 5, iterations: int = 100) -> Dict[str, Any]:
        """Run function concurrently and return results."""
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(iterations):
                    result = func()
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        return {
            "results": results,
            "errors": errors,
            "total_calls": len(results),
            "total_errors": len(errors),
            "success_rate": len(results) / (len(results) + len(errors)) if (len(results) + len(errors)) > 0 else 0
        }
    
    return run_concurrently


# Logging capture fixture
@pytest.fixture
def log_capture(caplog):
    """Enhanced log capture with better filtering."""
    import logging
    
    # Set up log capture
    caplog.set_level(logging.DEBUG)
    
    class LogCapture:
        def __init__(self, caplog):
            self.caplog = caplog
        
        def get_logs_by_level(self, level: str) -> list:
            """Get logs filtered by level."""
            return [record for record in self.caplog.records if record.levelname == level.upper()]
        
        def get_error_logs(self) -> list:
            """Get all error logs."""
            return self.get_logs_by_level("ERROR")
        
        def get_warning_logs(self) -> list:
            """Get all warning logs."""
            return self.get_logs_by_level("WARNING")
        
        def assert_no_errors(self) -> None:
            """Assert that no error logs were generated."""
            error_logs = self.get_error_logs()
            if error_logs:
                error_messages = [log.message for log in error_logs]
                raise AssertionError(f"Unexpected error logs: {error_messages}")
        
        def assert_no_warnings(self) -> None:
            """Assert that no warning logs were generated."""
            warning_logs = self.get_warning_logs()
            if warning_logs:
                warning_messages = [log.message for log in warning_logs]
                raise AssertionError(f"Unexpected warning logs: {warning_messages}")
    
    return LogCapture(caplog)
