"""Performance metrics collection for DSPY-GEPA integration.

This module provides comprehensive metrics collection and tracking for DSPY programs
within the GEPA framework. It captures execution performance, accuracy, cost, and
other relevant metrics to guide the evolutionary optimization process.
"""

from __future__ import annotations

import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager

from pydantic import BaseModel, Field

# Conditional DSPY import with direct type imports
try:
    import dspy
    from dspy import Module, Signature
    DSPY_AVAILABLE = True
    DSPY_TYPES_AVAILABLE = True
except ImportError:
    dspy = None
    DSPY_AVAILABLE = False
    DSPY_TYPES_AVAILABLE = False
    # Create dummy types for when DSPY is not available
    class DSPYFallbackModule:
        pass
    class DSPYFallbackSignature:
        pass

from gepa.core.candidate import ExecutionTrace, Candidate


@dataclass
class ResourceUsage:
    """Resource usage metrics during execution."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    peak_memory_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    

class DSPYMetrics(BaseModel):
    """Comprehensive metrics for DSPY program execution."""
    
    # Basic execution metrics
    execution_time: float = Field(description="Total execution time in seconds")
    success: bool = Field(description="Whether execution completed successfully")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Performance metrics
    total_predictions: int = Field(default=0, description="Number of predictions made")
    successful_predictions: int = Field(default=0, description="Number of successful predictions")
    failed_predictions: int = Field(default=0, description="Number of failed predictions")
    
    # Resource usage
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    
    # Cost metrics (estimated)
    estimated_cost: float = Field(default=0.0, description="Estimated monetary cost in USD")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    
    # Quality metrics
    accuracy: float = Field(default=0.0, description="Accuracy score (0-1)")
    error_rate: float = Field(default=0.0, description="Error rate (0-1)")
    
    # Custom metrics
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Additional metadata
    additional_data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class MetricCollector:
    """Collects and manages performance metrics for DSPY programs.
    
    This class provides comprehensive metrics collection including execution time,
    resource usage, cost estimation, accuracy tracking, and custom metrics. It integrates
    seamlessly with the GEPA framework's ExecutionTrace system.
    """
    
    def __init__(self, 
                 track_resources: bool = True,
                 track_costs: bool = True,
                 custom_metrics: Optional[Dict[str, Callable]] = None,
                 cost_per_token: float = 0.00002):  # $0.02 per 1K tokens
        """Initialize the metric collector.
        
        Args:
            track_resources: Whether to track CPU/memory usage
            track_costs: Whether to track estimated costs
            custom_metrics: Dictionary of custom metric functions
            cost_per_token: Cost per token for cost estimation
        """
        self.track_resources = track_resources
        self.track_costs = track_costs
        self.custom_metrics = custom_metrics or {}
        self.cost_per_token = cost_per_token
        
        # Resource monitoring
        self._resource_monitor = None
        self._monitoring_active = False
        self._resource_data = ResourceUsage()
        
        if self.track_resources:
            self._start_resource_monitoring()
    
    def evaluate_program(self, 
                        program: "dspy.Module",
                        eval_data: List[Dict[str, Any]],
                        candidate: Optional[Candidate] = None,
                        timeout: Optional[float] = None) -> DSPYMetrics:
        """Evaluate a DSPY program and collect comprehensive metrics.
        
        Args:
            program: DSPY program to evaluate
            eval_data: Evaluation dataset
            candidate: Optional GEPA candidate to update with metrics
            timeout: Optional timeout for evaluation
            
        Returns:
            Comprehensive metrics object
            
        Raises:
            ImportError: If DSPY is not available
            ValueError: If program is not a valid DSPY module
        """
        # Runtime validation
        if not DSPY_AVAILABLE:
            raise ImportError("DSPY is required but not available")
        
        # Additional runtime check if we have a way to validate DSPY modules
        if hasattr(program, '__class__') and not self._is_likely_dspy_module(program):
            # This is a soft check - we don't want to be too strict since we might
            # be working with mock objects or other compatible implementations
            pass
        
        start_time = time.time()
        
        # Reset resource monitoring
        if self.track_resources:
            self._reset_resource_tracking()
        
        # Initialize metrics
        metrics = DSPYMetrics(
            execution_time=0.0,
            success=False,
            total_predictions=len(eval_data),
            successful_predictions=0,
            failed_predictions=0
        )
        
        try:
            # Evaluate program on each example
            results = []
            total_tokens = {"input": 0, "output": 0, "total": 0}
            
            for i, example in enumerate(eval_data):
                example_start = time.time()
                
                try:
                    # Execute program
                    result = program(**example)
                    results.append(result)
                    metrics.successful_predictions += 1
                    
                    # Estimate token usage (simplified)
                    example_tokens = self._estimate_token_usage(example, result)
                    for key, value in example_tokens.items():
                        total_tokens[key] = total_tokens.get(key, 0) + value
                    
                except Exception as e:
                    results.append({"error": str(e), "example": i})
                    metrics.failed_predictions += 1
                
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
            
            # Calculate final metrics
            metrics.execution_time = time.time() - start_time
            metrics.success = True
            metrics.token_usage = total_tokens
            
            # Calculate accuracy (simplified - assumes ground truth in eval_data)
            metrics.accuracy = self._calculate_accuracy(results, eval_data)
            metrics.error_rate = 1.0 - metrics.accuracy
            
            # Calculate cost
            if self.track_costs:
                metrics.estimated_cost = total_tokens["total"] * self.cost_per_token
            
            # Collect resource usage
            if self.track_resources:
                metrics.resource_usage = self._get_current_resource_usage()
            
            # Calculate custom metrics
            for metric_name, metric_func in self.custom_metrics.items():
                try:
                    metric_value = metric_func(results, eval_data, program)
                    metrics.custom_metrics[metric_name] = float(metric_value)
                except Exception as e:
                    metrics.custom_metrics[metric_name] = 0.0
            
            # Additional data
            metrics.additional_data = {
                "num_examples": len(eval_data),
                "program_class": program.__class__.__name__,
                "has_signature": hasattr(program, 'signature'),
                "results_summary": self._summarize_results(results)
            }
            
        except Exception as e:
            # Handle evaluation failure
            metrics.execution_time = time.time() - start_time
            metrics.success = False
            metrics.additional_data["error"] = str(e)
        
        # Update candidate if provided
        if candidate:
            self._update_candidate_with_metrics(candidate, metrics)
        
        return metrics
    
    @contextmanager
    def collect_execution_metrics(self, candidate: Optional[Candidate] = None):
        """Context manager for collecting metrics during program execution.
        
        Args:
            candidate: Optional candidate to update with execution trace
            
        Yields:
            Metrics collection context
        """
        start_time = time.time()
        
        if self.track_resources:
            self._reset_resource_tracking()
        
        try:
            yield self
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            # Create execution trace
            execution_time = time.time() - start_time
            
            trace = ExecutionTrace(
                timestamp=datetime.now(),
                execution_time=execution_time,
                success=success,
                error=error
            )
            
            if self.track_resources:
                trace.metrics["resource_usage"] = self._get_current_resource_usage().__dict__
            
            if candidate:
                candidate.add_execution_trace(trace)
    
    def batch_evaluate(self,
                      candidates: List[Candidate],
                      eval_data: List[Dict[str, Any]],
                      max_workers: int = 4) -> List[DSPYMetrics]:
        """Evaluate multiple candidates in parallel.
        
        Args:
            candidates: List of GEPA candidates to evaluate
            eval_data: Evaluation dataset
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of metrics for each candidate
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        metrics_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit evaluation tasks
            future_to_candidate = {
                executor.submit(self._evaluate_single_candidate, candidate, eval_data): candidate
                for candidate in candidates
            }
            
            # Collect results
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    metrics = future.result()
                    metrics_list.append(metrics)
                except Exception as e:
                    # Create error metrics
                    error_metrics = DSPYMetrics(
                        execution_time=0.0,
                        success=False,
                        additional_data={"error": str(e)}
                    )
                    metrics_list.append(error_metrics)
        
        return metrics_list
    
    def _evaluate_single_candidate(self, candidate: Candidate, eval_data: List[Dict[str, Any]]) -> DSPYMetrics:
        """Evaluate a single candidate (used in parallel evaluation)."""
        try:
            # Import here to avoid circular imports
            from .dspy_adapter import DSPYAdapter
            
            adapter = DSPYAdapter()
            program = adapter.candidate_to_dspy(candidate)
            
            return self.evaluate_program(program, eval_data, candidate)
            
        except Exception as e:
            return DSPYMetrics(
                execution_time=0.0,
                success=False,
                additional_data={"error": str(e)}
            )
    
    def _start_resource_monitoring(self) -> None:
        """Start background resource monitoring."""
        def monitor_resources():
            process = psutil.Process()
            
            while self._monitoring_active:
                try:
                    # CPU and memory
                    cpu_percent = process.cpu_percent()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    memory_percent = process.memory_percent()
                    
                    # Update peak memory
                    self._resource_data.peak_memory_mb = max(
                        self._resource_data.peak_memory_mb, memory_mb
                    )
                    
                    # Update current values
                    self._resource_data.cpu_percent = cpu_percent
                    self._resource_data.memory_mb = memory_mb
                    self._resource_data.memory_percent = memory_percent
                    
                    # Disk I/O
                    io_counters = process.io_counters()
                    self._resource_data.disk_io_read_mb = io_counters.read_bytes / 1024 / 1024
                    self._resource_data.disk_io_write_mb = io_counters.write_bytes / 1024 / 1024
                    
                    # Network I/O
                    net_io = psutil.net_io_counters()
                    self._resource_data.network_io_recv_mb = net_io.bytes_recv / 1024 / 1024
                    self._resource_data.network_io_sent_mb = net_io.bytes_sent / 1024 / 1024
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                
                time.sleep(0.5)  # Monitor every 500ms
        
        self._monitoring_active = True
        self._resource_monitor = threading.Thread(target=monitor_resources, daemon=True)
        self._resource_monitor.start()
    
    def _reset_resource_tracking(self) -> None:
        """Reset resource tracking for new measurement."""
        self._resource_data = ResourceUsage()
        
        # Reset process counters
        try:
            process = psutil.Process()
            process.io_counters()  # This resets the counters
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    def _get_current_resource_usage(self) -> ResourceUsage:
        """Get current resource usage snapshot."""
        return self._resource_data
    
    def _estimate_token_usage(self, input_data: Dict[str, Any], output_data: Any) -> Dict[str, int]:
        """Estimate token usage for input/output data.
        
        This is a simplified estimation. In practice, you'd use actual tokenizer
        counts from the LLM provider.
        """
        # Rough estimation: ~4 characters per token
        input_text = str(input_data)
        output_text = str(output_data)
        
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4
        total_tokens = input_tokens + output_tokens
        
        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        }
    
    def _calculate_accuracy(self, results: List[Dict[str, Any]], eval_data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of results.
        
        This is a simplified implementation. Real accuracy calculation depends
        on the specific task and evaluation criteria.
        """
        if not results:
            return 0.0
        
        # For now, just calculate success rate
        successful = sum(1 for r in results if "error" not in r)
        return successful / len(results)
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of evaluation results."""
        summary = {
            "total_results": len(results),
            "successful_results": sum(1 for r in results if "error" not in r),
            "failed_results": sum(1 for r in results if "error" in r),
            "has_errors": any("error" in r for r in results)
        }
        
        # Sample of result types (if structured)
        if results and "error" not in results[0]:
            first_result = results[0]
            if isinstance(first_result, dict):
                summary["result_keys"] = list(first_result.keys())
                summary["result_types"] = {
                    key: type(value).__name__ for key, value in first_result.items()
                }
        
        return summary
    
    def _update_candidate_with_metrics(self, candidate: Candidate, metrics: DSPYMetrics) -> None:
        """Update candidate with collected metrics."""
        # Add fitness scores based on metrics
        candidate.add_fitness_score("accuracy", metrics.accuracy)
        candidate.add_fitness_score("efficiency", 1.0 / (1.0 + metrics.execution_time))
        
        if self.track_costs:
            candidate.add_fitness_score("cost", 1.0 / (1.0 + metrics.estimated_cost))
        
        # Add custom metrics as fitness scores
        for metric_name, metric_value in metrics.custom_metrics.items():
            candidate.add_fitness_score(metric_name, metric_value)
        
        # Update metadata
        candidate.metadata.update({
            "last_evaluation": metrics.timestamp.isoformat(),
            "execution_time": metrics.execution_time,
            "total_predictions": metrics.total_predictions,
            "success_rate": metrics.accuracy,
            "estimated_cost": metrics.estimated_cost,
        })
    
    def _is_likely_dspy_module(self, program: Any) -> bool:
        """Check if an object is likely a DSPY module without requiring DSPY import.
        
        This is a runtime check that looks for common DSPY patterns without
        requiring the dspy module to be imported for type checking.
        """
        if not DSPY_AVAILABLE:
            return False
        
        try:
            # If dspy is available, use the proper check
            return isinstance(program, dspy.Module)
        except (AttributeError, TypeError):
            # Fallback: check for common DSPY attributes
            return (hasattr(program, 'forward') or 
                   hasattr(program, 'predict') or
                   hasattr(program, 'generate') or
                   (hasattr(program, '__class__') and 
                    'dspy' in str(program.__class__.__module__).lower()))
    
    def __del__(self):
        """Cleanup when collector is destroyed."""
        self._monitoring_active = False
        if self._resource_monitor and self._resource_monitor.is_alive():
            self._resource_monitor.join(timeout=1.0)