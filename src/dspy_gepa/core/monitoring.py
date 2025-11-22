"""Monitoring framework for multi-objective GEPA optimization.

This module provides comprehensive monitoring capabilities including resource monitoring,
checkpoint management, optimization logging, and performance tracking.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import json
import pickle
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import hashlib

from .interfaces import (
    ResourceMonitor, CheckpointManager, OptimizationLogger,
    EvaluationResult, ObjectiveEvaluation
)

# Import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    api_calls: int = 0
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class SystemResourceMonitor(ResourceMonitor):
    """Concrete implementation of ResourceMonitor using psutil."""
    
    def __init__(self, monitoring_interval: float = 1.0, max_history: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.max_history = max_history
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Resource history and tracking
        self._resource_history: deque = deque(maxlen=max_history)
        self._api_calls_count = 0
        self._api_calls_lock = threading.Lock()
        self._custom_metrics: Dict[str, Any] = {}
        self._custom_metrics_lock = threading.Lock()
        
        # Process handle
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        if self._monitoring:
            return
            
        if not PSUTIL_AVAILABLE:
            logging.warning("Resource monitoring disabled: psutil not available")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated resource usage data."""
        if not self._monitoring:
            return {}
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        aggregates = self._calculate_aggregates()
        self._resource_history.clear()
        return aggregates
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage snapshot."""
        if not PSUTIL_AVAILABLE or not self._process:
            return {}
        
        snapshot = self._take_snapshot()
        return {
            "timestamp": snapshot.timestamp, "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent, "memory_used_mb": snapshot.memory_used_mb,
            "api_calls": self._api_calls_count, "custom_metrics": dict(self._custom_metrics)
        }
    
    def check_resource_limits(self, limits: Dict[str, Any]) -> bool:
        """Check if current usage exceeds specified limits."""
        current = self.get_current_usage()
        
        for metric, limit in limits.items():
            if metric in current and current[metric] > limit:
                return True
        
        return False
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.wait(self.monitoring_interval):
            try:
                snapshot = self._take_snapshot()
                self._resource_history.append(snapshot)
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a resource usage snapshot."""
        if not self._process:
            return ResourceSnapshot(timestamp=time.time(), cpu_percent=0.0, memory_percent=0.0, memory_used_mb=0.0)
        
        try:
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            cpu_percent = self._process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            memory_info, memory_percent, cpu_percent = None, 0.0, 0.0
        
        return ResourceSnapshot(
            timestamp=time.time(), cpu_percent=cpu_percent, memory_percent=memory_percent,
            memory_used_mb=memory_info.rss / 1024 / 1024 if memory_info else 0.0,
            api_calls=self._api_calls_count, custom_metrics=dict(self._custom_metrics)
        )
    
    def _calculate_aggregates(self) -> Dict[str, Any]:
        """Calculate aggregated resource usage statistics."""
        if not self._resource_history:
            return {}
        
        cpu_values = [s.cpu_percent for s in self._resource_history]
        memory_values = [s.memory_percent for s in self._resource_history]
        duration = (self._resource_history[-1].timestamp - self._resource_history[0].timestamp
                   if len(self._resource_history) > 1 else 0)
        
        return {
            "total_snapshots": len(self._resource_history),
            "duration_seconds": duration,
            "cpu": {"avg": sum(cpu_values) / len(cpu_values), "max": max(cpu_values), "min": min(cpu_values)},
            "memory": {"avg": sum(memory_values) / len(memory_values), "max": max(memory_values), "min": min(memory_values)},
            "total_api_calls": self._api_calls_count
        }
    
    def increment_api_calls(self) -> None:
        """Increment API call counter."""
        with self._api_calls_lock:
            self._api_calls_count += 1
    
    def set_custom_metric(self, name: str, value: Any) -> None:
        """Set a custom metric value."""
        with self._custom_metrics_lock:
            self._custom_metrics[name] = value


class FileCheckpointManager(CheckpointManager):
    """Concrete implementation of CheckpointManager with file-based persistence."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_format: str = "pickle",
        max_checkpoints: int = 50,
        auto_cleanup: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.checkpoint_format = checkpoint_format.lower()
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self._metadata_file = self.checkpoint_dir / "metadata.json"
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()
    
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_id: Optional[str] = None) -> str:
        """Save optimization state to checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = self._generate_checkpoint_id(state)
        
        file_path = self._get_checkpoint_path(checkpoint_id)
        
        # Save state
        if self.checkpoint_format == "json":
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        else:  # pickle format
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
        
        # Update metadata
        self._metadata[checkpoint_id] = {
            "timestamp": time.time(), "file_path": str(file_path),
            "file_size": file_path.stat().st_size, "format": self.checkpoint_format,
            "generation": state.get("generation", 0), "population_size": state.get("population_size", 0)
        }
        self._save_metadata()
        
        # Auto cleanup if needed
        if self.auto_cleanup:
            self._cleanup_old_checkpoints()
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load optimization state from checkpoint."""
        if checkpoint_id not in self._metadata:
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        file_path = Path(self._metadata[checkpoint_id]["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file {file_path} not found")
        
        metadata = self._metadata[checkpoint_id]
        
        # Load state
        if metadata["format"] == "json":
            with open(file_path, 'r') as f:
                return json.load(f)
        else:  # pickle format
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        return sorted(self._metadata.keys(), key=lambda x: self._metadata[x].get("timestamp", 0), reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        if checkpoint_id not in self._metadata:
            return False
        
        file_path = Path(self._metadata[checkpoint_id]["file_path"])
        try:
            if file_path.exists():
                file_path.unlink()
            del self._metadata[checkpoint_id]
            self._save_metadata()
            return True
        except Exception as e:
            logging.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    def _generate_checkpoint_id(self, state: Dict[str, Any]) -> str:
        """Generate unique checkpoint ID based on state."""
        state_hash = hashlib.md5(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_{timestamp}_{state_hash}"
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        extension = "json" if self.checkpoint_format == "json" else "pkl"
        return self.checkpoint_dir / f"{checkpoint_id}.{extension}"
    
    def _load_metadata(self) -> None:
        """Load checkpoint metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logging.error(f"Error loading checkpoint metadata: {e}")
                self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving checkpoint metadata: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        if len(self._metadata) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and remove oldest
        checkpoints_to_remove = sorted(self._metadata.keys(),
                                     key=lambda x: self._metadata[x].get("timestamp", 0))[:len(self._metadata) - self.max_checkpoints]
        
        for checkpoint_id in checkpoints_to_remove:
            self.delete_checkpoint(checkpoint_id)


class StructuredOptimizationLogger(OptimizationLogger):
    """Concrete implementation of OptimizationLogger with structured logging."""
    
    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        max_log_entries: int = 10000,
        include_console_output: bool = True
    ):
        self.log_dir = Path(log_dir or "logs")
        self.log_level = log_level.upper()
        self.max_log_entries = max_log_entries
        self.include_console_output = include_console_output
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._logs: deque = deque(maxlen=max_log_entries)
        self._log_file = self.log_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Setup console logging if enabled
        if self.include_console_output:
            self._setup_console_logging()
    
    def log_generation_start(self, generation: int, population_size: int) -> None:
        """Log the start of a generation."""
        entry = {
            "timestamp": time.time(),
            "type": "generation_start",
            "generation": generation,
            "population_size": population_size
        }
        self._add_log_entry(entry)
    
    def log_generation_end(self, generation: int, metrics: Dict[str, Any]) -> None:
        """Log the end of a generation."""
        entry = {
            "timestamp": time.time(),
            "type": "generation_end",
            "generation": generation,
            "metrics": metrics
        }
        self._add_log_entry(entry)
    
    def log_evaluation(self, solution_id: str, results: EvaluationResult) -> None:
        """Log solution evaluation results."""
        objectives_data = {
            name: {
                "score": eval_obj.score,
                "direction": eval_obj.direction.value,
                "evaluation_time": eval_obj.evaluation_time
            }
            for name, eval_obj in results.objectives.items()
        }
        
        entry = {
            "timestamp": time.time(),
            "type": "evaluation",
            "solution_id": solution_id,
            "objectives": objectives_data,
            "overall_score": results.overall_score,
            "evaluation_time": results.evaluation_time
        }
        self._add_log_entry(entry)
    
    def log_mutation(self, solution_id: str, mutation_type: str, result_id: str) -> None:
        """Log mutation operation."""
        entry = {
            "timestamp": time.time(),
            "type": "mutation",
            "solution_id": solution_id,
            "mutation_type": mutation_type,
            "result_id": result_id
        }
        self._add_log_entry(entry)
    
    def log_convergence(self, generation: int, convergence_score: float) -> None:
        """Log convergence information."""
        entry = {
            "timestamp": time.time(),
            "type": "convergence",
            "generation": generation,
            "convergence_score": convergence_score
        }
        self._add_log_entry(entry)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization logs."""
        if level is None:
            return list(self._logs)
        
        # Filter by log level (simplified - in practice would map to severity)
        return [log for log in self._logs if log.get("level", "INFO").upper() == level.upper()]
    
    def _add_log_entry(self, entry: Dict[str, Any]) -> None:
        """Add log entry to storage and file."""
        self._logs.append(entry)
        # Write to file
        try:
            with open(self._log_file, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            logging.error(f"Error writing to log file: {e}")
        
        if self.include_console_output:
            self._log_to_console(entry)
    
    def _setup_console_logging(self) -> None:
        """Setup console logging."""
        pass
    
    def _log_to_console(self, entry: Dict[str, Any]) -> None:
        """Log entry to console."""
        timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
        entry_type = entry["type"].upper()
        
        if entry_type == "GENERATION_START":
            msg = f"[{timestamp}] Generation {entry['generation']} started (pop: {entry['population_size']})"
        elif entry_type == "GENERATION_END":
            metrics = entry.get('metrics', {})
            msg = f"[{timestamp}] Generation {entry['generation']} completed"
            if 'best_score' in metrics:
                msg += f" (best: {metrics['best_score']:.3f})"
        elif entry_type == "EVALUATION":
            score = entry.get('overall_score', 'N/A')
            msg = f"[{timestamp}] Evaluated {entry['solution_id']} (score: {score})"
        elif entry_type == "CONVERGENCE":
            msg = f"[{timestamp}] Convergence at generation {entry['generation']}: {entry['convergence_score']:.3f}"
        else:
            msg = f"[{timestamp}] {entry_type}: {entry}"
        
        print(msg)


class MonitoringFramework:
    """Main orchestrator for all monitoring activities."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        resource_monitoring: bool = True,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        logger_config: Optional[Dict[str, Any]] = None,
        resource_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize components
        self.resource_monitor = (
            SystemResourceMonitor(**(resource_config or {}))
            if resource_monitoring else None
        )
        
        self.checkpoint_manager = FileCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            **(checkpoint_config or {})
        )
        
        self.optimization_logger = StructuredOptimizationLogger(
            log_dir=log_dir,
            **(logger_config or {})
        )
        
        # Monitoring state
        self._monitoring_active = False
        self._performance_history: List[Dict[str, Any]] = []
    
    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        logging.info("Monitoring framework started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop all monitoring and return aggregated data."""
        if not self._monitoring_active:
            return {}
        
        self._monitoring_active = False
        
        # Stop resource monitoring
        resource_data = {}
        if self.resource_monitor:
            resource_data = self.resource_monitor.stop_monitoring()
        
        result = {
            "resource_usage": resource_data,
            "performance_history": self._performance_history.copy(),
            "total_checkpoints": len(self.checkpoint_manager.list_checkpoints()),
            "total_logs": len(self.optimization_logger._logs)
        }
        
        logging.info("Monitoring framework stopped")
        return result
    
    def create_checkpoint(self, state: Dict[str, Any]) -> str:
        """Create optimization checkpoint."""
        return self.checkpoint_manager.save_checkpoint(state)
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load optimization checkpoint."""
        return self.checkpoint_manager.load_checkpoint(checkpoint_id)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get current monitoring summary."""
        summary = {
            "monitoring_active": self._monitoring_active,
            "resource_monitoring": self.resource_monitor is not None,
            "current_usage": self.resource_monitor.get_current_usage() if self.resource_monitor else {},
            "checkpoint_count": len(self.checkpoint_manager.list_checkpoints()),
            "log_count": len(self.optimization_logger._logs),
            "performance_snapshots": len(self._performance_history)
        }
        return summary
    
    def create_performance_checkpoint(self, generation: int, metrics: Dict[str, Any]) -> None:
        """Create performance checkpoint for tracking."""
        snapshot = {
            "generation": generation,
            "timestamp": time.time(),
            "metrics": metrics.copy(),
            "resource_usage": self.resource_monitor.get_current_usage() if self.resource_monitor else {}
        }
        self._performance_history.append(snapshot)
    
    # Delegate methods to maintain compatibility with interfaces
    def log_generation_start(self, generation: int, population_size: int) -> None:
        """Delegate to optimization logger."""
        self.optimization_logger.log_generation_start(generation, population_size)
    
    def log_generation_end(self, generation: int, metrics: Dict[str, Any]) -> None:
        """Delegate to optimization logger and create performance checkpoint."""
        self.optimization_logger.log_generation_end(generation, metrics)
        self.create_performance_checkpoint(generation, metrics)
    
    def log_evaluation(self, solution_id: str, results: EvaluationResult) -> None:
        """Delegate to optimization logger."""
        self.optimization_logger.log_evaluation(solution_id, results)
    
    def log_mutation(self, solution_id: str, mutation_type: str, result_id: str) -> None:
        """Delegate to optimization logger."""
        self.optimization_logger.log_mutation(solution_id, mutation_type, result_id)
    
    def log_convergence(self, generation: int, convergence_score: float) -> None:
        """Delegate to optimization logger."""
        self.optimization_logger.log_convergence(generation, convergence_score)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Delegate to optimization logger."""
        return self.optimization_logger.get_logs(level)
    
    def track_progress(self, metrics: Dict[str, Any]) -> None:
        """Track optimization progress."""
        generation = metrics.get('generation', 0)
        self.log_generation_end(generation, metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        metrics = {}
        
        # Get resource metrics
        if self.resource_monitor:
            metrics['resource_usage'] = self.resource_monitor.get_current_usage()
        
        # Get performance history summary
        if hasattr(self, '_performance_history') and self._performance_history:
            recent_metrics = self._performance_history[-1]
            metrics['latest_performance'] = recent_metrics.get('metrics', {})
        
        # Get checkpoint info
        metrics['checkpoint_count'] = len(self.checkpoint_manager.list_checkpoints()) if self.checkpoint_manager else 0
        
        return metrics
    
    def reset(self) -> None:
        """Reset monitoring state."""
        # Reset resource monitor
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            if hasattr(self.resource_monitor, '_resource_history'):
                self.resource_monitor._resource_history.clear()
        
        # Reset performance history
        if hasattr(self, '_performance_history'):
            self._performance_history.clear()
        
        # Reset logger
        if self.optimization_logger:
            if hasattr(self.optimization_logger, '_log_entries'):
                self.optimization_logger._log_entries.clear()


# Factory function for easy setup
def create_monitoring_framework(
    enable_all: bool = True,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    **config_overrides
) -> MonitoringFramework:
    """Factory function to create monitoring framework with sensible defaults."""
    if not enable_all:
        return MonitoringFramework(
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            resource_monitoring=False
        )
    
    default_config = {
        "checkpoint_config": {
            "checkpoint_format": "pickle",
            "max_checkpoints": 50,
            "auto_cleanup": True
        },
        "logger_config": {
            "log_level": "INFO",
            "max_log_entries": 10000,
            "include_console_output": True
        },
        "resource_config": {
            "monitoring_interval": 2.0,
            "max_history": 500
        }
    }
    
    # Apply overrides
    for key, value in config_overrides.items():
        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
            default_config[key].update(value)
        else:
            default_config[key] = value
    
    return MonitoringFramework(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        resource_monitoring=True,
        **default_config
    )


# Additional classes for architecture compliance

class ConvergenceDetector:
    """Detects convergence in multi-objective optimization."""
    
    def __init__(self, window_size: int = 10, tolerance: float = 0.01):
        self.window_size = window_size
        self.tolerance = tolerance
        self.hypervolume_history = []
        self.frontier_history = []
    
    def detect_convergence(self, hypervolume: float, frontier_size: int) -> bool:
        """Detect if optimization has converged."""
        self.hypervolume_history.append(hypervolume)
        self.frontier_history.append(frontier_size)
        
        # Keep only recent history
        if len(self.hypervolume_history) > self.window_size:
            self.hypervolume_history.pop(0)
            self.frontier_history.pop(0)
        
        # Check convergence based on hypervolume stability
        if len(self.hypervolume_history) >= self.window_size:
            recent_hv = self.hypervolume_history[-self.window_size:]
            hv_change = max(recent_hv) - min(recent_hv)
            return hv_change < self.tolerance
        
        return False
    
    def check_stability(self, metrics: Dict[str, List[float]]) -> bool:
        """Check if metrics are stable over time."""
        for metric_name, values in metrics.items():
            if len(values) >= self.window_size:
                recent_values = values[-self.window_size:]
                change = max(recent_values) - min(recent_values)
                if change > self.tolerance:
                    return False
        return True
    
    def analyze_trends(self, metrics: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze trends in metrics."""
        trends = {}
        for metric_name, values in metrics.items():
            if len(values) >= 3:
                recent_slope = (values[-1] - values[-3]) / 2
                if abs(recent_slope) < self.tolerance:
                    trends[metric_name] = "stable"
                elif recent_slope > 0:
                    trends[metric_name] = "improving"
                else:
                    trends[metric_name] = "declining"
            else:
                trends[metric_name] = "insufficient_data"
        return trends


class ParetoFrontierVisualizer:
    """Visualizes Pareto frontier for multi-objective optimization."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or "visualizations"
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def plot_2d(self, frontier: List[EvaluationResult], obj1: str, obj2: str, 
                save_path: Optional[str] = None) -> str:
        """Create 2D Pareto frontier plot."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract objective values
            obj1_values = [eval_result.get_objective_score(obj1) for eval_result in frontier]
            obj2_values = [eval_result.get_objective_score(obj2) for eval_result in frontier]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(obj1_values, obj2_values, alpha=0.7, s=50)
            plt.xlabel(obj1)
            plt.ylabel(obj2)
            plt.title(f"Pareto Frontier: {obj1} vs {obj2}")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                save_path = f"{self.output_dir}/pareto_2d_{obj1}_{obj2}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            return save_path
            
        except ImportError:
            # Fallback to text-based representation
            return self._create_text_plot(frontier, obj1, obj2)
    
    def plot_3d(self, frontier: List[EvaluationResult], obj1: str, obj2: str, obj3: str,
                save_path: Optional[str] = None) -> str:
        """Create 3D Pareto frontier plot."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Extract objective values
            obj1_values = [eval_result.get_objective_score(obj1) for eval_result in frontier]
            obj2_values = [eval_result.get_objective_score(obj2) for eval_result in frontier]
            obj3_values = [eval_result.get_objective_score(obj3) for eval_result in frontier]
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(obj1_values, obj2_values, obj3_values, alpha=0.7, s=50)
            ax.set_xlabel(obj1)
            ax.set_ylabel(obj2)
            ax.set_zlabel(obj3)
            ax.set_title(f"Pareto Frontier: {obj1}, {obj2}, {obj3}")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                save_path = f"{self.output_dir}/pareto_3d_{obj1}_{obj2}_{obj3}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            return save_path
            
        except ImportError:
            return f"3D plot not available without matplotlib"
    
    def plot_parallel_coordinates(self, frontier: List[EvaluationResult], objectives: List[str],
                                  save_path: Optional[str] = None) -> str:
        """Create parallel coordinates plot."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from pandas.plotting import parallel_coordinates
            
            # Prepare data
            data = []
            for eval_result in frontier:
                row = {obj: eval_result.get_objective_score(obj) for obj in objectives}
                row['solution'] = eval_result.solution_id
                data.append(row)
            
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(12, 8))
            parallel_coordinates(df, 'solution', colormap='viridis')
            plt.title("Pareto Frontier - Parallel Coordinates")
            plt.xticks(rotation=45)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                save_path = f"{self.output_dir}/pareto_parallel_coordinates.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            return save_path
            
        except ImportError:
            return f"Parallel coordinates plot not available without required dependencies"
    
    def _create_text_plot(self, frontier: List[EvaluationResult], obj1: str, obj2: str) -> str:
        """Create text-based plot representation."""
        output_path = f"{self.output_dir}/pareto_2d_text.txt"
        
        with open(output_path, 'w') as f:
            f.write(f"Pareto Frontier: {obj1} vs {obj2}\n")
            f.write("=" * 50 + "\n")
            for i, eval_result in enumerate(frontier):
                obj1_val = eval_result.get_objective_score(obj1)
                obj2_val = eval_result.get_objective_score(obj2)
                f.write(f"Solution {i+1}: {obj1}={obj1_val:.4f}, {obj2}={obj2_val:.4f}\n")
        
        return output_path


class OptimalStoppingEstimator:
    """Estimates optimal stopping points for optimization."""
    
    def __init__(self, min_generations: int = 5, patience: int = 10):
        self.min_generations = min_generations
        self.patience = patience
        self.best_hypervolume = 0.0
        self.generations_without_improvement = 0
    
    def estimate_stopping_point(self, hypervolume_history: List[float], 
                                generation: int) -> bool:
        """Estimate if optimization should stop."""
        if generation < self.min_generations:
            return False
        
        current_hv = hypervolume_history[-1] if hypervolume_history else 0.0
        
        if current_hv > self.best_hypervolume:
            self.best_hypervolume = current_hv
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        
        return self.generations_without_improvement >= self.patience
    
    def calculate_returns(self, hypervolume_history: List[float]) -> Dict[str, float]:
        """Calculate diminishing returns metrics."""
        if len(hypervolume_history) < 2:
            return {"improvement_rate": 0.0, "diminishing_returns": False}
        
        # Calculate recent improvement rate
        recent_improvements = []
        window_size = min(5, len(hypervolume_history) - 1)
        
        for i in range(window_size):
            idx = -(i + 1)
            improvement = hypervolume_history[idx] - hypervolume_history[idx - 1]
            recent_improvements.append(improvement)
        
        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        
        # Check for diminishing returns
        if len(hypervolume_history) >= 10:
            early_avg = sum(hypervolume_history[:5]) / 5
            recent_avg = sum(hypervolume_history[-5:]) / 5
            diminishing_returns = (recent_avg - early_avg) < (early_avg * 0.1)
        else:
            diminishing_returns = False
        
        return {
            "improvement_rate": avg_improvement,
            "diminishing_returns": diminishing_returns
        }
    
    def predict_optimal_time(self, hypervolume_history: List[float], 
                           time_history: List[float]) -> Optional[float]:
        """Predict optimal stopping time based on trends."""
        if len(hypervolume_history) < 5 or len(time_history) < 5:
            return None
        
        # Simple linear extrapolation
        import numpy as np
        
        try:
            # Fit linear model to hypervolume vs time
            X = np.array(time_history).reshape(-1, 1)
            y = np.array(hypervolume_history)
            
            # Calculate slope
            slope = np.polyfit(time_history, hypervolume_history, 1)[0]
            
            # If slope is very small, we're near optimal
            if abs(slope) < 0.001:
                return time_history[-1]
            
            # Estimate when improvement will be minimal
            current_hv = hypervolume_history[-1]
            target_hv = current_hv + 0.01  # Small improvement threshold
            
            if slope > 0:
                estimated_time = (target_hv - current_hv) / slope + time_history[-1]
                return max(0, estimated_time)
            
        except:
            pass
        
        return None