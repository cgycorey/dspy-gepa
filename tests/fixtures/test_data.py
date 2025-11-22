"""Test fixtures and data for dspy-gepa tests."""

from __future__ import annotations

import time
from typing import Any, Dict, List
from dataclasses import dataclass

from src.dspy_gepa.core.interfaces import (
    EvaluationResult, ObjectiveEvaluation, SolutionMetadata,
    OptimizationDirection
)


@dataclass
class MockProcess:
    """Mock psutil.Process for testing."""
    
    def __init__(self, cpu_percent: float = 50.0, memory_percent: float = 30.0, memory_rss: int = 1024*1024*100):
        self._cpu_percent = cpu_percent
        self._memory_percent = memory_percent
        self._memory_rss = memory_rss
    
    def cpu_percent(self) -> float:
        return self._cpu_percent
    
    def memory_percent(self) -> float:
        return self._memory_percent
    
    def memory_info(self) -> 'MockMemoryInfo':
        return MockMemoryInfo(self._memory_rss)


@dataclass
class MockMemoryInfo:
    """Mock memory info for testing."""
    
    def __init__(self, rss: int):
        self.rss = rss


def create_sample_evaluation_results(count: int = 10) -> List[EvaluationResult]:
    """Create sample evaluation results for testing."""
    results = []
    
    for i in range(count):
        solution_id = f"solution_{i:03d}"
        
        # Create objectives with varied scores
        objectives = {
            "accuracy": ObjectiveEvaluation(
                objective_name="accuracy",
                score=0.5 + (i % 5) * 0.1,  # 0.5 to 0.9
                direction=OptimizationDirection.MAXIMIZE,
                evaluation_time=0.1 + i * 0.01
            ),
            "efficiency": ObjectiveEvaluation(
                objective_name="efficiency", 
                score=1.0 - (i % 4) * 0.2,  # 1.0 to 0.2
                direction=OptimizationDirection.MINIMIZE,
                evaluation_time=0.05 + i * 0.005
            ),
            "complexity": ObjectiveEvaluation(
                objective_name="complexity",
                score=10 + (i % 3) * 5,  # 10 to 20
                direction=OptimizationDirection.MINIMIZE,
                evaluation_time=0.02 + i * 0.001
            )
        }
        
        # Calculate overall score (weighted average)
        overall_score = (
            objectives["accuracy"].score * 0.5 +
            (1.0 - objectives["efficiency"].score) * 0.3 +  # Invert for minimization
            (20.0 - objectives["complexity"].score) / 10.0 * 0.2  # Normalize and invert
        )
        
        metadata = SolutionMetadata(
            generation=i // 3,
            parent_ids=[f"solution_{max(0, i-1):03d}"],
            mutation_type="test_mutation",
            evaluation_time=objectives["accuracy"].evaluation_time,
            resource_usage={"cpu": 50.0 + i, "memory": 30.0 + i * 0.5}
        )
        
        results.append(EvaluationResult(
            solution_id=solution_id,
            objectives=objectives,
            overall_score=overall_score,
            evaluation_time=sum(obj.evaluation_time for obj in objectives.values()),
            metadata=metadata
        ))
    
    return results


def create_sample_optimization_state(generation: int = 5) -> Dict[str, Any]:
    """Create sample optimization state for testing."""
    population = create_sample_evaluation_results(20)
    frontier = population[:10]  # First 10 as frontier
    
    return {
        "generation": generation,
        "population": population,
        "frontier": frontier,
        "population_size": len(population),
        "frontier_size": len(frontier),
        "start_time": time.time() - 300,  # Started 5 minutes ago
        "evaluation_count": len(population) * generation,
        "metrics": {
            "hypervolume": 15.5 + generation * 0.5,
            "pairwise_distance": 2.5 - generation * 0.1,
            "diversity_score": 0.8 - generation * 0.05,
            "convergence_score": generation * 0.15,
            "best_accuracy": max(population, key=lambda p: p.get_objective_score("accuracy")).get_objective_score("accuracy"),
            "best_efficiency": min(population, key=lambda p: p.get_objective_score("efficiency")).get_objective_score("efficiency"),
            "prev_best_accuracy": 0.7 + (generation - 1) * 0.02,
            "prev_best_efficiency": 0.5 + (generation - 1) * 0.03
        },
        "resource_usage": {
            "cpu_percent": 45.0 + generation * 2.0,
            "memory_percent": 60.0 + generation * 1.0,
            "memory_used_mb": 1024.0 + generation * 50.0,
            "api_calls": 100 + generation * 20
        }
    }


def create_progress_data(generations: int = 20) -> Dict[str, List[float]]:
    """Create sample progress data for testing."""
    import random
    
    data = {
        "best_accuracy": [],
        "avg_efficiency": [], 
        "hypervolume": [],
        "diversity": []
    }
    
    for i in range(generations):
        # Simulate improvement with noise
        data["best_accuracy"].append(0.6 + i * 0.015 + random.uniform(-0.01, 0.01))
        data["avg_efficiency"].append(0.8 - i * 0.01 + random.uniform(-0.005, 0.005))
        data["hypervolume"].append(10.0 + i * 0.3 + random.uniform(-0.1, 0.1))
        data["diversity"].append(1.0 - i * 0.02 + random.uniform(-0.01, 0.01))
    
    return data


# Sample checkpoint data
SAMPLE_CHECKPOINT_STATE = {
    "generation": 10,
    "population": [
        {"solution_id": "sol_001", "score": 0.85, "objectives": {"accuracy": 0.9, "efficiency": 0.8}},
        {"solution_id": "sol_002", "score": 0.82, "objectives": {"accuracy": 0.85, "efficiency": 0.79}}
    ],
    "best_solution": {"solution_id": "sol_001", "score": 0.85},
    "metrics": {"hypervolume": 15.5, "diversity": 0.75},
    "parameters": {"mutation_rate": 0.1, "population_size": 50}
}


# Resource limit configurations
RESOURCE_LIMITS = {
    "conservative": {"cpu_percent": 70.0, "memory_percent": 80.0, "api_calls": 1000},
    "aggressive": {"cpu_percent": 90.0, "memory_percent": 95.0, "api_calls": 5000},
    "minimal": {"cpu_percent": 50.0, "memory_percent": 60.0, "api_calls": 500}
}


# Convergence detector configurations
CONVERGENCE_CONFIGS = {
    "pareto_stability": {"stability_threshold": 0.1, "window_size": 10},
    "hypervolume": {"improvement_threshold": 0.01, "window_size": 5},
    "diversity": {"diversity_threshold": 0.05, "window_size": 5},
    "improvement_plateau": {"plateau_threshold": 0.001, "window_size": 10}
}


# Optimal stopping estimator configurations
STOPPING_CONFIGS = {
    "statistical_trend": {"window_size": 10, "trend_threshold": 0.0},
    "resource_bounded": {"max_runtime_seconds": 3600, "max_memory_percent": 90.0, "max_evaluations": 10000},
    "diminishing_returns": {"window_size": 5, "improvement_threshold": 0.01}
}
