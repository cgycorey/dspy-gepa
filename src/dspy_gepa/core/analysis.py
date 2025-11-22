"""Convergence detection and optimal stopping analysis for multi-objective GEPA.

This module provides concrete implementations of convergence detection strategies
and optimal stopping estimators to determine when optimization should terminate
based on various statistical and resource-based criteria.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from .interfaces import (
    ConvergenceDetector, OptimalStoppingEstimator,
    EvaluationResult, OptimizationDirection
)
from ..utils.logging import get_logger


_logger = get_logger(__name__)


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking convergence behavior."""
    frontier_stability: float = 0.0
    hypervolume_change: float = 0.0
    diversity_score: float = 0.0
    improvement_rate: float = 0.0
    trend_slope: float = 0.0
    resource_efficiency: float = 0.0
    
    def get_overall_score(self) -> float:
        """Get overall convergence score (0.0 = no convergence, 1.0 = fully converged)."""
        return statistics.mean([
            self.frontier_stability,
            self.hypervolume_change,
            self.diversity_score,
            self.improvement_rate
        ])


class ParetoStabilityDetector(ConvergenceDetector):
    """Detects convergence based on Pareto frontier stability.
    
    Analyzes how much the Pareto frontier changes over recent generations.
    A stable frontier suggests convergence.
    """
    
    def __init__(self, stability_threshold: float = 0.1, window_size: int = 10):
        """Initialize Pareto stability detector.
        
        Args:
            stability_threshold: Threshold for considering frontier stable (0.0-1.0)
            window_size: Number of recent generations to analyze
        """
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self._frontier_history: deque = deque(maxlen=window_size)
        _logger.debug(f"ParetoStabilityDetector initialized: threshold={stability_threshold}, window={window_size}")
    
    def has_converged(self, optimization_state: Dict[str, Any]) -> bool:
        """Check if optimization has converged based on frontier stability."""
        current_frontier = optimization_state.get('frontier', [])
        
        if not current_frontier:
            return False
        
        # Add current frontier to history
        frontier_signature = self._compute_frontier_signature(current_frontier)
        self._frontier_history.append(frontier_signature)
        
        # Need history to assess stability
        if len(self._frontier_history) < self.window_size:
            return False
        
        # Calculate stability metrics
        stability_score = self.get_convergence_score(optimization_state)
        
        _logger.debug(f"Pareto stability score: {stability_score:.3f} (threshold: {self.stability_threshold})")
        return stability_score >= self.stability_threshold
    
    def get_convergence_metrics(self) -> List[str]:
        """Get metrics used for convergence detection."""
        return ['frontier_stability', 'frontier_size', 'frontier_turnover']
    
    def get_convergence_score(self, optimization_state: Dict[str, Any]) -> float:
        """Get convergence score based on frontier stability."""
        if len(self._frontier_history) < 2:
            return 0.0
        
        # Calculate stability between consecutive frontiers
        similarities = []
        for i in range(1, len(self._frontier_history)):
            similarity = self._compute_similarity(
                self._frontier_history[i-1],
                self._frontier_history[i]
            )
            similarities.append(similarity)
        
        # Return average similarity as stability score
        return statistics.mean(similarities)
    
    def _compute_frontier_signature(self, frontier: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute a signature for the frontier to track changes."""
        if not frontier:
            return {'size': 0, 'objective_sums': {}}
        
        # Sum objective scores for a quick signature
        objective_sums = {}
        objective_counts = {}
        
        for solution in frontier:
            for obj_name, obj_eval in solution.objectives.items():
                if obj_name not in objective_sums:
                    objective_sums[obj_name] = 0.0
                    objective_counts[obj_name] = 0
                objective_sums[obj_name] += obj_eval.score
                objective_counts[obj_name] += 1
        
        # Normalize sums
        for obj_name in objective_sums:
            if objective_counts[obj_name] > 0:
                objective_sums[obj_name] /= objective_counts[obj_name]
        
        return {
            'size': len(frontier),
            'objective_sums': objective_sums,
            'solution_ids': {sol.solution_id for sol in frontier}
        }
    
    def _compute_similarity(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> float:
        """Compute similarity between two frontier signatures."""
        # Size similarity
        size_sim = 1.0 - abs(sig1['size'] - sig2['size']) / max(sig1['size'], sig2['size'], 1)
        
        # Objective similarity
        obj_sim = 1.0
        all_objectives = set(sig1['objective_sums'].keys()) | set(sig2['objective_sums'].keys())
        
        if all_objectives:
            obj_diffs = []
            for obj_name in all_objectives:
                sum1 = sig1['objective_sums'].get(obj_name, 0.0)
                sum2 = sig2['objective_sums'].get(obj_name, 0.0)
                max_sum = max(sum1, sum2, 1e-6)
                obj_diffs.append(abs(sum1 - sum2) / max_sum)
            
            obj_sim = 1.0 - statistics.mean(obj_diffs)
        
        # Solution overlap
        overlap = len(sig1['solution_ids'] & sig2['solution_ids'])
        union = len(sig1['solution_ids'] | sig2['solution_ids'])
        overlap_sim = overlap / union if union > 0 else 1.0
        
        # Combined similarity
        return (size_sim + obj_sim + overlap_sim) / 3.0


class HypervolumeConvergenceDetector(ConvergenceDetector):
    """Detects convergence based on hypervolume improvement rate.
    
    Tracks how much the hypervolume indicator changes over time.
    Diminishing improvements suggest convergence.
    """
    
    def __init__(self, improvement_threshold: float = 0.01, window_size: int = 5):
        """Initialize hypervolume convergence detector.
        
        Args:
            improvement_threshold: Minimum improvement rate to consider non-converged
            window_size: Number of recent generations to analyze
        """
        self.improvement_threshold = improvement_threshold
        self.window_size = window_size
        self._hypervolume_history: deque = deque(maxlen=window_size + 1)
        _logger.debug(f"HypervolumeConvergenceDetector initialized: threshold={improvement_threshold}, window={window_size}")
    
    def has_converged(self, optimization_state: Dict[str, Any]) -> bool:
        """Check if optimization has converged based on hypervolume changes."""
        # Get current hypervolume
        current_hv = self._calculate_current_hypervolume(optimization_state)
        if current_hv is None:
            return False
        
        self._hypervolume_history.append(current_hv)
        
        # Need history to assess convergence
        if len(self._hypervolume_history) < self.window_size + 1:
            return False
        
        # Calculate improvement rate
        convergence_score = self.get_convergence_score(optimization_state)
        
        _logger.debug(f"Hypervolume convergence score: {convergence_score:.3f} (threshold: {1.0 - self.improvement_threshold})")
        return convergence_score >= (1.0 - self.improvement_threshold)
    
    def get_convergence_metrics(self) -> List[str]:
        """Get metrics used for convergence detection."""
        return ['hypervolume', 'hypervolume_improvement', 'improvement_rate']
    
    def get_convergence_score(self, optimization_state: Dict[str, Any]) -> float:
        """Get convergence score based on hypervolume improvement rate."""
        if len(self._hypervolume_history) < 2:
            return 0.0
        
        # Calculate relative improvements
        improvements = []
        for i in range(1, len(self._hypervolume_history)):
            prev_hv = self._hypervolume_history[i-1]
            curr_hv = self._hypervolume_history[i]
            
            if prev_hv > 0:
                rel_improvement = (curr_hv - prev_hv) / prev_hv
                improvements.append(rel_improvement)
        
        if not improvements:
            return 1.0
        
        # Calculate average absolute improvement
        avg_improvement = statistics.mean(abs(imp) for imp in improvements)
        
        # Convert to convergence score (lower improvement = higher convergence)
        convergence_score = max(0.0, 1.0 - (avg_improvement / self.improvement_threshold))
        return min(1.0, convergence_score)
    
    def _calculate_current_hypervolume(self, optimization_state: Dict[str, Any]) -> Optional[float]:
        """Calculate current hypervolume from optimization state."""
        frontier = optimization_state.get('frontier', [])
        if not frontier:
            return None
        
        # Try to get hypervolume from metrics if already calculated
        metrics = optimization_state.get('metrics', {})
        if 'hypervolume' in metrics:
            return metrics['hypervolume']
        
        # Calculate hypervolume using simple approximation
        return self._approximate_hypervolume(frontier)
    
    def _approximate_hypervolume(self, frontier: List[EvaluationResult]) -> float:
        """Simple hypervolume approximation."""
        if not frontier:
            return 0.0
        
        # For 2D case, use simple polygon area approximation
        # Get all objective names
        all_objectives = set()
        for solution in frontier:
            all_objectives.update(solution.objectives.keys())
        
        if len(all_objectives) == 0:
            return 0.0
        
        # For simplicity, use sum of normalized objective scores
        total_score = 0.0
        for solution in frontier:
            solution_score = 0.0
            for obj_name in all_objectives:
                score = solution.get_objective_score(obj_name) or 0.0
                solution_score += score
            total_score += solution_score / len(all_objectives)
        
        return total_score / len(frontier) if frontier else 0.0


class DiversityConvergenceDetector(ConvergenceDetector):
    """Detects convergence based on solution diversity in the population.
    
    Low diversity suggests the optimization has converged to a region
    of the search space.
    """
    
    def __init__(self, diversity_threshold: float = 0.05, window_size: int = 5):
        """Initialize diversity convergence detector.
        
        Args:
            diversity_threshold: Minimum diversity to consider non-converged
            window_size: Number of recent generations to analyze
        """
        self.diversity_threshold = diversity_threshold
        self.window_size = window_size
        self._diversity_history: deque = deque(maxlen=window_size)
        _logger.debug(f"DiversityConvergenceDetector initialized: threshold={diversity_threshold}, window={window_size}")
    
    def has_converged(self, optimization_state: Dict[str, Any]) -> bool:
        """Check if optimization has converged based on population diversity."""
        current_diversity = self._calculate_diversity(optimization_state)
        
        if current_diversity is None:
            return False
        
        self._diversity_history.append(current_diversity)
        
        # Need history to assess convergence
        if len(self._diversity_history) < self.window_size:
            return False
        
        convergence_score = self.get_convergence_score(optimization_state)
        
        _logger.debug(f"Diversity convergence score: {convergence_score:.3f} (threshold: {1.0 - self.diversity_threshold})")
        return convergence_score >= (1.0 - self.diversity_threshold)
    
    def get_convergence_metrics(self) -> List[str]:
        """Get metrics used for convergence detection."""
        return ['population_diversity', 'objective_variance', 'pairwise_distance']
    
    def get_convergence_score(self, optimization_state: Dict[str, Any]) -> float:
        """Get convergence score based on population diversity."""
        if len(self._diversity_history) < 2:
            return 0.0
        
        # Calculate average diversity over recent window
        avg_diversity = statistics.mean(self._diversity_history)
        
        # Convert to convergence score (lower diversity = higher convergence)
        convergence_score = max(0.0, 1.0 - (avg_diversity / self.diversity_threshold))
        return min(1.0, convergence_score)
    
    def _calculate_diversity(self, optimization_state: Dict[str, Any]) -> Optional[float]:
        """Calculate population diversity."""
        population = optimization_state.get('population', [])
        if len(population) < 2:
            return None
        
        # Get diversity metrics from state if available
        metrics = optimization_state.get('metrics', {})
        if 'pairwise_distance' in metrics:
            return metrics['pairwise_distance']
        
        # Calculate pairwise distance
        total_distance = 0.0
        count = 0
        
        for i, sol1 in enumerate(population):
            for sol2 in enumerate(population[i+1:], i+1):
                distance = self._objective_distance(sol1, sol2)
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _objective_distance(self, r1: EvaluationResult, r2: EvaluationResult) -> float:
        """Calculate Euclidean distance between two results in objective space."""
        obj_names = set(r1.objectives.keys()) | set(r2.objectives.keys())
        
        distance_squared = 0.0
        for obj_name in obj_names:
            score1 = r1.get_objective_score(obj_name) or 0.0
            score2 = r2.get_objective_score(obj_name) or 0.0
            distance_squared += (score1 - score2) ** 2
        
        return distance_squared ** 0.5


class ImprovementPlateauDetector(ConvergenceDetector):
    """Detects convergence based on improvement rate plateau.
    
    Analyzes the rate of improvement in objective scores over time.
    A plateau suggests convergence.
    """
    
    def __init__(self, plateau_threshold: float = 0.001, window_size: int = 10):
        """Initialize improvement plateau detector.
        
        Args:
            plateau_threshold: Minimum improvement rate to consider non-converged
            window_size: Number of recent generations to analyze
        """
        self.plateau_threshold = plateau_threshold
        self.window_size = window_size
        self._improvement_history: deque = deque(maxlen=window_size)
        _logger.debug(f"ImprovementPlateauDetector initialized: threshold={plateau_threshold}, window={window_size}")
    
    def has_converged(self, optimization_state: Dict[str, Any]) -> bool:
        """Check if optimization has converged based on improvement plateau."""
        current_improvement = self._calculate_improvement(optimization_state)
        
        if current_improvement is None:
            return False
        
        self._improvement_history.append(current_improvement)
        
        # Need history to assess plateau
        if len(self._improvement_history) < self.window_size:
            return False
        
        convergence_score = self.get_convergence_score(optimization_state)
        
        _logger.debug(f"Improvement plateau score: {convergence_score:.3f} (threshold: {1.0 - self.plateau_threshold})")
        return convergence_score >= (1.0 - self.plateau_threshold)
    
    def get_convergence_metrics(self) -> List[str]:
        """Get metrics used for convergence detection."""
        return ['improvement_rate', 'best_objective_scores', 'score_trends']
    
    def get_convergence_score(self, optimization_state: Dict[str, Any]) -> float:
        """Get convergence score based on improvement plateau."""
        if len(self._improvement_history) < 2:
            return 0.0
        
        # Calculate average improvement rate
        avg_improvement = statistics.mean(self._improvement_history)
        
        # Convert to convergence score (lower improvement = higher convergence)
        convergence_score = max(0.0, 1.0 - (avg_improvement / self.plateau_threshold))
        return min(1.0, convergence_score)
    
    def _calculate_improvement(self, optimization_state: Dict[str, Any]) -> Optional[float]:
        """Calculate improvement rate for current generation."""
        population = optimization_state.get('population', [])
        if not population:
            return None
        
        # Get best scores for each objective
        best_scores = {}
        for solution in population:
            for obj_name, obj_eval in solution.objectives.items():
                score = obj_eval.score
                if obj_name not in best_scores:
                    best_scores[obj_name] = score
                else:
                    if obj_eval.direction == OptimizationDirection.MAXIMIZE:
                        best_scores[obj_name] = max(best_scores[obj_name], score)
                    else:
                        best_scores[obj_name] = min(best_scores[obj_name], score)
        
        # Calculate improvement from previous best (stored in metrics)
        metrics = optimization_state.get('metrics', {})
        improvement = 0.0
        count = 0
        
        for obj_name, current_best in best_scores.items():
            prev_best_key = f'prev_best_{obj_name}'
            if prev_best_key in metrics:
                prev_best = metrics[prev_best_key]
                # Calculate relative improvement
                if prev_best != 0:
                    rel_improvement = abs(current_best - prev_best) / abs(prev_best)
                    improvement += rel_improvement
                    count += 1
        
        return improvement / count if count > 0 else 0.0


# Optimal Stopping Estimators


class StatisticalTrendEstimator(OptimalStoppingEstimator):
    """Estimates optimal stopping based on statistical trend analysis.
    
    Uses linear regression on recent best objective scores to predict
    future improvement. Stops when the trend slope falls below threshold.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        trend_threshold: float = 0.0,
        objective_name: Optional[str] = None
    ):
        """Initialize statistical trend estimator.
        
        Args:
            window_size: Number of recent generations to analyze
            trend_threshold: Minimum positive trend to continue (slope <= threshold stops)
            objective_name: Specific objective to track (None for first objective)
        """
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self.objective_name = objective_name
        self._score_history: deque = deque(maxlen=window_size)
        _logger.debug(
            f"StatisticalTrendEstimator initialized: window={window_size}, "
            f"threshold={trend_threshold}, objective={objective_name}"
        )
    
    def should_stop(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if optimization should stop based on trend."""
        score = self._extract_best_score(optimization_state)
        if score is None:
            return False
        
        self._score_history.append(score)
        
        if len(self._score_history) < self.window_size:
            return False
        
        # Compute linear regression slope
        x = np.arange(len(self._score_history))
        y = np.array(self._score_history)
        slope = np.polyfit(x, y, 1)[0]
        
        # Stop if slope <= threshold (no positive trend)
        _logger.debug(f"StatisticalTrendEstimator: slope={slope:.6f}, threshold={self.trend_threshold}")
        return slope <= self.trend_threshold
    
    def get_stopping_confidence(self) -> float:
        """Get confidence level for the stopping decision."""
        if len(self._score_history) < 2:
            return 0.0
        
        x = np.arange(len(self._score_history))
        y = np.array(self._score_history)
        slope = np.polyfit(x, y, 1)[0]
        
        # Confidence based on how negative the slope is relative to threshold
        if self.trend_threshold == 0:
            # Use absolute slope magnitude
            confidence = max(0.0, min(1.0, -slope / max(abs(slope), 1e-6)))
        else:
            confidence = max(0.0, min(1.0, -slope / abs(self.trend_threshold)))
        
        return confidence
    
    def get_stopping_reason(self) -> str:
        """Get human-readable reason for stopping recommendation."""
        slope = self._get_current_slope()
        return (
            f"StatisticalTrendEstimator: trend slope {slope:.6f} <= threshold "
            f"{self.trend_threshold} (no positive improvement trend)"
        )
    
    def get_predicted_improvement(self) -> float:
        """Get predicted improvement from continuing optimization."""
        return self._get_current_slope()
    
    def _extract_best_score(self, optimization_state: Dict[str, Any]) -> Optional[float]:
        """Extract best score for the target objective."""
        population = optimization_state.get('population', [])
        if not population:
            return None
        
        # Determine objective name
        obj_name = self.objective_name
        if obj_name is None:
            for sol in population:
                if sol.objectives:
                    obj_name = next(iter(sol.objectives))
                    break
        
        if obj_name is None:
            return None
        
        best_score = None
        for sol in population:
            obj_eval = sol.objectives.get(obj_name)
            if obj_eval:
                score = obj_eval.score
                if best_score is None:
                    best_score = score
                else:
                    if obj_eval.direction == OptimizationDirection.MAXIMIZE:
                        best_score = max(best_score, score)
                    else:
                        best_score = min(best_score, score)
        
        return best_score


# Additional classes for architecture compliance

class AnalysisEngine:
    """Main analysis engine for multi-objective optimization."""
    
    def __init__(self):
        self.pareto_analyzer = ParetoFrontierAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
    
    def analyze(self, frontier: List[EvaluationResult], 
                history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        results = {}
        
        # Pareto frontier analysis
        results["pareto"] = self.pareto_analyzer.analyze_frontier(frontier)
        
        # Performance analysis
        results["performance"] = self.performance_analyzer.analyze_performance(frontier, history)
        
        # Convergence analysis
        results["convergence"] = self.convergence_analyzer.analyze_convergence(history)
        
        return results
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate analysis report."""
        report_lines = ["Multi-Objective Optimization Analysis Report", "=" * 50]
        
        for section, data in analysis_results.items():
            report_lines.append(f"\n{section.upper()}:")
            report_lines.append("-" * len(section))
            for key, value in data.items():
                report_lines.append(f"  {key}: {value}")
        
        return "\n".join(report_lines)
    
    def get_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis."""
        insights = []
        
        # Extract insights from different sections
        if "pareto" in analysis_results:
            pareto_data = analysis_results["pareto"]
            if "hypervolume" in pareto_data:
                hypervolume = pareto_data["hypervolume"]
                if hypervolume > 0.8:
                    insights.append("Excellent Pareto frontier coverage")
                elif hypervolume > 0.5:
                    insights.append("Good Pareto frontier coverage")
                else:
                    insights.append("Pareto frontier needs improvement")
        
        if "convergence" in analysis_results:
            conv_data = analysis_results["convergence"]
            if "converged" in conv_data and conv_data["converged"]:
                insights.append("Optimization has converged")
            else:
                insights.append("Optimization still in progress")
        
        return insights


class ParetoFrontierAnalyzer:
    """Analyzes Pareto frontier properties."""
    
    def analyze_frontier(self, frontier: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze Pareto frontier."""
        if not frontier:
            return {"error": "Empty frontier"}
        
        results = {
            "frontier_size": len(frontier),
            "hypervolume": self.calculate_hypervolume(frontier),
            "diversity": self.assess_diversity(frontier),
            "convergence": self.assess_convergence(frontier)
        }
        
        return results
    
    def calculate_hypervolume(self, frontier: List[EvaluationResult]) -> float:
        """Calculate hypervolume of Pareto frontier."""
        if not frontier:
            return 0.0
        
        # Simple hypervolume approximation
        # In practice, this would use more sophisticated algorithms
        try:
            import numpy as np
            
            # Extract objective values
            objective_values = []
            for solution in frontier:
                values = []
                for obj_eval in solution.objectives.values():
                    values.append(obj_eval.score)
                if values:
                    objective_values.append(values)
            
            if not objective_values:
                return 0.0
            
            # Normalize to [0, 1] range
            objective_values = np.array(objective_values)
            normalized = (objective_values - objective_values.min(axis=0)) / (objective_values.max(axis=0) - objective_values.min(axis=0) + 1e-8)
            
            # Simple hypervolume estimation
            return float(np.mean(normalized))
            
        except ImportError:
            # Fallback simple calculation
            return len(frontier) / 100.0  # Simple proxy
    
    def assess_diversity(self, frontier: List[EvaluationResult]) -> float:
        """Assess solution diversity in frontier."""
        if len(frontier) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        try:
            import numpy as np
            
            distances = []
            for i, sol1 in enumerate(frontier):
                for j, sol2 in enumerate(frontier[i+1:], i+1):
                    # Extract objective vectors
                    vec1 = [obj_eval.score for obj_eval in sol1.objectives.values()]
                    vec2 = [obj_eval.score for obj_eval in sol2.objectives.values()]
                    
                    if len(vec1) == len(vec2) and vec1:
                        dist = np.linalg.norm(np.array(vec1) - np.array(vec2))
                        distances.append(dist)
            
            if distances:
                return float(np.mean(distances))
            
        except ImportError:
            pass
        
        # Fallback: count unique solutions
        unique_solutions = len(set(sol.solution_id for sol in frontier))
        return unique_solutions / len(frontier)
    
    def assess_convergence(self, frontier: List[EvaluationResult]) -> float:
        """Assess convergence state of frontier."""
        if len(frontier) < 3:
            return 0.0
        
        # Simple convergence assessment based on solution spread
        try:
            import numpy as np
            
            # Calculate variance in objective space
            all_values = []
            for solution in frontier:
                for obj_eval in solution.objectives.values():
                    all_values.append(obj_eval.score)
            
            if all_values:
                variance = np.var(all_values)
                # Lower variance suggests convergence
                convergence = 1.0 / (1.0 + variance)
                return float(convergence)
            
        except ImportError:
            pass
        
        return 0.5  # Neutral value


class PerformanceAnalyzer:
    """Analyzes performance of optimization."""
    
    def analyze_performance(self, frontier: List[EvaluationResult], 
                          history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze optimization performance."""
        results = {
            "best_solutions": self.get_best_solutions(frontier),
            "objective_ranges": self.get_objective_ranges(frontier),
            "improvement_trends": self.analyze_improvement_trends(history)
        }
        
        return results
    
    def benchmark_solutions(self, solutions: List[EvaluationResult]) -> Dict[str, float]:
        """Benchmark solutions against each other."""
        if not solutions:
            return {}
        
        benchmarks = {}
        
        # Calculate performance metrics for each solution
        for solution in solutions:
            # Average score across all objectives
            scores = [obj_eval.score for obj_eval in solution.objectives.values()]
            if scores:
                avg_score = sum(scores) / len(scores)
                benchmarks[solution.solution_id] = avg_score
        
        return benchmarks
    
    def compare_solutions(self, sol1: EvaluationResult, sol2: EvaluationResult) -> Dict[str, Any]:
        """Compare two solutions."""
        comparison = {
            "sol1_better": 0,
            "sol2_better": 0,
            "ties": 0,
            "detailed_comparison": {}
        }
        
        # Compare each objective
        for obj_name in set(sol1.objectives.keys()) | set(sol2.objectives.keys()):
            obj1_eval = sol1.objectives.get(obj_name)
            obj2_eval = sol2.objectives.get(obj_name)
            
            if obj1_eval and obj2_eval:
                if obj1_eval.score > obj2_eval.score:
                    comparison["sol1_better"] += 1
                elif obj1_eval.score < obj2_eval.score:
                    comparison["sol2_better"] += 1
                else:
                    comparison["ties"] += 1
                
                comparison["detailed_comparison"][obj_name] = {
                    "sol1_score": obj1_eval.score,
                    "sol2_score": obj2_eval.score,
                    "winner": "sol1" if obj1_eval.score > obj2_eval.score else "sol2" if obj1_eval.score < obj2_eval.score else "tie"
                }
        
        return comparison
    
    def get_best_solutions(self, frontier: List[EvaluationResult]) -> Dict[str, str]:
        """Get best solution for each objective."""
        best_solutions = {}
        
        # Collect all objective names
        all_objectives = set()
        for solution in frontier:
            all_objectives.update(solution.objectives.keys())
        
        # Find best solution for each objective
        for obj_name in all_objectives:
            best_solution = None
            best_score = None
            
            for solution in frontier:
                obj_eval = solution.objectives.get(obj_name)
                if obj_eval:
                    if best_score is None or obj_eval.score > best_score:
                        best_score = obj_eval.score
                        best_solution = solution.solution_id
            
            if best_solution:
                best_solutions[obj_name] = best_solution
        
        return best_solutions
    
    def get_objective_ranges(self, frontier: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """Get ranges of objective values."""
        ranges = {}
        
        # Collect all objective values
        objective_values = {}
        for solution in frontier:
            for obj_name, obj_eval in solution.objectives.items():
                if obj_name not in objective_values:
                    objective_values[obj_name] = []
                objective_values[obj_name].append(obj_eval.score)
        
        # Calculate ranges
        for obj_name, values in objective_values.items():
            if values:
                ranges[obj_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "range": max(values) - min(values)
                }
        
        return ranges
    
    def analyze_improvement_trends(self, history: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze improvement trends over time."""
        trends = {}
        
        for metric_name, values in history.items():
            if len(values) >= 3:
                # Simple trend analysis
                recent_avg = sum(values[-3:]) / 3
                earlier_avg = sum(values[:min(3, len(values))]) / min(3, len(values))
                
                if recent_avg > earlier_avg * 1.05:
                    trends[metric_name] = "improving"
                elif recent_avg < earlier_avg * 0.95:
                    trends[metric_name] = "declining"
                else:
                    trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"
        
        return trends


class ConvergenceAnalyzer:
    """Analyzes convergence behavior."""
    
    def analyze_convergence(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze convergence from history."""
        results = {
            "converged": False,
            "convergence_generation": None,
            "stability_metrics": {},
            "trend_analysis": {}
        }
        
        # Check each metric for convergence
        for metric_name, values in history.items():
            if len(values) >= 5:
                stability = self._check_stability(values)
                results["stability_metrics"][metric_name] = stability
                
                if stability > 0.9:  # High stability indicates convergence
                    results["converged"] = True
                    if results["convergence_generation"] is None:
                        results["convergence_generation"] = len(values)
        
        # Trend analysis
        results["trend_analysis"] = self._analyze_trends(history)
        
        return results
    
    def detect_plateau(self, values: List[float], window_size: int = 5, tolerance: float = 0.01) -> bool:
        """Detect if values have plateaued."""
        if len(values) < window_size:
            return False
        
        recent_values = values[-window_size:]
        variance = sum((x - sum(recent_values)/len(recent_values))**2 for x in recent_values) / len(recent_values)
        
        return variance < tolerance
    
    def estimate_convergence_time(self, history: Dict[str, List[float]]) -> Optional[int]:
        """Estimate when convergence occurred."""
        convergence_points = []
        
        for metric_name, values in history.items():
            if len(values) >= 5:
                for i in range(5, len(values)):
                    window = values[i-5:i]
                    if self.detect_plateau(window):
                        convergence_points.append(i)
                        break
        
        if convergence_points:
            return sum(convergence_points) // len(convergence_points)
        
        return None
    
    def _check_stability(self, values: List[float], window_size: int = 5) -> float:
        """Check stability of recent values."""
        if len(values) < window_size:
            return 0.0
        
        recent_values = values[-window_size:]
        mean_val = sum(recent_values) / len(recent_values)
        variance = sum((x - mean_val)**2 for x in recent_values) / len(recent_values)
        
        # Convert variance to stability score (0 = unstable, 1 = very stable)
        stability = 1.0 / (1.0 + variance)
        return stability
    
    def _analyze_trends(self, history: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze trends in metrics."""
        trends = {}
        
        for metric_name, values in history.items():
            if len(values) >= 3:
                # Calculate trend slope
                x = list(range(len(values)))
                try:
                    import numpy as np
                    slope = np.polyfit(x, values, 1)[0]
                    
                    if abs(slope) < 0.001:
                        trends[metric_name] = "stable"
                    elif slope > 0:
                        trends[metric_name] = "increasing"
                    else:
                        trends[metric_name] = "decreasing"
                except ImportError:
                    # Simple trend detection
                    if values[-1] > values[0]:
                        trends[metric_name] = "increasing"
                    elif values[-1] < values[0]:
                        trends[metric_name] = "decreasing"
                    else:
                        trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"
        
        return trends
    
    def _get_current_slope(self) -> float:
        """Get current trend slope."""
        if len(self._score_history) < 2:
            return 0.0
        x = np.arange(len(self._score_history))
        y = np.array(self._score_history)
        return np.polyfit(x, y, 1)[0]


class ResourceBoundedEstimator(OptimalStoppingEstimator):
    """Estimates optimal stopping based on resource constraints.
    
    Stops optimization when resource limits (time, memory, evaluations) are exceeded.
    """
    
    def __init__(
        self,
        max_runtime_seconds: Optional[float] = None,
        max_memory_percent: float = 90.0,
        max_evaluations: Optional[int] = None
    ):
        """Initialize resource-bounded estimator.
        
        Args:
            max_runtime_seconds: Maximum allowed runtime in seconds
            max_memory_percent: Maximum allowed memory usage percentage
            max_evaluations: Maximum allowed number of evaluations
        """
        self.max_runtime_seconds = max_runtime_seconds
        self.max_memory_percent = max_memory_percent
        self.max_evaluations = max_evaluations
        _logger.debug(
            f"ResourceBoundedEstimator initialized: max_time={max_runtime_seconds}, "
            f"max_mem={max_memory_percent}%, max_eval={max_evaluations}"
        )
    
    def should_stop(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if optimization should stop based on resource usage."""
        # Check runtime
        if self.max_runtime_seconds is not None:
            start_time = optimization_state.get('start_time')
            if start_time:
                import time
                elapsed = time.time() - start_time
                if elapsed >= self.max_runtime_seconds:
                    _logger.debug(f"ResourceBoundedEstimator: runtime {elapsed:.1f}s >= limit {self.max_runtime_seconds}s")
                    return True
        
        # Check memory usage
        resource_usage = optimization_state.get('resource_usage', {})
        memory_percent = resource_usage.get('memory_percent', 0)
        if memory_percent >= self.max_memory_percent:
            _logger.debug(f"ResourceBoundedEstimator: memory {memory_percent}% >= limit {self.max_memory_percent}%")
            return True
        
        # Check evaluation count
        if self.max_evaluations is not None:
            eval_count = optimization_state.get('evaluation_count', 0)
            if eval_count >= self.max_evaluations:
                _logger.debug(f"ResourceBoundedEstimator: evaluations {eval_count} >= limit {self.max_evaluations}")
                return True
        
        return False
    
    def get_stopping_confidence(self) -> float:
        """Get confidence level for the stopping decision."""
        # Resource limits are hard constraints, so confidence is high
        return 1.0
    
    def get_stopping_reason(self) -> str:
        """Get human-readable reason for stopping recommendation."""
        reasons = []
        if self.max_runtime_seconds is not None:
            reasons.append(f"runtime limit {self.max_runtime_seconds}s")
        if self.max_memory_percent is not None:
            reasons.append(f"memory limit {self.max_memory_percent}%")
        if self.max_evaluations is not None:
            reasons.append(f"evaluation limit {self.max_evaluations}")
        
        if not reasons:
            return "ResourceBoundedEstimator: No limits set"
        
        return f"ResourceBoundedEstimator: Exceeded {', '.join(reasons)}"
    
    def get_predicted_improvement(self) -> float:
        """Get predicted improvement from continuing optimization."""
        # Resource constraints limit further improvement
        return 0.0


class DiminishingReturnsEstimator(OptimalStoppingEstimator):
    """Estimates optimal stopping based on diminishing returns analysis.
    
    Stops when the rate of improvement falls below a threshold,
    indicating that further optimization yields minimal benefits.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        improvement_threshold: float = 0.01,
        objective_name: Optional[str] = None
    ):
        """Initialize diminishing returns estimator.
        
        Args:
            window_size: Number of recent improvements to analyze
            improvement_threshold: Minimum average improvement to continue
            objective_name: Specific objective to track (None for first objective)
        """
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.objective_name = objective_name
        self._improvement_history: deque = deque(maxlen=window_size)
        self._last_best_score: Optional[float] = None
        _logger.debug(
            f"DiminishingReturnsEstimator initialized: window={window_size}, "
            f"threshold={improvement_threshold}, objective={objective_name}"
        )
    
    def should_stop(self, optimization_state: Dict[str, Any]) -> bool:
        """Determine if optimization should stop based on diminishing returns."""
        current_best = self._extract_best_score(optimization_state)
        if current_best is None:
            return False
        
        # Calculate improvement from previous generation
        if self._last_best_score is not None:
            improvement = abs(current_best - self._last_best_score)
            if self._last_best_score != 0:
                rel_improvement = improvement / abs(self._last_best_score)
            else:
                rel_improvement = improvement
            self._improvement_history.append(rel_improvement)
        
        self._last_best_score = current_best
        
        if len(self._improvement_history) < self.window_size:
            return False
        
        avg_improvement = statistics.mean(self._improvement_history)
        _logger.debug(
            f"DiminishingReturnsEstimator: avg_improvement={avg_improvement:.6f}, "
            f"threshold={self.improvement_threshold}"
        )
        return avg_improvement <= self.improvement_threshold
    
    def get_stopping_confidence(self) -> float:
        """Get confidence level for the stopping decision."""
        if len(self._improvement_history) < self.window_size:
            return 0.0
        
        avg_improvement = statistics.mean(self._improvement_history)
        # Confidence based on how low the improvement is relative to threshold
        confidence = max(0.0, min(1.0, 1.0 - (avg_improvement / self.improvement_threshold)))
        return confidence
    
    def get_stopping_reason(self) -> str:
        """Get human-readable reason for stopping recommendation."""
        if len(self._improvement_history) >= self.window_size:
            avg_imp = statistics.mean(self._improvement_history)
            return (
                f"DiminishingReturnsEstimator: Average improvement {avg_imp:.6f} <= "
                f"threshold {self.improvement_threshold} (diminishing returns)"
            )
        return "DiminishingReturnsEstimator: Insufficient improvement history"
    
    def get_predicted_improvement(self) -> float:
        """Get predicted improvement from continuing optimization."""
        if not self._improvement_history:
            return 0.0
        return statistics.mean(self._improvement_history)
    
    def _extract_best_score(self, optimization_state: Dict[str, Any]) -> Optional[float]:
        """Extract best score for the target objective."""
        population = optimization_state.get('population', [])
        if not population:
            return None
        
        obj_name = self.objective_name
        if obj_name is None:
            for sol in population:
                if sol.objectives:
                    obj_name = next(iter(sol.objectives))
                    break
        
        if obj_name is None:
            return None
        
        best_score = None
        for sol in population:
            obj_eval = sol.objectives.get(obj_name)
            if obj_eval:
                score = obj_eval.score
                if best_score is None:
                    best_score = score
                else:
                    if obj_eval.direction == OptimizationDirection.MAXIMIZE:
                        best_score = max(best_score, score)
                    else:
                        best_score = min(best_score, score)
        
        return best_score


# Additional classes for architecture compliance

class AnalysisEngine:
    """Main analysis engine for multi-objective optimization."""
    
    def __init__(self):
        self.pareto_analyzer = ParetoFrontierAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
    
    def analyze(self, frontier: List[EvaluationResult], 
                history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        results = {}
        
        # Pareto frontier analysis
        results["pareto"] = self.pareto_analyzer.analyze_frontier(frontier)
        
        # Performance analysis
        results["performance"] = self.performance_analyzer.analyze_performance(frontier, history)
        
        # Convergence analysis
        results["convergence"] = self.convergence_analyzer.analyze_convergence(history)
        
        return results
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate analysis report."""
        report_lines = ["Multi-Objective Optimization Analysis Report", "=" * 50]
        
        for section, data in analysis_results.items():
            report_lines.append(f"\n{section.upper()}:")
            report_lines.append("-" * len(section))
            for key, value in data.items():
                report_lines.append(f"  {key}: {value}")
        
        return "\n".join(report_lines)
    
    def get_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis."""
        insights = []
        
        # Extract insights from different sections
        if "pareto" in analysis_results:
            pareto_data = analysis_results["pareto"]
            if "hypervolume" in pareto_data:
                hypervolume = pareto_data["hypervolume"]
                if hypervolume > 0.8:
                    insights.append("Excellent Pareto frontier coverage")
                elif hypervolume > 0.5:
                    insights.append("Good Pareto frontier coverage")
                else:
                    insights.append("Pareto frontier needs improvement")
        
        if "convergence" in analysis_results:
            conv_data = analysis_results["convergence"]
            if "converged" in conv_data and conv_data["converged"]:
                insights.append("Optimization has converged")
            else:
                insights.append("Optimization still in progress")
        
        return insights


class ParetoFrontierAnalyzer:
    """Analyzes Pareto frontier properties."""
    
    def analyze_frontier(self, frontier: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze Pareto frontier."""
        if not frontier:
            return {"error": "Empty frontier"}
        
        results = {
            "frontier_size": len(frontier),
            "hypervolume": self.calculate_hypervolume(frontier),
            "diversity": self.assess_diversity(frontier),
            "convergence": self.assess_convergence(frontier)
        }
        
        return results
    
    def calculate_hypervolume(self, frontier: List[EvaluationResult]) -> float:
        """Calculate hypervolume of Pareto frontier."""
        if not frontier:
            return 0.0
        
        # Simple hypervolume approximation
        # In practice, this would use more sophisticated algorithms
        try:
            import numpy as np
            
            # Extract objective values
            objective_values = []
            for solution in frontier:
                values = []
                for obj_eval in solution.objectives.values():
                    values.append(obj_eval.score)
                if values:
                    objective_values.append(values)
            
            if not objective_values:
                return 0.0
            
            # Normalize to [0, 1] range
            objective_values = np.array(objective_values)
            normalized = (objective_values - objective_values.min(axis=0)) / (objective_values.max(axis=0) - objective_values.min(axis=0) + 1e-8)
            
            # Simple hypervolume estimation
            return float(np.mean(normalized))
            
        except ImportError:
            # Fallback simple calculation
            return len(frontier) / 100.0  # Simple proxy
    
    def assess_diversity(self, frontier: List[EvaluationResult]) -> float:
        """Assess solution diversity in frontier."""
        if len(frontier) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        try:
            import numpy as np
            
            distances = []
            for i, sol1 in enumerate(frontier):
                for j, sol2 in enumerate(frontier[i+1:], i+1):
                    # Extract objective vectors
                    vec1 = [obj_eval.score for obj_eval in sol1.objectives.values()]
                    vec2 = [obj_eval.score for obj_eval in sol2.objectives.values()]
                    
                    if len(vec1) == len(vec2) and vec1:
                        dist = np.linalg.norm(np.array(vec1) - np.array(vec2))
                        distances.append(dist)
            
            if distances:
                return float(np.mean(distances))
            
        except ImportError:
            pass
        
        # Fallback: count unique solutions
        unique_solutions = len(set(sol.solution_id for sol in frontier))
        return unique_solutions / len(frontier)
    
    def assess_convergence(self, frontier: List[EvaluationResult]) -> float:
        """Assess convergence state of frontier."""
        if len(frontier) < 3:
            return 0.0
        
        # Simple convergence assessment based on solution spread
        try:
            import numpy as np
            
            # Calculate variance in objective space
            all_values = []
            for solution in frontier:
                for obj_eval in solution.objectives.values():
                    all_values.append(obj_eval.score)
            
            if all_values:
                variance = np.var(all_values)
                # Lower variance suggests convergence
                convergence = 1.0 / (1.0 + variance)
                return float(convergence)
            
        except ImportError:
            pass
        
        return 0.5  # Neutral value


class PerformanceAnalyzer:
    """Analyzes performance of optimization."""
    
    def analyze_performance(self, frontier: List[EvaluationResult], 
                          history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze optimization performance."""
        results = {
            "best_solutions": self.get_best_solutions(frontier),
            "objective_ranges": self.get_objective_ranges(frontier),
            "improvement_trends": self.analyze_improvement_trends(history)
        }
        
        return results
    
    def benchmark_solutions(self, solutions: List[EvaluationResult]) -> Dict[str, float]:
        """Benchmark solutions against each other."""
        if not solutions:
            return {}
        
        benchmarks = {}
        
        # Calculate performance metrics for each solution
        for solution in solutions:
            # Average score across all objectives
            scores = [obj_eval.score for obj_eval in solution.objectives.values()]
            if scores:
                avg_score = sum(scores) / len(scores)
                benchmarks[solution.solution_id] = avg_score
        
        return benchmarks
    
    def compare_solutions(self, sol1: EvaluationResult, sol2: EvaluationResult) -> Dict[str, Any]:
        """Compare two solutions."""
        comparison = {
            "sol1_better": 0,
            "sol2_better": 0,
            "ties": 0,
            "detailed_comparison": {}
        }
        
        # Compare each objective
        for obj_name in set(sol1.objectives.keys()) | set(sol2.objectives.keys()):
            obj1_eval = sol1.objectives.get(obj_name)
            obj2_eval = sol2.objectives.get(obj_name)
            
            if obj1_eval and obj2_eval:
                if obj1_eval.score > obj2_eval.score:
                    comparison["sol1_better"] += 1
                elif obj1_eval.score < obj2_eval.score:
                    comparison["sol2_better"] += 1
                else:
                    comparison["ties"] += 1
                
                comparison["detailed_comparison"][obj_name] = {
                    "sol1_score": obj1_eval.score,
                    "sol2_score": obj2_eval.score,
                    "winner": "sol1" if obj1_eval.score > obj2_eval.score else "sol2" if obj1_eval.score < obj2_eval.score else "tie"
                }
        
        return comparison
    
    def get_best_solutions(self, frontier: List[EvaluationResult]) -> Dict[str, str]:
        """Get best solution for each objective."""
        best_solutions = {}
        
        # Collect all objective names
        all_objectives = set()
        for solution in frontier:
            all_objectives.update(solution.objectives.keys())
        
        # Find best solution for each objective
        for obj_name in all_objectives:
            best_solution = None
            best_score = None
            
            for solution in frontier:
                obj_eval = solution.objectives.get(obj_name)
                if obj_eval:
                    if best_score is None or obj_eval.score > best_score:
                        best_score = obj_eval.score
                        best_solution = solution.solution_id
            
            if best_solution:
                best_solutions[obj_name] = best_solution
        
        return best_solutions
    
    def get_objective_ranges(self, frontier: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """Get ranges of objective values."""
        ranges = {}
        
        # Collect all objective values
        objective_values = {}
        for solution in frontier:
            for obj_name, obj_eval in solution.objectives.items():
                if obj_name not in objective_values:
                    objective_values[obj_name] = []
                objective_values[obj_name].append(obj_eval.score)
        
        # Calculate ranges
        for obj_name, values in objective_values.items():
            if values:
                ranges[obj_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "range": max(values) - min(values)
                }
        
        return ranges
    
    def analyze_improvement_trends(self, history: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze improvement trends over time."""
        trends = {}
        
        for metric_name, values in history.items():
            if len(values) >= 3:
                # Simple trend analysis
                recent_avg = sum(values[-3:]) / 3
                earlier_avg = sum(values[:min(3, len(values))]) / min(3, len(values))
                
                if recent_avg > earlier_avg * 1.05:
                    trends[metric_name] = "improving"
                elif recent_avg < earlier_avg * 0.95:
                    trends[metric_name] = "declining"
                else:
                    trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"
        
        return trends


class ConvergenceAnalyzer:
    """Analyzes convergence behavior."""
    
    def analyze_convergence(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze convergence from history."""
        results = {
            "converged": False,
            "convergence_generation": None,
            "stability_metrics": {},
            "trend_analysis": {}
        }
        
        # Check each metric for convergence
        for metric_name, values in history.items():
            if len(values) >= 5:
                stability = self._check_stability(values)
                results["stability_metrics"][metric_name] = stability
                
                if stability > 0.9:  # High stability indicates convergence
                    results["converged"] = True
                    if results["convergence_generation"] is None:
                        results["convergence_generation"] = len(values)
        
        # Trend analysis
        results["trend_analysis"] = self._analyze_trends(history)
        
        return results
    
    def detect_plateau(self, values: List[float], window_size: int = 5, tolerance: float = 0.01) -> bool:
        """Detect if values have plateaued."""
        if len(values) < window_size:
            return False
        
        recent_values = values[-window_size:]
        variance = sum((x - sum(recent_values)/len(recent_values))**2 for x in recent_values) / len(recent_values)
        
        return variance < tolerance
    
    def estimate_convergence_time(self, history: Dict[str, List[float]]) -> Optional[int]:
        """Estimate when convergence occurred."""
        convergence_points = []
        
        for metric_name, values in history.items():
            if len(values) >= 5:
                for i in range(5, len(values)):
                    window = values[i-5:i]
                    if self.detect_plateau(window):
                        convergence_points.append(i)
                        break
        
        if convergence_points:
            return sum(convergence_points) // len(convergence_points)
        
        return None
    
    def _check_stability(self, values: List[float], window_size: int = 5) -> float:
        """Check stability of recent values."""
        if len(values) < window_size:
            return 0.0
        
        recent_values = values[-window_size:]
        mean_val = sum(recent_values) / len(recent_values)
        variance = sum((x - mean_val)**2 for x in recent_values) / len(recent_values)
        
        # Convert variance to stability score (0 = unstable, 1 = very stable)
        stability = 1.0 / (1.0 + variance)
        return stability
    
    def _analyze_trends(self, history: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze trends in metrics."""
        trends = {}
        
        for metric_name, values in history.items():
            if len(values) >= 3:
                # Calculate trend slope
                x = list(range(len(values)))
                try:
                    import numpy as np
                    slope = np.polyfit(x, values, 1)[0]
                    
                    if abs(slope) < 0.001:
                        trends[metric_name] = "stable"
                    elif slope > 0:
                        trends[metric_name] = "increasing"
                    else:
                        trends[metric_name] = "decreasing"
                except ImportError:
                    # Simple trend detection
                    if values[-1] > values[0]:
                        trends[metric_name] = "increasing"
                    elif values[-1] < values[0]:
                        trends[metric_name] = "decreasing"
                    else:
                        trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"
        
        return trends