#!/usr/bin/env python3
"""
Monitoring Features Demo

This demonstrates the monitoring and analysis capabilities of the GEPA framework.
Shows real-time progress tracking, performance metrics, and visualization.

Usage:
    uv run python monitoring_demo.py
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class OptimizationMonitor:
    """Comprehensive monitoring system for optimization progress."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {
            "generation_data": [],
            "population_stats": [],
            "convergence_metrics": [],
            "diversity_metrics": [],
            "performance_stats": {},
            "alerts": []
        }
        self.generation_count = 0
        
    def start_monitoring(self):
        """Initialize monitoring."""
        self.start_time = time.time()
        self.metrics["performance_stats"]["start_time"] = datetime.now().isoformat()
        print("📊 Monitoring started...")
        
    def record_generation(self, generation: int, population: List[Dict], 
                         best_solution: Dict, objectives: List[str]):
        """Record data for a generation."""
        self.generation_count += 1
        current_time = time.time()
        
        # Calculate generation statistics
        scores = [sol.get('weighted_score', 0) for sol in population]
        objective_values = {obj: [] for obj in objectives}
        
        for sol in population:
            for obj in objectives:
                obj_val = sol.get('objectives', {}).get(obj, 0)
                objective_values[obj].append(obj_val)
        
        # Calculate diversity
        all_prompts = [sol.get('prompt', '') for sol in population]
        unique_words = set()
        total_words = 0
        for prompt in all_prompts:
            words = prompt.lower().split()
            unique_words.update(words)
            total_words += len(words)
        diversity = len(unique_words) / max(1, total_words)
        
        # Calculate convergence metrics
        if len(self.metrics["generation_data"]) > 0:
            prev_best = self.metrics["generation_data"][-1]["best_score"]
            improvement = (best_solution['weighted_score'] - prev_best) / max(0.001, prev_best)
        else:
            improvement = 1.0
            
        # Store generation data
        gen_data = {
            "generation": generation,
            "timestamp": current_time,
            "elapsed_time": current_time - self.start_time if self.start_time else 0,
            "population_size": len(population),
            "best_score": best_solution['weighted_score'],
            "best_prompt": best_solution.get('prompt', ''),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "score_std": self._calculate_std(scores) if len(scores) > 1 else 0,
            "improvement_rate": improvement,
            "diversity": diversity,
            "objective_stats": {}
        }
        
        # Calculate objective-specific stats
        for obj in objectives:
            obj_scores = objective_values[obj]
            if obj_scores:
                gen_data["objective_stats"][obj] = {
                    "best": max(obj_scores),
                    "avg": sum(obj_scores) / len(obj_scores),
                    "worst": min(obj_scores),
                    "std": self._calculate_std(obj_scores) if len(obj_scores) > 1 else 0
                }
        
        self.metrics["generation_data"].append(gen_data)
        self.metrics["diversity_metrics"].append(diversity)
        self.metrics["convergence_metrics"].append(improvement)
        
        # Check for alerts
        self._check_alerts(gen_data)
        
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def _check_alerts(self, gen_data: Dict):
        """Check for optimization alerts."""
        alerts = []
        
        # Stagnation alert
        if len(self.metrics["convergence_metrics"]) >= 3:
            recent_improvements = self.metrics["convergence_metrics"][-3:]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            if avg_improvement < 0.01:
                alerts.append({
                    "type": "stagnation",
                    "generation": gen_data["generation"],
                    "message": "Optimization stagnating - low improvement rate",
                    "severity": "warning"
                })
        
        # Low diversity alert
        if gen_data["diversity"] < 0.1:
            alerts.append({
                "type": "low_diversity",
                "generation": gen_data["generation"],
                "message": "Population diversity is very low",
                "severity": "warning"
            })
            
        # High performance alert
        if gen_data["best_score"] > 0.9:
            alerts.append({
                "type": "high_performance",
                "generation": gen_data["generation"],
                "message": "Excellent performance achieved",
                "severity": "info"
            })
            
        self.metrics["alerts"].extend(alerts)
        
        # Display recent alerts
        for alert in alerts[-3:]:  # Show last 3 alerts
            icon = "⚠️" if alert["severity"] == "warning" else "ℹ️"
            print(f"{icon} Alert (Gen {alert['generation']}): {alert['message']}")
            
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if not self.metrics["generation_data"]:
            return {"error": "No data to report"}
            
        report = {
            "summary": {
                "total_generations": len(self.metrics["generation_data"]),
                "total_time": self.metrics["generation_data"][-1]["elapsed_time"],
                "initial_score": self.metrics["generation_data"][0]["best_score"],
                "final_score": self.metrics["generation_data"][-1]["best_score"],
                "improvement": 0,
                "improvement_percentage": 0,
                "avg_diversity": sum(self.metrics["diversity_metrics"]) / len(self.metrics["diversity_metrics"]),
                "convergence_rate": 0,
                "total_alerts": len(self.metrics["alerts"])
            },
            "performance_trends": self._analyze_trends(),
            "alerts_summary": self._summarize_alerts(),
            "recommendations": self._generate_recommendations()
        }
        
        # Calculate improvement
        initial = report["summary"]["initial_score"]
        final = report["summary"]["final_score"]
        if initial > 0:
            report["summary"]["improvement"] = final - initial
            report["summary"]["improvement_percentage"] = ((final - initial) / initial) * 100
            
        # Calculate convergence rate
        if len(self.metrics["convergence_metrics"]) > 1:
            convergence_scores = [abs(x) for x in self.metrics["convergence_metrics"][1:]]
            report["summary"]["convergence_rate"] = sum(convergence_scores) / len(convergence_scores)
            
        return report
        
    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze performance trends."""
        if len(self.metrics["generation_data"]) < 3:
            return {"status": "insufficient_data"}
            
        scores = [gen["best_score"] for gen in self.metrics["generation_data"]]
        diversities = self.metrics["diversity_metrics"]
        
        # Score trend
        if scores[-1] > scores[-3]:
            score_trend = "improving"
        elif scores[-1] < scores[-3] * 0.95:
            score_trend = "declining"
        else:
            score_trend = "stable"
            
        # Diversity trend
        if diversities[-1] > diversities[-3]:
            diversity_trend = "increasing"
        elif diversities[-1] < diversities[-3] * 0.95:
            diversity_trend = "decreasing"
        else:
            diversity_trend = "stable"
            
        return {
            "score_trend": score_trend,
            "diversity_trend": diversity_trend,
            "overall_status": "healthy" if score_trend == "improving" else "attention_needed"
        }
        
    def _summarize_alerts(self) -> Dict[str, int]:
        """Summarize alerts by type."""
        alert_counts = {}
        for alert in self.metrics["alerts"]:
            alert_type = alert["type"]
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        return alert_counts
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not self.metrics["generation_data"]:
            return ["No data available for recommendations"]
            
        # Check recent performance
        if len(self.metrics["convergence_metrics"]) >= 3:
            recent_improvements = self.metrics["convergence_metrics"][-3:]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            if avg_improvement < 0.01:
                recommendations.append("Consider increasing mutation rate or population size")
                recommendations.append("Try different mutation strategies")
                
        # Check diversity
        avg_diversity = sum(self.metrics["diversity_metrics"]) / len(self.metrics["diversity_metrics"])
        if avg_diversity < 0.2:
            recommendations.append("Increase diversity through enhanced mutation operators")
            recommendations.append("Consider adding random immigrants to population")
            
        # Check final performance
        final_score = self.metrics["generation_data"][-1]["best_score"]
        if final_score < 0.7:
            recommendations.append("Consider running more generations")
            recommendations.append("Review and adjust objective weights")
            
        if not recommendations:
            recommendations.append("Optimization proceeding well - continue current strategy")
            
        return recommendations
        
    def save_report(self, filename: str = "monitoring_report.json"):
        """Save monitoring report to file."""
        report = self.generate_report()
        
        # Add raw data for detailed analysis
        report["raw_data"] = self.metrics
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📄 Report saved to {filename}")
        
    def print_realtime_status(self, generation: int, population: List[Dict], best_solution: Dict):
        """Print real-time status update."""
        if generation == 1:
            print("\n📊 Real-time Monitoring Dashboard")
            print("=" * 60)
            print(f"{'Gen':>4} | {'Best Score':>11} | {'Avg Score':>10} | {'Diversity':>9} | {'Improvement'}")
            print("-" * 65)
            
        scores = [sol.get('weighted_score', 0) for sol in population]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate diversity
        all_prompts = [sol.get('prompt', '') for sol in population]
        unique_words = set()
        total_words = 0
        for prompt in all_prompts:
            words = prompt.lower().split()
            unique_words.update(words)
            total_words += len(words)
        diversity = len(unique_words) / max(1, total_words)
        
        # Calculate improvement
        if len(self.metrics["generation_data"]) > 0:
            prev_score = self.metrics["generation_data"][-1]["best_score"]
            improvement = ((best_solution['weighted_score'] - prev_score) / max(0.001, prev_score)) * 100
            improvement_str = f"{improvement:+6.1f}%"
        else:
            improvement_str = "   +0.0%"
            
        print(f"{generation:>4} | {best_solution['weighted_score']:>11.4f} | {avg_score:>10.4f} | {diversity:>9.3f} | {improvement_str}")
        
        # Show best solution
        if len(best_solution.get('prompt', '')) <= 30:
            prompt_display = best_solution.get('prompt', '')
        else:
            prompt_display = best_solution.get('prompt', '')[:27] + "..."
        print(f"      Best: '{prompt_display}'")

def create_evaluation_function():
    """Create evaluation function for demo."""
    def evaluate(prompt: str) -> Dict[str, float]:
        """Evaluate prompt quality."""
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        # Multiple objectives
        effectiveness = 0.3
        quality_words = ["specific", "detailed", "step-by-step", "example", "clear", "comprehensive"]
        for word in quality_words:
            if word in prompt_lower:
                effectiveness += 0.1
        if len(words) >= 5:
            effectiveness += 0.1
        effectiveness = min(1.0, effectiveness)
        
        clarity = 0.5
        if len(words) <= 15:
            clarity += 0.2
        if '?' in prompt or '.' in prompt:
            clarity += 0.1
        clarity = min(1.0, clarity)
        
        efficiency = 0.7
        if len(words) <= 10:
            efficiency += 0.2
        elif len(words) > 20:
            efficiency -= 0.2
        efficiency = max(0.0, min(1.0, efficiency))
        
        return {
            "effectiveness": effectiveness,
            "clarity": clarity,
            "efficiency": efficiency
        }
    
    return evaluate

def demo_monitoring_features():
    """Demonstrate comprehensive monitoring features."""
    print("📊 Monitoring Features Demo")
    print("=" * 50)
    print("Real-time tracking, performance analysis, and alerts.")
    print()
    
    # Initialize monitor
    monitor = OptimizationMonitor()
    monitor.start_monitoring()
    
    # Setup optimization parameters
    objectives = {"effectiveness": 0.4, "clarity": 0.3, "efficiency": 0.3}
    evaluate = create_evaluation_function()
    
    print(f"🎯 Objectives: {list(objectives.keys())}")
    print(f"📊 Weights: {objectives}")
    print()
    
    # Simulate optimization with monitoring
    print("🚀 Starting monitored optimization...")
    print()
    
    current_prompt = "help me"
    current_objectives = evaluate(current_prompt)
    current_score = sum(current_objectives[obj] * objectives[obj] for obj in objectives)
    
    for generation in range(1, 8):  # 7 generations
        # Generate population
        if generation == 1:
            population_prompts = [
                "help me",
                "please help me",
                "help me please",
                "can you help me",
                "help me with this task",
                "I need help",
                "help me understand",
                "please help me understand"
            ]
        else:
            # Generate mutations
            mutations = []
            base_mutations = [
                f"please {current_prompt}",
                f"{current_prompt} clearly",
                f"explain {current_prompt}",
                f"describe {current_prompt}",
                f"show me {current_prompt}",
                f"help me understand {current_prompt}",
                f"{current_prompt} step by step",
                f"clearly {current_prompt}"
            ]
            population_prompts = base_mutations
        
        # Evaluate population
        population = []
        for prompt in population_prompts:
            obj_values = evaluate(prompt)
            weighted_score = sum(obj_values[obj] * objectives[obj] for obj in objectives)
            
            population.append({
                'prompt': prompt,
                'objectives': obj_values,
                'weighted_score': weighted_score
            })
        
        # Find best solution
        best_solution = max(population, key=lambda x: x['weighted_score'])
        current_prompt = best_solution['prompt']
        
        # Record generation data
        monitor.record_generation(generation, population, best_solution, list(objectives.keys()))
        
        # Print real-time status
        monitor.print_realtime_status(generation, population, best_solution)
        
        # Small delay for demo purposes
        time.sleep(0.5)
        
        # Early stopping if very high performance
        if best_solution['weighted_score'] > 0.95:
            print(f"\n🎯 High performance achieved - stopping early")
            break
    
    print()
    
    # Generate and display final report
    print("📈 Generating comprehensive report...")
    report = monitor.generate_report()
    
    print("\n" + "=" * 60)
    print("📊 MONITORING REPORT")
    print("=" * 60)
    
    # Summary
    summary = report["summary"]
    print(f"\n📋 Summary:")
    print(f"  • Generations: {summary['total_generations']}")
    print(f"  • Total time: {summary['total_time']:.2f}s")
    print(f"  • Score improvement: {summary['initial_score']:.4f} → {summary['final_score']:.4f}")
    print(f"  • Improvement: {summary['improvement_percentage']:.1f}%")
    print(f"  • Average diversity: {summary['avg_diversity']:.3f}")
    print(f"  • Convergence rate: {summary['convergence_rate']:.3f}")
    print(f"  • Total alerts: {summary['total_alerts']}")
    
    # Trends
    trends = report["performance_trends"]
    if trends.get("status") != "insufficient_data":
        print(f"\n📈 Performance Trends:")
        print(f"  • Score trend: {trends['score_trend']}")
        print(f"  • Diversity trend: {trends['diversity_trend']}")
        print(f"  • Overall status: {trends['overall_status']}")
    
    # Alerts
    if report["alerts_summary"]:
        print(f"\n⚠️  Alerts Summary:")
        for alert_type, count in report["alerts_summary"].items():
            print(f"  • {alert_type}: {count}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Save report
    monitor.save_report("demo_monitoring_report.json")
    
    return True

def main():
    """Main function."""
    print("📊 GEPA Framework Monitoring Demo")
    print("=" * 60)
    print("This demonstrates comprehensive monitoring and analysis features.")
    print()
    
    try:
        success = demo_monitoring_features()
        
        if success:
            print(f"\n✨ Monitoring demo completed successfully!")
            print(f"\n💡 Key monitoring features demonstrated:")
            print(f"   • Real-time progress tracking")
            print(f"   • Performance trend analysis")
            print(f"   • Diversity and convergence metrics")
            print(f"   • Automated alert system")
            print(f"   • Comprehensive reporting")
            print(f"   • Actionable recommendations")
            print(f"\n📄 Detailed report saved to: demo_monitoring_report.json")
        else:
            print(f"\n❌ Demo failed. Check the error messages above.")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)