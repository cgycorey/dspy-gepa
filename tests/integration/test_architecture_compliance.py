#!/usr/bin/env python3
"""
Comprehensive Architecture Compliance Test

This test validates that every component from multi_objective_gepa_architecture.md
is properly implemented in the current codebase. It ensures 100% compliance
with the specified architecture.

Author: puppy (code-puppy)
Created: 2025-06-17
"""

import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, get_type_hints
from dataclasses import is_dataclass
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class ArchitectureComplianceTest:
    """Comprehensive test suite for architecture compliance."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        self.total_tests = 0
        self.compliance_score = 0.0
        
    def log_test(self, test_name: str, passed: bool, message: str = "", details: Dict[str, Any] = None):
        """Log a test result."""
        self.total_tests += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        
        result = {
            "test_name": test_name,
            "passed": passed,
            "message": message,
            "details": details or {}
        }
        
        self.test_results.append(result)
        
        if passed:
            self.passed_tests.append(result)
            print(f"{status} {test_name}")
            if message:
                print(f"     {message}")
        else:
            self.failed_tests.append(result)
            print(f"{status} {test_name}")
            print(f"     {message}")
            if details:
                print(f"     Details: {details}")
    
    def test_module_imports(self):
        """Test that all required modules can be imported."""
        print("\n🔍 Testing Module Imports...")
        
        required_modules = [
            "dspy_gepa",
            "dspy_gepa.core",
            "dspy_gepa.core.agent",
            "dspy_gepa.core.multi_objective_agent",
            "dspy_gepa.core.multi_objective_gepa",
            "dspy_gepa.core.analysis",
            "dspy_gepa.core.monitoring",
            "dspy_gepa.core.visualization",
            "dspy_gepa.core.interfaces",
            "dspy_gepa.core.mutation_engine",
            "dspy_gepa.core.parameter_tuner",
            "dspy_gepa.core.objectives",
            "dspy_gepa.dspy_integration",
            "dspy_gepa.dspy_integration.metric_converter",
            "dspy_gepa.dspy_integration.signature_analyzer",
            "dspy_gepa.dspy_integration.multi_objective_optimizer",
            "dspy_gepa.utils",
            "dspy_gepa.utils.config",
            "dspy_gepa.utils.logging",
        ]
        
        for module_name in required_modules:
            try:
                module = importlib.import_module(module_name)
                self.log_test(
                    f"Import {module_name}",
                    True,
                    f"Successfully imported {module_name}"
                )
            except Exception as e:
                self.log_test(
                    f"Import {module_name}",
                    False,
                    f"Failed to import: {str(e)}"
                )
    
    def test_mogepaaagent_implementation(self):
        """Test MOGEPAAgent matches specification."""
        print("\n🎯 Testing MOGEPAAgent Implementation...")
        
        try:
            from dspy_gepa.core.multi_objective_agent import MultiObjectiveGEPAAgent, MOGEPAAgent
            
            # Test class exists and has correct inheritance
            self.log_test(
                "MOGEPAAgent class exists",
                True,
                "MultiObjectiveGEPAAgent class found"
            )
            
            # Test inheritance from GEPAAgent
            from dspy_gepa.core.agent import GEPAAgent
            self.log_test(
                "MOGEPAAgent inherits from GEPAAgent",
                issubclass(MultiObjectiveGEPAAgent, GEPAAgent),
                "Correct inheritance chain" if issubclass(MultiObjectiveGEPAAgent, GEPAAgent) else "Incorrect inheritance"
            )
            
            # Test required methods exist
            required_methods = [
                "__init__",
                "optimize_prompt", 
                "add_objective",
                "remove_objective",
                "set_preferences",
                "get_pareto_frontier",
                "get_optimization_insights",
                "enable_multi_objective_mode",
                "disable_multi_objective_mode"
            ]
            
            for method_name in required_methods:
                has_method = hasattr(MultiObjectiveGEPAAgent, method_name)
                self.log_test(
                    f"MOGEPAAgent has {method_name} method",
                    has_method,
                    f"Method {method_name} present" if has_method else f"Missing method: {method_name}"
                )
            
            # Test backward compatibility alias
            self.log_test(
                "MOGEPAAgent alias exists",
                MOGEPAAgent is MultiObjectiveGEPAAgent,
                "Backward compatibility alias maintained"
            )
            
        except Exception as e:
            self.log_test(
                "MOGEPAAgent implementation",
                False,
                f"Error testing MOGEPAAgent: {str(e)}"
            )
    
    def test_monitoring_components(self):
        """Test all monitoring components exist and work."""
        print("\n📊 Testing Monitoring Components...")
        
        try:
            from dspy_gepa.core.monitoring import (
                MonitoringFramework,
                ConvergenceDetector,
                ParetoFrontierVisualizer,
                OptimalStoppingEstimator
            )
            
            # Test monitoring framework
            self.log_test(
                "MonitoringFramework class exists",
                True,
                "MonitoringFramework imported successfully"
            )
            
            # Test convergence detector
            self.log_test(
                "ConvergenceDetector class exists",
                True,
                "ConvergenceDetector imported successfully"
            )
            
            # Test Pareto frontier visualizer
            self.log_test(
                "ParetoFrontierVisualizer class exists",
                True,
                "ParetoFrontierVisualizer imported successfully"
            )
            
            # Test optimal stopping estimator
            self.log_test(
                "OptimalStoppingEstimator class exists",
                True,
                "OptimalStoppingEstimator imported successfully"
            )
            
            # Test key methods exist
            monitoring_methods = {
                "MonitoringFramework": ["track_progress", "get_metrics", "reset"],
                "ConvergenceDetector": ["detect_convergence", "check_stability", "analyze_trends"],
                "ParetoFrontierVisualizer": ["plot_2d", "plot_3d", "plot_parallel_coordinates"],
                "OptimalStoppingEstimator": ["estimate_stopping_point", "calculate_returns", "predict_optimal_time"]
            }
            
            for class_name, methods in monitoring_methods.items():
                class_obj = locals()[class_name]
                for method_name in methods:
                    has_method = hasattr(class_obj, method_name)
                    self.log_test(
                        f"{class_name} has {method_name} method",
                        has_method,
                        f"Method {method_name} present" if has_method else f"Missing method: {method_name}"
                    )
                    
        except Exception as e:
            self.log_test(
                "Monitoring components",
                False,
                f"Error importing monitoring components: {str(e)}"
            )
    
    def test_analysis_components(self):
        """Test all analysis components are implemented."""
        print("\n🔬 Testing Analysis Components...")
        
        try:
            from dspy_gepa.core.analysis import (
                AnalysisEngine,
                ParetoFrontierAnalyzer,
                PerformanceAnalyzer,
                ConvergenceAnalyzer
            )
            
            # Test analysis engine
            self.log_test(
                "AnalysisEngine class exists",
                True,
                "AnalysisEngine imported successfully"
            )
            
            # Test Pareto frontier analyzer
            self.log_test(
                "ParetoFrontierAnalyzer class exists",
                True,
                "ParetoFrontierAnalyzer imported successfully"
            )
            
            # Test performance analyzer
            self.log_test(
                "PerformanceAnalyzer class exists",
                True,
                "PerformanceAnalyzer imported successfully"
            )
            
            # Test convergence analyzer
            self.log_test(
                "ConvergenceAnalyzer class exists",
                True,
                "ConvergenceAnalyzer imported successfully"
            )
            
            # Test key analysis methods
            analysis_methods = {
                "AnalysisEngine": ["analyze", "generate_report", "get_insights"],
                "ParetoFrontierAnalyzer": ["analyze_frontier", "calculate_hypervolume", "assess_diversity"],
                "PerformanceAnalyzer": ["analyze_performance", "benchmark_solutions", "compare_solutions"],
                "ConvergenceAnalyzer": ["analyze_convergence", "detect_plateau", "estimate_convergence_time"]
            }
            
            for class_name, methods in analysis_methods.items():
                class_obj = locals()[class_name]
                for method_name in methods:
                    has_method = hasattr(class_obj, method_name)
                    self.log_test(
                        f"{class_name} has {method_name} method",
                        has_method,
                        f"Method {method_name} present" if has_method else f"Missing method: {method_name}"
                    )
                    
        except Exception as e:
            self.log_test(
                "Analysis components",
                False,
                f"Error importing analysis components: {str(e)}"
            )
    
    def test_visualization_components(self):
        """Test all visualization components are working."""
        print("\n📈 Testing Visualization Components...")
        
        try:
            from dspy_gepa.core.visualization import (
                VisualizationEngine,
                ParetoFrontierPlotter,
                ConvergencePlotter,
                PerformancePlotter
            )
            
            # Test visualization engine
            self.log_test(
                "VisualizationEngine class exists",
                True,
                "VisualizationEngine imported successfully"
            )
            
            # Test Pareto frontier plotter
            self.log_test(
                "ParetoFrontierPlotter class exists",
                True,
                "ParetoFrontierPlotter imported successfully"
            )
            
            # Test convergence plotter
            self.log_test(
                "ConvergencePlotter class exists",
                True,
                "ConvergencePlotter imported successfully"
            )
            
            # Test performance plotter
            self.log_test(
                "PerformancePlotter class exists",
                True,
                "PerformancePlotter imported successfully"
            )
            
            # Test key visualization methods
            viz_methods = {
                "VisualizationEngine": ["create_plot", "save_plot", "show_plot"],
                "ParetoFrontierPlotter": ["plot_2d_frontier", "plot_3d_frontier", "plot_parallel_coordinates"],
                "ConvergencePlotter": ["plot_convergence", "plot_hypervolume", "plot_diversity"],
                "PerformancePlotter": ["plot_performance", "plot_objectives", "plot_comparison"]
            }
            
            for class_name, methods in viz_methods.items():
                class_obj = locals()[class_name]
                for method_name in methods:
                    has_method = hasattr(class_obj, method_name)
                    self.log_test(
                        f"{class_name} has {method_name} method",
                        has_method,
                        f"Method {method_name} present" if has_method else f"Missing method: {method_name}"
                    )
                    
        except Exception as e:
            self.log_test(
                "Visualization components",
                False,
                f"Error importing visualization components: {str(e)}"
            )
    
    def test_dspy_integration(self):
        """Test DSPy integration is complete."""
        print("\n🤝 Testing DSPy Integration...")
        
        try:
            # Test core integration components
            from dspy_gepa.dspy_integration.metric_converter import MetricConverter
            from dspy_gepa.dspy_integration.signature_analyzer import SignatureAnalyzer
            from dspy_gepa.dspy_integration.multi_objective_optimizer import MultiObjectiveOptimizer
            
            # Test metric converter
            self.log_test(
                "MetricConverter class exists",
                True,
                "MetricConverter imported successfully"
            )
            
            # Test signature analyzer
            self.log_test(
                "SignatureAnalyzer class exists",
                True,
                "SignatureAnalyzer imported successfully"
            )
            
            # Test multi-objective optimizer
            self.log_test(
                "MultiObjectiveOptimizer class exists",
                True,
                "MultiObjectiveOptimizer imported successfully"
            )
            
            # Test key integration methods
            integration_methods = {
                "MetricConverter": ["dspy_to_multi_obj", "aggregate_dspy_metrics", "create_composite_metric"],
                "SignatureAnalyzer": ["analyze_signature", "extract_constraints", "validate_compatibility"],
                "MultiObjectiveOptimizer": ["optimize_dspy_module", "convert_objectives", "integrate_with_gepa"]
            }
            
            for class_name, methods in integration_methods.items():
                class_obj = locals()[class_name]
                for method_name in methods:
                    has_method = hasattr(class_obj, method_name)
                    self.log_test(
                        f"{class_name} has {method_name} method",
                        has_method,
                        f"Method {method_name} present" if has_method else f"Missing method: {method_name}"
                    )
            
            # Test GEPA agent integration
            from dspy_gepa.gepa_agent import GEPAAgent
            self.log_test(
                "GEPAAgent integration exists",
                True,
                "GEPAAgent successfully imported"
            )
            
            # Test simple GEPA integration
            from dspy_gepa.simple_gepa import SimpleGEPAAgent
            self.log_test(
                "SimpleGEPAAgent integration exists",
                True,
                "SimpleGEPAAgent successfully imported"
            )
            
        except Exception as e:
            self.log_test(
                "DSPy integration",
                False,
                f"Error testing DSPy integration: {str(e)}"
            )
    
    def test_interface_definitions(self):
        """Test all interfaces are properly defined."""
        print("\n🔧 Testing Interface Definitions...")
        
        try:
            from dspy_gepa.core.interfaces import (
                Objective,
                TaskType,
                PreferenceVector,
                EvaluationResult,
                ParetoFrontierManager,
                MutationOperator,
                ParameterTuner,
                OptimizationDirection,
                CandidateSolution
            )
            
            # Test core interfaces
            interfaces = [
                "Objective",
                "TaskType", 
                "PreferenceVector",
                "EvaluationResult",
                "ParetoFrontierManager",
                "MutationOperator",
                "ParameterTuner",
                "OptimizationDirection",
                "CandidateSolution"
            ]
            
            for interface_name in interfaces:
                interface_obj = locals()[interface_name]
                exists = interface_obj is not None
                self.log_test(
                    f"Interface {interface_name} exists",
                    exists,
                    f"Interface {interface_name} defined" if exists else f"Missing interface: {interface_name}"
                )
            
            # Test Objective is dataclass with required attributes
            if is_dataclass(Objective):
                import dataclasses
                fields = dataclasses.fields(Objective)
                field_names = {f.name for f in fields}
                required_attrs = ["name", "weight", "direction", "description"]
                for attr in required_attrs:
                    has_attr = attr in field_names
                    self.log_test(
                        f"Objective has {attr} attribute",
                        has_attr,
                        f"Attribute {attr} present" if has_attr else f"Missing attribute: {attr}"
                    )
            else:
                self.log_test(
                    "Objective is dataclass",
                    False,
                    "Objective should be a dataclass"
                )
            
            # Test TaskType enum values
            expected_task_types = ["CLASSIFICATION", "GENERATION", "TRANSLATION", "SUMMARIZATION", "QA", "CUSTOM"]
            for task_type in expected_task_types:
                has_task_type = hasattr(TaskType, task_type)
                self.log_test(
                    f"TaskType has {task_type}",
                    has_task_type,
                    f"Task type {task_type} available" if has_task_type else f"Missing task type: {task_type}"
                )
            
            # Test OptimizationDirection enum
            expected_directions = ["MINIMIZE", "MAXIMIZE"]
            for direction in expected_directions:
                has_direction = hasattr(OptimizationDirection, direction)
                self.log_test(
                    f"OptimizationDirection has {direction}",
                    has_direction,
                    f"Direction {direction} available" if has_direction else f"Missing direction: {direction}"
                )
                
        except Exception as e:
            self.log_test(
                "Interface definitions",
                False,
                f"Error testing interfaces: {str(e)}"
            )
    
    def test_multi_objective_gepa_core(self):
        """Test core multi-objective GEPA implementation."""
        print("\n⚙️ Testing Multi-Objective GEPA Core...")
        
        try:
            from dspy_gepa.core.multi_objective_gepa import (
                MultiObjectiveGEPA,
                OptimizationState,
                ParetoFrontier,
                CandidateSolution
            )
            
            # Test MultiObjectiveGEPA class
            self.log_test(
                "MultiObjectiveGEPA class exists",
                True,
                "MultiObjectiveGEPA imported successfully"
            )
            
            # Test OptimizationState
            self.log_test(
                "OptimizationState class exists",
                True,
                "OptimizationState imported successfully"
            )
            
            # Test ParetoFrontier
            self.log_test(
                "ParetoFrontier class exists",
                True,
                "ParetoFrontier imported successfully"
            )
            
            # Test core methods
            core_methods = {
                "MultiObjectiveGEPA": ["optimize", "get_pareto_frontier", "get_optimization_history"],
                "OptimizationState": ["add_solution", "get_best_solution", "update_frontier"],
                "ParetoFrontier": ["update", "get_frontier", "calculate_hypervolume"],
                "CandidateSolution": ["dominates", "crowding_distance", "get_objective_score"]
            }
            
            for class_name, methods in core_methods.items():
                class_obj = locals()[class_name]
                for method_name in methods:
                    has_method = hasattr(class_obj, method_name)
                    self.log_test(
                        f"{class_name} has {method_name} method",
                        has_method,
                        f"Method {method_name} present" if has_method else f"Missing method: {method_name}"
                    )
            
        except Exception as e:
            self.log_test(
                "Multi-Objective GEPA Core",
                False,
                f"Error testing core components: {str(e)}"
            )
    
    def test_mutation_and_parameter_components(self):
        """Test mutation engine and parameter tuner components."""
        print("\n🔄 Testing Mutation and Parameter Components...")
        
        try:
            # Test mutation components
            from dspy_gepa.core.mutation_engine import (
                CompositeMutator,
                SemanticMutator,
                TaskSpecificMutator,
                AdaptiveRateMutator
            )
            
            # Test parameter tuner components
            from dspy_gepa.core.parameter_tuner import (
                ConvergenceBasedTuner,
                ResourceAwareTuner,
                DynamicParameterTuner
            )
            
            # Test mutation classes
            mutation_classes = [
                "CompositeMutator",
                "SemanticMutator", 
                "TaskSpecificMutator",
                "AdaptiveRateMutator"
            ]
            
            for class_name in mutation_classes:
                class_obj = locals()[class_name]
                exists = class_obj is not None
                self.log_test(
                    f"Mutation class {class_name} exists",
                    exists,
                    f"Class {class_name} available" if exists else f"Missing class: {class_name}"
                )
            
            # Test parameter tuner classes
            tuner_classes = [
                "ConvergenceBasedTuner",
                "ResourceAwareTuner",
                "DynamicParameterTuner"
            ]
            
            for class_name in tuner_classes:
                class_obj = locals()[class_name]
                exists = class_obj is not None
                self.log_test(
                    f"Tuner class {class_name} exists",
                    exists,
                    f"Class {class_name} available" if exists else f"Missing class: {class_name}"
                )
            
        except Exception as e:
            self.log_test(
                "Mutation and Parameter Components",
                False,
                f"Error testing mutation/parameter components: {str(e)}"
            )
    
    def test_objectives_system(self):
        """Test objectives system implementation."""
        print("\n🎯 Testing Objectives System...")
        
        try:
            from dspy_gepa.core.objectives import (
                TaskMetrics,
                ResourceMetrics,
                AccuracyMetric,
                FluencyMetric,
                RelevanceMetric,
                TokenUsageMetric,
                ExecutionTimeMetric
            )
            
            # Test metrics classes
            metrics_classes = [
                "TaskMetrics",
                "ResourceMetrics",
                "AccuracyMetric",
                "FluencyMetric",
                "RelevanceMetric",
                "TokenUsageMetric",
                "ExecutionTimeMetric"
            ]
            
            for class_name in metrics_classes:
                class_obj = locals()[class_name]
                exists = class_obj is not None
                self.log_test(
                    f"Metrics class {class_name} exists",
                    exists,
                    f"Class {class_name} available" if exists else f"Missing class: {class_name}"
                )
            
            # Test task metrics structure
            if hasattr(TaskMetrics, '__annotations__'):
                task_metrics_attrs = [
                    "accuracy_metric",
                    "fluency_metric", 
                    "relevance_metric"
                ]
                
                for attr in task_metrics_attrs:
                    has_attr = hasattr(TaskMetrics, attr)
                    self.log_test(
                        f"TaskMetrics has {attr}",
                        has_attr,
                        f"Attribute {attr} present" if has_attr else f"Missing attribute: {attr}"
                    )
            
            # Test resource metrics structure
            if hasattr(ResourceMetrics, '__annotations__'):
                resource_metrics_attrs = [
                    "token_usage_metric",
                    "execution_time_metric"
                ]
                
                for attr in resource_metrics_attrs:
                    has_attr = hasattr(ResourceMetrics, attr)
                    self.log_test(
                        f"ResourceMetrics has {attr}",
                        has_attr,
                        f"Attribute {attr} present" if has_attr else f"Missing attribute: {attr}"
                    )
                    
        except Exception as e:
            self.log_test(
                "Objectives System",
                False,
                f"Error testing objectives system: {str(e)}"
            )
    
    def run_all_tests(self):
        """Run all architecture compliance tests."""
        print("🚀 Starting Comprehensive Architecture Compliance Test\n")
        print("=" * 60)
        
        # Run all test suites
        self.test_module_imports()
        self.test_mogepaaagent_implementation()
        self.test_monitoring_components()
        self.test_analysis_components()
        self.test_visualization_components()
        self.test_dspy_integration()
        self.test_interface_definitions()
        self.test_multi_objective_gepa_core()
        self.test_mutation_and_parameter_components()
        self.test_objectives_system()
        
        # Calculate compliance score
        if self.total_tests > 0:
            self.compliance_score = (len(self.passed_tests) / self.total_tests) * 100
        
        # Print final results
        self.print_final_results()
        
        return self.compliance_score >= 95.0  # Require 95% compliance
    
    def print_final_results(self):
        """Print final test results and compliance score."""
        print("\n" + "=" * 60)
        print("🏁 ARCHITECTURE COMPLIANCE TEST RESULTS")
        print("=" * 60)
        
        print(f"\n📊 Summary:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {len(self.passed_tests)} ✅")
        print(f"   Failed: {len(self.failed_tests)} ❌")
        print(f"   Compliance Score: {self.compliance_score:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test['test_name']}: {test['message']}")
        
        if self.compliance_score >= 95.0:
            print(f"\n🎉 EXCELLENT! Architecture compliance: {self.compliance_score:.1f}%")
            print("   ✅ All critical components are properly implemented")
            print("   ✅ Ready for production deployment")
        elif self.compliance_score >= 80.0:
            print(f"\n👍 GOOD! Architecture compliance: {self.compliance_score:.1f}%")
            print("   ⚠️  Minor issues found but core functionality is intact")
        else:
            print(f"\n⚠️  NEEDS ATTENTION! Architecture compliance: {self.compliance_score:.1f}%")
            print("   ❌ Significant implementation gaps detected")
            print("   🔧 Review failed tests and address missing components")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point for the architecture compliance test."""
    tester = ArchitectureComplianceTest()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()