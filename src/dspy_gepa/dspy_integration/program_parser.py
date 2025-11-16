"""DSPY Program Parser for analysis and optimization.

This module provides comprehensive parsing and analysis capabilities for DSPY programs
within the GEPA framework. It extracts program structure, identifies optimization targets,
and provides detailed analysis to guide the evolutionary optimization process.
"""

from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

try:
    import dspy
except ImportError:
    dspy = None


class ComponentType(Enum):
    """Types of DSPY components."""
    PREDICTOR = "predictor"
    SIGNATURE = "signature"
    MODULE = "module"
    CHAIN = "chain"
    RETRIEVE = "retrieve"
    ASSERT = "assert"
    SYNTHESIZE = "synthesize"
    UNKNOWN = "unknown"


class OptimizationTarget(Enum):
    """Types of optimization targets."""
    INSTRUCTIONS = "instructions"  # Prompt instructions
    DEMOS = "demos"  # Few-shot examples
    SIGNATURE = "signature"  # Input/output signature
    ARCHITECTURE = "architecture"  # Program structure
    PARAMETERS = "parameters"  # Module parameters


@dataclass
class DSPYComponent:
    """Represents a component in a DSPY program."""
    name: str
    component_type: ComponentType
    class_name: str
    module_path: str
    instructions: Optional[str] = None
    signature: Optional[str] = None
    demos: Optional[List[Dict[str, Any]]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    children: List['DSPYComponent'] = field(default_factory=list)
    parent_name: Optional[str] = None
    line_number: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.component_type == ComponentType.UNKNOWN:
            self._infer_component_type()
    
    def _infer_component_type(self) -> None:
        """Infer component type from class name and structure."""
        class_lower = self.class_name.lower()
        
        if 'predict' in class_lower:
            self.component_type = ComponentType.PREDICTOR
        elif 'signature' in class_lower:
            self.component_type = ComponentType.SIGNATURE
        elif 'chain' in class_lower:
            self.component_type = ComponentType.CHAIN
        elif 'retrieve' in class_lower:
            self.component_type = ComponentType.RETRIEVE
        elif 'assert' in class_lower:
            self.component_type = ComponentType.ASSERT
        elif 'synthesize' in class_lower:
            self.component_type = ComponentType.SYNTHESIZE
        elif 'module' in class_lower:
            self.component_type = ComponentType.MODULE


class DSPYProgramInfo(BaseModel):
    """Comprehensive information about a DSPY program."""
    
    # Basic information
    program_name: str
    program_class: str
    module_path: str
    source_code: Optional[str] = None
    
    # Structural information
    components: List[Dict[str, Any]] = Field(default_factory=list)
    component_hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    total_components: int = Field(default=0)
    
    # Optimization targets
    optimization_targets: List[str] = Field(default_factory=list)
    mutable_instructions: List[str] = Field(default_factory=list)
    mutable_demos: List[str] = Field(default_factory=list)
    mutable_signatures: List[str] = Field(default_factory=list)
    
    # Program characteristics
    complexity_score: float = Field(default=0.0)
    num_parameters: int = Field(default=0)
    estimated_tokens: int = Field(default=0)
    
    # Analysis results
    analysis_warnings: List[str] = Field(default_factory=list)
    analysis_suggestions: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ProgramParser:
    """Parser and analyzer for DSPY programs.
    
    This class provides comprehensive analysis of DSPY programs to identify
    optimization opportunities, extract program structure, and guide the GEPA
    evolutionary optimization process.
    """
    
    def __init__(self, 
                 include_source_analysis: bool = True,
                 include_complexity_analysis: bool = True,
                 include_optimization_suggestions: bool = True):
        """Initialize the program parser.
        
        Args:
            include_source_analysis: Whether to analyze source code
            include_complexity_analysis: Whether to calculate complexity metrics
            include_optimization_suggestions: Whether to generate optimization suggestions
        """
        self.include_source_analysis = include_source_analysis
        self.include_complexity_analysis = include_complexity_analysis
        self.include_optimization_suggestions = include_optimization_suggestions
        
        if dspy is None:
            raise ImportError("DSPY is required but not installed")
    
    def parse_program(self, program: "dspy.Module") -> DSPYProgramInfo:
        """Parse and analyze a DSPY program.
        
        Args:
            program: DSPY program to parse
            
        Returns:
            Comprehensive program information
        """
        # Extract basic information
        program_info = DSPYProgramInfo(
            program_name=getattr(program, 'name', program.__class__.__name__),
            program_class=program.__class__.__name__,
            module_path=program.__class__.__module__
        )
        
        # Get source code if requested
        if self.include_source_analysis:
            try:
                program_info.source_code = inspect.getsource(program)
            except OSError:
                program_info.source_code = None
        
        # Extract components
        components = self._extract_components(program)
        program_info.components = [comp.__dict__ for comp in components]
        program_info.total_components = len(components)
        
        # Build hierarchy
        program_info.component_hierarchy = self._build_component_hierarchy(components)
        
        # Identify optimization targets
        optimization_targets = self._identify_optimization_targets(components)
        program_info.optimization_targets = [target.value for target in optimization_targets]
        program_info.mutable_instructions = [
            comp.name for comp in components if comp.instructions
        ]
        program_info.mutable_demos = [
            comp.name for comp in components if comp.demos
        ]
        program_info.mutable_signatures = [
            comp.name for comp in components if comp.signature
        ]
        
        # Calculate complexity
        if self.include_complexity_analysis:
            program_info.complexity_score = self._calculate_complexity(components, program_info.source_code)
            program_info.num_parameters = self._count_parameters(program)
            program_info.estimated_tokens = self._estimate_tokens(components)
        
        # Generate analysis and suggestions
        if self.include_optimization_suggestions:
            warnings, suggestions = self._analyze_program(components, program_info)
            program_info.analysis_warnings = warnings
            program_info.analysis_suggestions = suggestions
        
        return program_info
    
    def extract_mutation_targets(self, program_info: DSPYProgramInfo) -> Dict[str, List[Dict[str, Any]]]:
        """Extract specific targets for mutation operations.
        
        Args:
            program_info: Parsed program information
            
        Returns:
            Dictionary of mutation targets by type
        """
        targets = {
            "instructions": [],
            "demos": [],
            "signatures": [],
            "parameters": []
        }
        
        for comp_dict in program_info.components:
            comp = DSPYComponent(**comp_dict)
            
            # Instructions
            if comp.instructions:
                targets["instructions"].append({
                    "component_name": comp.name,
                    "component_type": comp.component_type.value,
                    "current_instructions": comp.instructions,
                    "length": len(comp.instructions),
                    "optimization_priority": self._calculate_instruction_priority(comp)
                })
            
            # Demos
            if comp.demos:
                targets["demos"].append({
                    "component_name": comp.name,
                    "component_type": comp.component_type.value,
                    "current_demos": comp.demos,
                    "num_demos": len(comp.demos),
                    "optimization_priority": self._calculate_demo_priority(comp)
                })
            
            # Signatures
            if comp.signature:
                targets["signatures"].append({
                    "component_name": comp.name,
                    "component_type": comp.component_type.value,
                    "current_signature": comp.signature,
                    "optimization_priority": self._calculate_signature_priority(comp)
                })
            
            # Parameters
            if comp.parameters:
                targets["parameters"].append({
                    "component_name": comp.name,
                    "component_type": comp.component_type.value,
                    "current_parameters": comp.parameters,
                    "optimization_priority": self._calculate_parameter_priority(comp)
                })
        
        # Sort targets by priority
        for target_type in targets:
            targets[target_type].sort(key=lambda x: x["optimization_priority"], reverse=True)
        
        return targets
    
    def _extract_components(self, program: "dspy.Module") -> List[DSPYComponent]:
        """Extract all components from a DSPY program."""
        components = []
        
        # Use DSPY's named_modules to get all components
        for name, module in program.named_modules():
            component = DSPYComponent(
                name=name,
                component_type=ComponentType.UNKNOWN,
                class_name=module.__class__.__name__,
                module_path=module.__class__.__module__
            )
            
            # Extract instructions
            if hasattr(module, 'instructions'):
                component.instructions = module.instructions
            
            # Extract signature
            if hasattr(module, 'signature'):
                component.signature = str(module.signature)
            
            # Extract demos
            if hasattr(module, 'demos'):
                component.demos = module.demos or []
            
            # Extract parameters
            if hasattr(module, 'dump_state'):
                try:
                    component.parameters = module.dump_state()
                except Exception:
                    pass
            
            # Get line number if source is available
            try:
                source_lines = inspect.getsourcelines(module)
                component.line_number = source_lines[1]
            except (OSError, TypeError):
                pass
            
            components.append(component)
        
        return components
    
    def _build_component_hierarchy(self, components: List[DSPYComponent]) -> Dict[str, List[str]]:
        """Build hierarchical relationship between components."""
        hierarchy = {}
        
        # Simple hierarchy based on naming conventions
        for comp in components:
            # Find potential children (nested components)
            children = []
            for other_comp in components:
                if other_comp.name.startswith(comp.name + "."):
                    children.append(other_comp.name)
            
            hierarchy[comp.name] = children
        
        return hierarchy
    
    def _identify_optimization_targets(self, components: List[DSPYComponent]) -> List[OptimizationTarget]:
        """Identify what can be optimized in the program."""
        targets = set()
        
        for comp in components:
            if comp.instructions:
                targets.add(OptimizationTarget.INSTRUCTIONS)
            if comp.demos:
                targets.add(OptimizationTarget.DEMOS)
            if comp.signature:
                targets.add(OptimizationTarget.SIGNATURE)
            if comp.parameters:
                targets.add(OptimizationTarget.PARAMETERS)
        
        # Always include architecture as a potential target
        targets.add(OptimizationTarget.ARCHITECTURE)
        
        return list(targets)
    
    def _calculate_complexity(self, components: List[DSPYComponent], source_code: Optional[str]) -> float:
        """Calculate program complexity score."""
        complexity = 0.0
        
        # Component-based complexity
        complexity += len(components) * 1.0
        
        # Instruction complexity
        for comp in components:
            if comp.instructions:
                complexity += len(comp.instructions.split()) * 0.1
            if comp.demos:
                complexity += len(comp.demos) * 0.5
        
        # Source code complexity if available
        if source_code:
            try:
                tree = ast.parse(source_code)
                complexity += self._calculate_ast_complexity(tree)
            except SyntaxError:
                pass
        
        return complexity
    
    def _calculate_ast_complexity(self, node: ast.AST) -> float:
        """Calculate complexity from AST."""
        complexity = 0.0
        
        if isinstance(node, ast.FunctionDef):
            complexity += 1.0
            # Add complexity for each statement
            complexity += len(node.body) * 0.1
        elif isinstance(node, ast.ClassDef):
            complexity += 2.0
            complexity += len(node.body) * 0.1
        elif isinstance(node, ast.If):
            complexity += 0.5
        elif isinstance(node, ast.For):
            complexity += 0.5
        elif isinstance(node, ast.While):
            complexity += 0.5
        
        # Recursively process child nodes
        for child in ast.iter_child_nodes(node):
            complexity += self._calculate_ast_complexity(child)
        
        return complexity
    
    def _count_parameters(self, program: "dspy.Module") -> int:
        """Count total parameters in the program."""
        param_count = 0
        
        for _, module in program.named_modules():
            if hasattr(module, 'dump_state'):
                try:
                    state = module.dump_state()
                    param_count += len(str(state))  # Rough estimate
                except Exception:
                    pass
        
        return param_count
    
    def _estimate_tokens(self, components: List[DSPYComponent]) -> int:
        """Estimate total tokens used by the program."""
        total_tokens = 0
        
        for comp in components:
            if comp.instructions:
                total_tokens += len(comp.instructions.split())
            if comp.signature:
                total_tokens += len(comp.signature.split())
            if comp.demos:
                for demo in comp.demos:
                    total_tokens += len(str(demo).split())
        
        return total_tokens
    
    def _analyze_program(self, 
                         components: List[DSPYComponent], 
                         program_info: DSPYProgramInfo) -> Tuple[List[str], List[str]]:
        """Analyze program and generate warnings and suggestions."""
        warnings = []
        suggestions = []
        
        # Check for common issues
        if len(components) == 0:
            warnings.append("No components found in program")
        
        # Check for missing instructions
        components_without_instructions = [c for c in components if not c.instructions]
        if components_without_instructions:
            warnings.append(f"{len(components_without_instructions)} components lack instructions")
            suggestions.append("Consider adding instructions to improve performance")
        
        # Check for missing demos
        components_without_demos = [c for c in components if not c.demos]
        predictors_without_demos = [c for c in components_without_demos 
                                   if c.component_type == ComponentType.PREDICTOR]
        if predictors_without_demos:
            suggestions.append("Consider adding few-shot examples to predictors")
        
        # Complexity warnings
        if program_info.complexity_score > 50:
            warnings.append("Program has high complexity, consider simplification")
        
        # Optimization suggestions
        if program_info.mutable_instructions:
            suggestions.append("Instructions can be optimized for better performance")
        
        if program_info.mutable_demos:
            suggestions.append("Few-shot examples can be optimized for better accuracy")
        
        # Architecture suggestions
        if len(components) > 10:
            suggestions.append("Consider modularizing large programs")
        
        return warnings, suggestions
    
    def _calculate_instruction_priority(self, component: DSPYComponent) -> float:
        """Calculate optimization priority for instructions."""
        if not component.instructions:
            return 0.0
        
        priority = 1.0
        
        # Higher priority for predictors
        if component.component_type == ComponentType.PREDICTOR:
            priority += 1.0
        
        # Higher priority for shorter instructions (more room for improvement)
        instruction_length = len(component.instructions)
        if instruction_length < 50:
            priority += 0.5
        elif instruction_length < 100:
            priority += 0.25
        
        # Lower priority if already very detailed
        if instruction_length > 500:
            priority -= 0.5
        
        return priority
    
    def _calculate_demo_priority(self, component: DSPYComponent) -> float:
        """Calculate optimization priority for demos."""
        if not component.demos:
            return 0.0
        
        priority = 1.0
        
        # Higher priority for predictors
        if component.component_type == ComponentType.PREDICTOR:
            priority += 1.0
        
        # Priority based on number of demos
        num_demos = len(component.demos)
        if num_demos < 3:
            priority += 0.5  # Need more demos
        elif num_demos > 8:
            priority += 0.3  # Too many demos, might need optimization
        
        return priority
    
    def _calculate_signature_priority(self, component: DSPYComponent) -> float:
        """Calculate optimization priority for signatures."""
        if not component.signature:
            return 0.0
        
        priority = 0.5  # Lower priority generally
        
        # Higher priority if signature is simple
        if len(component.signature) < 100:
            priority += 0.3
        
        return priority
    
    def _calculate_parameter_priority(self, component: DSPYComponent) -> float:
        """Calculate optimization priority for parameters."""
        if not component.parameters:
            return 0.0
        
        priority = 0.3  # Generally lower priority
        
        # Higher priority for complex parameters
        if len(str(component.parameters)) > 200:
            priority += 0.2
        
        return priority
    
    def generate_mutation_plan(self, program_info: DSPYProgramInfo) -> Dict[str, Any]:
        """Generate a comprehensive mutation plan for the program.
        
        Args:
            program_info: Parsed program information
            
        Returns:
            Mutation plan with priorities and recommendations
        """
        targets = self.extract_mutation_targets(program_info)
        
        # Calculate overall mutation priorities
        plan = {
            "program_name": program_info.program_name,
            "complexity_score": program_info.complexity_score,
            "mutation_targets": targets,
            "recommended_mutations": [],
            "mutation_sequence": []
        }
        
        # Generate recommended mutations
        for target_type, target_list in targets.items():
            if target_list:
                best_target = target_list[0]  # Highest priority
                plan["recommended_mutations"].append({
                    "type": target_type,
                    "target": best_target,
                    "reasoning": self._generate_mutation_reasoning(target_type, best_target)
                })
        
        # Suggest mutation sequence
        plan["mutation_sequence"] = self._suggest_mutation_sequence(targets, program_info)
        
        return plan
    
    def _generate_mutation_reasoning(self, target_type: str, target: Dict[str, Any]) -> str:
        """Generate reasoning for why this target should be mutated."""
        component_name = target["component_name"]
        priority = target["optimization_priority"]
        
        if target_type == "instructions":
            return f"Instructions in {component_name} have high optimization potential (priority: {priority:.2f})"
        elif target_type == "demos":
            return f"Few-shot examples in {component_name} can be improved for better accuracy (priority: {priority:.2f})"
        elif target_type == "signatures":
            return f"Signature in {component_name} can be refined (priority: {priority:.2f})"
        elif target_type == "parameters":
            return f"Parameters in {component_name} can be optimized (priority: {priority:.2f})"
        
        return f"Target {component_name} has optimization potential"
    
    def _suggest_mutation_sequence(self, targets: Dict[str, List[Dict[str, Any]]], program_info: DSPYProgramInfo) -> List[str]:
        """Suggest an optimal sequence of mutations."""
        sequence = []
        
        # Start with instructions (highest impact)
        if targets["instructions"]:
            sequence.append("Optimize instructions")
        
        # Then demos (improves accuracy)
        if targets["demos"]:
            sequence.append("Optimize few-shot examples")
        
        # Then signatures (structural improvement)
        if targets["signatures"]:
            sequence.append("Refine signatures")
        
        # Finally parameters (fine-tuning)
        if targets["parameters"]:
            sequence.append("Adjust parameters")
        
        # Architecture changes for complex programs
        if program_info.complexity_score > 30:
            sequence.append("Consider architectural modifications")
        
        return sequence