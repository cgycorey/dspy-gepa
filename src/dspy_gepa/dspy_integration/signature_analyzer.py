"""Signature analyzer for DSPy signature-aware optimization.

This module provides analysis of DSPy module signatures to enable
signature-aware optimization strategies that consider the structure
and constraints of DSPy modules during optimization.

Copyright (c) 2025 cgycorey. All rights reserved.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..core.interfaces import TaskType, MutationOperator
from ..core.mutation_engine import SemanticMutator, TaskSpecificMutator
from ..utils.logging import get_logger


_logger = get_logger(__name__)


class SignatureType(Enum):
    """Types of DSPy signatures."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    MULTI_STEP = "multi_step"
    SINGLE_STEP = "single_step"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"
    TOOL_USING = "tool_using"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


@dataclass
class FieldAnalysis:
    """Analysis of a DSPy signature field."""
    name: str
    field_type: str
    description: Optional[str]
    is_input: bool
    is_output: bool
    constraints: List[str]
    examples: List[Any]
    complexity_score: float
    
    def __post_init__(self):
        """Calculate complexity score."""
        if self.complexity_score == 0.0:
            self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate complexity score for this field."""
        score = 0.0
        
        # Base score for field type
        if "str" in self.field_type.lower():
            score += 1.0
        elif "list" in self.field_type.lower() or "dict" in self.field_type.lower():
            score += 2.0
        
        # Add score for constraints
        score += len(self.constraints) * 0.5
        
        # Add score for description length
        if self.description:
            score += min(len(self.description) / 100, 2.0)
        
        # Add score for examples
        score += len(self.examples) * 0.3
        
        return score


@dataclass
class SignatureAnalysis:
    """Complete analysis of a DSPy signature."""
    signature_name: str
    signature_type: SignatureType
    task_type: TaskType
    fields: Dict[str, FieldAnalysis]
    input_fields: List[str]
    output_fields: List[str]
    complexity_score: float
    mutation_hints: List[str]
    optimization_suggestions: List[str]
    
    def __post_init__(self):
        """Calculate derived properties."""
        self.input_fields = [name for name, field in self.fields.items() if field.is_input]
        self.output_fields = [name for name, field in self.fields.items() if field.is_output]
        
        if self.complexity_score == 0.0:
            self.complexity_score = sum(field.complexity_score for field in self.fields.values())


class DSPySignatureAnalyzer:
    """Analyzer for DSPy signatures to enable signature-aware optimization.
    
    This class analyzes DSPy module signatures to extract useful information
    for optimization, including field analysis, complexity metrics, and
    optimization suggestions based on the signature structure.
    """
    
    def __init__(self):
        """Initialize the signature analyzer."""
        # Patterns for different signature types
        self.signature_patterns = {
            'chain_of_thought': {
                'keywords': ['think', 'reason', 'step', 'explain'],
                'output_patterns': ['thought', 'reasoning', 'steps'],
                'input_patterns': ['question', 'problem', 'query']
            },
            'react': {
                'keywords': ['action', 'observation', 'thought'],
                'output_patterns': ['action', 'action_input', 'observation'],
                'input_patterns': ['question', 'task', 'goal']
            },
            'multi_step': {
                'keywords': ['step', 'phase', 'stage'],
                'output_patterns': ['step', 'result', 'intermediate'],
                'input_patterns': ['input', 'context', 'previous']
            },
            'retrieval_augmented': {
                'keywords': ['retrieve', 'search', 'context'],
                'output_patterns': ['answer', 'response'],
                'input_patterns': ['question', 'query', 'context']
            },
            'tool_using': {
                'keywords': ['tool', 'function', 'call'],
                'output_patterns': ['tool_call', 'function_call'],
                'input_patterns': ['task', 'request', 'parameters']
            }
        }
        
        # Task type mappings
        self.task_type_mappings = {
            ('question', 'answer'): TaskType.QUESTION_ANSWERING,
            ('translate', 'translation'): TaskType.TRANSLATION,
            ('summarize', 'summary'): TaskType.SUMMARIZATION,
            ('classify', 'classification'): TaskType.CLASSIFICATION,
            ('generate', 'generation'): TaskType.GENERATION,
            ('code', 'program'): TaskType.CODE_GENERATION
        }
        
        _logger.info("DSPySignatureAnalyzer initialized")
    
    def analyze_module(self, module: Any) -> SignatureAnalysis:
        """Analyze a DSPy module's signature.
        
        Args:
            module: DSPy module to analyze
            
        Returns:
            Complete signature analysis
        """
        try:
            # Extract signature information
            signature_info = self._extract_signature_info(module)
            
            # Analyze fields
            fields = self._analyze_fields(signature_info)
            
            # Determine signature type
            signature_type = self._determine_signature_type(signature_info, fields)
            
            # Determine task type
            task_type = self._determine_task_type(signature_info, fields)
            
            # Generate optimization hints
            mutation_hints = self._generate_mutation_hints(signature_type, fields)
            optimization_suggestions = self._generate_optimization_suggestions(
                signature_type, task_type, fields
            )
            
            # Create analysis result
            analysis = SignatureAnalysis(
                signature_name=getattr(module, '__class__', {}).get('__name__', 'Unknown'),
                signature_type=signature_type,
                task_type=task_type,
                fields=fields,
                input_fields=[],  # Will be calculated in __post_init__
                output_fields=[],  # Will be calculated in __post_init__
                complexity_score=0.0,  # Will be calculated in __post_init__
                mutation_hints=mutation_hints,
                optimization_suggestions=optimization_suggestions
            )
            
            _logger.info(f"Analyzed module: {analysis.signature_name} ({signature_type.value})")
            return analysis
            
        except Exception as e:
            _logger.error(f"Failed to analyze module: {e}")
            # Return default analysis
            return self._create_default_analysis(module)
    
    def _extract_signature_info(self, module: Any) -> Dict[str, Any]:
        """Extract signature information from a DSPy module."""
        info = {
            'class_name': getattr(module, '__class__', {}).get('__name__', ''),
            'module_name': getattr(module, '__module__', ''),
            'signature': None,
            'docstring': getattr(module, '__doc__', ''),
            'attributes': {}
        }
        
        # Try to extract DSPy signature
        if hasattr(module, 'signature'):
            info['signature'] = module.signature
        elif hasattr(module, '__dict__') and 'signature' in module.__dict__:
            info['signature'] = module.__dict__['signature']
        
        # Extract other attributes
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(module, attr_name)
                    if not callable(attr_value):
                        info['attributes'][attr_name] = str(attr_value)
                except:
                    continue
        
        return info
    
    def _analyze_fields(self, signature_info: Dict[str, Any]) -> Dict[str, FieldAnalysis]:
        """Analyze fields in the signature."""
        fields = {}
        
        signature = signature_info.get('signature')
        if not signature:
            return fields
        
        try:
            # Handle different signature formats
            if hasattr(signature, 'fields'):
                # DSPy Signature object
                for field_name, field_obj in signature.fields.items():
                    fields[field_name] = self._analyze_field(field_name, field_obj)
            
            elif hasattr(signature, '__dict__'):
                # Dict-like signature
                for field_name, field_value in signature.__dict__.items():
                    if not field_name.startswith('_'):
                        fields[field_name] = self._analyze_field_value(field_name, field_value)
            
            elif isinstance(signature, dict):
                # Plain dict signature
                for field_name, field_value in signature.items():
                    fields[field_name] = self._analyze_field_value(field_name, field_value)
        
        except Exception as e:
            _logger.warning(f"Failed to analyze signature fields: {e}")
        
        return fields
    
    def _analyze_field(self, name: str, field_obj: Any) -> FieldAnalysis:
        """Analyze a single field object."""
        try:
            # Extract field properties
            field_type = getattr(field_obj, '__class__', {}).get('__name__', str(type(field_obj)))
            description = getattr(field_obj, '__doc__', None) or getattr(field_obj, 'desc', None)
            
            # Determine if it's input/output
            is_input = self._is_input_field(name, field_obj)
            is_output = self._is_output_field(name, field_obj)
            
            # Extract constraints
            constraints = self._extract_field_constraints(field_obj)
            
            # Extract examples
            examples = getattr(field_obj, 'examples', [])
            
            return FieldAnalysis(
                name=name,
                field_type=field_type,
                description=description,
                is_input=is_input,
                is_output=is_output,
                constraints=constraints,
                examples=examples,
                complexity_score=0.0  # Will be calculated in __post_init__
            )
            
        except Exception as e:
            _logger.warning(f"Failed to analyze field {name}: {e}")
            return FieldAnalysis(
                name=name,
                field_type='unknown',
                description=None,
                is_input=False,
                is_output=False,
                constraints=[],
                examples=[],
                complexity_score=0.0
            )
    
    def _analyze_field_value(self, name: str, field_value: Any) -> FieldAnalysis:
        """Analyze a field value (when field is not an object)."""
        field_type = type(field_value).__name__
        description = str(field_value) if field_value else None
        is_input = self._is_input_field(name, field_value)
        is_output = self._is_output_field(name, field_value)
        
        return FieldAnalysis(
            name=name,
            field_type=field_type,
            description=description,
            is_input=is_input,
            is_output=is_output,
            constraints=[],
            examples=[],
            complexity_score=0.0
        )
    
    def _is_input_field(self, name: str, field_obj: Any) -> bool:
        """Determine if a field is an input field."""
        name_lower = name.lower()
        
        # Common input field patterns
        input_patterns = ['question', 'query', 'input', 'context', 'problem', 'task', 'prompt']
        
        return any(pattern in name_lower for pattern in input_patterns)
    
    def _is_output_field(self, name: str, field_obj: Any) -> bool:
        """Determine if a field is an output field."""
        name_lower = name.lower()
        
        # Common output field patterns
        output_patterns = ['answer', 'response', 'output', 'result', 'solution', 'generated']
        
        return any(pattern in name_lower for pattern in output_patterns)
    
    def _extract_field_constraints(self, field_obj: Any) -> List[str]:
        """Extract constraints from a field object."""
        constraints = []
        
        try:
            # Look for constraint attributes
            if hasattr(field_obj, 'constraints'):
                constraints.extend(field_obj.constraints)
            
            if hasattr(field_obj, 'min_length'):
                constraints.append(f"min_length: {field_obj.min_length}")
            
            if hasattr(field_obj, 'max_length'):
                constraints.append(f"max_length: {field_obj.max_length}")
            
            if hasattr(field_obj, 'required') and field_obj.required:
                constraints.append("required")
            
        except Exception:
            pass
        
        return constraints
    
    def _determine_signature_type(self, signature_info: Dict[str, Any], fields: Dict[str, FieldAnalysis]) -> SignatureType:
        """Determine the type of signature."""
        # Analyze signature name and description
        name_lower = signature_info['class_name'].lower()
        docstring_lower = signature_info['docstring'].lower()
        
        # Analyze field names
        all_field_names = ' '.join(fields.keys()).lower()
        
        # Calculate scores for each signature type
        scores = {}
        for sig_type, patterns in self.signature_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in name_lower:
                    score += 3
                if keyword in docstring_lower:
                    score += 2
                if keyword in all_field_names:
                    score += 1
            
            # Check field patterns
            input_patterns = patterns['input_patterns']
            output_patterns = patterns['output_patterns']
            
            for field_name in fields:
                field_name_lower = field_name.lower()
                if any(pattern in field_name_lower for pattern in input_patterns):
                    score += 2
                if any(pattern in field_name_lower for pattern in output_patterns):
                    score += 2
            
            scores[sig_type] = score
        
        # Determine the best match
        if not any(scores.values()):
            return SignatureType.UNKNOWN
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        # Require a minimum score to be confident
        if max_score < 3:
            return SignatureType.MULTI_STEP  # Default fallback
        
        return SignatureType(best_type)
    
    def _determine_task_type(self, signature_info: Dict[str, Any], fields: Dict[str, FieldAnalysis]) -> TaskType:
        """Determine the task type based on signature analysis."""
        # Combine all text for analysis
        all_text = ' '.join([
            signature_info['class_name'],
            signature_info['docstring'],
            ' '.join(fields.keys()),
            ' '.join(field.description or '' for field in fields.values())
        ]).lower()
        
        # Check for task type indicators
        for task_keywords, task_type in self.task_type_mappings.items():
            if all(keyword in all_text for keyword in task_keywords.split(',')):
                return task_type
        
        # Default to custom
        return TaskType.CUSTOM
    
    def _generate_mutation_hints(self, signature_type: SignatureType, fields: Dict[str, FieldAnalysis]) -> List[str]:
        """Generate mutation hints based on signature analysis."""
        hints = []
        
        # Base hints for all signature types
        hints.append("Use semantic mutations for prompt text")
        
        # Type-specific hints
        if signature_type == SignatureType.CHAIN_OF_THOUGHT:
            hints.append("Preserve reasoning structure in mutations")
            hints.append("Focus on clarity of thinking steps")
        elif signature_type == SignatureType.REACT:
            hints.append("Maintain action-observation cycle")
            hints.append("Preserve tool interaction patterns")
        elif signature_type == SignatureType.RETRIEVAL_AUGMENTED:
            hints.append("Consider context integration in mutations")
            hints.append("Focus on utilizing retrieved information")
        elif signature_type == SignatureType.TOOL_USING:
            hints.append("Preserve tool calling syntax")
            hints.append("Maintain parameter extraction patterns")
        
        # Field-specific hints
        complex_fields = [name for name, field in fields.items() if field.complexity_score > 2.0]
        if complex_fields:
            hints.append(f"Handle complex fields carefully: {', '.join(complex_fields)}")
        
        return hints
    
    def _generate_optimization_suggestions(
        self,
        signature_type: SignatureType,
        task_type: TaskType,
        fields: Dict[str, FieldAnalysis]
    ) -> List[str]:
        """Generate optimization suggestions based on signature analysis."""
        suggestions = []
        
        # General suggestions
        suggestions.append("Monitor field-level performance during optimization")
        suggestions.append("Consider field interdependencies in mutations")
        
        # Type-specific suggestions
        if signature_type == SignatureType.CHAIN_OF_THOUGHT:
            suggestions.append("Optimize for reasoning clarity and step completeness")
        elif signature_type == SignatureType.REACT:
            suggestions.append("Balance action selection and observation integration")
        elif signature_type == SignatureType.MULTI_STEP:
            suggestions.append("Ensure step continuity and logical progression")
        
        # Task-specific suggestions
        if task_type == TaskType.QUESTION_ANSWERING:
            suggestions.append("Focus on answer accuracy and relevance")
        elif task_type == TaskType.TRANSLATION:
            suggestions.append("Preserve meaning while optimizing fluency")
        elif task_type == TaskType.CODE_GENERATION:
            suggestions.append("Maintain syntax validity during optimization")
        
        # Field-level suggestions
        if len(fields) > 5:
            suggestions.append("Consider reducing signature complexity for better optimization")
        
        constrained_fields = [name for name, field in fields.items() if field.constraints]
        if constrained_fields:
            suggestions.append(f"Respect field constraints: {', '.join(constrained_fields)}")
        
        return suggestions
    
    def _create_default_analysis(self, module: Any) -> SignatureAnalysis:
        """Create a default analysis when analysis fails."""
        return SignatureAnalysis(
            signature_name=getattr(module, '__class__', {}).get('__name__', 'Unknown'),
            signature_type=SignatureType.UNKNOWN,
            task_type=TaskType.CUSTOM,
            fields={},
            input_fields=[],
            output_fields=[],
            complexity_score=1.0,
            mutation_hints=["Use conservative mutations for unknown signature type"],
            optimization_suggestions=["Monitor optimization progress carefully"]
        )
    
    def get_analysis_summary(self, analysis: SignatureAnalysis) -> str:
        """Get a readable summary of the signature analysis."""
        summary = f"Signature: {analysis.signature_name}\n"
        summary += f"Type: {analysis.signature_type.value}\n"
        summary += f"Task: {analysis.task_type.value}\n"
        summary += f"Complexity: {analysis.complexity_score:.2f}\n"
        summary += f"Input fields: {len(analysis.input_fields)}\n"
        summary += f"Output fields: {len(analysis.output_fields)}\n"
        
        if analysis.mutation_hints:
            summary += f"\nMutation hints:\n"
            for hint in analysis.mutation_hints[:3]:  # Show top 3
                summary += f"  - {hint}\n"
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return f"DSPySignatureAnalyzer(signature_patterns={len(self.signature_patterns)})"


# Alias for backward compatibility and architecture compliance
SignatureAnalyzer = DSPySignatureAnalyzer