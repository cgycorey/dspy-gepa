"""DSPY Adapter for converting between DSPY programs and GEPA candidates.

This module provides the main adapter class that handles conversion between DSPY programs
and GEPA candidates, enabling evolutionary optimization of DSPY programs through the GEPA
framework.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import time
from typing import Any, Dict, List, Optional, Type, Union, Callable
from datetime import datetime

from pydantic import BaseModel, Field

try:
    import dspy
except ImportError:
    dspy = None

from gepa.core.candidate import Candidate, ExecutionTrace, MutationRecord


class DSPYProgramMetadata(BaseModel):
    """Metadata for a DSPY program."""
    
    program_name: str
    program_type: str  # e.g., "ChainOfThought", "ReAct", "MultiChainComparison"
    num_steps: int
    required_inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    signature: Optional[str] = None
    module_info: Dict[str, Any] = Field(default_factory=dict)
    

class DSPYAdapter:
    """Adapter for converting between DSPY programs and GEPA candidates.
    
    This class provides bidirectional conversion between DSPY programs and GEPA
    candidates, enabling evolutionary optimization of DSPY programs. It handles
    serialization, deserialization, and maintains program structure throughout
    the optimization process.
    """
    
    def __init__(self, 
                 serialize_program_code: bool = True,
                 include_metadata: bool = True,
                 fitness_objectives: Optional[List[str]] = None):
        """Initialize the DSPY adapter.
        
        Args:
            serialize_program_code: Whether to serialize the full program code
            include_metadata: Whether to include detailed metadata in candidates
            fitness_objectives: List of fitness objectives to track
        """
        self.serialize_program_code = serialize_program_code
        self.include_metadata = include_metadata
        self.fitness_objectives = fitness_objectives or [
            "accuracy", "efficiency", "cost", "complexity"
        ]
        
        if dspy is None:
            raise ImportError("DSPY is required but not installed. "
                            "Install with: pip install dspy")
    
    def dspy_to_candidate(self, 
                          program: "dspy.Module", 
                          generation: int = 0,
                          parent_ids: Optional[List[str]] = None,
                          initial_fitness: Optional[Dict[str, float]] = None) -> Candidate:
        """Convert a DSPY program to a GEPA candidate.
        
        Args:
            program: DSPY program to convert
            generation: Generation number for the candidate
            parent_ids: List of parent candidate IDs
            initial_fitness: Initial fitness scores
            
        Returns:
            GEPA candidate representing the DSPY program
        """
        # Extract program information
        program_info = self._extract_program_info(program)
        
        # Serialize program content
        content = self._serialize_program(program, program_info)
        
        # Create metadata
        metadata = {
            "program_type": "dspy",
            "program_name": program_info.program_name,
            "program_class": program.__class__.__name__,
            "serialization_timestamp": datetime.now().isoformat(),
            "adapter_version": "0.1.0",
        }
        
        if self.include_metadata:
            metadata.update({
                "dspy_metadata": program_info.dict(),
                "num_parameters": len(self._get_predictor_parameters(program)),
                "predictor_info": self._get_predictor_info(program),
            })
        
        # Create candidate
        candidate = Candidate(
            content=content,
            fitness_scores=initial_fitness or {},
            generation=generation,
            parent_ids=parent_ids or [],
            metadata=metadata
        )
        
        return candidate
    
    def candidate_to_dspy(self, candidate: Candidate) -> "dspy.Module":
        """Convert a GEPA candidate back to a DSPY program.
        
        Args:
            candidate: GEPA candidate to convert
            
        Returns:
            DSPY program
            
        Raises:
            ValueError: If candidate cannot be converted to DSPY program
        """
        # Verify this is a DSPY candidate
        if candidate.metadata.get("program_type") != "dspy":
            raise ValueError("Candidate is not a DSPY program")
        
        try:
            # Deserialize program
            program = self._deserialize_program(candidate.content, candidate.metadata)
            return program
        except Exception as e:
            raise ValueError(f"Failed to convert candidate to DSPY program: {e}")
    
    def evaluate_candidate(self, 
                          candidate: Candidate,
                          eval_data: List[Dict[str, Any]],
                          metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """Evaluate a candidate's fitness on evaluation data.
        
        Args:
            candidate: Candidate to evaluate
            eval_data: Evaluation dataset
            metrics: Custom metrics to compute
            
        Returns:
            Dictionary of fitness scores
        """
        start_time = time.time()
        execution_trace = ExecutionTrace(
            timestamp=datetime.now(),
            execution_time=0.0,
            success=False
        )
        
        try:
            # Convert candidate to DSPY program
            program = self.candidate_to_dspy(candidate)
            
            # Evaluate program
            results = []
            total_cost = 0.0
            
            for example in eval_data:
                try:
                    # Execute program
                    result = program(**example)
                    results.append(result)
                    
                    # Track cost (estimate based on token usage if available)
                    if hasattr(program, 'predict'):
                        # This is a rough estimate - actual cost tracking would need
                        # more sophisticated monitoring
                        total_cost += 0.001  # Placeholder cost per prediction
                        
                except Exception as e:
                    results.append({"error": str(e)})
            
            # Calculate fitness scores
            fitness_scores = self._calculate_fitness_scores(
                results, eval_data, total_cost, time.time() - start_time, metrics
            )
            
            # Update candidate
            for objective, score in fitness_scores.items():
                candidate.add_fitness_score(objective, score)
            
            # Record successful execution
            execution_trace.success = True
            execution_trace.execution_time = time.time() - start_time
            execution_trace.output = f"Evaluated {len(eval_data)} examples"
            execution_trace.metrics = {
                "num_examples": len(eval_data),
                "success_count": sum(1 for r in results if "error" not in r),
                "total_cost": total_cost,
                "fitness_scores": fitness_scores
            }
            
        except Exception as e:
            execution_trace.success = False
            execution_trace.error = str(e)
            execution_trace.execution_time = time.time() - start_time
            
            # Set penalty fitness scores
            fitness_scores = {obj: 0.0 for obj in self.fitness_objectives}
        
        # Add execution trace
        candidate.add_execution_trace(execution_trace)
        
        return fitness_scores
    
    def _extract_program_info(self, program: "dspy.Module") -> DSPYProgramMetadata:
        """Extract information from a DSPY program."""
        program_name = getattr(program, 'name', program.__class__.__name__)
        program_type = program.__class__.__name__
        
        # Try to get signature information
        signature = None
        required_inputs = []
        outputs = []
        
        if hasattr(program, 'signature'):
            sig = program.signature
            signature = str(sig)
            if hasattr(sig, 'instructions'):
                # Try to extract input/output fields
                # This is a simplified approach - real implementation would be more sophisticated
                inputs_str = str(sig)
                if "->" in inputs_str:
                    parts = inputs_str.split("->")
                    if len(parts) == 2:
                        required_inputs = [p.strip() for p in parts[0].split(",")]
                        outputs = [p.strip() for p in parts[1].split(",")]
        
        # Count steps/predictors
        num_steps = len(self._get_predictor_parameters(program))
        
        # Get module info
        module_info = {
            "class_name": program.__class__.__name__,
            "module": program.__class__.__module__,
            "has_signature": hasattr(program, 'signature'),
            "has_predict": hasattr(program, 'predict'),
            "has_forward": hasattr(program, 'forward'),
        }
        
        return DSPYProgramMetadata(
            program_name=program_name,
            program_type=program_type,
            num_steps=num_steps,
            required_inputs=required_inputs,
            outputs=outputs,
            signature=signature,
            module_info=module_info
        )
    
    def _serialize_program(self, 
                          program: "dspy.Module", 
                          program_info: DSPYProgramMetadata) -> str:
        """Serialize a DSPY program to string format."""
        if self.serialize_program_code:
            # Get source code if possible
            try:
                source = inspect.getsource(program)
                return source
            except OSError:
                # Fallback for dynamically generated programs
                pass
        
        # Fallback: serialize as JSON
        try:
            # Get parameters safely
            parameters = self._get_predictor_parameters(program)
            # Ensure all values are JSON serializable
            serializable_params = {}
            for key, value in parameters.items():
                try:
                    json.dumps(value)  # Test if it's serializable
                    serializable_params[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable objects to strings
                    serializable_params[key] = str(value)
            
            program_dict = {
                "class_name": program.__class__.__name__,
                "module": program.__class__.__module__,
                "metadata": program_info.dict() if hasattr(program_info, 'dict') else str(program_info),
                "parameters": serializable_params,
            }
            
            return json.dumps(program_dict, indent=2)
        except Exception as e:
            # Last resort: return simple string representation
            return f"DSPY program: {program.__class__.__name__}"
    
    def _deserialize_program(self, content: str, metadata: Dict[str, Any]) -> "dspy.Module":
        """Deserialize a program from string content."""
        try:
            # Try to parse as JSON first
            program_dict = json.loads(content)
            
            # Get class information
            class_name = program_dict.get("class_name")
            module_name = program_dict.get("module")
            
            if class_name and module_name:
                # Import and instantiate the class
                module = importlib.import_module(module_name)
                program_class = getattr(module, class_name)
                
                # Create instance
                program = program_class()
                
                # Restore parameters if available
                parameters = program_dict.get("parameters", {})
                self._set_predictor_parameters(program, parameters)
                
                return program
            else:
                raise ValueError("Invalid program serialization format")
                
        except json.JSONDecodeError:
            # Try to evaluate as Python code
            try:
                # This is unsafe - in production, use proper sandboxing
                namespace = {}
                exec(content, namespace)
                
                # Find the program class in the namespace
                for name, obj in namespace.items():
                    if isinstance(obj, type) and issubclass(obj, dspy.Module):
                        return obj()
                        
                raise ValueError("No DSPY program found in deserialized code")
                
            except Exception as e:
                raise ValueError(f"Failed to deserialize program: {e}")
    
    def _get_predictor_parameters(self, program: "dspy.Module") -> Dict[str, Any]:
        """Extract parameters from DSPY predictors in the program."""
        parameters = {}
        
        # Handle case where named_modules might not be available
        if hasattr(program, 'named_modules'):
            for name, module in program.named_modules():
                if hasattr(module, 'dump_state'):
                    try:
                        parameters[name] = module.dump_state()
                    except Exception:
                        # Skip if unable to dump state
                        continue
                elif hasattr(module, 'instructions'):
                    parameters[name] = {
                        "instructions": module.instructions,
                    }
        else:
            # Fallback: check module attributes directly
            for attr_name in dir(program):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr_value = getattr(program, attr_name)
                    if hasattr(attr_value, 'dump_state'):
                        try:
                            parameters[attr_name] = attr_value.dump_state()
                        except Exception:
                            continue
                    elif hasattr(attr_value, 'instructions'):
                        parameters[attr_name] = {
                            "instructions": attr_value.instructions,
                        }
                except Exception:
                    continue
        
        return parameters
    
    def _set_predictor_parameters(self, 
                                 program: "dspy.Module", 
                                 parameters: Dict[str, Any]) -> None:
        """Set parameters for DSPY predictors in the program."""
        # Handle case where named_modules might not be available
        if hasattr(program, 'named_modules'):
            for name, module in program.named_modules():
                if name in parameters:
                    param_data = parameters[name]
                    
                    if hasattr(module, 'load_state') and hasattr(module, 'dump_state'):
                        try:
                            module.load_state(param_data)
                        except Exception:
                            continue
        else:
            # Fallback: set module attributes directly
            for attr_name, param_data in parameters.items():
                if hasattr(program, attr_name):
                    attr_value = getattr(program, attr_name)
                    if hasattr(attr_value, 'load_state'):
                        try:
                            attr_value.load_state(param_data)
                        except Exception:
                            continue
                elif hasattr(module, 'instructions') and "instructions" in param_data:
                    module.instructions = param_data["instructions"]
    
    def _get_predictor_info(self, program: "dspy.Module") -> Dict[str, Any]:
        """Get information about predictors in the program."""
        info = {}
        
        # Handle case where named_modules might not be available
        if hasattr(program, 'named_modules'):
            modules = program.named_modules()
        else:
            # Fallback: check module attributes directly
            modules = []
            for attr_name in dir(program):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr_value = getattr(program, attr_name)
                    if hasattr(attr_value, '__class__'):
                        modules.append((attr_name, attr_value))
                except Exception:
                    continue
        
        for name, module in modules:
            info[name] = {
                "class": module.__class__.__name__,
                "has_instructions": hasattr(module, 'instructions'),
                "has_dump_state": hasattr(module, 'dump_state'),
                "has_load_state": hasattr(module, 'load_state'),
            }
            
            if hasattr(module, 'demos'):
                info[name]["num_demos"] = len(module.demos) if module.demos else 0
        
        return info
    
    def _calculate_fitness_scores(self, 
                                 results: List[Dict[str, Any]],
                                 eval_data: List[Dict[str, Any]],
                                 total_cost: float,
                                 execution_time: float,
                                 custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """Calculate fitness scores for evaluation results."""
        scores = {}
        
        # Default accuracy metric
        if "accuracy" in self.fitness_objectives:
            success_count = sum(1 for r in results if "error" not in r)
            scores["accuracy"] = success_count / len(results) if results else 0.0
        
        # Efficiency (inverse of execution time)
        if "efficiency" in self.fitness_objectives:
            scores["efficiency"] = 1.0 / (1.0 + execution_time)
        
        # Cost (inverse of total cost)
        if "cost" in self.fitness_objectives:
            scores["cost"] = 1.0 / (1.0 + total_cost)
        
        # Complexity (based on result consistency)
        if "complexity" in self.fitness_objectives:
            # Lower complexity is better, so we invert it
            error_rate = sum(1 for r in results if "error" in r) / len(results) if results else 1.0
            scores["complexity"] = 1.0 - error_rate
        
        # Custom metrics
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                try:
                    score = metric_func(results, eval_data)
                    scores[metric_name] = float(score)
                except Exception:
                    scores[metric_name] = 0.0
        
        return scores
    
    def create_mutated_candidate(self, 
                               base_candidate: Candidate,
                               mutation_type: str,
                               description: str,
                               changes_made: Optional[Dict[str, Any]] = None) -> Candidate:
        """Create a mutated candidate from a base candidate.
        
        Args:
            base_candidate: Base candidate to mutate
            mutation_type: Type of mutation applied
            description: Description of the mutation
            changes_made: Dictionary describing changes made
            
        Returns:
            New mutated candidate
        """
        # Copy the base candidate
        mutated = base_candidate.copy()
        
        # Update generation and parent relationship
        mutated.generation = base_candidate.generation + 1
        mutated.parent_ids = [base_candidate.id]
        
        # Record mutation (this will be called by the mutator)
        # mutated.update_content(...) would be called by the actual mutation logic
        
        return mutated