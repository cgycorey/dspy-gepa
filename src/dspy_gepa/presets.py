"""Preset configurations for common optimization scenarios.

This module provides ready-to-use configurations for the most common
prompt optimization use cases, making it easy for beginners to get started.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .core.interfaces import Objective, TaskType, OptimizationDirection
from .core.objectives import (
    AccuracyMetric, FluencyMetric, RelevanceMetric, TokenUsageMetric,
    ExecutionTimeMetric, create_default_task_objectives,
    create_efficiency_focused_objectives, create_quality_focused_objectives
)


@dataclass
class OptimizationPreset:
    """Preset configuration for optimization scenarios."""
    name: str
    description: str
    objectives: List[Objective]
    max_generations: int
    population_size: int
    mutation_rate: float
    elitism_rate: float
    convergence_threshold: float
    max_time_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary."""
        return {
            'objectives': self.objectives,
            'max_generations': self.max_generations,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'elitism_rate': self.elitism_rate,
            'convergence_threshold': self.convergence_threshold,
            'max_time_minutes': self.max_time_minutes
        }


class PresetRegistry:
    """Registry of optimization presets."""
    
    def __init__(self):
        self._presets = self._create_builtin_presets()
    
    def _create_builtin_presets(self) -> Dict[str, OptimizationPreset]:
        """Create built-in preset configurations."""
        presets = {}
        
        # Quick optimization preset
        presets['quick'] = OptimizationPreset(
            name='quick',
            description='Fast optimization for quick experiments and prototyping',
            objectives=create_default_task_objectives(TaskType.GENERATION),
            max_generations=5,
            population_size=3,
            mutation_rate=0.3,
            elitism_rate=0.2,
            convergence_threshold=0.01,
            max_time_minutes=2
        )
        
        # Balanced optimization preset
        presets['balanced'] = OptimizationPreset(
            name='balanced',
            description='Balanced optimization for general use cases',
            objectives=create_default_task_objectives(TaskType.GENERATION),
            max_generations=10,
            population_size=5,
            mutation_rate=0.2,
            elitism_rate=0.3,
            convergence_threshold=0.005,
            max_time_minutes=5
        )
        
        # Quality-focused preset
        presets['quality'] = OptimizationPreset(
            name='quality',
            description='Focus on output quality and accuracy',
            objectives=create_quality_focused_objectives(TaskType.GENERATION),
            max_generations=15,
            population_size=7,
            mutation_rate=0.15,
            elitism_rate=0.4,
            convergence_threshold=0.003,
            max_time_minutes=10
        )
        
        # Efficiency-focused preset
        presets['efficiency'] = OptimizationPreset(
            name='efficiency',
            description='Optimize for speed and resource usage',
            objectives=create_efficiency_focused_objectives(TaskType.GENERATION),
            max_generations=8,
            population_size=4,
            mutation_rate=0.25,
            elitism_rate=0.25,
            convergence_threshold=0.008,
            max_time_minutes=3
        )
        
        # Translation preset
        presets['translation'] = OptimizationPreset(
            name='translation',
            description='Optimized for translation tasks',
            objectives=[
                AccuracyMetric(weight=0.6),
                FluencyMetric(weight=0.3),
                TokenUsageMetric(weight=0.1)
            ],
            max_generations=12,
            population_size=6,
            mutation_rate=0.2,
            elitism_rate=0.35,
            convergence_threshold=0.004,
            max_time_minutes=8
        )
        
        # Summarization preset
        presets['summarization'] = OptimizationPreset(
            name='summarization',
            description='Optimized for text summarization',
            objectives=[
                AccuracyMetric(weight=0.4),
                RelevanceMetric(weight=0.4),
                FluencyMetric(weight=0.2)
            ],
            max_generations=10,
            population_size=5,
            mutation_rate=0.18,
            elitism_rate=0.3,
            convergence_threshold=0.005,
            max_time_minutes=6
        )
        
        # Code generation preset
        presets['code_generation'] = OptimizationPreset(
            name='code_generation',
            description='Optimized for code generation tasks',
            objectives=[
                AccuracyMetric(weight=0.7),
                TokenUsageMetric(weight=0.2),
                ExecutionTimeMetric(weight=0.1)
            ],
            max_generations=8,
            population_size=4,
            mutation_rate=0.22,
            elitism_rate=0.3,
            convergence_threshold=0.006,
            max_time_minutes=7
        )
        
        return presets
    
    def get_preset(self, name: str) -> OptimizationPreset:
        """Get a preset by name."""
        if name not in self._presets:
            available = ', '.join(self._presets.keys())
            raise ValueError(
                f"Unknown preset '{name}'. Available presets: {available}"
            )
        return self._presets[name]
    
    def list_presets(self) -> Dict[str, str]:
        """List all available presets with descriptions."""
        return {name: preset.description for name, preset in self._presets.items()}
    
    def register_preset(self, preset: OptimizationPreset):
        """Register a custom preset."""
        self._presets[preset.name] = preset
    
    def get_preset_for_task_type(self, task_type: TaskType, focus: str = 'balanced') -> OptimizationPreset:
        """Get a preset based on task type and focus."""
        if task_type == TaskType.TRANSLATION:
            return self.get_preset('translation')
        elif task_type == TaskType.SUMMARIZATION:
            return self.get_preset('summarization')
        elif task_type == TaskType.CODE_GENERATION:
            return self.get_preset('code_generation')
        else:
            return self.get_preset(focus)


# Global preset registry
preset_registry = PresetRegistry()


# Convenience functions
def get_preset(name: str) -> OptimizationPreset:
    """Get a preset by name."""
    return preset_registry.get_preset(name)


def list_presets() -> Dict[str, str]:
    """List all available presets."""
    return preset_registry.list_presets()


def get_quick_preset() -> OptimizationPreset:
    """Get the quick optimization preset."""
    return preset_registry.get_preset('quick')


def get_balanced_preset() -> OptimizationPreset:
    """Get the balanced optimization preset."""
    return preset_registry.get_preset('balanced')


def get_quality_preset() -> OptimizationPreset:
    """Get the quality-focused optimization preset."""
    return preset_registry.get_preset('quality')


def get_efficiency_preset() -> OptimizationPreset:
    """Get the efficiency-focused optimization preset."""
    return preset_registry.get_preset('efficiency')
