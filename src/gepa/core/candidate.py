"""Candidate class for representing text components in GEPA algorithm.

This module defines the Candidate class which represents individual text components
(prompts, code, specifications) with their fitness scores, metadata, and evolutionary
history as part of the GEPA (Genetic-Pareto Algorithm) framework.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class ExecutionTrace(BaseModel):
    """Represents the execution trace of a candidate."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time: float
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class MutationRecord(BaseModel):
    """Represents a mutation operation in the candidate's history."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    mutation_type: str
    description: str
    parent_id: Optional[str] = None
    changes_made: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class Candidate(BaseModel):
    """Represents a text component in the GEPA algorithm.
    
    A Candidate contains text content (prompts, code, specifications), fitness
    scores for multiple objectives, metadata about its generation and evolution,
    execution traces, and feedback information.
    
    Attributes:
        id: Unique identifier for the candidate
        content: The text content of the candidate
        fitness_scores: Dictionary of fitness scores for different objectives
        generation: Generation number in the evolutionary process
        parent_ids: List of parent candidate IDs (for sexual reproduction)
        mutation_history: List of mutation operations applied to this candidate
        execution_traces: List of execution traces for this candidate
        metadata: Additional metadata about the candidate
        created_at: Timestamp when candidate was created
        last_modified: Timestamp when candidate was last modified
    """
    
    id: str = Field(default_factory=lambda: f"candidate_{int(time.time())}_{uuid.uuid4().hex[:8]}")
    content: str
    fitness_scores: Dict[str, float] = Field(default_factory=dict)
    generation: int = Field(default=0, ge=0)
    parent_ids: List[str] = Field(default_factory=list)
    mutation_history: List[MutationRecord] = Field(default_factory=list)
    execution_traces: List[ExecutionTrace] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    
    # Additional fields for tracking algorithm state
    is_elite: bool = Field(default=False)
    rank: Optional[int] = Field(default=None)  # Pareto rank
    crowding_distance: float = Field(default=0.0)
    domination_count: int = Field(default=0)
    dominated_solutions: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        validate_assignment=True
    )
    
    def __hash__(self) -> int:
        """Hash based on content and ID for use in sets."""
        return hash((self.id, self.content))
    
    def __eq__(self, other: object) -> bool:
        """Equality based on ID and content."""
        if not isinstance(other, Candidate):
            return False
        return self.id == other.id and self.content == other.content
    
    def update_content(self, new_content: str, mutation_type: str, description: str, 
                      changes_made: Optional[Dict[str, Any]] = None) -> None:
        """Update the candidate's content and record the mutation.
        
        Args:
            new_content: The new text content
            mutation_type: Type of mutation applied
            description: Description of the mutation
            changes_made: Dictionary of changes made during mutation
        """
        old_content = self.content
        self.content = new_content
        self.last_modified = datetime.now()
        
        # Record mutation
        mutation_record = MutationRecord(
            mutation_type=mutation_type,
            description=description,
            changes_made=changes_made or {}
        )
        self.mutation_history.append(mutation_record)
        
        # Update content hash in metadata
        self.metadata["content_hash"] = hashlib.md5(new_content.encode()).hexdigest()
        self.metadata["content_length"] = len(new_content)
    
    def add_fitness_score(self, objective: str, score: float) -> None:
        """Add or update a fitness score for an objective.
        
        Args:
            objective: Name of the objective
            score: Fitness score for the objective
        """
        self.fitness_scores[objective] = score
        self.last_modified = datetime.now()
    
    def set_fitness_scores(self, scores: Dict[str, float]) -> None:
        """Set all fitness scores at once.
        
        Args:
            scores: Dictionary of fitness scores
        """
        self.fitness_scores = scores.copy()
        self.last_modified = datetime.now()
    
    def get_fitness(self, objective: str) -> Optional[float]:
        """Get fitness score for a specific objective.
        
        Args:
            objective: Name of the objective
            
        Returns:
            Fitness score for the objective, or None if not set
        """
        return self.fitness_scores.get(objective)
    
    def get_objectives(self) -> List[str]:
        """Get list of all objective names.
        
        Returns:
            List of objective names
        """
        return list(self.fitness_scores.keys())
    
    def dominates(self, other: Candidate, objectives: Optional[List[str]] = None) -> bool:
        """Check if this candidate dominates another candidate in Pareto sense.
        
        A candidate x dominates y if:
        1. x is no worse than y in all objectives
        2. x is strictly better than y in at least one objective
        
        Args:
            other: Other candidate to compare against
            objectives: List of objectives to consider (default: all common objectives)
            
        Returns:
            True if this candidate dominates the other, False otherwise
        """
        if objectives is None:
            # Use common objectives between both candidates
            objectives = list(set(self.fitness_scores.keys()) & set(other.fitness_scores.keys()))
        
        if not objectives:
            return False
        
        # Check if this candidate is at least as good in all objectives
        at_least_as_good = True
        strictly_better = False
        
        for obj in objectives:
            self_score = self.fitness_scores.get(obj)
            other_score = other.fitness_scores.get(obj)
            
            if self_score is None or other_score is None:
                at_least_as_good = False
                break
            
            # Assuming higher fitness is better
            if self_score < other_score:
                at_least_as_good = False
                break
            elif self_score > other_score:
                strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def add_execution_trace(self, trace: ExecutionTrace) -> None:
        """Add an execution trace to the candidate.
        
        Args:
            trace: Execution trace to add
        """
        self.execution_traces.append(trace)
        self.last_modified = datetime.now()
    
    def get_latest_execution_trace(self) -> Optional[ExecutionTrace]:
        """Get the most recent execution trace.
        
        Returns:
            The most recent execution trace, or None if no traces exist
        """
        if not self.execution_traces:
            return None
        return max(self.execution_traces, key=lambda t: t.timestamp)
    
    def calculate_complexity(self) -> int:
        """Calculate a simple complexity metric based on content.
        
        Returns:
            Complexity score based on content length, unique tokens, etc.
        """
        if not self.content:
            return 0
        
        words = self.content.split()
        unique_words = set(word.lower() for word in words)
        lines = self.content.split('\n')
        
        # Simple complexity heuristic
        return len(words) + len(unique_words) + len(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary representation.
        
        Returns:
            Dictionary representation of the candidate
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Candidate:
        """Create candidate from dictionary representation.
        
        Args:
            data: Dictionary representation of the candidate
            
        Returns:
            Candidate instance
        """
        return cls.model_validate(data)
    
    def copy(self) -> Candidate:
        """Create a deep copy of the candidate.
        
        Returns:
            Deep copy of the candidate
        """
        # Create new candidate with same content but new ID
        new_candidate = Candidate(
            content=self.content,
            fitness_scores=self.fitness_scores.copy(),
            generation=self.generation,
            parent_ids=[self.id],  # This candidate is the parent
            metadata=self.metadata.copy(),
        )
        
        # Copy mutation history but update parent relationship
        new_candidate.mutation_history = self.mutation_history.copy()
        
        return new_candidate
    
    def __str__(self) -> str:
        """String representation of the candidate."""
        return (f"Candidate(id={self.id[:8]}..., fitness={self.fitness_scores}, "
                f"generation={self.generation}, is_elite={self.is_elite})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the candidate."""
        return (f"Candidate(id='{self.id}', content_length={len(self.content)}, "
                f"fitness_scores={self.fitness_scores}, generation={self.generation}, "
                f"rank={self.rank}, crowding_distance={self.crowding_distance})")


# Example usage
if __name__ == "__main__":
    # Create a simple candidate
    candidate = Candidate(
        content="Write a function that sorts a list of numbers.",
        fitness_scores={"accuracy": 0.8, "efficiency": 0.7},
        generation=0
    )
    
    print(f"Created candidate: {candidate}")
    print(f"ID: {candidate.id}")
    print(f"Content: {candidate.content}")
    print(f"Fitness scores: {candidate.fitness_scores}")
    print(f"Complexity: {candidate.calculate_complexity()}")
    
    # Test mutation
    candidate.update_content(
        "Write an efficient function that sorts a list of numbers using quicksort.",
        mutation_type="LLM_Guided",
        description="Added sorting algorithm specification"
    )
    
    print(f"\nAfter mutation:")
    print(f"Content: {candidate.content}")
    print(f"Mutation history: {len(candidate.mutation_history)} mutations")
    
    # Test Pareto dominance
    other_candidate = Candidate(
        content="Write a simple function to sort numbers.",
        fitness_scores={"accuracy": 0.7, "efficiency": 0.9},
        generation=0
    )
    
    print(f"\nDominance test:")
    print(f"Candidate 1 dominates Candidate 2: {candidate.dominates(other_candidate)}")
    print(f"Candidate 2 dominates Candidate 1: {other_candidate.dominates(candidate)}")