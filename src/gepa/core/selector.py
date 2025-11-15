"""Pareto-optimal selection for GEPA algorithm.

This module implements the ParetoSelector class which performs Pareto-optimal
selection for multi-objective optimization in the GEPA (Genetic-Pareto Algorithm)
framework. It implements NSGA-II style selection with crowding distance.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from .candidate import Candidate


class ParetoSelector:
    """Implements Pareto-optimal selection for multi-objective optimization.
    
    This selector uses non-dominated sorting and crowding distance calculation
    to select diverse, high-quality candidates for the next generation.
    It implements the core concepts from NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    Attributes:
        objectives: List of objective names to optimize
        maximize_objectives: Dictionary indicating whether each objective should be maximized
    """
    
    def __init__(self, objectives: List[str], maximize_objectives: Optional[Dict[str, bool]] = None):
        """Initialize the Pareto selector.
        
        Args:
            objectives: List of objective names to consider for selection
            maximize_objectives: Dictionary indicating whether to maximize each objective.
                                  If None, all objectives are assumed to be maximized.
        """
        self.objectives = objectives
        self.maximize_objectives = maximize_objectives or {obj: True for obj in objectives}
        
        # Validate that all objectives have maximize settings
        for obj in objectives:
            if obj not in self.maximize_objectives:
                self.maximize_objectives[obj] = True
    
    def dominates(self, candidate_a: Candidate, candidate_b: Candidate) -> bool:
        """Check if candidate_a dominates candidate_b.
        
        A candidate a dominates b if:
        1. a is no worse than b in all objectives
        2. a is strictly better than b in at least one objective
        
        Args:
            candidate_a: First candidate
            candidate_b: Second candidate
            
        Returns:
            True if candidate_a dominates candidate_b, False otherwise
        """
        at_least_as_good = True
        strictly_better = False
        
        for obj in self.objectives:
            a_score = candidate_a.get_fitness(obj)
            b_score = candidate_b.get_fitness(obj)
            
            # Skip if either candidate doesn't have this objective score
            if a_score is None or b_score is None:
                continue
            
            maximize = self.maximize_objectives[obj]
            
            if maximize:
                # Higher is better
                if a_score < b_score:
                    at_least_as_good = False
                    break
                elif a_score > b_score:
                    strictly_better = True
            else:
                # Lower is better (e.g., error rate, latency)
                if a_score > b_score:
                    at_least_as_good = False
                    break
                elif a_score < b_score:
                    strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def non_dominated_sort(self, population: List[Candidate]) -> List[List[Candidate]]:
        """Perform non-dominated sorting to organize population into Pareto fronts.
        
        Args:
            population: List of candidates to sort
            
        Returns:
            List of fronts, where each front is a list of non-dominated candidates
        """
        # Reset domination counts and dominated solutions
        for candidate in population:
            candidate.domination_count = 0
            candidate.dominated_solutions = []
            candidate.rank = None
        
        fronts: List[List[Candidate]] = []
        first_front: List[Candidate] = []
        
        # Find domination relationships
        for i, candidate_a in enumerate(population):
            for j, candidate_b in enumerate(population):
                if i == j:
                    continue
                
                if self.dominates(candidate_a, candidate_b):
                    candidate_a.dominated_solutions.append(candidate_b.id)
                elif self.dominates(candidate_b, candidate_a):
                    candidate_a.domination_count += 1
            
            # If not dominated by anyone, add to first front
            if candidate_a.domination_count == 0:
                candidate_a.rank = 0
                first_front.append(candidate_a)
        
        fronts.append(first_front)
        
        # Build subsequent fronts
        i = 0
        while i < len(fronts):
            next_front: List[Candidate] = []
            
            for candidate in fronts[i]:
                for dominated_id in candidate.dominated_solutions:
                    # Find the dominated candidate
                    dominated_candidate = next(
                        (c for c in population if c.id == dominated_id), 
                        None
                    )
                    if dominated_candidate:
                        dominated_candidate.domination_count -= 1
                        if dominated_candidate.domination_count == 0:
                            dominated_candidate.rank = i + 1
                            next_front.append(dominated_candidate)
            
            if next_front:
                fronts.append(next_front)
            
            i += 1
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[Candidate]) -> None:
        """Calculate crowding distance for candidates in a front.
        
        Crowding distance measures how dense the neighborhood around a candidate
        is. Higher crowding distance means the candidate is in a less crowded area.
        
        Args:
            front: List of candidates in the same Pareto front
        """
        if len(front) == 0:
            return
        
        # Initialize crowding distances
        for candidate in front:
            candidate.crowding_distance = 0.0
        
        # If front has only 1 or 2 candidates, give them high crowding distance
        if len(front) <= 2:
            for candidate in front:
                candidate.crowding_distance = float('inf')
            return
        
        # Calculate crowding distance for each objective
        for obj in self.objectives:
            # Sort candidates by this objective
            sorted_candidates = sorted(
                [c for c in front if c.get_fitness(obj) is not None],
                key=lambda c: c.get_fitness(obj),
                reverse=not self.maximize_objectives[obj]
            )
            
            if len(sorted_candidates) < 2:
                continue
            
            # Set boundary candidates to infinity
            sorted_candidates[0].crowding_distance = float('inf')
            sorted_candidates[-1].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_min = sorted_candidates[0].get_fitness(obj)
            obj_max = sorted_candidates[-1].get_fitness(obj)
            
            if obj_max == obj_min:
                continue  # Avoid division by zero
            
            # Update crowding distance for intermediate candidates
            for i in range(1, len(sorted_candidates) - 1):
                if sorted_candidates[i].crowding_distance != float('inf'):
                    distance = (
                        sorted_candidates[i + 1].get_fitness(obj) - 
                        sorted_candidates[i - 1].get_fitness(obj)
                    ) / (obj_max - obj_min)
                    sorted_candidates[i].crowding_distance += distance
    
    def environmental_selection(self, population: List[Candidate], 
                               target_size: int) -> List[Candidate]:
        """Select candidates for the next generation using environmental selection.
        
        Args:
            population: Current population
            target_size: Desired population size
            
        Returns:
            Selected candidates for the next generation
        """
        if len(population) <= target_size:
            return population.copy()
        
        # Perform non-dominated sorting
        fronts = self.non_dominated_sort(population)
        
        # Calculate crowding distances for each front
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select fronts until we exceed target size
        selected: List[Candidate] = []
        
        for front in fronts:
            if len(selected) + len(front) <= target_size:
                selected.extend(front)
            else:
                # Sort current front by crowding distance (descending)
                front_sorted = sorted(
                    front, 
                    key=lambda c: c.crowding_distance, 
                    reverse=True
                )
                
                # Fill remaining slots with best crowding distance
                remaining_slots = target_size - len(selected)
                selected.extend(front_sorted[:remaining_slots])
                break
        
        return selected
    
    def tournament_selection(self, population: List[Candidate], 
                            tournament_size: int = 2) -> Candidate:
        """Perform tournament selection to choose a parent candidate.
        
        Args:
            population: Population to select from
            tournament_size: Number of candidates in each tournament
            
        Returns:
            Selected candidate
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        
        # Randomly select tournament participants
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Perform non-dominated sort on tournament
        fronts = self.non_dominated_sort(tournament)
        
        # Select best candidate (lowest rank, then highest crowding distance)
        best_candidates = fronts[0]  # First (best) front
        
        if len(best_candidates) == 1:
            return best_candidates[0]
        
        # If multiple candidates in best front, select by crowding distance
        best_candidates.sort(key=lambda c: c.crowding_distance, reverse=True)
        return best_candidates[0]
    
    def select_parents(self, population: List[Candidate], 
                       num_parents: int, 
                       tournament_size: int = 2) -> List[Candidate]:
        """Select parent candidates for reproduction.
        
        Args:
            population: Current population
            num_parents: Number of parents to select
            tournament_size: Size of selection tournaments
            
        Returns:
            List of selected parent candidates
        """
        parents: List[Candidate] = []
        
        for _ in range(num_parents):
            parent = self.tournament_selection(population, tournament_size)
            parents.append(parent)
        
        return parents
    
    def get_pareto_front(self, population: List[Candidate]) -> List[Candidate]:
        """Get the Pareto-optimal front from a population.
        
        Args:
            population: Population to extract Pareto front from
            
        Returns:
            List of Pareto-optimal candidates
        """
        fronts = self.non_dominated_sort(population)
        return fronts[0] if fronts else []
    
    def calculate_hypervolume(self, population: List[Candidate], 
                             reference_point: Optional[Dict[str, float]] = None) -> float:
        """Calculate hypervolume indicator for a set of candidates.
        
        Hypervolume measures the volume of objective space dominated by
        the population and bounded by a reference point.
        
        Args:
            population: Set of candidates
            reference_point: Reference point for hypervolume calculation.
                           If None, uses worst values in population.
            
        Returns:
            Hypervolume value
        """
        if not population or len(self.objectives) != 2:
            # Simplified hypervolume for 2 objectives only
            return 0.0
        
        obj1, obj2 = self.objectives[0], self.objectives[1]
        
        # Find reference point if not provided
        if reference_point is None:
            scores_obj1 = [c.get_fitness(obj1) for c in population if c.get_fitness(obj1) is not None]
            scores_obj2 = [c.get_fitness(obj2) for c in population if c.get_fitness(obj2) is not None]
            
            if not scores_obj1 or not scores_obj2:
                return 0.0
            
            reference_point = {
                obj1: min(scores_obj1) if not self.maximize_objectives[obj1] else 0,
                obj2: min(scores_obj2) if not self.maximize_objectives[obj2] else 0
            }
        
        # Calculate hypervolume for 2D case
        pareto_front = self.get_pareto_front(population)
        
        if not pareto_front:
            return 0.0
        
        # Sort by first objective
        sorted_front = sorted(
            pareto_front,
            key=lambda c: c.get_fitness(obj1) or 0,
            reverse=self.maximize_objectives[obj1]
        )
        
        hypervolume = 0.0
        prev_y = reference_point[obj2]
        
        for candidate in sorted_front:
            x = candidate.get_fitness(obj1)
            y = candidate.get_fitness(obj2)
            
            if x is None or y is None:
                continue
            
            if self.maximize_objectives[obj1]:
                width = x - reference_point[obj1]
            else:
                width = reference_point[obj1] - x
            
            if self.maximize_objectives[obj2]:
                height = y - reference_point[obj2]
            else:
                height = reference_point[obj2] - y
            
            hypervolume += width * (prev_y - y if self.maximize_objectives[obj2] else y - prev_y)
            prev_y = y
        
        return max(0, hypervolume)


# Example usage
if __name__ == "__main__":
    # Create some test candidates
    candidates = [
        Candidate(content="A", fitness_scores={"accuracy": 0.8, "complexity": 0.3}),
        Candidate(content="B", fitness_scores={"accuracy": 0.9, "complexity": 0.7}),
        Candidate(content="C", fitness_scores={"accuracy": 0.7, "complexity": 0.2}),
        Candidate(content="D", fitness_scores={"accuracy": 0.85, "complexity": 0.4}),
    ]
    
    # Create selector
    selector = ParetoSelector(
        objectives=["accuracy", "complexity"],
        maximize_objectives={"accuracy": True, "complexity": False}
    )
    
    # Get Pareto front
    pareto_front = selector.get_pareto_front(candidates)
    print(f"Pareto front size: {len(pareto_front)}")
    for candidate in pareto_front:
        print(f"  {candidate}")
    
    # Environmental selection
    selected = selector.environmental_selection(candidates, target_size=2)
    print(f"\nSelected candidates: {len(selected)}")
    for candidate in selected:
        print(f"  {candidate}")
    
    # Test dominance
    candidate_a = candidates[0]
    candidate_b = candidates[1]
    print(f"\nDominance test:")
    print(f"Candidate A dominates B: {selector.dominates(candidate_a, candidate_b)}")
    print(f"Candidate B dominates A: {selector.dominates(candidate_b, candidate_a)}")
    
    # Test hypervolume
    hypervolume = selector.calculate_hypervolume(candidates)
    print(f"\nHypervolume: {hypervolume:.4f}")