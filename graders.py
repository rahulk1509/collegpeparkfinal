# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grading functions for the CollegePark Parking Environment.

Computes scores based on parking efficiency, specifically minimizing
the number of vehicle reshuffles required during departures.
"""

from typing import Dict, Any


def grade_easy(reshuffles: int, departures: int) -> float:
    """Grade an easy task episode.
    
    Score = 1.0 - (reshuffles / departures)
    
    Args:
        reshuffles: Total reshuffles that occurred
        departures: Total vehicle departures
        
    Returns:
        Score in range [0.0, 1.0], rounded to 4 decimals
    """
    if departures == 0:
        return 1.0  # Perfect score if no departures yet
    
    score = 1.0 - (reshuffles / departures)
    return round(max(0.0, min(1.0, score)), 4)


def grade_medium(reshuffles: int, departures: int) -> float:
    """Grade a medium task episode.
    
    Score = 1.0 - (reshuffles / departures)
    
    Args:
        reshuffles: Total reshuffles that occurred
        departures: Total vehicle departures
        
    Returns:
        Score in range [0.0, 1.0], rounded to 4 decimals
    """
    if departures == 0:
        return 1.0
    
    score = 1.0 - (reshuffles / departures)
    return round(max(0.0, min(1.0, score)), 4)


def grade_hard(reshuffles: int, departures: int) -> float:
    """Grade a hard task episode.
    
    Score = 1.0 - (1.5 * reshuffles / departures)
    
    Harder tasks have a stricter penalty for reshuffles.
    
    Args:
        reshuffles: Total reshuffles that occurred
        departures: Total vehicle departures
        
    Returns:
        Score in range [0.0, 1.0], rounded to 4 decimals
    """
    if departures == 0:
        return 1.0
    
    score = 1.0 - (1.5 * reshuffles / departures)
    return round(max(0.0, min(1.0, score)), 4)


# Grader registry
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_episode(task_id: str, reshuffles: int, departures: int) -> float:
    """Grade an episode based on task difficulty.
    
    Args:
        task_id: Task identifier ('easy', 'medium', 'hard')
        reshuffles: Total reshuffles during episode
        departures: Total departures during episode
        
    Returns:
        Score in range [0.0, 1.0]
        
    Raises:
        ValueError: If task_id is not recognized
    """
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of {list(GRADERS.keys())}")
    
    return GRADERS[task_id](reshuffles, departures)


def compute_step_reward(
    action_valid: bool,
    vehicle_parked: bool,
    reshuffles_caused: int = 0,
    optimal_placement: bool = False
) -> float:
    """Compute reward for a single step.
    
    Args:
        action_valid: Whether the action was valid
        vehicle_parked: Whether a vehicle was successfully parked
        reshuffles_caused: Number of reshuffles this action may have contributed to
        optimal_placement: Whether the placement was optimal (close to exit, good departure order)
        
    Returns:
        Step reward value
    """
    if not action_valid:
        return -0.5  # Invalid action penalty
    
    if not vehicle_parked:
        return -0.1  # Failed to park
    
    # Base reward for successful parking
    reward = 0.5
    
    # Bonus for optimal placement
    if optimal_placement:
        reward += 0.5
    
    return reward


def get_episode_summary(
    task_id: str,
    reshuffles: int,
    departures: int,
    steps_taken: int,
    max_steps: int,
    vehicles_parked: int,
    total_vehicles: int
) -> Dict[str, Any]:
    """Generate a summary of episode performance.
    
    Args:
        task_id: Task identifier
        reshuffles: Total reshuffles
        departures: Total departures
        steps_taken: Steps taken in episode
        max_steps: Maximum allowed steps
        vehicles_parked: Vehicles successfully parked
        total_vehicles: Total vehicles to process
        
    Returns:
        Dictionary with performance metrics
    """
    score = grade_episode(task_id, reshuffles, departures)
    
    return {
        "task_id": task_id,
        "score": score,
        "reshuffles": reshuffles,
        "departures": departures,
        "reshuffle_rate": round(reshuffles / max(1, departures), 4),
        "steps_taken": steps_taken,
        "max_steps": max_steps,
        "efficiency": round(steps_taken / max_steps, 4) if max_steps > 0 else 1.0,
        "vehicles_parked": vehicles_parked,
        "total_vehicles": total_vehicles,
        "completion_rate": round(vehicles_parked / max(1, total_vehicles), 4),
        "success": score >= 0.5 and vehicles_parked >= total_vehicles * 0.8
    }
