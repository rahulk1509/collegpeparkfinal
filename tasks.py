# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task configurations for the CollegePark Parking Environment.

Defines three difficulty levels with different lot sizes, vehicle counts,
and departure time constraints.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class ParkingTask:
    """Configuration for a parking task."""
    
    task_id: str
    rows: int
    slots_per_row: int
    num_vehicles: int
    max_steps: int
    departure_time_range: Tuple[int, int]  # (min, max) time steps
    description: str
    
    @property
    def total_capacity(self) -> int:
        return self.rows * self.slots_per_row


# Task Definitions
TASK_EASY = ParkingTask(
    task_id="easy",
    rows=3,
    slots_per_row=4,
    num_vehicles=8,
    max_steps=50,
    departure_time_range=(10, 30),
    description="Small lot (3x4), 8 vehicles, generous departure windows"
)

TASK_MEDIUM = ParkingTask(
    task_id="medium",
    rows=5,
    slots_per_row=6,
    num_vehicles=20,
    max_steps=100,
    departure_time_range=(8, 25),
    description="Medium lot (5x6), 20 vehicles, moderate time constraints"
)

TASK_HARD = ParkingTask(
    task_id="hard",
    rows=8,
    slots_per_row=10,
    num_vehicles=50,
    max_steps=200,
    departure_time_range=(5, 20),
    description="Large lot (8x10), 50 vehicles, tight departure times"
)


# Task registry
TASKS: Dict[str, ParkingTask] = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}


def get_task(task_id: str) -> ParkingTask:
    """Get a task configuration by ID.
    
    Args:
        task_id: One of 'easy', 'medium', 'hard'
        
    Returns:
        ParkingTask configuration
        
    Raises:
        ValueError: If task_id is not recognized
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of {list(TASKS.keys())}")
    return TASKS[task_id]


def get_available_tasks() -> List[str]:
    """Get list of available task IDs."""
    return list(TASKS.keys())


def generate_vehicle_queue(task: ParkingTask, seed: int) -> List[Dict]:
    """Generate a deterministic queue of vehicles for a task.
    
    Args:
        task: ParkingTask configuration
        seed: Random seed for reproducibility
        
    Returns:
        List of vehicle dictionaries with vehicle_id and departure_time
    """
    rng = random.Random(seed)
    
    vehicles = []
    min_dep, max_dep = task.departure_time_range
    
    for i in range(task.num_vehicles):
        vehicle_id = f"V{i+1:03d}"
        # Departure times are spread across the range
        # Earlier vehicles tend to have earlier departures to create realistic flow
        base_time = min_dep + (i * (max_dep - min_dep)) // task.num_vehicles
        jitter = rng.randint(-3, 3)
        departure_time = max(min_dep, min(max_dep, base_time + jitter))
        
        vehicles.append({
            "vehicle_id": vehicle_id,
            "departure_time": departure_time
        })
    
    # Shuffle arrival order (but keep departure times as assigned)
    rng.shuffle(vehicles)
    
    return vehicles


def create_empty_lot(task: ParkingTask) -> List[List[None]]:
    """Create an empty parking lot grid.
    
    Args:
        task: ParkingTask configuration
        
    Returns:
        2D list of None values representing empty slots
    """
    return [[None for _ in range(task.slots_per_row)] for _ in range(task.rows)]
