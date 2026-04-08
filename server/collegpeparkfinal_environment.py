# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CollegePark Parking Environment Implementation.

A parking lot simulation where an agent must efficiently assign arriving
vehicles to parking spots while minimizing reshuffles when vehicles depart.

Key Mechanics:
- Parking lot is a 2D grid (rows x slots_per_row)
- Slot 0 in each row is closest to the exit
- Vehicles have estimated departure times
- When a vehicle needs to leave, all vehicles blocking its path must be reshuffled
- Goal: Minimize total reshuffles by placing vehicles optimally
"""

from typing import Dict, Any, Optional, List, Tuple
from uuid import uuid4
import copy

try:
    from ..models import CollegeParkAction, CollegeParkObservation, CollegeParkState
    from ..tasks import get_task, get_available_tasks, generate_vehicle_queue, create_empty_lot, ParkingTask
    from ..graders import grade_episode, compute_step_reward, get_episode_summary
except ImportError:
    from models import CollegeParkAction, CollegeParkObservation, CollegeParkState
    from tasks import get_task, get_available_tasks, generate_vehicle_queue, create_empty_lot, ParkingTask
    from graders import grade_episode, compute_step_reward, get_episode_summary


class CollegeParkEnvironment:
    """
    Parking lot optimization environment.
    
    The agent's goal is to park vehicles efficiently to minimize reshuffles
    when vehicles need to depart. Each row has a single exit point (slot 0),
    so vehicles parked deeper in the row will block vehicles closer to the exit.
    
    Optimal Strategy:
    - Park vehicles with later departure times deeper in rows
    - Park vehicles with earlier departure times closer to exits
    - Avoid placing early-departing vehicles behind late-departing ones
    
    Example:
        >>> env = CollegeParkEnvironment()
        >>> obs = env.reset(task_id="easy", seed=42)
        >>> print(obs.queue)  # View waiting vehicles
        >>> action = CollegeParkAction(vehicle_id="V001", row=0, slot=0)
        >>> obs = env.step(action)
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    def __init__(self):
        """Initialize the parking environment."""
        self._state: Optional[CollegeParkState] = None
        self._task: Optional[ParkingTask] = None
        self._lot: List[List[Optional[str]]] = []
        self._queue: List[Dict[str, Any]] = []
        self._parked: Dict[str, Dict[str, Any]] = {}  # vehicle_id -> {row, slot, departure_time}
        self._reshuffles: int = 0
        self._departures: int = 0
        self._current_time: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._seed: int = 0
        self._total_vehicles: int = 0
        self._all_vehicles_processed: bool = False
    
    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        **kwargs
    ) -> CollegeParkObservation:
        """
        Reset the environment for a new episode.
        
        Args:
            task_id: Task difficulty ('easy', 'medium', 'hard')
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation with empty lot and vehicle queue
        """
        # Get task configuration
        self._task = get_task(task_id)
        self._seed = seed if seed is not None else 42
        
        # Initialize lot
        self._lot = create_empty_lot(self._task)
        
        # Generate vehicle queue
        self._queue = generate_vehicle_queue(self._task, self._seed)
        self._total_vehicles = len(self._queue)
        
        # Reset counters
        self._parked = {}
        self._reshuffles = 0
        self._departures = 0
        self._current_time = 0
        self._step_count = 0
        self._done = False
        self._all_vehicles_processed = False
        
        # Create state
        self._state = CollegeParkState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            lot=copy.deepcopy(self._lot),
            queue=copy.deepcopy(self._queue),
            parked_vehicles={},
            reshuffles=0,
            departures=0,
            current_time=0,
            done=False,
            total_vehicles=self._total_vehicles
        )
        
        return self._create_observation()
    
    def step(self, action: CollegeParkAction) -> CollegeParkObservation:
        """
        Execute a parking action.
        
        Args:
            action: CollegeParkAction specifying vehicle_id, row, and slot
            
        Returns:
            Observation with updated lot state and metrics
        """
        if self._done:
            return self._create_observation()
        
        self._step_count += 1
        self._current_time += 1
        
        # Validate and execute action
        reward, action_valid, error_msg = self._execute_action(action)
        
        # Process departures based on current time
        self._process_departures()
        
        # Check if episode is done
        self._check_done()
        
        # Update state
        self._update_state()
        
        obs = self._create_observation()
        obs.reward = reward
        
        if error_msg:
            obs.metadata = obs.metadata or {}
            obs.metadata["error"] = error_msg
        
        return obs
    
    def _execute_action(self, action: CollegeParkAction) -> Tuple[float, bool, Optional[str]]:
        """Execute a parking action and return (reward, valid, error_message)."""
        vehicle_id = action.vehicle_id
        row = action.row
        slot = action.slot
        
        # Find vehicle in queue
        vehicle_data = None
        for v in self._queue:
            if v["vehicle_id"] == vehicle_id:
                vehicle_data = v
                break
        
        if vehicle_data is None:
            return -0.5, False, f"Vehicle {vehicle_id} not in queue"
        
        # Validate position
        if row < 0 or row >= self._task.rows:
            return -0.5, False, f"Invalid row {row}. Must be 0-{self._task.rows - 1}"
        
        if slot < 0 or slot >= self._task.slots_per_row:
            return -0.5, False, f"Invalid slot {slot}. Must be 0-{self._task.slots_per_row - 1}"
        
        if self._lot[row][slot] is not None:
            return -0.5, False, f"Slot ({row}, {slot}) is already occupied"
        
        # Park the vehicle
        self._lot[row][slot] = vehicle_id
        self._parked[vehicle_id] = {
            "row": row,
            "slot": slot,
            "departure_time": vehicle_data["departure_time"]
        }
        
        # Remove from queue
        self._queue = [v for v in self._queue if v["vehicle_id"] != vehicle_id]
        
        # Calculate reward based on placement quality
        optimal = self._is_optimal_placement(vehicle_id, row, slot, vehicle_data["departure_time"])
        reward = compute_step_reward(
            action_valid=True,
            vehicle_parked=True,
            optimal_placement=optimal
        )
        
        return reward, True, None
    
    def _is_optimal_placement(
        self, 
        vehicle_id: str, 
        row: int, 
        slot: int, 
        departure_time: int
    ) -> bool:
        """Check if a placement is optimal (won't cause future reshuffles).
        
        Optimal if no vehicle with later departure time is between this slot and exit.
        """
        # Check vehicles between this slot and exit (slot 0)
        for s in range(slot):
            blocking_vid = self._lot[row][s]
            if blocking_vid and blocking_vid in self._parked:
                blocking_dep = self._parked[blocking_vid]["departure_time"]
                if blocking_dep > departure_time:
                    # A vehicle with later departure is blocking path to exit
                    return False
        return True
    
    def _process_departures(self):
        """Process vehicle departures based on current time."""
        departing = []
        
        for vid, info in self._parked.items():
            if info["departure_time"] <= self._current_time:
                departing.append(vid)
        
        for vid in departing:
            self._depart_vehicle(vid)
    
    def _depart_vehicle(self, vehicle_id: str):
        """Handle a vehicle departing, including any required reshuffles."""
        if vehicle_id not in self._parked:
            return
        
        info = self._parked[vehicle_id]
        row, slot = info["row"], info["slot"]
        
        # Count vehicles blocking the exit path
        blocking_count = 0
        for s in range(slot):
            if self._lot[row][s] is not None:
                blocking_count += 1
        
        # Add reshuffles (each blocking vehicle must be moved)
        self._reshuffles += blocking_count
        
        # Remove the departing vehicle
        self._lot[row][slot] = None
        del self._parked[vehicle_id]
        self._departures += 1
    
    def _check_done(self):
        """Check if episode should end."""
        # Done if max steps reached
        if self._step_count >= self._task.max_steps:
            self._done = True
            return
        
        # Done if all vehicles have been processed (parked and departed)
        all_parked = len(self._queue) == 0
        all_departed = len(self._parked) == 0 and self._departures > 0
        
        if all_parked and all_departed:
            self._done = True
            self._all_vehicles_processed = True
    
    def _update_state(self):
        """Update internal state object."""
        if self._state:
            self._state.step_count = self._step_count
            self._state.lot = copy.deepcopy(self._lot)
            self._state.queue = copy.deepcopy(self._queue)
            self._state.parked_vehicles = copy.deepcopy(self._parked)
            self._state.reshuffles = self._reshuffles
            self._state.departures = self._departures
            self._state.current_time = self._current_time
            self._state.done = self._done
    
    def _create_observation(self) -> CollegeParkObservation:
        """Create an observation from current state."""
        # Calculate final score if done
        final_reward = None
        if self._done:
            final_reward = grade_episode(
                self._task.task_id,
                self._reshuffles,
                self._departures
            )
        
        return CollegeParkObservation(
            lot=copy.deepcopy(self._lot),
            queue=copy.deepcopy(self._queue),
            reshuffles_so_far=self._reshuffles,
            step_count=self._step_count,
            task_id=self._task.task_id if self._task else "easy",
            max_steps=self._task.max_steps if self._task else 50,
            pending_count=len(self._queue),
            departed_count=self._departures,
            parked_count=len(self._parked),
            done=self._done,
            reward=final_reward if final_reward is not None else 0.0,
            metadata={
                "current_time": self._current_time,
                "total_vehicles": self._total_vehicles,
                "reshuffles": self._reshuffles,
                "departures": self._departures,
                "episode_id": self._state.episode_id if self._state else "",
            }
        )
    
    @property
    def state(self) -> CollegeParkState:
        """Get current environment state."""
        if self._state is None:
            return CollegeParkState(episode_id="", step_count=0)
        return self._state
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current/completed episode."""
        return get_episode_summary(
            task_id=self._task.task_id if self._task else "easy",
            reshuffles=self._reshuffles,
            departures=self._departures,
            steps_taken=self._step_count,
            max_steps=self._task.max_steps if self._task else 50,
            vehicles_parked=self._departures + len(self._parked),
            total_vehicles=self._total_vehicles
        )


# Backward compatibility alias
CollegpeparkfinalEnvironment = CollegeParkEnvironment
