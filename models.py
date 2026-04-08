# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the CollegePark Parking Environment.

This environment simulates a parking lot where an agent must assign
arriving vehicles to parking spots while minimizing reshuffles when
vehicles need to depart.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Vehicle:
    """Represents a vehicle in the parking lot or queue."""
    
    def __init__(self, vehicle_id: str, departure_time: int):
        self.vehicle_id = vehicle_id
        self.departure_time = departure_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "departure_time": self.departure_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vehicle":
        return cls(
            vehicle_id=data["vehicle_id"],
            departure_time=data["departure_time"]
        )


class CollegeParkAction(BaseModel):
    """Action for parking a vehicle in the lot.
    
    The agent selects a vehicle from the queue and assigns it to a specific
    row and slot in the parking lot.
    """
    
    vehicle_id: str = Field(..., description="ID of the vehicle to park (from queue)")
    row: int = Field(..., description="Row index in the parking lot (0-indexed)")
    slot: int = Field(..., description="Slot index within the row (0-indexed, 0 is closest to exit)")


class CollegeParkObservation(BaseModel):
    """Observation from the parking environment.
    
    Contains the current state of the parking lot, vehicle queue,
    and performance metrics.
    """
    
    lot: List[List[Optional[str]]] = Field(
        default_factory=list,
        description="2D array of parking lot. Each cell is vehicle_id or null/empty string for empty."
    )
    queue: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of vehicles waiting to be parked: [{vehicle_id, departure_time}, ...]"
    )
    reshuffles_so_far: int = Field(
        default=0,
        description="Total number of reshuffles that have occurred this episode"
    )
    step_count: int = Field(
        default=0,
        description="Current step number in the episode"
    )
    task_id: str = Field(
        default="easy",
        description="Task difficulty identifier: easy, medium, or hard"
    )
    max_steps: int = Field(
        default=100,
        description="Maximum steps allowed for this episode"
    )
    pending_count: int = Field(
        default=0,
        description="Number of vehicles still in queue waiting to be parked"
    )
    departed_count: int = Field(
        default=0,
        description="Number of vehicles that have departed"
    )
    parked_count: int = Field(
        default=0,
        description="Number of vehicles currently parked in the lot"
    )
    done: bool = Field(default=False, description="Whether episode is complete")
    reward: float = Field(default=0.0, description="Reward for this step")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom serialization: convert None to empty string in lot."""
        data = super().model_dump(*args, **kwargs)
        # Convert None to "" in lot for JSON compatibility
        if data.get("lot"):
            data["lot"] = [
                [cell if cell is not None else "" for cell in row]
                for row in data["lot"]
            ]
        return data


class CollegeParkState(BaseModel):
    """Full environment state including internal tracking data."""
    
    episode_id: str = Field(default="", description="Unique episode identifier")
    step_count: int = Field(default=0, description="Current step count")
    task_id: str = Field(default="easy", description="Current task difficulty")
    lot: List[List[Optional[str]]] = Field(
        default_factory=list,
        description="Current parking lot state"
    )
    queue: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Vehicles waiting to park"
    )
    parked_vehicles: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Map of vehicle_id to {row, slot, departure_time}"
    )
    reshuffles: int = Field(default=0, description="Total reshuffles so far")
    departures: int = Field(default=0, description="Total departures so far")
    current_time: int = Field(default=0, description="Current simulation time step")
    done: bool = Field(default=False, description="Whether episode is complete")
    total_vehicles: int = Field(default=0, description="Total vehicles to process")


# Backward compatibility aliases
CollegpeparkfinalAction = CollegeParkAction
CollegpeparkfinalObservation = CollegeParkObservation
