# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CollegePark Parking Environment Client."""

from typing import Dict, Any, Optional, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CollegeParkAction, CollegeParkObservation, CollegeParkState


class CollegeParkEnv(
    EnvClient[CollegeParkAction, CollegeParkObservation, CollegeParkState]
):
    """
    Client for the CollegePark Parking Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CollegeParkEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset(task_id="easy", seed=42)
        ...     print(f"Queue: {result.observation.queue}")
        ...
        ...     # Park a vehicle
        ...     result = client.step(CollegeParkAction(
        ...         vehicle_id="V001", row=0, slot=0
        ...     ))
        ...     print(f"Reward: {result.reward}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CollegeParkEnv.from_docker_image("collegpark-env:latest")
        >>> try:
        ...     result = client.reset(task_id="medium", seed=123)
        ...     for v in result.observation.queue[:3]:
        ...         result = client.step(CollegeParkAction(
        ...             vehicle_id=v["vehicle_id"], row=0, slot=0
        ...         ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CollegeParkAction) -> Dict[str, Any]:
        """
        Convert CollegeParkAction to JSON payload for step message.

        Args:
            action: CollegeParkAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "vehicle_id": action.vehicle_id,
            "row": action.row,
            "slot": action.slot,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CollegeParkObservation]:
        """
        Parse server response into StepResult[CollegeParkObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CollegeParkObservation
        """
        obs_data = payload.get("observation", {})
        
        # Parse lot, converting "" back to None
        lot_data = obs_data.get("lot", [])
        lot = [
            [cell if cell != "" else None for cell in row]
            for row in lot_data
        ]
        
        observation = CollegeParkObservation(
            lot=lot,
            queue=obs_data.get("queue", []),
            reshuffles_so_far=obs_data.get("reshuffles_so_far", 0),
            step_count=obs_data.get("step_count", 0),
            task_id=obs_data.get("task_id", "easy"),
            max_steps=obs_data.get("max_steps", 50),
            pending_count=obs_data.get("pending_count", 0),
            departed_count=obs_data.get("departed_count", 0),
            parked_count=obs_data.get("parked_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CollegeParkState:
        """
        Parse server response into CollegeParkState object.

        Args:
            payload: JSON response from state request

        Returns:
            CollegeParkState object with full environment state
        """
        return CollegeParkState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            lot=payload.get("lot", []),
            queue=payload.get("queue", []),
            parked_vehicles=payload.get("parked_vehicles", {}),
            reshuffles=payload.get("reshuffles", 0),
            departures=payload.get("departures", 0),
            current_time=payload.get("current_time", 0),
            done=payload.get("done", False),
            total_vehicles=payload.get("total_vehicles", 0),
        )


# Backward compatibility alias
CollegpeparkfinalEnv = CollegeParkEnv
