# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# Ensure project root is in path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from server.collegpeparkfinal_environment import CollegeParkEnvironment
from models import CollegeParkAction, CollegeParkObservation, CollegeParkState
from tasks import TASKS, get_task, get_available_tasks
from graders import grade_episode, get_episode_summary

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Environment variables configuration
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

app = FastAPI(title="CollegePark Parking Environment", version="1.0.0")
_env = CollegeParkEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {"task_id": "easy", "seed": 42}
        }


class StepRequest(BaseModel):
    vehicle_id: str
    row: int
    slot: int

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": "V001",
                "row": 0,
                "slot": 2
            }
        }


class ResetResponse(BaseModel):
    episode_id: str
    observation: Dict[str, Any]
    reward: float
    done: bool
    available_tasks: List[str]
    info: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


# ============================================================================
# Required OpenEnv validator endpoints
# ============================================================================

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint - must return 'healthy' for OpenEnv validator."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Environment metadata - required by OpenEnv validator."""
    return {
        "name": "CollegePark",
        "description": "An OpenEnv-compliant environment for parking lot vehicle assignment optimization. The agent must park vehicles to minimize reshuffles when vehicles depart.",
        "version": "1.0.0",
        "author": "CollegePark Maintainers",
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Action/observation/state schemas - required by OpenEnv validator."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "vehicle_id": {"type": "string", "description": "ID of vehicle to park from queue"},
                "row": {"type": "integer", "description": "Row index (0-indexed)"},
                "slot": {"type": "integer", "description": "Slot index within row (0 = closest to exit)"},
            },
            "required": ["vehicle_id", "row", "slot"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "lot": {"type": "array", "description": "2D array of parking lot state"},
                "queue": {"type": "array", "description": "Vehicles waiting to park"},
                "reshuffles_so_far": {"type": "integer"},
                "step_count": {"type": "integer"},
                "task_id": {"type": "string"},
                "max_steps": {"type": "integer"},
                "pending_count": {"type": "integer"},
                "departed_count": {"type": "integer"},
                "parked_count": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "task_id": {"type": "string"},
                "lot": {"type": "array"},
                "queue": {"type": "array"},
                "reshuffles": {"type": "integer"},
                "departures": {"type": "integer"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.get("/tasks")
async def tasks() -> List[Dict[str, Any]]:
    """List all tasks with grader info - required by OpenEnv validator."""
    return [
        {
            "id": "easy",
            "name": "Easy Parking",
            "description": "Small lot (3x4), 8 vehicles, generous departure windows (10-30 time steps)",
            "difficulty": "easy",
            "grader": {
                "type": "python",
                "path": "graders.py",
                "function": "grade_easy",
            },
        },
        {
            "id": "medium",
            "name": "Medium Parking",
            "description": "Medium lot (5x6), 20 vehicles, moderate time constraints (8-25 time steps)",
            "difficulty": "medium",
            "grader": {
                "type": "python",
                "path": "graders.py",
                "function": "grade_medium",
            },
        },
        {
            "id": "hard",
            "name": "Hard Parking",
            "description": "Large lot (8x10), 50 vehicles, tight departure times (5-20 time steps)",
            "difficulty": "hard",
            "grader": {
                "type": "python",
                "path": "graders.py",
                "function": "grade_hard",
            },
        },
    ]


@app.post("/mcp")
async def mcp(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """MCP JSON-RPC endpoint - required by OpenEnv validator."""
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "CollegePark",
            "description": "Parking lot optimization environment",
        },
    }


# ============================================================================
# Main API endpoints
# ============================================================================

@app.get("/web", response_class=HTMLResponse)
async def home():
    """Serve the homepage."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CollegePark Parking Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
            max-width: 700px;
            text-align: center;
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 2.5em; }
        .emoji { font-size: 3em; display: block; margin: 20px 0; }
        p { color: #666; line-height: 1.8; margin: 15px 0; }
        .tasks { background: #f5f5f5; border-radius: 8px; padding: 20px; margin: 25px 0; text-align: left; }
        .tasks h2 { color: #333; margin-bottom: 15px; font-size: 1.2em; }
        .task-item { padding: 10px 0; border-bottom: 1px solid #ddd; }
        .task-item:last-child { border-bottom: none; }
        .task-item strong { color: #1e3c72; }
        .task-item span { color: #888; font-size: 0.9em; }
        .links { margin-top: 30px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        a { display: inline-block; padding: 10px 20px; background: #1e3c72; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
        a:hover { background: #2a5298; }
        .info { background: #e8f4f8; border-radius: 8px; padding: 15px; margin: 20px 0; text-align: left; font-size: 0.9em; }
        .info code { background: #ddd; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <span class="emoji">P</span>
        <h1>CollegePark</h1>
        <p><strong>OpenEnv environment for parking lot vehicle assignment optimization</strong></p>
        <p>Park vehicles efficiently to minimize reshuffles when they need to depart!</p>
        
        <div class="tasks">
            <h2>Available Tasks</h2>
            <div class="task-item">
                <strong>Easy</strong> - 3x4 lot, 8 vehicles <span>(departure: 10-30 steps)</span>
            </div>
            <div class="task-item">
                <strong>Medium</strong> - 5x6 lot, 20 vehicles <span>(departure: 8-25 steps)</span>
            </div>
            <div class="task-item">
                <strong>Hard</strong> - 8x10 lot, 50 vehicles <span>(departure: 5-20 steps)</span>
            </div>
        </div>
        
        <div class="info">
            <strong>How it works:</strong><br>
            - Slot 0 in each row is closest to the exit<br>
            - When a vehicle departs, blocking vehicles must be reshuffled<br>
            - Score = <code>1.0 - (reshuffles / departures)</code>
        </div>
        
        <div class="links">
            <a href="/docs">API Docs</a>
            <a href="/tasks">View Tasks</a>
            <a href="/schema">Schema</a>
        </div>
    </div>
</body>
</html>"""


@app.post("/reset", response_model=ResetResponse)
async def reset(body: Optional[ResetRequest] = Body(None)):
    """Reset the environment and start a new episode."""
    try:
        task_id = body.task_id if body and body.task_id else "easy"
        seed = body.seed if body and body.seed else 42
        
        logger.info(f"[START] task={task_id} env=collegpark model={MODEL_NAME} seed={seed}")
        
        obs = _env.reset(task_id=task_id, seed=seed)
        
        obs_dict = obs.model_dump()
        # Convert None to "" in lot for JSON
        if obs_dict.get("lot"):
            obs_dict["lot"] = [
                [cell if cell is not None else "" for cell in row]
                for row in obs_dict["lot"]
            ]
        
        return ResetResponse(
            episode_id=_env.state.episode_id,
            observation=obs_dict,
            reward=0.0,
            done=False,
            available_tasks=get_available_tasks(),
            info={"episode_id": _env.state.episode_id, "task_id": task_id, "seed": seed}
        )
    except Exception as e:
        logger.error(f"[RESET ERROR] {str(e)}")
        raise


@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest = Body(...)):
    """Execute a parking action."""
    try:
        action = CollegeParkAction(
            vehicle_id=body.vehicle_id,
            row=body.row,
            slot=body.slot
        )
        
        obs = _env.step(action)
        
        action_str = f"park {body.vehicle_id} -> row {body.row}, slot {body.slot}"
        error_msg = obs.metadata.get("error") if obs.metadata else None
        
        logger.info(
            f"[STEP] step={obs.step_count} action={action_str} "
            f"reward={obs.reward:.2f} done={str(obs.done).lower()} "
            f"error={error_msg if error_msg else 'null'}"
        )
        
        obs_dict = obs.model_dump()
        # Convert None to "" in lot for JSON
        if obs_dict.get("lot"):
            obs_dict["lot"] = [
                [cell if cell is not None else "" for cell in row]
                for row in obs_dict["lot"]
            ]
        
        # Calculate score if done
        score = None
        if obs.done:
            score = grade_episode(
                _env._task.task_id,
                _env._reshuffles,
                _env._departures
            )
            logger.info(
                f"[END] task_id={_env._task.task_id} steps={obs.step_count} "
                f"reshuffles={_env._reshuffles} departures={_env._departures} "
                f"score={score:.4f}"
            )
        
        return StepResponse(
            observation=obs_dict,
            reward=obs.reward,
            done=obs.done,
            truncated=obs.step_count >= obs.max_steps,
            info={
                "reshuffles": _env._reshuffles,
                "departures": _env._departures,
                "parked": len(_env._parked),
                "score": score,
                "error": error_msg
            }
        )
    except Exception as e:
        logger.error(f"[STEP ERROR] {str(e)}")
        raise


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the current environment state."""
    env_state = _env.state
    return {
        "episode_id": env_state.episode_id,
        "step_count": env_state.step_count,
        "task_id": env_state.task_id,
        "lot": [
            [cell if cell is not None else "" for cell in row]
            for row in env_state.lot
        ] if env_state.lot else [],
        "queue": env_state.queue,
        "parked_vehicles": env_state.parked_vehicles,
        "reshuffles": env_state.reshuffles,
        "departures": env_state.departures,
        "current_time": env_state.current_time,
        "done": env_state.done,
        "total_vehicles": env_state.total_vehicles,
    }


@app.get("/summary")
async def summary() -> Dict[str, Any]:
    """Get episode summary with score."""
    return _env.get_episode_summary()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
