#!/usr/bin/env python3
"""
Inference Script for CollegePark Parking Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if using from_docker_image()

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import requests
from openai import OpenAI

from models import CollegeParkAction, CollegeParkObservation

# Environment configuration
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"
BENCHMARK = "collegpark"
MAX_STEPS = 100
TEMPERATURE = 0.1
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["easy", "medium", "hard"]
SEEDS = {"easy": 101, "medium": 202, "hard": 303}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a parking lot optimizer. Your goal is to minimize reshuffles.

    RULES:
    1. Each row has slot 0 closest to the exit
    2. When a vehicle departs, all vehicles blocking its path must be reshuffled
    3. Park vehicles with EARLIER departure times CLOSER to the exit (lower slot numbers)
    4. Park vehicles with LATER departure times DEEPER in the row (higher slot numbers)

    Respond with a JSON action: {"vehicle_id": "V001", "row": 0, "slot": 2}
    Always choose from the vehicles in the queue.
    Choose an empty slot (shown as ____).
""").strip()


@dataclass
class EnvResult:
    """Result from environment reset/step."""
    observation: Dict[str, Any]
    reward: float
    done: bool


class CollegeParkEnv:
    """Async wrapper for CollegePark environment HTTP client."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None) -> "CollegeParkEnv":
        """Create environment client (connects to running Docker container)."""
        base_url = os.getenv("ENV_URL") or "http://localhost:7860"
        return cls(base_url)
    
    async def reset(self, task_id: str = "easy", seed: int = 42) -> EnvResult:
        """Reset environment."""
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return EnvResult(
            observation=data.get("observation", {}),
            reward=data.get("reward", 0.0),
            done=data.get("done", False)
        )
    
    async def step(self, action: CollegeParkAction) -> EnvResult:
        """Execute action."""
        response = requests.post(
            f"{self.base_url}/step",
            json={"vehicle_id": action.vehicle_id, "row": action.row, "slot": action.slot},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return EnvResult(
            observation=data.get("observation", {}),
            reward=data.get("reward", 0.0),
            done=data.get("done", False)
        )
    
    async def close(self) -> None:
        """Cleanup (no-op for HTTP client)."""
        pass


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def format_lot_for_prompt(lot: List[List[str]]) -> str:
    """Format the parking lot for LLM prompt."""
    lines = ["Parking Lot (slot 0 = exit):"]
    for i, row in enumerate(lot):
        row_str = " | ".join(cell if cell else "____" for cell in row)
        lines.append(f"  Row {i}: [{row_str}]")
    return "\n".join(lines)


def format_queue_for_prompt(queue: List[Dict]) -> str:
    """Format the vehicle queue for LLM prompt."""
    if not queue:
        return "Queue: (empty)"
    lines = ["Vehicle Queue (vehicle_id, departure_time):"]
    for v in queue[:10]:
        lines.append(f"  - {v['vehicle_id']}: departs at t={v['departure_time']}")
    if len(queue) > 10:
        lines.append(f"  ... and {len(queue) - 10} more vehicles")
    return "\n".join(lines)


def build_user_prompt(step: int, obs: Dict[str, Any]) -> str:
    lot = obs.get("lot", [])
    queue = obs.get("queue", [])
    lot_str = format_lot_for_prompt(lot)
    queue_str = format_queue_for_prompt(queue)
    
    return textwrap.dedent(f"""
        Current state (step {step}):

        {lot_str}

        {queue_str}

        Reshuffles so far: {obs.get('reshuffles_so_far', 0)}
        Parked vehicles: {obs.get('parked_count', 0)}
        Departed vehicles: {obs.get('departed_count', 0)}

        Which vehicle should be parked where? Respond with JSON: {{"vehicle_id": "...", "row": N, "slot": M}}
    """).strip()


def get_model_action(client: OpenAI, step: int, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get parking action from LLM."""
    user_prompt = build_user_prompt(step, obs)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Parse JSON from response
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            action = json.loads(json_str)
            if "vehicle_id" in action and "row" in action and "slot" in action:
                return {
                    "vehicle_id": str(action["vehicle_id"]),
                    "row": int(action["row"]),
                    "slot": int(action["slot"])
                }
    except Exception as exc:
        pass  # Fall through to heuristic
    
    # Fallback: use heuristic
    queue = obs.get("queue", [])
    lot = obs.get("lot", [])
    if queue:
        sorted_queue = sorted(queue, key=lambda v: v["departure_time"])
        vehicle = sorted_queue[0]
        for row_idx, row in enumerate(lot):
            for slot_idx, cell in enumerate(row):
                if cell is None or cell == "":
                    return {
                        "vehicle_id": vehicle["vehicle_id"],
                        "row": row_idx,
                        "slot": slot_idx
                    }
    return None


async def run_task(client: OpenAI, env: CollegeParkEnv, task_id: str, seed: int) -> Dict[str, Any]:
    """Run a single task episode."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = await env.reset(task_id=task_id, seed=seed)
        obs = result.observation
        done = result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            
            queue = obs.get("queue", [])
            if not queue:
                break

            action_dict = get_model_action(client, step, obs)
            if action_dict is None:
                log_step(step=step, action="null", reward=0.0, done=False, error="Could not parse action")
                break

            # Execute action
            action = CollegeParkAction(**action_dict)
            result = await env.step(action)
            
            obs = result.observation
            reward = result.reward
            done = result.done
            error = obs.get("metadata", {}).get("error") if isinstance(obs.get("metadata"), dict) else None

            rewards.append(reward)
            steps_taken = step

            action_str = f"park({action_dict['vehicle_id']},{action_dict['row']},{action_dict['slot']})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Calculate score
        reshuffles = obs.get("reshuffles_so_far", 0)
        departures = obs.get("departed_count", 0)
        
        if departures > 0:
            if task_id == "hard":
                score = max(0.0, min(1.0, 1.0 - (1.5 * reshuffles / departures)))
            else:
                score = max(0.0, min(1.0, 1.0 - (reshuffles / departures)))
        else:
            score = 1.0 if reshuffles == 0 else 0.5
        
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "steps_taken": steps_taken,
        "score": round(score, 4),
        "success": success,
        "rewards": rewards
    }


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await CollegeParkEnv.from_docker_image(IMAGE_NAME)

    results = []
    
    for task_id in TASKS:
        seed = SEEDS[task_id]
        try:
            result = await run_task(client, env, task_id, seed)
            results.append(result)
        except Exception as e:
            log_end(success=False, steps=0, score=0.0, rewards=[])
            results.append({
                "task_id": task_id,
                "steps_taken": 0,
                "score": 0.0,
                "success": False,
                "error": str(e)
            })

    # Cleanup
    try:
        await env.close()
    except Exception:
        pass

    # Final summary
    total_score = sum(r['score'] for r in results)
    avg_score = total_score / len(results) if results else 0.0
    print(f"\n# Overall Average Score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
