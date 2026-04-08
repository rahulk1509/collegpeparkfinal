---
title: CollegePark-v1
emoji: 🅿
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# CollegePark Parking Environment

A reinforcement learning environment for **parking lot vehicle assignment optimization**. The agent's goal is to efficiently assign arriving vehicles to parking spots while minimizing reshuffles when vehicles need to depart.

## Problem Description

- **Parking Lot**: 2D grid of rows x slots per row
- **Slot 0** in each row is closest to the exit
- **Vehicles** arrive with estimated departure times
- **Challenge**: When a vehicle departs, all vehicles blocking its path to the exit must be reshuffled
- **Goal**: Park vehicles to minimize total reshuffles

### Optimal Strategy

- Park vehicles with **earlier departure times** closer to the exit (lower slot numbers)
- Park vehicles with **later departure times** deeper in rows (higher slot numbers)
- Never place an early-departing vehicle behind a late-departing one

## Quick Start

### Using the Client

```python
from collegpark import CollegeParkAction, CollegeParkEnv

# Connect to running server
with CollegeParkEnv(base_url="http://localhost:7860") as env:
    # Reset with easy task
    result = env.reset(task_id="easy", seed=42)
    print(f"Queue: {len(result.observation.queue)} vehicles waiting")
    
    # Park vehicles from queue
    for vehicle in result.observation.queue:
        vid = vehicle["vehicle_id"]
        dep_time = vehicle["departure_time"]
        
        # Simple strategy: park in first available slot
        for row_idx, row in enumerate(result.observation.lot):
            for slot_idx, cell in enumerate(row):
                if cell is None or cell == "":
                    result = env.step(CollegeParkAction(
                        vehicle_id=vid,
                        row=row_idx,
                        slot=slot_idx
                    ))
                    break
            if result.done:
                break
    
    print(f"Final score: {result.reward}")
```

### Running the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py --port 7860

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Using Docker

```bash
# Build the image
docker build -t collegpark-env:latest .

# Run the container
docker run -p 7860:7860 collegpark-env:latest
```

## Task Difficulties

| Task | Lot Size | Vehicles | Max Steps | Departure Range | Description |
|------|----------|----------|-----------|-----------------|-------------|
| **easy** | 3x4 | 8 | 50 | 10-30 | Small lot, generous time windows |
| **medium** | 5x6 | 20 | 100 | 8-25 | Medium lot, moderate constraints |
| **hard** | 8x10 | 50 | 200 | 5-20 | Large lot, tight departure times |

## API Endpoints

### POST /reset

Initialize a new episode.

**Request:**
```json
{
  "task_id": "easy",
  "seed": 42
}
```

**Response:**
```json
{
  "episode_id": "uuid",
  "observation": {
    "lot": [["", "", "", ""], ["", "", "", ""], ["", "", "", ""]],
    "queue": [
      {"vehicle_id": "V001", "departure_time": 12},
      {"vehicle_id": "V002", "departure_time": 18}
    ],
    "reshuffles_so_far": 0,
    "step_count": 0,
    "task_id": "easy",
    "max_steps": 50,
    "pending_count": 8,
    "departed_count": 0,
    "parked_count": 0
  },
  "reward": 0.0,
  "done": false,
  "available_tasks": ["easy", "medium", "hard"],
  "info": {"episode_id": "uuid"}
}
```

### POST /step

Execute a parking action.

**Request:**
```json
{
  "vehicle_id": "V001",
  "row": 0,
  "slot": 2
}
```

**Response:**
```json
{
  "observation": {
    "lot": [["", "", "V001", ""], ["", "", "", ""], ["", "", "", ""]],
    "queue": [{"vehicle_id": "V002", "departure_time": 18}],
    "reshuffles_so_far": 0,
    "step_count": 1,
    "pending_count": 7,
    "parked_count": 1,
    "departed_count": 0
  },
  "reward": 0.5,
  "done": false
}
```

### GET /state

Get current environment state.

### GET /health

Health check endpoint.

## Scoring

Scores are calculated based on reshuffle efficiency:

| Task | Formula | Score Range |
|------|---------|-------------|
| **easy** | `1.0 - (reshuffles / departures)` | [0.0, 1.0] |
| **medium** | `1.0 - (reshuffles / departures)` | [0.0, 1.0] |
| **hard** | `1.0 - (1.5 * reshuffles / departures)` | [0.0, 1.0] |

A score of **1.0** means zero reshuffles (perfect). A score below **0.5** is considered failure.

## Inference Agent

The included `inference.py` uses an LLM to make parking decisions:

```bash
# Set environment variables
export HF_TOKEN="your-huggingface-token"
export ENV_URL="http://localhost:7860"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Run inference
python inference.py
```

**Output Format:**
```
[START]
task_id: easy
model: meta-llama/Llama-3.1-8B-Instruct
seed: 101

[STEP]
step: 1
action: park V001 -> row 0, slot 2
reward: 0.85
done: false
error: null

[END]
task_id: easy
steps_taken: 25
final_reward: 0.85
total_reward: 15.30
episode_done: true
score: 0.8750
success: true
```

## Project Structure

```
collegpark/
├── models.py                  # Pydantic models (Action, Observation, State)
├── tasks.py                   # Task configurations (easy, medium, hard)
├── graders.py                 # Scoring functions
├── client.py                  # HTTP client for environment
├── inference.py               # LLM-based parking agent
├── run.py                     # Server entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # Project metadata
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI application
    ├── collegpeparkfinal_environment.py  # Core environment logic
    ├── Dockerfile             # Alternative Dockerfile
    └── requirements.txt       # Server-specific dependencies
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_URL` | `http://localhost:7860` | Parking server URL |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | LLM model for inference |
| `HF_TOKEN` | - | Hugging Face API token |

## License

BSD-style license. See LICENSE file for details.
