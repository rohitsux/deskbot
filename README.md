---
title: DeskBot
emoji: 🦾
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# DeskBot — OpenEnv Robot Desk Cleaning Environment

A simulated dual-arm desktop robot learns to clean and organise a cluttered desk.
Built for the **Meta PyTorch OpenEnv Hackathon** (April 2026).


---

## What It Does

Train a reinforcement learning agent to control a dual-arm robot that picks up, moves,
and places objects on a messy desk. Three difficulty levels with progressively harder
manipulation challenges:

| Task | Objects | Challenge |
|------|---------|-----------|
| Easy | 5 | Pick and place to target positions |
| Medium | 8 | Objects block each other — order matters |
| Hard | 12 | Material constraints + fragile objects |

---

## Quick Start

```bash
# Install
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Start the server
uvicorn deskbot.server:app --host 0.0.0.0 --port 8000

# Reset environment (task: easy / medium / hard)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'

# Send an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "pick", "object_id": "cube_1", "arm": "right"}'

# Get episode state
curl http://localhost:8000/state

# List all tasks and action schema
curl http://localhost:8000/tasks
```

---

## API Reference

### POST /reset

Spawns a new episode with the desk and objects.

```json
// Request
{ "task": "easy", "seed": 42 }

// Response — DeskObservation
{
  "episode_id": "ep_abc123",
  "objects": [
    { "id": "cube_1", "position": [0.15, 0.10, 0.03], "held": false }
  ],
  "joint_states": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "gripper_states": [false, false],
  "targets": [
    { "object_id": "cube_1", "position": [0.30, 0.20, 0.03] }
  ],
  "step_count": 0
}
```

### POST /step

Execute one action and advance the simulation.

```json
// Request — DeskAction
{
  "action_type": "pick",   // pick | place | push
  "object_id": "cube_1",  // for pick and push
  "arm": "right",         // left | right
  "target": [0.3, 0.2, 0.03]  // for place
}

// Response — StepResult
{
  "observation": { ... },  // DeskObservation (updated state)
  "reward": 0.045,
  "done": false,
  "info": { "cleanliness": 0.2, "order": 1.0, "safety": 1.0 }
}
```

### GET /state

```json
{
  "episode_id": "ep_abc123",
  "step_count": 7
}
```

### GET /tasks

Returns task list and action schema required by OpenEnv spec.

### GET /baseline

Triggers baseline agent run and returns scores for all 3 tasks.

### POST /grader

Returns grader score after episode completion.

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `pick` | `object_id`, `arm` | Grasp an object with the specified arm |
| `place` | `x`, `y`, `z`, `arm` | Place the held object at a desk position |
| `push` | `object_id`, `direction [dx, dy]` | Push an object without grasping |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `objects` | list | All objects — position (x,y,z), held status, fragility |
| `joint_states` | list[float] | 12 joint angles in radians (6 per arm) |
| `gripper_states` | list[bool] | Open/close state for each gripper [left, right] |
| `targets` | list | Target position for each object |
| `step_count` | int | Steps taken this episode |
| `episode_id` | str | Unique episode identifier |

---

## Reward Function

Every step returns a dense reward signal — the agent always gets feedback:

**Easy:** `reward = cleanliness_score`
**Medium:** `reward = 0.6 × cleanliness + 0.4 × order`
**Hard:** `reward = 0.5 × cleanliness + 0.3 × order + 0.2 × safety`

| Component | Formula | Range |
|-----------|---------|-------|
| Cleanliness | objects_at_target / total_objects | 0.0–1.0 |
| Order | 1.0 - (extra_moves / optimal_moves) | 0.0–1.0 |
| Safety | 1.0 - (violations × penalty) | 0.0–1.0 |

**Hard penalty:** Destroying a fragile object → `done=True`, `reward=0.0`

Step penalty: `-0.01` per step (time pressure to solve efficiently).

---

## Robot Arm Spec (Qubit SO-100 Inspired)

| Property | Value |
|----------|-------|
| Arms | 2 (left + right) |
| Joints per arm | 6 (base, shoulder pitch, shoulder roll, elbow, wrist pitch, wrist roll) |
| Gripper | Binary open/close per arm |
| Total DoF | 12 (5 arm joints + 1 gripper per arm) |
| Workspace | 60cm × 40cm desk |
| Reach per arm | ~25cm radius |
| Control | Position control, CPU-only MuJoCo |

---

## Tasks

### Easy — Simple Arrangement (max_steps=50)
5 randomly placed objects, each with a fixed target position.
The agent must pick each object and place it at its target.
Score = fraction of objects correctly placed.

### Medium — Sequenced Cleaning (max_steps=100)
8 objects where some block access to others.
Agent must figure out the right removal order.
Wrong order = inefficiency penalty on order score.

### Hard — Collision-Aware Arrangement (max_steps=150)
12 objects with material constraints:
- Glass cannot touch metal
- Heavy items cannot be stacked on fragile objects
- Dropping a fragile object ends the episode immediately

---

## Baseline Scores (LLM agent, gpt-4o-mini, 10 episodes, seed=42)

| Task | Mean Score | Std |
|------|-----------|-----|
| Easy | ~0.60 | ±0.08 |
| Medium | ~0.40 | ±0.10 |
| Hard | ~0.25 | ±0.09 |

Run the baseline yourself:
```bash
export OPENAI_API_KEY=your_key_here
python baseline.py
```

---

## Running with Docker

```bash
# Build
docker build -t deskbot .

# Run (HF Spaces port)
docker run -p 7860:7860 deskbot

# Validate OpenEnv spec
openenv validate
```

---

## Project Structure

```
deskbot/
├── server.py          # FastAPI app (reset/step/state/baseline/grader/tasks)
├── models.py          # All Pydantic models — single source of truth
├── environment.py     # BaseEnvironment — ties simulation to API
├── simulation/
│   ├── scene.py       # MuJoCo desk scene
│   ├── robot.py       # Dual-arm robot, IK solver, grippers
│   └── objects.py     # Object spawner from catalogue
├── tasks/
│   ├── easy.py        # 5 objects, simple placement
│   ├── medium.py      # 8 objects, blocking dependencies
│   └── hard.py        # 12 objects, material constraints
├── reward/
│   ├── cleanliness.py # Position-based scoring
│   ├── order.py       # Move efficiency scoring
│   └── safety.py      # Collision/constraint scoring
└── graders/
    └── graders.py     # grade_easy / grade_medium / grade_hard
config/
├── objects.json       # 20+ object catalogue (mass, fragility, material)
├── constraints.json   # Material interaction rules
├── task1_easy.yaml
├── task2_medium.yaml
└── task3_hard.yaml
urdf/
└── deskbot.urdf       # Dual-arm robot model
baseline.py            # LLM baseline agent (OpenAI API)
openenv.yaml           # OpenEnv manifest
Dockerfile             # HF Spaces deployment
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Physics | MuJoCo 3.x (CPU mode) |
| Robot model | Custom URDF (Qubit SO-100 inspired) |
| API server | FastAPI + Uvicorn |
| Types | Pydantic v2 |
| Baseline agent | OpenAI API (gpt-4o-mini) |
| Container | Docker (python:3.11-slim) |
| Deployment | Hugging Face Spaces |

---

## Setup

```bash
git clone https://huggingface.co/spaces/SupremeM/deskbot
cd deskbot
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uvicorn deskbot.server:app --port 8000
```

---

## License

MIT — free to use, build on, and submit agents.

Built by Rohit Suthar for the Meta PyTorch OpenEnv Hackathon, April 2026.
