# DeskBot — OpenEnv Robot Desk Cleaning Environment

> Source of truth for system design. Read this before making architectural decisions.
> Inspired by [Qubit](https://github.com/0xaiwhisperer/qubit) dual-arm desktop robot.

**Goal:** Train an AI agent to clean and organise a cluttered desk using a simulated dual-arm robot, scoring 0.0–1.0 across cleanliness, order, and safety.

---

## L1 — System Context

### Actors

| Actor | Who | What They Do |
|-------|-----|-------------|
| **AI Agent** | LLM or RL model | Sends actions (pick, place, push) and receives observations (object states, joint angles) |
| **Hackathon Judges** | Meta / HF evaluators | Run `openenv validate`, check Docker, review baseline scores |
| **Developers** | Future contributors | Build better agents, add objects, extend tasks |

### Platform Domains

| Domain | Purpose | Key Tech |
|--------|---------|----------|
| **Physics Simulation** | Simulate desk, objects, and dual-arm robot with realistic physics | PyBullet, URDF models |
| **Environment Server** | Expose OpenEnv API (reset/step/state/baseline/grader/tasks) over HTTP | FastAPI, Pydantic |
| **Object System** | Define objects with properties (mass, fragility, stackability, material) | Python dataclasses, JSON configs |
| **Reward Engine** | Compute 3-component reward (cleanliness + order + safety) per step | Pure Python scoring logic |
| **Task Manager** | Configure 3 difficulty levels, manage episode boundaries | YAML task configs |

### External Systems

| System | Role | Integration |
|--------|------|-------------|
| **Hugging Face Spaces** | Deployment target | Docker container, port 7860 |
| **OpenEnv CLI** | Validation (`openenv validate`) | openenv.yaml manifest |
| **PyBullet** | Physics engine | Python API, URDF loading |

### Key Data Flows (L1)

| From | To | What | Format |
|------|----|------|--------|
| AI Agent | Environment Server | Action (pick/place/push + target coords) | JSON POST |
| Environment Server | Physics Simulation | Execute action as joint commands | MuJoCo API calls |
| Physics Simulation | Reward Engine | Object positions, collision events | Python dicts |
| Reward Engine | Environment Server | StepResult (observation + reward + done) | Pydantic model |

---

## L2 — Containers & Data Flow

### Physics Simulation

| Container | Tech | What It Does |
|-----------|------|-------------|
| **DeskScene** | PyBullet | Loads desk plane, spawns objects, steps physics |
| **DualArmRobot** | URDF + PyBullet | Qubit-style dual SO-100 arms, 6 joints each, gripper end-effectors |
| **ObjectSpawner** | Python | Generates random desk clutter from object catalogue |

### Environment Server

| Container | Tech | What It Does |
|-----------|------|-------------|
| **FastAPI App** | FastAPI + Uvicorn | HTTP server exposing /reset, /step, /state, /baseline, /grader, /tasks |
| **SessionManager** | Python | Tracks episode state, step count, task config per session |

### Object System

| Container | Tech | What It Does |
|-----------|------|-------------|
| **ObjectCatalogue** | JSON + dataclasses | Defines 20+ objects with physics properties and constraints |
| **ConstraintEngine** | Python | Evaluates object interaction rules (fragile + heavy = violation) |

### Reward Engine

| Container | Tech | What It Does |
|-----------|------|-------------|
| **CleanlinessScorer** | Python | objects_in_target / total_objects, weighted by priority |
| **OrderScorer** | Python | 1.0 - (extra_moves / optimal_moves), penalise backtracking |
| **SafetyScorer** | Python | 1.0 - (violations * penalty), hard zero on fragile destruction |
| **RewardCombiner** | Python | task-weighted combination of the 3 components |

### Cross-Domain Data Flows (L2)

| From | To | Data | Purpose |
|------|----|------|---------|
| FastAPI App | DeskScene | Action command | Execute robot movement |
| DeskScene | ObjectSpawner | Reset signal | Generate new clutter layout |
| DeskScene | ConstraintEngine | Object positions + contact points | Check for violations |
| ConstraintEngine | SafetyScorer | Violation events | Compute safety penalty |
| All Scorers | RewardCombiner | Component scores | Compute final reward |

---

## L3 — Core Module Internals

### DualArmRobot (Critical — Qubit-inspired)

| Component | What It Does |
|-----------|-------------|
| **URDFLoader** | Loads robot URDF with 2 arms, 12 joints (6 per arm), gripper meshes |
| **JointController** | Position control for each joint, velocity limits, torque limits |
| **GripperController** | Binary open/close per gripper, contact force detection |
| **IKSolver** | Inverse kinematics — target (x,y,z) → joint angles for reach |
| **CollisionDetector** | Per-step collision query between gripper/arm and all objects |

### Reward Pipeline

| Stage | Input | Output |
|-------|-------|--------|
| **Position Check** | Current object positions vs target positions | Distance scores per object |
| **Move Counter** | Action history for episode | Move efficiency ratio |
| **Collision Log** | Contact events from PyBullet | Violation count + severity |
| **Combiner** | Three component scores | Single float 0.0–1.0 |

---

## Resilience & Failure Handling

| Component Down | Impact | Degradation |
|---------------|--------|-------------|
| **PyBullet crashes** | Episode fails | Return done=True, reward=0.0, log error |
| **Invalid action** | Robot can't execute | Return current observation unchanged, small penalty |
| **Object falls off desk** | Object lost | Mark as failed placement, cleanliness score reduced |

---

## Architecture Principles

1. **OpenEnv spec first** — every design decision must keep reset/step/state API clean
2. **Deterministic seeds** — same seed = same desk layout = reproducible scores
3. **No GPU required** — PyBullet CPU mode only, runs anywhere Docker runs
4. **Objects are config, not code** — add new objects via JSON, no Python changes
5. **Reward is always dense** — every step returns a meaningful signal, never sparse
6. **Qubit-compatible kinematics** — simulated arm matches SO-100 joint limits and reach
7. **Docker-first** — if it doesn't work in Docker, it doesn't work

---

## Robot Arm Spec (Qubit SO-100 Simulated)

| Property | Value |
|----------|-------|
| Arms | 2 (left + right) |
| Joints per arm | 6 (base rotation, shoulder pitch, shoulder roll, elbow pitch, wrist pitch, wrist roll) |
| Gripper | 1 per arm, binary open/close |
| Total DoF | 14 (12 joints + 2 grippers) |
| Reach radius | ~25cm per arm |
| Workspace | 60cm × 40cm desk surface |
| Joint limits | Matched to STS3215 servo range (0–300°) |
| Control mode | Position control with configurable velocity |

---

## Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| Physics | MuJoCo 3.x (CPU mode) |
| Robot model | Custom URDF (Qubit SO-100 inspired) |
| API server | FastAPI + Uvicorn |
| Types | Pydantic v2 (models.py — single source of truth) |
| Task config | YAML |
| Object config | JSON |
| Container | Docker (python:3.11-slim) |
| Deployment | Hugging Face Spaces |
| Testing | pytest |
| Baseline | OpenAI API (gpt-4o-mini) |

---

## Related Documents

| Doc | What |
|-----|------|
| `docs/RELEASE_ROADMAP.md` | Features per release version (v0.1 → v1.0) |
| `docs/TESTING_AND_VALIDATION.md` | Test cases per release |
| `CLAUDE.md` | AI agent session context for Claude Code |
| `openenv.yaml` | OpenEnv manifest |
