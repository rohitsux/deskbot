# DeskBot Release Roadmap

> Each version delivers working features. Every release is a shippable increment.
> Testing & validation: See [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md)

**Platform goal:** Ship a complete OpenEnv environment for the Meta PyTorch Hackathon by April 8, 2026 — dual-arm robot cleans a messy desk, 3 tasks, deployed on HF Spaces.

**Platform context:**
- Physics: MuJoCo 3.x (CPU mode)
- Server: FastAPI + Pydantic v2
- Robot: Custom URDF (Qubit SO-100 inspired, 2 arms × 6 joints)
- Deployment: Docker → Hugging Face Spaces
- Team: Rohit + brother (2 people, 4 days)

**Build strategy:** Full AI-assisted build with Claude Code. One session per workstream. Parallel sessions for simulation vs server vs tasks.

---

## v0.1 — Walking Skeleton

**Status:** BUILD
**Delivers to:** Developer (yourself)
**Theme:** MuJoCo loads a desk with one object, robot exists, OpenEnv API responds
**Timeline:** Day 1 (first half)

### Features
- [ ] PyBullet scene: flat desk plane + one cube object
- [ ] Robot URDF: single arm with 6 joints loads into scene
- [ ] FastAPI server boots and exposes /reset, /step, /state, /tasks
- [ ] reset() spawns desk + object + robot, returns initial observation
- [ ] step() accepts an action, steps physics, returns StepResult
- [ ] state() returns episode_id + step_count
- [ ] Pydantic models: DeskAction, DeskObservation, DeskState (all in models.py)
- [ ] openenv.yaml manifest exists and passes `openenv validate`

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Server starts | `uvicorn deskbot.server:app` responds on :8000 |
| Reset works | POST /reset returns valid DeskObservation JSON |
| Step works | POST /step with action returns StepResult with reward float |
| openenv.yaml valid | `openenv validate` passes |

---

## v0.2 — Dual Arms + Object Catalogue

**Status:** BUILD
**Delivers to:** Developer
**Theme:** Full Qubit-style dual arms, 10+ objects with physics properties
**Timeline:** Day 1 (second half)

### Features
- [ ] Robot URDF: 2 arms (left + right), 12 joints, 2 grippers
- [ ] Inverse kinematics: target (x,y,z) → joint angles
- [ ] Gripper: open/close with contact detection
- [ ] Object catalogue: 10+ objects in JSON config
- [ ] Object properties: mass, size, fragility (bool), stackable (bool), material
- [ ] ObjectSpawner: random desk layouts from catalogue with seed
- [ ] Action space: pick(obj_id), place(x,y,z), push(obj_id, direction)
- [ ] Observation includes: all object positions + robot joint states + available actions

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Both arms move | Step with joint commands, both arms respond |
| IK works | Send target xyz, arm reaches within 2cm |
| Gripper grasps | Pick action on object, object lifts with gripper |
| 10 objects defined | objects.json has 10+ entries with all properties |
| Seed determinism | Same seed → same layout (run twice, compare) |

---

## v0.3 — Task 1: Simple Arrangement (Easy)

**Status:** BUILD
**Delivers to:** AI Agent (baseline)
**Theme:** Agent picks and places objects to match a target arrangement
**Timeline:** Day 2 (first half)

### Features
- [ ] Task config: task1_easy.yaml with target positions for 5 objects
- [ ] Cleanliness scorer: objects_in_target / total (within configurable threshold)
- [ ] Dense reward: +0.02 per object moved closer to target, -0.01 per step (time pressure)
- [ ] Episode ends: all objects in target OR max_steps reached
- [ ] Grader: score 0.0–1.0 based on final cleanliness
- [ ] max_steps: 50

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Task loads | reset(task="easy") spawns correct layout |
| Reward is dense | Each step returns non-zero reward signal |
| Episode terminates | Reaches done=True within max_steps |
| Grader scores | grade_easy(trajectory) returns float 0.0–1.0 |
| Perfect score possible | Manual optimal actions achieve 1.0 |

---

## v0.4 — Task 2: Sequenced Cleaning (Medium)

**Status:** BUILD
**Delivers to:** AI Agent (baseline)
**Theme:** Objects block each other — agent must figure out the right order
**Timeline:** Day 2 (second half)

### Features
- [ ] Task config: task2_medium.yaml with 8 objects, some blocking others
- [ ] Blocking detection: object A can't reach target until object B is moved
- [ ] Order scorer: 1.0 - (extra_moves / optimal_moves)
- [ ] Combined reward: 0.6 * cleanliness + 0.4 * order
- [ ] Dependency graph: auto-computed from initial vs target positions
- [ ] Episode ends: all placed OR max_steps (80)
- [ ] Grader: weighted cleanliness + order score

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Blocking works | Object behind another can't be directly reached |
| Order scored | Moving in wrong order penalises score |
| Optimal path exists | Known optimal sequence achieves 1.0 |
| Grader scores | grade_medium(trajectory) returns 0.0–1.0 |

---

## v0.5 — Task 3: Collision-Aware Arrangement (Hard)

**Status:** BUILD
**Delivers to:** AI Agent (baseline)
**Theme:** Objects have constraints — fragile, heavy, incompatible materials
**Timeline:** Day 3 (first half)

### Features
- [ ] Task config: task3_hard.yaml with 12 objects, mixed constraints
- [ ] Constraint engine: fragile can't go under heavy, glass can't touch metal, etc.
- [ ] Safety scorer: 1.0 - (violations * penalty_weight)
- [ ] Hard penalty: destroying fragile object = episode reward zeroed
- [ ] Combined reward: 0.5 * cleanliness + 0.3 * order + 0.2 * safety
- [ ] Constraint definitions in constraints.json (configurable, not hardcoded)
- [ ] Episode ends: all placed OR max_steps (120) OR fragile destroyed
- [ ] Grader: full 3-component score

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Constraints enforced | Placing glass on metal triggers violation |
| Fragile destruction works | Dropping fragile object → done=True, reward=0 |
| Full reward computed | 3 components combine correctly |
| Grader scores | grade_hard(trajectory) returns 0.0–1.0 |

---

## v0.6 — Baseline Agent + Reproducible Scores

**Status:** BUILD
**Delivers to:** Hackathon judges
**Theme:** LLM baseline agent (OpenAI API) runs all 3 tasks, prints reproducible scores
**Timeline:** Day 3 (second half)

> **Hackathon requirement:** Baseline script must use OpenAI API client and read OPENAI_API_KEY from environment variables.

### Features
- [ ] baseline.py: connects to environment via HTTP, runs 10 episodes per task using OpenAI chat completions
- [ ] Agent receives observation JSON, outputs action JSON via gpt-4o-mini (or configurable model)
- [ ] Random agent: untrained rollout as lower bound baseline
- [ ] Heuristic agent: rule-based fallback (pick nearest unplaced object, move to target)
- [ ] Score reporter: prints mean ± std per task
- [ ] Seed pinning: baseline uses fixed seeds for reproducibility
- [ ] OPENAI_API_KEY read from environment (never hardcoded)
- [ ] Expected baseline scores documented in README

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Baseline runs | `python baseline.py` completes without error |
| Scores printed | Output shows mean score per task |
| Reproducible | Run twice with same seed → same scores |
| Scores in range | Easy ~0.6, Medium ~0.4, Hard ~0.25 (challenging but learnable) |

---

## v0.7 — Docker + HF Spaces Deployment

**Status:** BUILD
**Delivers to:** Hackathon judges
**Theme:** Dockerfile works, deploys to Hugging Face Spaces, passes openenv validate
**Timeline:** Day 4 (first half)

### Features
- [ ] Dockerfile: python:3.11-slim, installs deps, copies code, runs server
- [ ] No GPU required (PyBullet CPU mode)
- [ ] Port 7860 (HF Spaces default)
- [ ] Health check endpoint: GET / returns 200
- [ ] `openenv validate` passes against running container
- [ ] HF Space created and accessible

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Docker builds | `docker build -t deskbot .` succeeds |
| Docker runs | `docker run -p 7860:7860 deskbot` serves API |
| HF Space live | URL accessible, /reset returns valid response |
| openenv validate | CLI passes all checks |

---

## v1.0 — README + Polish + Submit

**Status:** PLANNED
**Delivers to:** Hackathon judges (final submission)
**Theme:** Complete submission with documentation, clean code, compelling README
**Timeline:** Day 4 (second half)

### Features
- [ ] README.md: environment description, setup, action/observation spaces, baseline scores
- [ ] Architecture diagram (simple ASCII or D2)
- [ ] All code linted and clean
- [ ] All tests pass
- [ ] Final baseline scores documented
- [ ] Submission to hackathon portal

### Key Deliverables

| Deliverable | Verification |
|-------------|-------------|
| README complete | Covers all required sections from hackathon spec |
| Tests pass | `pytest` green |
| Docker + HF live | Environment accessible on HF Spaces |
| Baseline documented | Scores in README match actual runs |
| Submitted | Hackathon portal confirms submission |

---

## v1.0+ Future

- Voice command input (speak to robot in Hindi → Sunlo pipeline → action)
- Camera-based observation (RGB image from desk camera)
- More objects (50+ catalogue)
- Multi-desk environments (kitchen, garage, workshop)
- Real Qubit hardware integration (sim-to-real transfer)

---

## Release Process

1. All features checked off in the version section above
2. All tests pass in TESTING_AND_VALIDATION.md for that version
3. Docker build succeeds
4. `openenv validate` passes
5. Baseline script reproduces documented scores
6. Submit
