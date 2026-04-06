# DeskBot Testing & Validation Guide

> Acceptance criteria, test cases, and verification steps for every release version.
> AI agents MUST consult the relevant section and verify all items before marking work complete.

**Platform context:**
- Physics: MuJoCo 3.x (CPU mode)
- Server: FastAPI + Uvicorn
- Types: Pydantic v2 (all models in deskbot/models.py)
- Test framework: pytest
- Container: Docker (python:3.11-slim)
- Deployment: Hugging Face Spaces

---

## v0.1 — Walking Skeleton

### Acceptance Criteria
- [ ] PyBullet scene loads without errors
- [ ] Robot URDF loads with 6 joints visible
- [ ] FastAPI server starts on port 8000
- [ ] POST /reset returns valid DeskObservation JSON
- [ ] POST /step with valid action returns StepResult JSON
- [ ] GET /state returns episode_id and step_count
- [ ] GET /tasks returns task list
- [ ] openenv.yaml passes `openenv validate`

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_server_starts | `GET /` | `200 {"status": "ok"}` | Status code 200 |
| test_reset | `POST /reset {"task": "easy"}` | `DeskObservation` with objects list | objects list length > 0 |
| test_step_valid | `POST /step {"action_type": "pick", "object_id": "cube_1", "target": [0,0,0.1]}` | `StepResult` with reward float | reward is float, done is bool |
| test_step_invalid | `POST /step {"action_type": "fly"}` | `422` or graceful error | Error message, no crash |
| test_state | `GET /state` after 3 steps | `{"episode_id": str, "step_count": 3}` | step_count == 3 |
| test_tasks | `GET /tasks` | TasksResponse with 3 tasks | easy/medium/hard all present |
| test_openenv_yaml | `openenv validate` | Pass | Exit code 0 |

### Performance Targets

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Server startup | < 5s | Time from `uvicorn` to first 200 response |
| reset() latency | < 500ms | Time for POST /reset to respond |
| step() latency | < 100ms | Time for POST /step to respond |

---

## v0.2 — Dual Arms + Object Catalogue

### Acceptance Criteria
- [ ] Both arms load with 6 joints each (12 total)
- [ ] IK solver reaches target within 2cm accuracy
- [ ] Gripper opens and closes, detects contact
- [ ] 10+ objects in catalogue with all required properties
- [ ] Same seed produces identical layout on two consecutive resets

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_dual_arms | Load robot | 12 movable joints | Joint count == 12 |
| test_ik_accuracy | target=[0.2, 0.1, 0.05] | End effector position | Distance to target < 0.02m |
| test_gripper_close | close_gripper() on object | Object attached to gripper | Object moves with gripper |
| test_gripper_open | open_gripper() | Object released | Object falls under gravity |
| test_object_catalogue | Load objects.json | 10+ objects with all fields | Each has: mass, size, fragility, stackable, material |
| test_seed_determinism | reset(seed=42) twice | Same object positions | Position diff < 1e-6 |
| test_pick_action | pick(obj_id="mug_1") | Robot reaches, grasps mug | Object in gripper after action |
| test_place_action | place(x=0.3, y=0.2, z=0) | Object placed at target | Object within 3cm of target |

### Data Validation
- Check: All objects in catalogue have mass > 0
- Check: All objects have valid material enum (glass, metal, plastic, wood, ceramic, paper)
- Check: Fragile objects have fragility=true
- Check: Joint limits within STS3215 servo range (0–300°)

---

## v0.3 — Task 1: Simple Arrangement (Easy)

### Acceptance Criteria
- [ ] reset(task="easy") spawns 5 objects with target positions
- [ ] Every step returns non-zero reward signal (dense)
- [ ] Episode terminates at max_steps=50 if not done
- [ ] Episode terminates early if all objects in target
- [ ] Grader returns float 0.0–1.0
- [ ] Perfect manual sequence achieves score 1.0

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_task1_loads | reset(task="easy") | 5 objects + 5 targets | len(objects) == 5, len(targets) == 5 |
| test_reward_dense | Move object closer to target | reward > 0 | reward > 0.0 on approach |
| test_reward_negative_step | Do nothing for 1 step | Small negative | reward < 0 (time penalty) |
| test_episode_done_success | Place all 5 objects correctly | done=True, high reward | done==True, reward > 0.9 |
| test_episode_done_timeout | Wait 50 steps | done=True | done==True at step 50 |
| test_grader_perfect | Optimal trajectory | score=1.0 | grade_easy() == 1.0 |
| test_grader_zero | No objects moved | score near 0.0 | grade_easy() < 0.1 |
| test_grader_partial | 3 of 5 objects placed | score ~0.6 | 0.5 < grade_easy() < 0.7 |

---

## v0.4 — Task 2: Sequenced Cleaning (Medium)

### Acceptance Criteria
- [ ] 8 objects spawn with blocking dependencies
- [ ] Agent can't reach blocked object directly
- [ ] Order score penalises unnecessary moves
- [ ] Combined reward uses 0.6 cleanliness + 0.4 order
- [ ] Grader returns 0.0–1.0 with order component

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_blocking | Try to pick blocked object | Action fails or object unreachable | Object not picked up |
| test_unblock_then_pick | Move blocker first, then pick | Object picked successfully | Object in gripper |
| test_order_optimal | Move in optimal sequence | order_score=1.0 | order component == 1.0 |
| test_order_suboptimal | Move in random sequence | order_score < 1.0 | order < 0.5 |
| test_grader_medium | Run full episode | Score 0.0–1.0 | grade_medium() in range |
| test_dependency_graph | Auto-compute from layout | Valid DAG | No circular dependencies |

---

## v0.5 — Task 3: Collision-Aware Arrangement (Hard)

### Acceptance Criteria
- [ ] 12 objects with material constraints
- [ ] Fragile object destruction triggers episode end
- [ ] Constraint violations reduce safety score
- [ ] Full 3-component reward: cleanliness + order + safety
- [ ] Grader returns 0.0–1.0 with all 3 components

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_fragile_drop | Drop glass object from height | done=True, reward=0 | Episode terminates, zero reward |
| test_constraint_violation | Place glass on metal | safety penalty | safety_score < 1.0 |
| test_no_violation | Place glass on wood | No penalty | safety_score == 1.0 |
| test_heavy_on_fragile | Place heavy book on glass | Destruction event | done=True, reward=0 |
| test_full_reward | Complete episode cleanly | 3-component score | All components > 0 |
| test_grader_hard | Run full episode | Score 0.0–1.0 | grade_hard() in range |
| test_constraints_from_json | Load constraints.json | Valid constraint rules | All materials paired correctly |

---

## v0.6 — Baseline Agent + Reproducible Scores

### Acceptance Criteria
- [ ] baseline.py runs end-to-end without errors
- [ ] Uses OpenAI API client, reads OPENAI_API_KEY from env
- [ ] Heuristic agent completes all 3 tasks
- [ ] Scores are reproducible (same seeds → same scores)
- [ ] Score ranges are reasonable (easy > medium > hard)

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_baseline_easy | Run 10 episodes task=easy | Mean score printed | 0.5 < mean < 0.8 |
| test_baseline_medium | Run 10 episodes task=medium | Mean score printed | 0.3 < mean < 0.6 |
| test_baseline_hard | Run 10 episodes task=hard | Mean score printed | 0.1 < mean < 0.4 |
| test_reproducibility | Run baseline twice with seed=42 | Same scores | Scores match exactly |

---

## v0.7 — Docker + HF Spaces

### Acceptance Criteria
- [ ] `docker build` succeeds
- [ ] `docker run` starts server on port 7860
- [ ] POST /reset works inside container
- [ ] `openenv validate` passes against container
- [ ] HF Space URL accessible

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_docker_build | `docker build -t deskbot .` | Build success | Exit code 0 |
| test_docker_health | `curl localhost:7860/` | `200 OK` | Status 200 |
| test_docker_reset | `curl -X POST localhost:7860/reset` | Valid JSON | Contains objects list |
| test_docker_step | POST /step with action | Valid StepResult | Contains reward float |
| test_openenv_validate | `openenv validate` | All checks pass | Exit code 0 |

---

## v1.0 — README + Submit

### Acceptance Criteria
- [ ] README has: description, setup, action space, observation space, tasks, baseline scores
- [ ] All previous version tests still pass (regression)
- [ ] Docker builds clean
- [ ] HF Space is live and responding
- [ ] Submitted to hackathon portal

### Critical Tests

| Test | Input | Expected Output | Pass Criteria |
|------|-------|-----------------|---------------|
| test_full_regression | `pytest` | All green | 0 failures |
| test_docker_e2e | Build + run + baseline | Scores printed | Matches documented scores |
| test_readme_sections | Parse README | Required sections present | All hackathon requirements covered |

---

## Cross-Release Validation Rules

### Determinism
- [ ] Same seed always produces same layout
- [ ] Same action sequence always produces same trajectory
- [ ] Baseline scores are reproducible across machines

### Robustness
- [ ] Invalid actions don't crash the server
- [ ] Missing fields in action return 422, not 500
- [ ] PyBullet errors are caught and return done=True gracefully

### OpenEnv Compliance
- [ ] openenv.yaml has name, description, version, tasks
- [ ] Each task has name, difficulty, max_steps
- [ ] reset/step/state all return typed Pydantic models
- [ ] `openenv validate` passes at every version

### Code Quality
- [ ] No hardcoded file paths (use relative or env vars)
- [ ] No secrets in code
- [ ] All config is in YAML/JSON, not Python
- [ ] Tests exist for every grader function
