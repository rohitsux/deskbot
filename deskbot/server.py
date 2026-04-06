"""
DeskBot FastAPI server — OpenEnv-compatible interface.

Endpoints
---------
GET  /          health check  → {"status": "healthy"}
GET  /web       interactive web UI
WS   /ws        WebSocket session (persistent, isolated per connection)
POST /reset     start new episode  (stateless HTTP fallback)
POST /step      advance one step   (stateless HTTP fallback)
GET  /state     current episode state
GET  /tasks     list all tasks and their schemas
GET  /baseline  run heuristic agent, return scores
POST /grader    score a completed trajectory
"""
from __future__ import annotations

import json
import math
import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from deskbot.models import (
    BaselineScores,
    DeskAction,
    DeskObservation,
    DeskState,
    GraderRequest,
    GraderResponse,
    ObjectState,
    StepInfo,
    StepResult,
    TargetState,
    TaskSchema,
    TasksResponse,
)
from deskbot.environment import BaseEnvironment
from deskbot.graders.graders import grade_easy, grade_medium, grade_hard

app = FastAPI(
    title="DeskBot OpenEnv",
    description="Dual-arm robot desk cleaning environment — Meta PyTorch OpenEnv Hackathon",
    version="0.1.0",
)

# ── shared HTTP fallback environment (stateless) ──────────────────────────────
_http_env = BaseEnvironment()

# ── task metadata ─────────────────────────────────────────────────────────────
_TASK_OBJECTS: dict[str, list[str]] = {
    "easy": ["cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic", "stapler_metal"],
    "medium": [
        "cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic",
        "stapler_metal", "notebook_paper", "pen_holder_plastic", "scissors_metal",
    ],
    "hard": [
        "cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic",
        "stapler_metal", "notebook_paper", "pen_holder_plastic", "scissors_metal",
        "glass_cup_glass", "vase_fragile_ceramic", "monitor_stand_metal", "keyboard_plastic",
    ],
}

_TASK_TARGETS: dict[str, dict[str, list[float]]] = {
    "easy": {
        "cube_plastic":   [0.45, 0.30, 0.78],
        "mug_ceramic":    [0.10, 0.30, 0.78],
        "book_paper":     [0.45, 0.10, 0.78],
        "bottle_plastic": [0.10, 0.10, 0.78],
        "stapler_metal":  [0.28, 0.20, 0.78],
    },
    "medium": {
        "cube_plastic":       [0.45, 0.35, 0.78],
        "mug_ceramic":        [0.10, 0.35, 0.78],
        "book_paper":         [0.45, 0.15, 0.78],
        "bottle_plastic":     [0.10, 0.15, 0.78],
        "stapler_metal":      [0.28, 0.25, 0.78],
        "notebook_paper":     [0.30, 0.10, 0.78],
        "pen_holder_plastic": [0.50, 0.20, 0.78],
        "scissors_metal":     [0.15, 0.20, 0.78],
    },
    "hard": {
        "cube_plastic":         [0.45, 0.35, 0.78],
        "mug_ceramic":          [0.10, 0.35, 0.78],
        "book_paper":           [0.45, 0.15, 0.78],
        "bottle_plastic":       [0.10, 0.15, 0.78],
        "stapler_metal":        [0.28, 0.25, 0.78],
        "notebook_paper":       [0.30, 0.10, 0.78],
        "pen_holder_plastic":   [0.50, 0.20, 0.78],
        "scissors_metal":       [0.15, 0.20, 0.78],
        "glass_cup_glass":      [0.55, 0.40, 0.78],
        "vase_fragile_ceramic": [0.05, 0.40, 0.78],
        "monitor_stand_metal":  [0.28, 0.42, 0.78],
        "keyboard_plastic":     [0.28, 0.05, 0.78],
    },
}

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
_TASK_FILES = {
    "easy":   "task1_easy.yaml",
    "medium": "task2_medium.yaml",
    "hard":   "task3_hard.yaml",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _stub_obs(task: str, episode_id: str, step_count: int = 0) -> DeskObservation:
    object_ids = _TASK_OBJECTS.get(task, _TASK_OBJECTS["easy"])
    targets_map = _TASK_TARGETS.get(task, _TASK_TARGETS["easy"])
    objects = []
    for i, oid in enumerate(object_ids):
        angle = 2 * math.pi * i / len(object_ids)
        x = 0.28 + 0.15 * math.cos(angle)
        y = 0.20 + 0.15 * math.sin(angle)
        fragile = "glass" in oid or "fragile" in oid
        material = oid.rsplit("_", 1)[-1] if "_" in oid else "plastic"
        objects.append(ObjectState(
            id=oid, position=[round(x, 3), round(y, 3), 0.78],
            held=False, fragile=fragile, material=material,
        ))
    targets = [TargetState(object_id=k, position=v) for k, v in targets_map.items()]
    return DeskObservation(
        episode_id=episode_id, objects=objects,
        joint_states=[0.0] * 12, gripper_states=[False, False],
        targets=targets, step_count=step_count,
    )


def _load_task_config(task: str) -> dict:
    import yaml
    path = os.path.join(_CONFIG_DIR, _TASK_FILES[task])
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {
            "targets": _TASK_TARGETS.get(task, {}),
            "placement_threshold": 0.05,
            "num_objects": len(_TASK_OBJECTS.get(task, [])),
            "reward_weights": {"cleanliness": 1.0, "order": 0.0, "safety": 0.0},
        }


# ── WebSocket session handler ─────────────────────────────────────────────────

async def _ws_session(websocket: WebSocket) -> None:
    """
    One isolated environment instance per WebSocket connection.
    Message protocol (JSON):
        Client → {"type": "reset", "task": "easy", "seed": 42}
        Client → {"type": "step",  "action_type": "pick", "object_id": "mug", "arm": "right"}
        Client → {"type": "state"}
        Server → StepResult / DeskObservation / DeskState as JSON dict
    """
    session_env = BaseEnvironment()
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            msg_type = data.get("type")

            if msg_type == "reset":
                task = data.get("task", "easy")
                seed = int(data.get("seed", 42))
                try:
                    obs = session_env.reset(task=task, seed=seed)
                except NotImplementedError:
                    obs = _stub_obs(task, session_env._episode_id or f"ep_{uuid.uuid4().hex[:8]}")
                await websocket.send_json(obs.model_dump())

            elif msg_type == "step":
                try:
                    action = DeskAction(
                        action_type=data.get("action_type", "pick"),
                        object_id=data.get("object_id"),
                        arm=data.get("arm", "right"),
                        target=data.get("target"),
                        direction=data.get("direction"),
                    )
                    result = session_env.step(action)
                except NotImplementedError:
                    episode_id = session_env._episode_id or f"ep_{uuid.uuid4().hex[:8]}"
                    task = session_env._task or "easy"
                    session_env._step_count += 1
                    obs = _stub_obs(task, episode_id, session_env._step_count)
                    result = StepResult(
                        observation=obs, reward=0.0, done=False,
                        info=StepInfo(cleanliness=0.0, order=1.0, safety=1.0),
                    )
                except Exception as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue
                await websocket.send_json(result.model_dump())

            elif msg_type == "state":
                await websocket.send_json(session_env.state().model_dump())

            else:
                await websocket.send_json({"error": f"unknown type: {msg_type!r}"})

    except WebSocketDisconnect:
        pass  # client closed — session_env is garbage-collected automatically


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/web", response_class=HTMLResponse)
def web_ui() -> HTMLResponse:
    """Interactive browser UI for testing the environment."""
    html = """
<!DOCTYPE html>
<html>
<head>
  <title>DeskBot — OpenEnv</title>
  <style>
    body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 24px; }
    h1   { color: #00d4ff; }
    button { background: #00d4ff; color: #000; border: none; padding: 8px 16px;
             margin: 4px; cursor: pointer; border-radius: 4px; font-weight: bold; }
    button:hover { background: #00aacc; }
    pre  { background: #0d0d1a; padding: 16px; border-radius: 8px;
           max-height: 400px; overflow-y: auto; white-space: pre-wrap; }
    input, select { background: #0d0d1a; color: #e0e0e0; border: 1px solid #333;
                    padding: 6px; margin: 4px; border-radius: 4px; }
    .row { margin: 12px 0; }
    .label { color: #888; margin-right: 8px; }
  </style>
</head>
<body>
  <h1>🦾 DeskBot — OpenEnv Interactive</h1>
  <div class="row">
    <span class="label">Task:</span>
    <select id="task"><option>easy</option><option>medium</option><option>hard</option></select>
    <span class="label">Seed:</span>
    <input id="seed" type="number" value="42" style="width:70px">
    <button onclick="doReset()">Reset</button>
  </div>
  <div class="row">
    <span class="label">Action:</span>
    <select id="action_type"><option>pick</option><option>place</option><option>push</option><option>home</option></select>
    <span class="label">Object:</span>
    <input id="object_id" placeholder="e.g. mug_ceramic" style="width:150px">
    <span class="label">Arm:</span>
    <select id="arm"><option>right</option><option>left</option></select>
    <button onclick="doStep()">Step</button>
    <button onclick="doState()">State</button>
  </div>
  <pre id="out">Connect and reset to start...</pre>

  <script>
    let ws = null;
    const out = document.getElementById('out');

    function log(obj) {
      out.textContent = JSON.stringify(obj, null, 2);
    }

    function connect() {
      if (ws && ws.readyState === WebSocket.OPEN) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(proto + '://' + location.host + '/ws');
      ws.onmessage = e => log(JSON.parse(e.data));
      ws.onerror   = e => log({error: 'WebSocket error'});
      ws.onclose   = () => log({status: 'disconnected'});
    }

    function send(msg) {
      connect();
      if (ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(msg)); }
      else { ws.onopen = () => ws.send(JSON.stringify(msg)); }
    }

    function doReset() {
      send({ type:'reset', task: document.getElementById('task').value,
             seed: parseInt(document.getElementById('seed').value) });
    }
    function doStep() {
      send({ type:'step',
             action_type: document.getElementById('action_type').value,
             object_id:   document.getElementById('object_id').value || null,
             arm:         document.getElementById('arm').value });
    }
    function doState() { send({ type:'state' }); }
    connect();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Persistent WebSocket session — one isolated environment per connection."""
    await _ws_session(websocket)


class _ResetBody(BaseModel):
    task: str = "easy"
    seed: int = 42

@app.post("/reset", response_model=DeskObservation)
def reset(body: _ResetBody = _ResetBody()) -> DeskObservation:
    task = body.task
    seed = body.seed
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=422, detail=f"Unknown task: {task!r}")
    try:
        return _http_env.reset(task=task, seed=seed)
    except NotImplementedError:
        return _stub_obs(task, _http_env._episode_id or f"ep_{uuid.uuid4().hex[:8]}")


@app.post("/step", response_model=StepResult)
def step(action: DeskAction) -> StepResult:
    try:
        return _http_env.step(action)
    except NotImplementedError:
        episode_id = _http_env._episode_id or f"ep_{uuid.uuid4().hex[:8]}"
        task = _http_env._task or "easy"
        _http_env._step_count += 1
        obs = _stub_obs(task, episode_id, _http_env._step_count)
        return StepResult(
            observation=obs, reward=0.0, done=False,
            info=StepInfo(cleanliness=0.0, order=1.0, safety=1.0),
        )


@app.get("/state", response_model=DeskState)
def state() -> DeskState:
    return _http_env.state()


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    action_fields = ["action_type", "object_id", "arm", "target", "direction"]
    # action_type values: pick | place | push | home
    return TasksResponse(tasks=[
        TaskSchema(name="easy",   display_name="Simple Arrangement",    difficulty="easy",
                   max_steps=50,  num_objects=5,  action_fields=action_fields),
        TaskSchema(name="medium", display_name="Sequenced Cleaning",    difficulty="medium",
                   max_steps=80,  num_objects=8,  action_fields=action_fields),
        TaskSchema(name="hard",   display_name="Collision-Aware Arrangement", difficulty="hard",
                   max_steps=120, num_objects=12, action_fields=action_fields),
    ])


@app.get("/baseline")
def baseline() -> dict:
    def _dist(a: list, b: list) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    scores: dict[str, float] = {}
    for task in ["easy", "medium", "hard"]:
        obs = _http_env.reset(task=task, seed=42)
        total_reward = 0.0
        arm_state: dict[str, str | None] = {"held_left": None, "held_right": None}
        for _ in range(30):
            objects = [o.model_dump() for o in obs.objects]
            targets = {t.object_id: t.position for t in obs.targets}
            step_num = obs.step_count
            arm = "left" if step_num % 2 == 0 else "right"
            held = arm_state.get(f"held_{arm}")
            if held and held in targets:
                action = DeskAction(action_type="place", arm=arm, target=targets[held])
                arm_state[f"held_{arm}"] = None
            else:
                held_ids = {arm_state.get("held_left"), arm_state.get("held_right")} - {None}
                best_obj, best_dist = None, -1.0
                for obj in objects:
                    if obj["id"] in held_ids:
                        continue
                    tgt = targets.get(obj["id"])
                    if tgt is None:
                        continue
                    d = _dist(obj["position"], tgt)
                    if d > best_dist:
                        best_dist, best_obj = d, obj
                if best_obj and best_dist > 0.04:
                    arm_state[f"held_{arm}"] = best_obj["id"]
                    action = DeskAction(action_type="pick", object_id=best_obj["id"], arm=arm)
                else:
                    action = DeskAction(
                        action_type="push",
                        arm=arm,
                        object_id=(objects[0]["id"] if objects else None),
                        direction=[0.0, 0.0],
                    )
            result = _http_env.step(action)
            total_reward += result.reward
            obs = result.observation
            if result.done:
                break
        scores[task] = round(total_reward, 4)
    return {"scores": scores, "agent": "heuristic", "episodes": 1}


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest) -> GraderResponse:
    task = request.task
    if task not in _TASK_FILES:
        raise HTTPException(status_code=422, detail=f"Unknown task: {task!r}")
    task_config = _load_task_config(task)
    graders_map = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
    score = graders_map[task](request.trajectory, task_config)
    return GraderResponse(task=task, score=score, components={"score": score})
