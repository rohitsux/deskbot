"""
DeskBot inference script — OpenEnv hackathon submission format.

Runs an LLM agent on all 3 tasks and logs in the required format:
  [START]
  [STEP] {"step": N, "action_type": "...", "reward": 0.0, "done": false, ...}
  [END] Final Score: 0.XX, Steps taken: N

Environment variables:
  API_BASE_URL  — LLM API base URL (default: OpenAI)
  MODEL_NAME    — model to use (default: gpt-4o-mini)
  HF_TOKEN      — Hugging Face token (used as bearer if API_BASE_URL points to HF)
  DESKBOT_URL   — running environment URL (default: http://localhost:7860)
"""
from __future__ import annotations

import json
import math
import os
import sys

import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL     = os.getenv("DESKBOT_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME   = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN     = os.getenv("HF_TOKEN")
SEED         = 42

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are controlling a dual-arm desktop robot that cleans a cluttered desk.
Output ONLY a JSON action object — no extra text, no markdown.

Action format (pick one):
{"action_type": "pick",  "object_id": "<id>", "arm": "left"|"right"}
{"action_type": "place", "arm": "left"|"right", "target": [x, y, z]}
{"action_type": "push",  "object_id": "<id>", "direction": [dx, dy]}

Strategy: pick objects that are furthest from their targets, then place them.
"""


def _obs_to_prompt(obs: dict) -> str:
    targets = {t["object_id"]: t["position"] for t in obs.get("targets", [])}
    lines = [f"Step {obs.get('step_count', 0)}. Objects on desk:"]
    for obj in obs.get("objects", []):
        oid = obj["id"]
        dist = math.sqrt(sum(
            (a - b) ** 2
            for a, b in zip(obj["position"], targets.get(oid, obj["position"]))
        ))
        lines.append(
            f"  {oid}: pos={[round(v,3) for v in obj['position']]} "
            f"target={targets.get(oid,'?')} dist={dist:.3f} held={obj.get('held', False)}"
        )
    lines.append("Output JSON action:")
    return "\n".join(lines)


def _llm_action(obs: dict) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _obs_to_prompt(obs)},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "{}").strip()
        text = text.lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(text)
    except Exception as exc:
        print(f"  LLM error: {exc}", file=sys.stderr)
        return {"action_type": "push", "direction": [0.01, 0.01]}


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    # Reset environment
    resp = requests.post(f"{BASE_URL}/reset",
                         json={"task": task_id, "seed": SEED}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()
    episode_id = obs.get("episode_id", "unknown")

    print("[START]", flush=True)

    step_count = 0
    done = False
    last_reward = 0.0
    trajectory: list[dict] = []

    while not done:
        action = _llm_action(obs)
        step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
        if step_resp.status_code != 200:
            print(f"  step error {step_resp.status_code}", file=sys.stderr)
            break

        data       = step_resp.json()
        last_reward = data.get("reward", 0.0)
        done        = data.get("done", False)
        obs         = data.get("observation", obs)
        step_count += 1

        log = {
            "step":        step_count,
            "action_type": action.get("action_type"),
            "reward":      round(last_reward, 4),
            "done":        done,
            "objects_at_target": sum(
                1 for obj in obs.get("objects", [])
                if obj.get("at_target", False)
            ),
            "step_count": obs.get("step_count", step_count),
        }
        print(f"[STEP] {json.dumps(log)}", flush=True)
        trajectory.append({"observation": obs, "action": action,
                            "reward": last_reward, "done": done})

    # Grader
    score = 0.0
    try:
        grader_resp = requests.post(
            f"{BASE_URL}/grader",
            json={"task": task_id, "episode_id": episode_id, "trajectory": trajectory},
            timeout=30,
        )
        if grader_resp.status_code == 200:
            score = float(grader_resp.json().get("score", 0.0))
    except Exception as exc:
        print(f"  grader error: {exc}", file=sys.stderr)

    print(f"[END] Final Score: {score:.4f}, Steps taken: {step_count}", flush=True)
    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def run_inference() -> None:
    tasks = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}

    for task_id in tasks:
        print(f"\n=== Task: {task_id} ===", flush=True)
        try:
            scores[task_id] = run_task(task_id)
        except Exception as exc:
            print(f"[END] Final Score: 0.0000, Steps taken: 0", flush=True)
            print(f"  task error: {exc}", file=sys.stderr)
            scores[task_id] = 0.0

    total = sum(scores.values()) / len(scores)
    print(f"\n=== Overall Mean Score: {total:.4f} ===", flush=True)
    print(f"  easy={scores.get('easy',0):.4f} "
          f"medium={scores.get('medium',0):.4f} "
          f"hard={scores.get('hard',0):.4f}", flush=True)


if __name__ == "__main__":
    run_inference()
