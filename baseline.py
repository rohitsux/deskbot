"""
DeskBot baseline agent.

Two modes (set BASELINE_MODE env var):
  heuristic  (default) — rule-based, free, no API key needed
                         Greedy: pick object furthest from target, place it.
  llm                  — OpenAI API (needs OPENAI_API_KEY + billing)

Usage:
    python baseline.py                          # heuristic, no key needed
    BASELINE_MODE=llm python baseline.py        # LLM mode
"""
from __future__ import annotations

import json
import math
import os
import sys

import requests

BASE_URL = os.getenv("DESKBOT_URL", "http://localhost:8000")
EPISODES = 10
SEED     = 42
MODE     = os.getenv("BASELINE_MODE", "heuristic")


# ── Heuristic agent ───────────────────────────────────────────────────────────

def _dist(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def heuristic_action(obs: dict, state: dict) -> dict:
    """
    Simple greedy pick-and-place:
      - If an arm is holding something → place it at its target.
      - Otherwise → pick the object that is furthest from its target.
    Alternates arms every step for efficiency.
    """
    objects   = obs.get("objects", [])
      # dict from last step, tracks what each arm holds
    targets   = {t["object_id"]: t["position"] for t in obs.get("targets", [])}
    step      = obs.get("step_count", 0)
    arm       = "left" if step % 2 == 0 else "right"

    # Check if this arm is holding something
    held = state.get(f"held_{arm}")
    if held and held in targets:
        return {
            "action_type": "place",
            "arm": arm,
            "target": targets[held],
        }

    # Find object furthest from target that's not held by either arm
    held_ids = {state.get("held_left"), state.get("held_right")} - {None}
    best_obj, best_dist = None, -1.0
    for obj in objects:
        if obj["id"] in held_ids:
            continue
        tgt = targets.get(obj["id"])
        if tgt is None:
            continue
        d = _dist(obj["position"], tgt)
        if d > best_dist:
            best_dist = d
            best_obj  = obj

    if best_obj and best_dist > 0.04:
        state[f"held_{arm}"] = best_obj["id"]
        return {
            "action_type": "pick",
            "object_id": best_obj["id"],
            "arm": arm,
        }

    # Everything at target — no-op push
    return {"action_type": "push", "arm": arm,
            "object_id": (objects[0]["id"] if objects else None),
            "direction": [0.0, 0.0]}


# ── LLM agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are controlling a dual-arm desktop robot that cleans a cluttered desk.
Output ONLY a JSON action object — no extra text.

Action format:
{"action_type": "pick"|"place"|"push", "object_id": "<id>", "arm": "left"|"right",
 "target": [x,y,z], "direction": [dx,dy]}

Strategy: pick objects far from their targets, place at target positions.
"""

def _obs_prompt(obs: dict) -> str:
    targets = {t["object_id"]: t["position"] for t in obs.get("targets", [])}
    lines   = [f"Step {obs.get('step_count', 0)}"]
    for obj in obs.get("objects", []):
        oid = obj["id"]
        lines.append(f"  {oid}: pos={obj['position']} target={targets.get(oid,'?')} held={obj.get('held',False)}")
    lines.append("Output JSON action:")
    return "\n".join(lines)

def _parse(content: str) -> dict:
    text = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"action_type": "push", "direction": [0.01, 0.01]}

def llm_action(client, obs: dict) -> dict:
    from openai import OpenAI  # only import if needed
    try:
        resp = client.chat.completions.create(
            model    = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _obs_prompt(obs)},
            ],
            temperature = 0.2,
            max_tokens  = 150,
        )
        return _parse(resp.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"  LLM error: {exc}", file=sys.stderr)
        return {"action_type": "push", "direction": [0.01, 0.01]}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task: str, episode_seed: int, client=None) -> float:
    resp = requests.post(f"{BASE_URL}/reset",
                         json={"task": task, "seed": episode_seed}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    trajectory: list[dict] = []
    episode_id = obs.get("episode_id", "unknown")
    arm_state: dict = {}   # tracks what each arm is holding (heuristic)
    done = False

    while not done:
        if MODE == "llm" and client:
            action = llm_action(client, obs)
        else:
            action = heuristic_action(obs, arm_state)

        step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
        if step_resp.status_code != 200:
            print(f"  step error {step_resp.status_code}", file=sys.stderr)
            break

        step_data = step_resp.json()
        reward    = step_data.get("reward", 0.0)
        done      = step_data.get("done", False)
        obs       = step_data.get("observation", obs)

        # Update held state from observation
        for obj in obs.get("objects", []):
            if obj.get("held"):
                # figure out which arm holds it from gripper_states
                pass  # server tracks this; heuristic tracks locally via arm_state

        trajectory.append({"observation": obs, "action": action,
                            "reward": reward, "done": done})

    grader = requests.post(f"{BASE_URL}/grader",
                           json={"task": task, "episode_id": episode_id,
                                 "trajectory": trajectory}, timeout=30)
    return float(grader.json().get("score", 0.0)) if grader.status_code == 200 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = None
    if MODE == "llm":
        from openai import OpenAI
        client = OpenAI()
        print("Mode: LLM (gpt-4o-mini)")
    else:
        print("Mode: heuristic (no API key needed)")

    all_scores: dict[str, float] = {}
    for task in ["easy", "medium", "hard"]:
        print(f"\n── Task: {task} ({EPISODES} episodes) ──")
        scores: list[float] = []
        for i in range(EPISODES):
            try:
                score = run_episode(task, SEED + i, client)
            except Exception as exc:
                print(f"  ep {i} error: {exc}", file=sys.stderr)
                score = 0.0
            scores.append(score)
            print(f"  ep {i+1:2d}: {score:.3f}")

        mean = sum(scores) / len(scores)
        std  = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        print(f"  → mean={mean:.3f} ± {std:.3f}")
        all_scores[task] = mean

    print(f"\n{'='*40}")
    print(f"  easy:   {all_scores.get('easy',0):.3f}")
    print(f"  medium: {all_scores.get('medium',0):.3f}")
    print(f"  hard:   {all_scores.get('hard',0):.3f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
