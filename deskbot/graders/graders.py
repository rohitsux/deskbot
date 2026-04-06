"""
Graders for DeskBot tasks.

Each grader takes a trajectory and task config, returns a score in [0.0, 1.0].

trajectory format:
    list of {"observation": dict, "action": dict, "reward": float, "done": bool}
"""
from __future__ import annotations

from deskbot.reward.cleanliness import compute_cleanliness
from deskbot.reward.order import compute_order
from deskbot.reward.safety import compute_safety, check_fragile_destroyed


def _extract_final_positions(trajectory: list[dict]) -> dict[str, list[float]]:
    """Pull {object_id: [x,y,z]} from the last observation in the trajectory."""
    if not trajectory:
        return {}
    last_obs = trajectory[-1].get("observation", {})
    objects = last_obs.get("objects", [])
    return {obj["id"]: obj["position"] for obj in objects if "id" in obj and "position" in obj}


def _extract_targets(task_config: dict, trajectory: list[dict] | None = None) -> dict[str, list[float]]:
    """Pull target positions — prefer trajectory obs targets over YAML config."""
    if trajectory:
        for step in reversed(trajectory):
            obs  = step.get("observation", {})
            tgts = obs.get("targets", [])
            if tgts:
                return {t["object_id"]: list(t["position"]) for t in tgts}
    raw = task_config.get("targets", {})
    return {k: list(v) for k, v in raw.items()}


def _count_moves(trajectory: list[dict]) -> int:
    """Count total steps taken (each trajectory entry = one step)."""
    return len(trajectory)


def _collect_violations(trajectory: list[dict]) -> list[dict]:
    """Aggregate all safety violations across the trajectory."""
    violations: list[dict] = []
    for step in trajectory:
        obs = step.get("observation", {})
        step_violations = obs.get("violations", [])
        if isinstance(step_violations, list):
            violations.extend(step_violations)
    return violations


def grade_easy(trajectory: list[dict], task_config: dict) -> float:
    """Score a completed easy episode. Returns 0.0–1.0.

    Uses final observation's object positions vs targets.
    Only cleanliness is scored (weight = 1.0).
    """
    final_positions = _extract_final_positions(trajectory)
    targets   = _extract_targets(task_config, trajectory)
    threshold = task_config.get("placement_threshold", 0.05)

    cleanliness = compute_cleanliness(final_positions, targets, threshold=threshold)

    weights = task_config.get("reward_weights", {"cleanliness": 1.0})
    w_clean = weights.get("cleanliness", 1.0)

    return float(min(1.0, max(0.0, w_clean * cleanliness)))


def grade_medium(trajectory: list[dict], task_config: dict) -> float:
    """Score medium episode: 0.6 * final_cleanliness + 0.4 * order_score."""
    final_positions = _extract_final_positions(trajectory)
    targets   = _extract_targets(task_config, trajectory)
    threshold = task_config.get("placement_threshold", 0.05)

    cleanliness = compute_cleanliness(final_positions, targets, threshold=threshold)

    num_objects = task_config.get("num_objects", len(targets))
    moves_taken = _count_moves(trajectory)
    # Optimal = 2 moves per object (pick + place)
    optimal_moves = max(1, num_objects * 2)
    order = compute_order(moves_taken, optimal_moves)

    weights = task_config.get("reward_weights", {"cleanliness": 0.6, "order": 0.4})
    w_clean = weights.get("cleanliness", 0.6)
    w_order = weights.get("order", 0.4)

    score = w_clean * cleanliness + w_order * order
    return float(min(1.0, max(0.0, score)))


def grade_hard(trajectory: list[dict], task_config: dict) -> float:
    """Score hard episode: 0.5 * cleanliness + 0.3 * order + 0.2 * safety.

    Returns 0.0 immediately if any step has done=True and reward==0.0
    (fragile destroyed).
    """
    # Early-exit for destruction events
    for step in trajectory:
        if step.get("done", False) and step.get("reward", 1.0) == 0.0:
            return 0.0

    final_positions = _extract_final_positions(trajectory)
    targets   = _extract_targets(task_config, trajectory)
    threshold = task_config.get("placement_threshold", 0.05)

    cleanliness = compute_cleanliness(final_positions, targets, threshold=threshold)

    num_objects = task_config.get("num_objects", len(targets))
    moves_taken = _count_moves(trajectory)
    optimal_moves = max(1, num_objects * 2)
    order = compute_order(moves_taken, optimal_moves)

    violations = _collect_violations(trajectory)
    safety = compute_safety(violations)

    weights = task_config.get("reward_weights", {"cleanliness": 0.5, "order": 0.3, "safety": 0.2})
    w_clean = weights.get("cleanliness", 0.5)
    w_order = weights.get("order", 0.3)
    w_safe = weights.get("safety", 0.2)

    score = w_clean * cleanliness + w_order * order + w_safe * safety
    return float(min(1.0, max(0.0, score)))
