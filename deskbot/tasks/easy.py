"""
Task module for the easy (Simple Arrangement) task.
"""
from __future__ import annotations

import yaml


def load_task(config_path: str) -> dict:
    """Load YAML config, return task dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_object_ids(task: dict) -> list[str]:
    """Return list of object IDs for this task."""
    return list(task.get("object_ids", []))


def get_targets(task: dict) -> dict[str, list[float]]:
    """Return {object_id: [x, y, z]} target positions."""
    return dict(task.get("targets", {}))


def is_done(step_count: int, objects_placed: int, task: dict, destroyed: bool = False) -> bool:
    """Return True if episode should end.

    Ends when:
    - All objects are placed at their targets, OR
    - step_count has reached max_steps, OR
    - A fragile object was destroyed (destroyed=True)
    """
    if destroyed:
        return True
    max_steps = task.get("max_steps", 50)
    num_objects = task.get("num_objects", len(task.get("object_ids", [])))
    if step_count >= max_steps:
        return True
    if objects_placed >= num_objects:
        return True
    return False
