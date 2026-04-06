"""
Tests for deskbot/graders/graders.py.
"""
from __future__ import annotations

import pytest

from deskbot.graders.graders import grade_easy, grade_medium, grade_hard


# ------------------------------------------------------------------ #
# Shared fixtures / helpers                                           #
# ------------------------------------------------------------------ #

EASY_CONFIG = {
    "name": "easy",
    "max_steps": 50,
    "num_objects": 5,
    "object_ids": ["cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic", "stapler_metal"],
    "targets": {
        "cube_plastic":   [0.45, 0.30, 0.03],
        "mug_ceramic":    [0.10, 0.30, 0.03],
        "book_paper":     [0.45, 0.10, 0.03],
        "bottle_plastic": [0.10, 0.10, 0.03],
        "stapler_metal":  [0.28, 0.20, 0.03],
    },
    "reward_weights": {"cleanliness": 1.0, "order": 0.0, "safety": 0.0},
    "placement_threshold": 0.05,
}

MEDIUM_CONFIG = {
    "name": "medium",
    "max_steps": 100,
    "num_objects": 8,
    "object_ids": [
        "cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic",
        "stapler_metal", "notebook_paper", "pen_holder_plastic", "scissors_metal",
    ],
    "targets": {
        "cube_plastic":       [0.45, 0.35, 0.03],
        "mug_ceramic":        [0.10, 0.35, 0.03],
        "book_paper":         [0.45, 0.15, 0.03],
        "bottle_plastic":     [0.10, 0.15, 0.03],
        "stapler_metal":      [0.28, 0.25, 0.03],
        "notebook_paper":     [0.30, 0.10, 0.03],
        "pen_holder_plastic": [0.50, 0.20, 0.03],
        "scissors_metal":     [0.15, 0.20, 0.03],
    },
    "reward_weights": {"cleanliness": 0.6, "order": 0.4, "safety": 0.0},
    "placement_threshold": 0.05,
}

HARD_CONFIG = {
    "name": "hard",
    "max_steps": 150,
    "num_objects": 5,
    "object_ids": ["cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic", "stapler_metal"],
    "targets": {
        "cube_plastic":   [0.45, 0.35, 0.03],
        "mug_ceramic":    [0.10, 0.35, 0.03],
        "book_paper":     [0.45, 0.15, 0.03],
        "bottle_plastic": [0.10, 0.15, 0.03],
        "stapler_metal":  [0.28, 0.25, 0.03],
    },
    "reward_weights": {"cleanliness": 0.5, "order": 0.3, "safety": 0.2},
    "placement_threshold": 0.05,
}


def _make_obs(positions: dict, targets: dict, step: int = 0) -> dict:
    """Build a minimal observation dict."""
    objects = [{"id": oid, "position": pos} for oid, pos in positions.items()]
    target_list = [{"object_id": oid, "position": pos} for oid, pos in targets.items()]
    return {"objects": objects, "targets": target_list, "step_count": step}


def _make_trajectory(positions: dict, targets: dict, steps: int = 10) -> list[dict]:
    """Build a trajectory where the final observation has the given positions."""
    trajectory = []
    for i in range(steps):
        obs = _make_obs(positions, targets, step=i + 1)
        trajectory.append({
            "observation": obs,
            "action": {"action_type": "pick"},
            "reward": 0.1,
            "done": i == steps - 1,
        })
    return trajectory


# ------------------------------------------------------------------ #
# Easy grader tests                                                   #
# ------------------------------------------------------------------ #

def test_grade_easy_perfect():
    """All objects at target → score == 1.0."""
    targets = EASY_CONFIG["targets"]
    traj = _make_trajectory(targets.copy(), targets, steps=10)
    score = grade_easy(traj, EASY_CONFIG)
    assert score == pytest.approx(1.0)


def test_grade_easy_zero():
    """Nothing placed → score < 0.1."""
    targets = EASY_CONFIG["targets"]
    far = {k: [5.0, 5.0, 0.0] for k in targets}
    traj = _make_trajectory(far, targets, steps=5)
    score = grade_easy(traj, EASY_CONFIG)
    assert score < 0.1


def test_grade_easy_range():
    """grade_easy always returns [0, 1]."""
    targets = EASY_CONFIG["targets"]
    positions = dict(list(targets.items())[:3])  # only 3 objects present
    # Add remaining objects far away
    for k in targets:
        if k not in positions:
            positions[k] = [5.0, 5.0, 0.0]
    traj = _make_trajectory(positions, targets, steps=8)
    score = grade_easy(traj, EASY_CONFIG)
    assert 0.0 <= score <= 1.0


def test_grade_easy_empty_trajectory():
    """Empty trajectory → 0.0 (no positions → nothing placed)."""
    score = grade_easy([], EASY_CONFIG)
    assert score == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# Medium grader tests                                                 #
# ------------------------------------------------------------------ #

def test_grade_medium_range():
    """grade_medium always returns float in [0, 1]."""
    targets = MEDIUM_CONFIG["targets"]
    # Place half at target, half far away
    positions = {}
    for i, (k, v) in enumerate(targets.items()):
        positions[k] = v if i % 2 == 0 else [5.0, 5.0, 0.0]
    traj = _make_trajectory(positions, targets, steps=20)
    score = grade_medium(traj, MEDIUM_CONFIG)
    assert 0.0 <= score <= 1.0


def test_grade_medium_perfect():
    """All placed, optimal moves → score close to 1.0."""
    targets = MEDIUM_CONFIG["targets"]
    # num_objects=8, optimal=16 moves (pick+place each)
    traj = _make_trajectory(targets.copy(), targets, steps=16)
    score = grade_medium(traj, MEDIUM_CONFIG)
    assert score == pytest.approx(1.0)


def test_grade_medium_returns_float():
    """Return type is float."""
    targets = MEDIUM_CONFIG["targets"]
    traj = _make_trajectory(targets.copy(), targets, steps=10)
    score = grade_medium(traj, MEDIUM_CONFIG)
    assert isinstance(score, float)


# ------------------------------------------------------------------ #
# Hard grader tests                                                   #
# ------------------------------------------------------------------ #

def test_grade_hard_fragile_destroyed():
    """Step with done=True and reward==0.0 → grade is 0.0."""
    targets = HARD_CONFIG["targets"]
    positions = targets.copy()
    trajectory = [
        {
            "observation": _make_obs(positions, targets, step=1),
            "action": {"action_type": "pick"},
            "reward": 0.0,
            "done": True,   # fragile destroyed
        }
    ]
    score = grade_hard(trajectory, HARD_CONFIG)
    assert score == pytest.approx(0.0)


def test_grade_hard_range():
    """grade_hard always returns float in [0, 1]."""
    targets = HARD_CONFIG["targets"]
    positions = {}
    for i, (k, v) in enumerate(targets.items()):
        positions[k] = v if i % 2 == 0 else [5.0, 5.0, 0.0]
    traj = _make_trajectory(positions, targets, steps=15)
    score = grade_hard(traj, HARD_CONFIG)
    assert 0.0 <= score <= 1.0


def test_grade_hard_perfect():
    """All placed, no violations, near-optimal moves → high score."""
    targets = HARD_CONFIG["targets"]
    # num_objects=5, optimal=10 moves
    traj = _make_trajectory(targets.copy(), targets, steps=10)
    score = grade_hard(traj, HARD_CONFIG)
    assert score > 0.7


def test_grade_hard_done_with_nonzero_reward_not_destroyed():
    """done=True with positive reward should NOT trigger the destruction zero-out."""
    targets = HARD_CONFIG["targets"]
    positions = targets.copy()
    trajectory = [
        {
            "observation": _make_obs(positions, targets, step=1),
            "action": {"action_type": "place"},
            "reward": 1.0,
            "done": True,   # completed normally
        }
    ]
    score = grade_hard(trajectory, HARD_CONFIG)
    assert score > 0.0
