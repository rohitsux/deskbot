"""
Tests for deskbot/reward/ modules.
"""
from __future__ import annotations

import pytest

from deskbot.reward.cleanliness import compute_cleanliness, compute_dense_cleanliness
from deskbot.reward.order import compute_order
from deskbot.reward.safety import compute_safety, check_fragile_destroyed


# ------------------------------------------------------------------ #
# Cleanliness                                                         #
# ------------------------------------------------------------------ #

TARGETS = {
    "cube_plastic":   [0.45, 0.30, 0.03],
    "mug_ceramic":    [0.10, 0.30, 0.03],
    "book_paper":     [0.45, 0.10, 0.03],
    "bottle_plastic": [0.10, 0.10, 0.03],
    "stapler_metal":  [0.28, 0.20, 0.03],
}


def test_cleanliness_perfect():
    """All objects exactly at target → 1.0."""
    score = compute_cleanliness(TARGETS.copy(), TARGETS.copy(), threshold=0.05)
    assert score == pytest.approx(1.0)


def test_cleanliness_zero():
    """All objects far from target → near 0.0."""
    far_positions = {k: [5.0, 5.0, 0.0] for k in TARGETS}
    score = compute_cleanliness(far_positions, TARGETS, threshold=0.05)
    assert score == pytest.approx(0.0)


def test_cleanliness_partial():
    """3 of 5 objects at target → 0.6."""
    current = TARGETS.copy()
    # Move 2 objects far away
    current["cube_plastic"]   = [5.0, 5.0, 0.0]
    current["mug_ceramic"]    = [5.0, 5.0, 0.0]
    score = compute_cleanliness(current, TARGETS, threshold=0.05)
    assert score == pytest.approx(3 / 5)


def test_cleanliness_empty_targets():
    """Empty targets → 0.0 (guard against ZeroDivisionError)."""
    score = compute_cleanliness({}, {}, threshold=0.05)
    assert score == 0.0


def test_dense_cleanliness_perfect():
    """All objects at target → 1.0 for dense signal."""
    score = compute_dense_cleanliness(TARGETS.copy(), TARGETS.copy())
    assert score == pytest.approx(1.0)


def test_dense_cleanliness_returns_float():
    """Dense cleanliness always returns a float in [0, 1]."""
    far = {k: [5.0, 5.0, 0.0] for k in TARGETS}
    score = compute_dense_cleanliness(far, TARGETS)
    assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------ #
# Order                                                               #
# ------------------------------------------------------------------ #

def test_order_perfect():
    """moves_taken == optimal_moves → 1.0."""
    assert compute_order(10, 10) == pytest.approx(1.0)


def test_order_suboptimal():
    """Extra moves → score < 1.0."""
    score = compute_order(15, 10)
    assert score < 1.0
    assert score >= 0.0


def test_order_double_moves():
    """Twice optimal → 0.0."""
    assert compute_order(20, 10) == pytest.approx(0.0)


def test_order_zero_optimal():
    """optimal_moves == 0 → 1.0 (no moves needed, guard against division by zero)."""
    assert compute_order(0, 0) == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Safety                                                              #
# ------------------------------------------------------------------ #

def test_safety_no_violations():
    """No violations → 1.0."""
    assert compute_safety([]) == pytest.approx(1.0)


def test_safety_with_violation():
    """One 0.3 penalty → 0.7."""
    violations = [{"type": "glass_on_metal", "penalty": 0.3}]
    assert compute_safety(violations) == pytest.approx(0.7)


def test_safety_multiple_violations():
    """Multiple penalties sum up, floored at 0.0."""
    violations = [
        {"type": "glass_on_metal", "penalty": 0.3},
        {"type": "drop_fragile",   "penalty": 0.4},
    ]
    assert compute_safety(violations) == pytest.approx(0.3)


def test_safety_floor_at_zero():
    """Total penalty > 1.0 → clamped at 0.0."""
    violations = [{"type": "destruction", "penalty": 1.5}]
    assert compute_safety(violations) == pytest.approx(0.0)


def test_check_fragile_destroyed_true():
    """penalty == 1.0 → destroyed."""
    violations = [{"type": "fragile_destroyed", "penalty": 1.0}]
    assert check_fragile_destroyed(violations) is True


def test_check_fragile_destroyed_false():
    """No full-penalty violation → not destroyed."""
    violations = [{"type": "near_miss", "penalty": 0.3}]
    assert check_fragile_destroyed(violations) is False


def test_check_fragile_destroyed_empty():
    """Empty violations → not destroyed."""
    assert check_fragile_destroyed([]) is False
