"""
Safety reward: penalises violations involving fragile / constraint-sensitive objects.
"""
from __future__ import annotations


def compute_safety(
    violations: list[dict],
) -> float:
    """1.0 - sum(penalties), floored at 0.0.

    Args:
        violations: list of dicts, each containing at minimum:
            {"type": str, "penalty": float}
            e.g. {"type": "glass_on_metal", "penalty": 0.3}

    Returns:
        Safety score in [0.0, 1.0].
    """
    total_penalty = sum(v.get("penalty", 0.0) for v in violations)
    return max(0.0, 1.0 - total_penalty)


def check_fragile_destroyed(violations: list[dict]) -> bool:
    """Return True if any violation has penalty == 1.0 (destruction event).

    Args:
        violations: same format as compute_safety.

    Returns:
        True if a destruction event occurred.
    """
    return any(v.get("penalty", 0.0) >= 1.0 for v in violations)
