"""
Cleanliness reward: measures how many objects are at their target positions.
"""
from __future__ import annotations

import math


def _euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def compute_cleanliness(
    current_positions: dict[str, list[float]],
    target_positions: dict[str, list[float]],
    threshold: float = 0.05,
) -> float:
    """objects_within_threshold / total_objects → [0.0, 1.0]

    Args:
        current_positions: {object_id: [x, y, z]}
        target_positions:  {object_id: [x, y, z]}
        threshold: distance in metres below which an object is "placed"

    Returns:
        Fraction of objects that are within threshold of their target.
        Returns 0.0 if target_positions is empty.
    """
    if not target_positions:
        return 0.0
    placed = 0
    for obj_id, target in target_positions.items():
        current = current_positions.get(obj_id)
        if current is not None and _euclidean(current, target) <= threshold:
            placed += 1
    return placed / len(target_positions)


def compute_dense_cleanliness(
    current_positions: dict[str, list[float]],
    target_positions: dict[str, list[float]],
) -> float:
    """Dense signal: mean of (1 - normalized_distance) per object.

    Distances are normalised by the desk diagonal (~0.6 m) so the signal
    stays in [0, 1].  Objects not present in current_positions contribute 0.

    Args:
        current_positions: {object_id: [x, y, z]}
        target_positions:  {object_id: [x, y, z]}

    Returns:
        Mean dense cleanliness score in [0.0, 1.0].
    """
    if not target_positions:
        return 0.0

    DESK_DIAGONAL = 0.6  # normalisation constant (metres)

    total = 0.0
    for obj_id, target in target_positions.items():
        current = current_positions.get(obj_id)
        if current is not None:
            dist = _euclidean(current, target)
            normalised = min(dist / DESK_DIAGONAL, 1.0)
            total += 1.0 - normalised
        # missing object contributes 0

    return total / len(target_positions)
