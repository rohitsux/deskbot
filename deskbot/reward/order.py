"""
Order reward: measures efficiency relative to optimal move count.
"""
from __future__ import annotations


def compute_order(
    moves_taken: int,
    optimal_moves: int,
) -> float:
    """1.0 - clamp((moves_taken - optimal_moves) / optimal_moves, 0, 1)

    Returns 1.0 when moves_taken == optimal_moves (perfect efficiency).
    Degrades linearly to 0.0 when moves_taken >= 2 * optimal_moves.

    Args:
        moves_taken:   total moves the agent used
        optimal_moves: minimum moves needed (lower bound)

    Returns:
        Order score in [0.0, 1.0].
    """
    if optimal_moves <= 0:
        # Guard against zero division; if no moves needed, score is 1.0
        return 1.0

    excess_ratio = (moves_taken - optimal_moves) / optimal_moves
    clamped = max(0.0, min(1.0, excess_ratio))
    return 1.0 - clamped
