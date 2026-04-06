"""
Integration test: exercises the FULL physics path (no mocks).
Requires mujoco to be installed — skipped automatically otherwise.
"""
from __future__ import annotations

import pytest


def test_environment_full_reset_object_ids():
    """Integration test: BaseEnvironment.reset() returns correct object IDs from YAML config."""
    pytest.importorskip("mujoco")
    from deskbot.environment import BaseEnvironment
    env = BaseEnvironment()
    obs = env.reset(task="easy", seed=42)

    expected_ids = {"cube_plastic", "mug_ceramic", "book_paper", "bottle_plastic", "stapler_metal"}
    returned_ids = {o.id for o in obs.objects}
    assert returned_ids == expected_ids, f"Object ID mismatch: expected {expected_ids}, got {returned_ids}"

    # Targets should be on the desk (z ≈ 0.78, not 0.03)
    for t in obs.targets:
        assert t.position[2] > 0.5, f"Target z={t.position[2]} too low — should be ~0.78 (desk height)"
