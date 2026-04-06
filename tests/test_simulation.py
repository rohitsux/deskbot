"""
Tests for the DeskBot simulation layer.

Run with: pytest tests/test_simulation.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from deskbot.simulation.scene import DeskScene
from deskbot.simulation.robot import DualArmRobot
from deskbot.environment import BaseEnvironment
from deskbot.models import DeskAction


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def scene():
    """Create one DeskScene for the module and close it after all tests."""
    s = DeskScene()
    yield s
    s.close()


@pytest.fixture(scope="module")
def reset_scene(scene):
    """Reset the scene once per module and return the initial state dict."""
    state = scene.reset(
        task="easy",
        object_ids=["mug_ceramic", "notebook", "pen_holder", "wooden_block", "scissors"],
        targets={"mug_ceramic": [0.1, 0.1, scene.DESK_HEIGHT + scene.DESK_THICKNESS / 2.0 + 0.05]},
        seed=42,
    )
    return state


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestSceneLoads:
    def test_scene_loads(self):
        """DeskScene() should create without errors and close cleanly."""
        s = DeskScene()
        assert s is not None
        s.close()


class TestResetReturnsObjects:
    def test_reset_returns_objects(self, reset_scene):
        """reset() should return a dict with a non-empty objects list."""
        state = reset_scene
        assert isinstance(state, dict), "reset() must return a dict"
        assert "objects" in state, "State must contain 'objects' key"
        assert len(state["objects"]) > 0, "Objects list must not be empty"

    def test_reset_returns_joint_states(self, reset_scene):
        """State dict must include joint_states of length 12."""
        joints = reset_scene.get("joint_states", [])
        assert len(joints) == 12, f"Expected 12 joint states, got {len(joints)}"

    def test_reset_returns_gripper_states(self, reset_scene):
        """State dict must include gripper_states of length 2."""
        grippers = reset_scene.get("gripper_states", [])
        assert len(grippers) == 2, f"Expected 2 gripper states, got {len(grippers)}"

    def test_reset_returns_targets(self, reset_scene):
        """State dict must include targets list."""
        assert "targets" in reset_scene

    def test_object_state_has_position(self, reset_scene):
        """Each object entry must have an 'id' and a 3-element 'position'."""
        for obj in reset_scene["objects"]:
            assert "id" in obj
            assert "position" in obj
            assert len(obj["position"]) == 3


class TestSeedDeterminism:
    def test_seed_determinism(self):
        """Same seed must produce identical object positions (diff < 1e-6)."""
        object_ids = ["mug_ceramic", "notebook", "pen_holder"]

        s1 = DeskScene()
        state1 = s1.reset(task="easy", object_ids=object_ids, targets={}, seed=7)
        pos1 = {obj["id"]: obj["position"] for obj in state1["objects"]}
        s1.close()

        s2 = DeskScene()
        state2 = s2.reset(task="easy", object_ids=object_ids, targets={}, seed=7)
        pos2 = {obj["id"]: obj["position"] for obj in state2["objects"]}
        s2.close()

        for oid in object_ids:
            assert oid in pos1, f"{oid} missing from first run"
            assert oid in pos2, f"{oid} missing from second run"
            diff = np.linalg.norm(np.array(pos1[oid]) - np.array(pos2[oid]))
            assert diff < 1e-6, (
                f"Object '{oid}' positions differ by {diff:.2e} between same-seed runs"
            )

    def test_different_seeds_produce_different_positions(self):
        """Different seeds should generally produce different object positions."""
        object_ids = ["mug_ceramic", "notebook", "pen_holder", "wooden_block"]

        s1 = DeskScene()
        state1 = s1.reset(task="easy", object_ids=object_ids, targets={}, seed=1)
        pos1 = {obj["id"]: obj["position"] for obj in state1["objects"]}
        s1.close()

        s2 = DeskScene()
        state2 = s2.reset(task="easy", object_ids=object_ids, targets={}, seed=99)
        pos2 = {obj["id"]: obj["position"] for obj in state2["objects"]}
        s2.close()

        # At least one object should be in a different position
        any_diff = any(
            np.linalg.norm(np.array(pos1[oid]) - np.array(pos2[oid])) > 1e-6
            for oid in object_ids
        )
        assert any_diff, "Different seeds produced identical positions — RNG not seeded"


class TestPickAction:
    def test_pick_action(self, scene, reset_scene):
        """pick action should return a valid state dict without errors."""
        # Identify a spawned object and its position
        obj = reset_scene["objects"][0]
        pos = obj["position"]

        result = scene.step(
            action_type="pick",
            object_id=obj["id"],
            arm="right",
            target=pos,
            direction=None,
        )
        assert isinstance(result, dict), "step() must return a dict"
        assert "objects" in result
        assert "joint_states" in result
        assert "done" in result
        assert "violation" in result
        assert result["violation"] is None, f"Unexpected violation: {result['violation']}"

    def test_place_action(self, scene):
        """place action should return a valid state dict without errors."""
        # Place somewhere on the desk
        desk_z = scene.DESK_HEIGHT + scene.DESK_THICKNESS / 2.0 + 0.05
        result = scene.step(
            action_type="place",
            object_id=None,
            arm="right",
            target=[0.05, 0.05, desk_z],
            direction=None,
        )
        assert isinstance(result, dict)
        assert result["violation"] is None

    def test_push_action(self, scene, reset_scene):
        """push action should return a valid state dict without errors."""
        obj = reset_scene["objects"][0]
        pos = obj["position"]
        result = scene.step(
            action_type="push",
            object_id=obj["id"],
            arm="left",
            target=pos,
            direction=[1.0, 0.0],
        )
        assert isinstance(result, dict)
        assert "objects" in result


class TestDualArmsJoints:
    def test_dual_arms_joints(self, scene, reset_scene):
        """Robot must report exactly 12 movable joint angles."""
        joints = reset_scene.get("joint_states", [])
        assert len(joints) == 12, f"Expected 12 joints, got {len(joints)}"

    def test_joint_values_are_floats(self, scene, reset_scene):
        """All joint state values must be finite floats."""
        joints = reset_scene.get("joint_states", [])
        for i, j in enumerate(joints):
            assert isinstance(j, float), f"Joint {i} is not a float: {type(j)}"
            assert np.isfinite(j), f"Joint {i} is not finite: {j}"

    def test_gripper_states_are_bool(self, scene, reset_scene):
        """Gripper states must be booleans."""
        grippers = reset_scene.get("gripper_states", [])
        for i, g in enumerate(grippers):
            assert isinstance(g, bool), f"Gripper {i} is not bool: {type(g)}"


class TestIKAccuracy:
    def test_ik_accuracy_left(self, scene, reset_scene):
        """_ik_solve must return a 6-element list for the left arm."""
        desk_z = scene.DESK_HEIGHT + scene.DESK_THICKNESS / 2.0 + 0.05
        angles = scene._robot._ik_solve([0.1, 0.05, desk_z], "left")
        assert isinstance(angles, list), "IK result must be a list"
        assert len(angles) == 6, f"Expected 6 angles for left arm, got {len(angles)}"

    def test_ik_accuracy_right(self, scene, reset_scene):
        """_ik_solve must return a 6-element list for the right arm."""
        desk_z = scene.DESK_HEIGHT + scene.DESK_THICKNESS / 2.0 + 0.05
        angles = scene._robot._ik_solve([0.1, -0.05, desk_z], "right")
        assert isinstance(angles, list), "IK result must be a list"
        assert len(angles) == 6, f"Expected 6 angles for right arm, got {len(angles)}"

    def test_ik_returns_finite_values(self, scene, reset_scene):
        """IK solution must contain only finite values."""
        desk_z = scene.DESK_HEIGHT + scene.DESK_THICKNESS / 2.0 + 0.05
        angles = scene._robot._ik_solve([0.05, 0.05, desk_z], "right")
        for i, a in enumerate(angles):
            assert np.isfinite(a), f"IK angle {i} is not finite: {a}"

    def test_ik_unreachable_graceful(self, scene, reset_scene):
        """IK for an unreachable target must return current joint angles (graceful degradation)."""
        # Far-away target: 10 metres away — definitely unreachable
        angles = scene._robot._ik_solve([10.0, 10.0, 10.0], "left")
        assert len(angles) == 6, "Should still return 6 angles for unreachable target"


class TestEnvironmentIntegration:
    def test_environment_reset(self):
        """BaseEnvironment.reset() must return a DeskObservation."""
        from deskbot.models import DeskObservation
        env = BaseEnvironment()
        obs = env.reset(task="easy", seed=42)
        assert isinstance(obs, DeskObservation)
        assert len(obs.objects) > 0
        assert len(obs.joint_states) == 12
        env._scene.close()

    def test_environment_step(self):
        """BaseEnvironment.step() must return a StepResult."""
        from deskbot.models import StepResult
        env = BaseEnvironment()
        env.reset(task="easy", seed=42)
        action = DeskAction(action_type="pick", arm="right", target=[0.05, 0.05, 0.77])
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        env._scene.close()
