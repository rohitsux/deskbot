"""
DeskScene — MuJoCo desk scene: plane + desk + dual-arm robot + objects.

Physics engine: MuJoCo 3.x (convex contact optimisation, CPU-only).
Rendering:      OSMesa headless (no X11 / GPU required).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import mujoco
import numpy as np
import yaml

from deskbot.reward.cleanliness import compute_dense_cleanliness
from deskbot.reward.safety import compute_safety, check_fragile_destroyed
from deskbot.simulation.objects import ObjectSpawner
from deskbot.simulation.robot import DualArmRobot


_TASK_OBJECT_COUNTS: Dict[str, int] = {"easy": 5, "medium": 10, "hard": 15}

_DEFAULT_OBJECT_POOL = [
    "mug_ceramic", "notebook_paper", "pen_holder_plastic", "wooden_block", "scissors_metal",
    "apple", "banana", "bowl", "foam_brick", "rubiks_cube",
    "wood_block", "orange", "lemon", "fork", "spoon",
    "large_marker", "cracker_box", "sugar_box", "mustard_bottle", "soup_can",
]

DESK_WIDTH: float = 1.00   # y-axis: covers -0.50 to 0.50 (targets go to y=0.42)
DESK_DEPTH: float = 1.40   # x-axis: covers -0.70 to 0.70 (targets go to x=0.55)
DESK_HEIGHT: float = 0.74
DESK_THICKNESS: float = 0.02
_MAX_STEPS: Dict[str, int] = {"easy": 50, "medium": 100, "hard": 200}


_CYLINDER_GEOM_IDS = {"pen_holder_plastic", "bottle_plastic", "vase_fragile_ceramic", "glass_cup_glass"}


def _build_xml(object_specs: List[dict]) -> str:
    obj_bodies = ""
    for spec in object_specs:
        if spec['id'] in _CYLINDER_GEOM_IDS:
            geom_str = (
                f'type="cylinder" '
                f'size="{spec["sx"]/2:.4f} {spec["sz"]/2:.4f}"'
            )
        else:
            geom_str = (
                f'type="box" '
                f'size="{spec["sx"]/2:.4f} {spec["sy"]/2:.4f} {spec["sz"]/2:.4f}"'
            )
        obj_bodies += f"""
        <body name="obj_{spec['id']}" pos="{spec['x']} {spec['y']} {spec['z']}">
          <freejoint name="fj_{spec['id']}"/>
          <geom {geom_str}
                mass="{spec['mass']}"
                friction="{spec['friction']} 0.005 0.0001"
                rgba="{spec['r']} {spec['g']} {spec['b']} 1"
                euler="0 0 {spec['yaw']}"/>
        </body>"""

    desk_z = DESK_HEIGHT - DESK_THICKNESS / 2
    arm_z  = DESK_HEIGHT + 0.01

    return f"""<mujoco model="deskbot">
  <option gravity="0 0 -9.81" timestep="0.004" integrator="implicitfast"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom condim="4" contype="1" conaffinity="1" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.7 0.65 0.55 1" friction="0.8 0.005 0.0001"/>
    <body name="desk" pos="0 0 {desk_z:.4f}">
      <geom type="box" size="{DESK_DEPTH/2:.4f} {DESK_WIDTH/2:.4f} {DESK_THICKNESS/2:.4f}"
            rgba="0.25 0.22 0.20 1" friction="0.7 0.005 0.0001"/>
    </body>
    <body name="left_base" pos="-0.15 0 {arm_z:.4f}">
      <geom type="cylinder" size="0.025 0.01" rgba="0.2 0.2 0.2 1" mass="0.5" contype="0" conaffinity="0"/>
      <body name="left_link0" pos="0 0 0.02">
        <joint name="left_j0" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom type="capsule" size="0.018 0.04" rgba="0.85 0.85 0.85 1" mass="0.3" contype="0" conaffinity="0"/>
        <body name="left_link1" pos="0 0 0.08">
          <joint name="left_j1" type="hinge" axis="0 1 0" range="-1.5 0.5"/>
          <geom type="capsule" size="0.015 0.055" rgba="0.85 0.85 0.85 1" mass="0.25" contype="0" conaffinity="0"/>
          <body name="left_link2" pos="0 0 0.11">
            <joint name="left_j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" size="0.013 0.045" rgba="0.85 0.85 0.85 1" mass="0.2" contype="0" conaffinity="0"/>
            <body name="left_link3" pos="0 0 0.09">
              <joint name="left_j3" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
              <geom type="capsule" size="0.012 0.04" rgba="0.7 0.7 0.7 1" mass="0.15" contype="0" conaffinity="0"/>
              <body name="left_link4" pos="0 0 0.08">
                <joint name="left_j4" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                <geom type="capsule" size="0.011 0.03" rgba="0.7 0.7 0.7 1" mass="0.1" contype="0" conaffinity="0"/>
                <site name="left_ee" pos="0 0 0.06" size="0.01"/>
                <body name="left_gripper" pos="0 0 0.06">
                  <joint name="left_j5" type="slide" axis="0 1 0" range="0 0.04"/>
                  <geom type="box" size="0.01 0.015 0.02" rgba="0.4 0.4 0.4 1" mass="0.05" contype="1" conaffinity="1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="right_base" pos="0.15 0 {arm_z:.4f}">
      <geom type="cylinder" size="0.025 0.01" rgba="0.2 0.2 0.2 1" mass="0.5" contype="0" conaffinity="0"/>
      <body name="right_link0" pos="0 0 0.02">
        <joint name="right_j0" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom type="capsule" size="0.018 0.04" rgba="0.85 0.85 0.85 1" mass="0.3" contype="0" conaffinity="0"/>
        <body name="right_link1" pos="0 0 0.08">
          <joint name="right_j1" type="hinge" axis="0 1 0" range="-1.5 0.5"/>
          <geom type="capsule" size="0.015 0.055" rgba="0.85 0.85 0.85 1" mass="0.25" contype="0" conaffinity="0"/>
          <body name="right_link2" pos="0 0 0.11">
            <joint name="right_j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" size="0.013 0.045" rgba="0.85 0.85 0.85 1" mass="0.2" contype="0" conaffinity="0"/>
            <body name="right_link3" pos="0 0 0.09">
              <joint name="right_j3" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
              <geom type="capsule" size="0.012 0.04" rgba="0.7 0.7 0.7 1" mass="0.15" contype="0" conaffinity="0"/>
              <body name="right_link4" pos="0 0 0.08">
                <joint name="right_j4" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
                <geom type="capsule" size="0.011 0.03" rgba="0.7 0.7 0.7 1" mass="0.1" contype="0" conaffinity="0"/>
                <site name="right_ee" pos="0 0 0.06" size="0.01"/>
                <body name="right_gripper" pos="0 0 0.06">
                  <joint name="right_j5" type="slide" axis="0 1 0" range="0 0.04"/>
                  <geom type="box" size="0.01 0.015 0.02" rgba="0.4 0.4 0.4 1" mass="0.05" contype="1" conaffinity="1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    {obj_bodies}
  </worldbody>
  <actuator>
    <position name="left_a0"  joint="left_j0"  kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.57 1.57"/>
    <position name="left_a1"  joint="left_j1"  kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.5 0.5"/>
    <position name="left_a2"  joint="left_j2"  kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.5 1.5"/>
    <position name="left_a3"  joint="left_j3"  kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.5 1.5"/>
    <position name="left_a4"  joint="left_j4"  kp="20" kv="6"  ctrllimited="true" ctrlrange="-1.57 1.57"/>
    <position name="left_a5"  joint="left_j5"  kp="15" kv="4"  ctrllimited="true" ctrlrange="0 0.04"/>
    <position name="right_a0" joint="right_j0" kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.57 1.57"/>
    <position name="right_a1" joint="right_j1" kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.5 0.5"/>
    <position name="right_a2" joint="right_j2" kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.5 1.5"/>
    <position name="right_a3" joint="right_j3" kp="30" kv="8"  ctrllimited="true" ctrlrange="-1.5 1.5"/>
    <position name="right_a4" joint="right_j4" kp="20" kv="6"  ctrllimited="true" ctrlrange="-1.57 1.57"/>
    <position name="right_a5" joint="right_j5" kp="15" kv="4"  ctrllimited="true" ctrlrange="0 0.04"/>
  </actuator>
</mujoco>"""


class DeskScene:
    """MuJoCo desk simulation. Rebuilds MJCF XML each episode reset.

    Parameters
    ----------
    fast_mode : bool
        If True, actions are applied instantly via teleport (no IK physics).
        Use for RL training where physics realism during action execution
        is not needed. The physics still runs for settling / contact detection.
    """

    # Class-level aliases so tests can access scene.DESK_HEIGHT etc.
    DESK_WIDTH     = DESK_WIDTH
    DESK_DEPTH     = DESK_DEPTH
    DESK_HEIGHT    = DESK_HEIGHT
    DESK_THICKNESS = DESK_THICKNESS

    def __init__(self, fast_mode: bool = False) -> None:
        self._fast_mode = fast_mode
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, "..", ".."))
        self._catalogue_path = os.path.join(_root, "config", "objects.json")
        with open(self._catalogue_path) as f:
            raw = json.load(f)
        self._catalogue: Dict[str, dict] = {item["id"]: item for item in raw}

        self._model: Optional[mujoco.MjModel] = None
        self._data: Optional[mujoco.MjData] = None
        self._robot: Optional[DualArmRobot] = None
        self._spawner: Optional[ObjectSpawner] = None

        self._current_object_ids: List[str] = []
        self._targets: Dict[str, List[float]] = {}
        self._step_count: int = 0
        self._task: str = "easy"
        self._blocking_pairs: List[dict] = []
        self._fragile_objects: List[str] = []
        self._violations: List[dict] = []
        self._destroyed: bool = False

        # Load constraint rules from config
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, "..", ".."))
        _constraints_path = os.path.join(_root, "config", "constraints.json")
        try:
            with open(_constraints_path) as f:
                self._constraints_cfg: dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._constraints_cfg = {}

        self._reload_model([])

    def reset(self, task: str, object_ids: List[str],
              targets: Dict[str, List[float]], seed: int) -> dict:
        self._task = task
        self._step_count = 0
        self._targets = dict(targets)
        self._blocking_pairs = []
        self._fragile_objects = []
        self._violations = []
        self._destroyed = False
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, "..", ".."))
        _task_files = {"easy": "task1_easy.yaml", "medium": "task2_medium.yaml", "hard": "task3_hard.yaml"}
        try:
            with open(os.path.join(_root, "config", _task_files[task])) as f:
                _cfg = yaml.safe_load(f) or {}
            self._blocking_pairs = _cfg.get("blocking_pairs", [])
            self._fragile_objects = _cfg.get("fragile_objects", [])
        except (KeyError, FileNotFoundError):
            pass
        rng = np.random.default_rng(seed)

        if not object_ids:
            n = _TASK_OBJECT_COUNTS.get(task, 5)
            object_ids = _DEFAULT_OBJECT_POOL[:n]
        self._current_object_ids = list(object_ids)

        specs = self._make_object_specs(object_ids, rng)
        self._reload_model(specs)
        for _ in range(200):
            mujoco.mj_step(self._model, self._data)
        return self.get_state()

    def step(self, action_type: str, object_id: Optional[str], arm: str,
             target: Optional[List[float]], direction: Optional[List[float]]) -> dict:
        self._step_count += 1
        arm = (arm or "right").lower()
        violation: Optional[str] = None

        if self._fast_mode:
            violation = self._step_fast(action_type, object_id, arm, target, direction)
        else:
            violation = self._step_physics(action_type, object_id, arm, target, direction)
            # Settle with carry
            for _ in range(60):
                mujoco.mj_step(self._model, self._data)
                self._robot.update_carry(self._spawner)

        state = self.get_state()
        state["reward_components"] = self._compute_reward_components()
        state["done"] = self._is_done()
        state["violation"] = violation
        return state

    def _step_physics(self, action_type: str, object_id: Optional[str], arm: str,
                      target: Optional[List[float]],
                      direction: Optional[List[float]]) -> Optional[str]:
        """Full IK + physics execution (server / demo mode)."""
        if action_type == "pick":
            if object_id and object_id not in self._current_object_ids:
                return f"Unknown object '{object_id}'"
            # Blocking check: cannot pick object_id if a blocker is not yet at its target
            if object_id and self._blocking_pairs:
                for pair in self._blocking_pairs:
                    if pair.get("blocks") == object_id:
                        blocker = pair.get("blocked_by")
                        if blocker and self._targets.get(blocker):
                            bp = self._get_object_position(blocker)
                            tgt = self._targets[blocker]
                            if bp is not None and float(np.linalg.norm(np.array(bp) - np.array(tgt))) > 0.05:
                                return f"Object '{object_id}' is blocked by '{blocker}'"
            pos = self._get_object_position(object_id) if object_id else target
            if pos is None:
                return "No target position for pick"
            prev = self._robot.get_held(arm)
            if prev:
                self._robot.release(arm)
            self._robot.pick(pos, arm)
            if object_id:
                self._robot.set_held(arm, object_id)

        elif action_type == "place":
            if target is None:
                return "No target position for place"
            held_oid = self._robot.get_held(arm)
            self._robot.place(target, arm, spawner=self._spawner)
            if held_oid:
                self._check_constraint_violation(held_oid, target)

        elif action_type == "push":
            pos = self._get_object_position(object_id) if object_id else target
            if pos is None:
                return "No target position for push"
            dx = float(direction[0]) if direction and len(direction) > 0 else 1.0
            dy = float(direction[1]) if direction and len(direction) > 1 else 0.0
            self._robot.push(pos, [pos[0] + dx * 0.10, pos[1] + dy * 0.10, pos[2]], arm)
            if object_id:
                self._nudge_object(object_id, [dx * 0.35, dy * 0.35, 0.0])

        elif action_type == "home":
            self._robot.home(arm)

        else:
            return f"Unknown action_type '{action_type}'"

        return None

    def _step_fast(self, action_type: str, object_id: Optional[str], arm: str,
                   target: Optional[List[float]],
                   direction: Optional[List[float]]) -> Optional[str]:
        """
        Instant teleport-based actions for RL training (no IK, no physics settling).
        Objects move immediately; a single mj_forward updates contacts + positions.
        """
        if action_type == "pick":
            if object_id and object_id not in self._current_object_ids:
                return f"Unknown object '{object_id}'"
            if object_id:
                prev = self._robot.get_held(arm)
                if prev:
                    self._robot.release(arm)
                self._robot.set_held(arm, object_id)

        elif action_type == "place":
            if target is None:
                return "No target position for place"
            held = self._robot.get_held(arm)
            if held and self._spawner:
                self._spawner.teleport(held, target)
                self._robot.release(arm)

        elif action_type == "push":
            pos = self._get_object_position(object_id) if object_id else target
            if pos is None:
                return "No target position for push"
            dx = float(direction[0]) if direction and len(direction) > 0 else 1.0
            dy = float(direction[1]) if direction and len(direction) > 1 else 0.0
            if object_id:
                new_pos = [pos[0] + dx * 0.10, pos[1] + dy * 0.10, pos[2]]
                self._teleport_object(object_id, new_pos)

        elif action_type == "home":
            pass  # no-op in fast mode

        else:
            return f"Unknown action_type '{action_type}'"

        # One forward pass — updates xpos/xmat, enough for reward computation
        mujoco.mj_forward(self._model, self._data)
        return None

    def get_state(self) -> dict:
        objects_list = [
            {
                "id": oid,
                "position": self._get_object_position(oid) or [0.0, 0.0, 0.0],
                "held": (self._robot.get_held("left") == oid or
                         self._robot.get_held("right") == oid),
                "fragile": self._catalogue.get(oid, {}).get("fragility", False),
                "material": self._catalogue.get(oid, {}).get("material", "plastic"),
            }
            for oid in self._current_object_ids
        ]
        return {
            "objects": objects_list,
            "joint_states": self._robot.get_joint_states(),
            "gripper_states": self._robot.get_gripper_states(),
            "targets": [{"object_id": oid, "position": pos}
                        for oid, pos in self._targets.items()],
            "violations": list(self._violations),
        }

    def close(self) -> None:
        self._model = None
        self._data = None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _reload_model(self, specs: List[dict]) -> None:
        xml = _build_xml(specs)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)
        self._robot = DualArmRobot(self._model, self._data)
        self._spawner = ObjectSpawner(self._model, self._data, self._catalogue, specs)

    def _make_object_specs(self, object_ids: List[str],
                           rng: np.random.Generator) -> List[dict]:
        placed: List[tuple] = []
        specs = []
        for oid in object_ids:
            cat = self._catalogue.get(oid, {})
            sx, sy, sz = cat.get("size", [0.05, 0.05, 0.07])
            color = cat.get("color", [0.6, 0.4, 0.2, 1.0])
            for _ in range(50):
                x = rng.uniform(0.05, 0.50)
                y = rng.uniform(0.05, 0.40)
                if all(abs(x - px) > 0.08 or abs(y - py) > 0.08 for px, py in placed):
                    break
            placed.append((x, y))
            specs.append({
                "id": oid,
                "x": round(x, 4), "y": round(y, 4),
                "z": round(DESK_HEIGHT + float(sz) / 2.0 + 0.005, 4),
                "sx": float(sx), "sy": float(sy), "sz": float(sz),
                "mass": float(cat.get("mass", 0.2)),
                "friction": float(cat.get("friction", 0.6)),
                "r": float(color[0]), "g": float(color[1]), "b": float(color[2]),
                "yaw": round(rng.uniform(0, 6.28), 4),
            })
        return specs

    def _get_object_position(self, oid: str) -> Optional[List[float]]:
        return self._spawner.get_position(oid) if self._spawner else None

    def _teleport_object(self, oid: str, pos: List[float]) -> None:
        if self._spawner:
            self._spawner.teleport(oid, pos)

    def _nudge_object(self, oid: str, vel: List[float]) -> None:
        if self._spawner:
            self._spawner.nudge(oid, vel)

    def _check_constraint_violation(self, oid: str, placed_pos: List[float]) -> None:
        """Check and record material constraint violations for a placed object."""
        obj_material = self._catalogue.get(oid, {}).get("material", "plastic")
        obj_mass = float(self._catalogue.get(oid, {}).get("mass", 0.2))
        heavy_threshold = float(self._constraints_cfg.get("heavy_mass_threshold", 0.4))
        forbidden_pairs = self._constraints_cfg.get("forbidden_contact_pairs", [])

        for other_oid in self._current_object_ids:
            if other_oid == oid:
                continue
            other_material = self._catalogue.get(other_oid, {}).get("material", "plastic")
            other_pos = self._get_object_position(other_oid)
            if other_pos is None:
                continue
            dist = float(np.linalg.norm(np.array(placed_pos) - np.array(other_pos)))
            if dist > 0.12:
                continue
            for pair in forbidden_pairs:
                mats = {pair.get("a"), pair.get("b")}
                if obj_material in mats and other_material in mats and obj_material != other_material:
                    penalty = float(pair.get("penalty", 0.2))
                    self._violations.append({"type": f"{obj_material}_near_{other_material}",
                                             "penalty": penalty, "object_id": oid, "other_id": other_oid})
                    if penalty >= 1.0 and oid in self._fragile_objects:
                        self._destroyed = True
                    break
            # heavy on fragile
            if obj_mass >= heavy_threshold and other_oid in self._fragile_objects:
                self._violations.append({"type": "heavy_on_fragile", "penalty": 1.0,
                                         "object_id": oid, "other_id": other_oid})
                self._destroyed = True

    def _compute_reward_components(self) -> dict:
        total = len(self._current_object_ids)
        if total == 0:
            return {"cleanliness": 0.0, "order": 1.0, "safety": 1.0, "violations": []}

        if self._targets:
            current_positions = {
                oid: self._get_object_position(oid) or [0.0, 0.0, 0.0]
                for oid in self._current_object_ids
            }
            cleanliness = compute_dense_cleanliness(current_positions, self._targets)
        else:
            in_bounds = sum(
                1 for oid in self._current_object_ids
                if (p := self._get_object_position(oid)) is not None
                and abs(p[0]) <= DESK_DEPTH / 2
                and abs(p[1]) <= DESK_WIDTH / 2
                and p[2] >= DESK_HEIGHT - 0.05
            )
            cleanliness = in_bounds / total

        order = 1.0
        if self._targets:
            at_target = sum(
                1 for oid, tgt in self._targets.items()
                if (p := self._get_object_position(oid)) is not None
                and float(np.linalg.norm(np.array(p) - np.array(tgt))) < 0.05
            )
            order = at_target / len(self._targets)

        safety = compute_safety(self._violations)

        return {"cleanliness": cleanliness, "order": order, "safety": safety,
                "violations": list(self._violations)}

    def _is_done(self) -> bool:
        if self._destroyed:
            return True
        if self._step_count >= _MAX_STEPS.get(self._task, 100):
            return True
        if not self._targets:
            return False
        return all(
            (p := self._get_object_position(oid)) is not None
            and float(np.linalg.norm(np.array(p) - np.array(tgt))) < 0.05
            for oid, tgt in self._targets.items()
        )
