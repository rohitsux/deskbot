"""
DeskBotGymEnv — Gymnasium wrapper around DeskScene for RL training.

Observation space (flat Box):
  per object : [x, y, z, held, fragile]          → N_OBJ × 5
  per target : [x, y, z]                          → N_OBJ × 3
  joint states                                     → 12
  gripper states                                   → 2
  left EE pos                                      → 3
  right EE pos                                     → 3
  Total easy(5):  5*5 + 5*3 + 12 + 2 + 3 + 3 = 60

Action space (MultiDiscrete):
  [action_type, arm, object_idx]
  action_type : 0=pick  1=place_to_target  2=push_x+  3=push_x-  4=push_y+  5=push_y-
  arm         : 0=left  1=right
  object_idx  : 0..N_OBJ-1

Reward (scalar, range ~[0, 1]):
  0.4 * cleanliness + 0.4 * order + 0.2 * safety
  + 0.05 shaping per object newly placed at target (to reduce sparsity)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from deskbot.simulation.scene import DeskScene, DESK_HEIGHT

# Max objects any task can have
_TASK_N_OBJ = {"easy": 5, "medium": 8, "hard": 12}

# Object IDs per task (must match config/objects.json catalogue)
_TASK_OBJECTS: Dict[str, List[str]] = {
    "easy": ["mug_ceramic", "notebook", "pen_holder", "wooden_block", "scissors"],
    "medium": ["mug_ceramic", "notebook", "pen_holder", "wooden_block", "scissors",
               "apple", "banana", "bowl"],
    "hard": ["mug_ceramic", "notebook", "pen_holder", "wooden_block", "scissors",
             "apple", "banana", "bowl", "foam_brick", "rubiks_cube",
             "wood_block", "orange"],
}

# Target positions per task: where each object should end up
_TASK_TARGETS: Dict[str, Dict[str, List[float]]] = {
    "easy": {
        "mug_ceramic":  [ 0.10,  0.20, DESK_HEIGHT + 0.04],
        "notebook":     [-0.10,  0.20, DESK_HEIGHT + 0.04],
        "pen_holder":   [ 0.10, -0.10, DESK_HEIGHT + 0.04],
        "wooden_block": [-0.10, -0.10, DESK_HEIGHT + 0.04],
        "scissors":     [ 0.00,  0.00, DESK_HEIGHT + 0.04],
    },
    "medium": {
        "mug_ceramic":  [ 0.12,  0.22, DESK_HEIGHT + 0.04],
        "notebook":     [-0.12,  0.22, DESK_HEIGHT + 0.04],
        "pen_holder":   [ 0.12, -0.12, DESK_HEIGHT + 0.04],
        "wooden_block": [-0.12, -0.12, DESK_HEIGHT + 0.04],
        "scissors":     [ 0.00,  0.00, DESK_HEIGHT + 0.04],
        "apple":        [ 0.12,  0.00, DESK_HEIGHT + 0.04],
        "banana":       [-0.12,  0.00, DESK_HEIGHT + 0.04],
        "bowl":         [ 0.00,  0.22, DESK_HEIGHT + 0.04],
    },
    "hard": {
        "mug_ceramic":  [ 0.12,  0.22, DESK_HEIGHT + 0.04],
        "notebook":     [-0.12,  0.22, DESK_HEIGHT + 0.04],
        "pen_holder":   [ 0.12, -0.12, DESK_HEIGHT + 0.04],
        "wooden_block": [-0.12, -0.12, DESK_HEIGHT + 0.04],
        "scissors":     [ 0.00,  0.00, DESK_HEIGHT + 0.04],
        "apple":        [ 0.12,  0.00, DESK_HEIGHT + 0.04],
        "banana":       [-0.12,  0.00, DESK_HEIGHT + 0.04],
        "bowl":         [ 0.00,  0.22, DESK_HEIGHT + 0.04],
        "foam_brick":   [ 0.00, -0.20, DESK_HEIGHT + 0.04],
        "rubiks_cube":  [ 0.15,  0.10, DESK_HEIGHT + 0.04],
        "wood_block":   [-0.15,  0.10, DESK_HEIGHT + 0.04],
        "orange":       [ 0.15, -0.10, DESK_HEIGHT + 0.04],
    },
}
_MAX_OBJ    = 12   # pad observations to this length for uniform space

# Desk surface bounds (for obs normalisation)
_X_RANGE = 0.30   # ± half-depth
_Y_RANGE = 0.30   # ± half-width
_Z_MIN   = DESK_HEIGHT - 0.05
_Z_MAX   = DESK_HEIGHT + 0.50

# Push directions (dx, dy) indexed by action_type - 2
_PUSH_DIRS = [
    [1.0,  0.0],   # 2: push +x
    [-1.0, 0.0],   # 3: push -x
    [0.0,  1.0],   # 4: push +y
    [0.0, -1.0],   # 5: push -y
]

_OBS_DIM = _MAX_OBJ * 5 + _MAX_OBJ * 3 + 12 + 2 + 3 + 3   # = 107


class DeskBotGymEnv(gym.Env):
    """
    Single-task Gymnasium environment for DeskBot.

    Parameters
    ----------
    task : "easy" | "medium" | "hard"
    max_steps : episode step limit (overrides scene default)
    """

    metadata = {"render_modes": []}

    def __init__(self, task: str = "easy", max_steps: int = 50) -> None:
        super().__init__()
        self.task      = task
        self.max_steps = max_steps
        self.n_obj     = _TASK_N_OBJ[task]

        self._scene     = DeskScene(fast_mode=True)
        self._step_cnt  = 0
        self._prev_at_target: int = 0   # for shaping reward
        self._prev_dist_sum: float = 1e6  # for dense distance reward

        # Spaces
        self.observation_space = spaces.Box(
            low  = -1.0,
            high =  1.0,
            shape = (_OBS_DIM,),
            dtype = np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [6, 2, self.n_obj]   # [action_type, arm, object_idx]
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed

        raw = self._scene.reset(
            task       = self.task,
            object_ids = _TASK_OBJECTS[self.task],
            targets    = _TASK_TARGETS[self.task],
            seed     = rng_seed,
        )
        self._step_cnt       = 0
        self._raw_state      = raw
        self._prev_at_target = self._count_at_target(raw)
        self._prev_dist_sum  = self._total_dist(raw)
        return self._obs(raw), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        act_type, arm_idx, obj_idx = int(action[0]), int(action[1]), int(action[2])
        arm = "left" if arm_idx == 0 else "right"

        objects = self._raw_state.get("objects", [])
        obj_id  = objects[obj_idx]["id"] if obj_idx < len(objects) else None
        obj_pos = objects[obj_idx]["position"] if obj_idx < len(objects) else None

        target_pos: Optional[List[float]] = None
        if act_type == 1 and obj_id:
            target_pos = self._get_target_for(obj_id)

        direction: Optional[List[float]] = None
        if act_type >= 2:
            direction = _PUSH_DIRS[act_type - 2]

        raw = self._scene.step(
            action_type = ["pick", "place", "push", "push", "push", "push"][act_type],
            object_id   = obj_id if act_type != 1 else None,
            arm         = arm,
            target      = target_pos,
            direction   = direction,
        )
        self._raw_state = raw
        self._step_cnt += 1

        reward = self._reward(raw)
        done   = bool(raw.get("done", False)) or self._step_cnt >= self.max_steps
        obs    = self._obs(raw)
        info   = {
            "violation": raw.get("violation"),
            **raw.get("reward_components", {}),
        }
        return obs, reward, done, False, info

    def render(self) -> None:
        pass  # headless — use DeskScene viewer separately

    def close(self) -> None:
        self._scene.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _obs(self, raw: dict) -> np.ndarray:
        objects = raw.get("objects", [])
        targets_map = {t["object_id"]: t["position"]
                       for t in raw.get("targets", [])}

        obj_feats = np.zeros((_MAX_OBJ, 5), dtype=np.float32)
        tgt_feats = np.zeros((_MAX_OBJ, 3), dtype=np.float32)

        for i, obj in enumerate(objects[:_MAX_OBJ]):
            x, y, z = obj["position"]
            obj_feats[i] = [
                x / _X_RANGE,
                y / _Y_RANGE,
                (z - DESK_HEIGHT) / (_Z_MAX - DESK_HEIGHT),
                float(obj.get("held", False)),
                float(obj.get("fragile", False)),
            ]
            if obj["id"] in targets_map:
                tx, ty, tz = targets_map[obj["id"]]
                tgt_feats[i] = [
                    tx / _X_RANGE,
                    ty / _Y_RANGE,
                    (tz - DESK_HEIGHT) / (_Z_MAX - DESK_HEIGHT),
                ]

        joints   = np.array(raw.get("joint_states", [0.0]*12), dtype=np.float32) / np.pi
        grippers = np.array(raw.get("gripper_states", [False, False]), dtype=np.float32)

        # EE positions — from scene robot if available, else zeros
        ee_l = np.zeros(3, dtype=np.float32)
        ee_r = np.zeros(3, dtype=np.float32)
        if hasattr(self._scene, "_robot") and self._scene._robot is not None:
            ee_l = (self._scene._robot.get_ee_position("left")  / [_X_RANGE, _Y_RANGE, _Z_MAX]).astype(np.float32)
            ee_r = (self._scene._robot.get_ee_position("right") / [_X_RANGE, _Y_RANGE, _Z_MAX]).astype(np.float32)

        obs = np.concatenate([
            obj_feats.ravel(),   # _MAX_OBJ * 5
            tgt_feats.ravel(),   # _MAX_OBJ * 3
            joints,              # 12
            grippers,            # 2
            ee_l,                # 3
            ee_r,                # 3
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _reward(self, raw: dict) -> float:
        comps = raw.get("reward_components", {})
        safety = float(comps.get("safety", 1.0))

        # Dense: reward improvement in total distance to targets
        dist_sum = self._total_dist(raw)
        dist_improvement = (self._prev_dist_sum - dist_sum) / max(self.n_obj, 1)
        self._prev_dist_sum = dist_sum

        # Sparse: +0.2 per newly placed object
        at_target = self._count_at_target(raw)
        placed_bonus = 0.2 * max(0, at_target - self._prev_at_target)
        self._prev_at_target = at_target

        # Safety penalty
        safety_pen = -0.5 * (1.0 - safety)

        return float(dist_improvement + placed_bonus + safety_pen)

    def _total_dist(self, raw: dict) -> float:
        """Sum of distances from each object to its target."""
        targets_map = {t["object_id"]: np.array(t["position"])
                       for t in raw.get("targets", [])}
        total = 0.0
        for obj in raw.get("objects", []):
            tgt = targets_map.get(obj["id"])
            if tgt is not None:
                total += float(np.linalg.norm(np.array(obj["position"]) - tgt))
        return total

    def _count_at_target(self, raw: dict) -> int:
        targets_map = {t["object_id"]: np.array(t["position"])
                       for t in raw.get("targets", [])}
        count = 0
        for obj in raw.get("objects", []):
            tgt = targets_map.get(obj["id"])
            if tgt is not None:
                dist = np.linalg.norm(np.array(obj["position"]) - tgt)
                if dist < 0.05:
                    count += 1
        return count

    def _get_target_for(self, object_id: str) -> Optional[List[float]]:
        targets = self._raw_state.get("targets", [])
        for t in targets:
            if t["object_id"] == object_id:
                return t["position"]
        return None
