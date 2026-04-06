"""
BaseEnvironment — glue between FastAPI server and PyBullet simulation.

Session 1 (simulation) implements: _physics_reset, _physics_step
Session 2 (server)     calls:      reset(), step(), state()
"""

from __future__ import annotations

import uuid
from typing import Optional

from deskbot.models import (
    DeskAction,
    DeskObservation,
    DeskState,
    ObjectState,
    StepInfo,
    StepResult,
    TargetState,
)
from deskbot.simulation.scene import DeskScene


class BaseEnvironment:
    """
    Ties the HTTP layer (server.py) to the physics layer (simulation/).

    Subclass or instantiate directly — Session 1 fills in
    _physics_reset() and _physics_step() via DeskScene.
    """

    def __init__(self) -> None:
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._task: Optional[str] = None
        self._scene: DeskScene = DeskScene()

    # ------------------------------------------------------------------ #
    # Public OpenEnv interface (called by server.py)                       #
    # ------------------------------------------------------------------ #

    def reset(self, task: str = "easy", seed: int = 42) -> DeskObservation:
        self._episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        self._step_count = 0
        self._task = task
        return self._physics_reset(task=task, seed=seed)

    def step(self, action: DeskAction) -> StepResult:
        result = self._physics_step(action)
        self._step_count += 1
        return result

    def state(self) -> DeskState:
        return DeskState(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
        )

    # ------------------------------------------------------------------ #
    # Physics hooks — implemented via DeskScene                            #
    # ------------------------------------------------------------------ #

    def _physics_reset(self, task: str, seed: int) -> DeskObservation:
        import os, yaml
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, ".."))
        _task_files = {
            "easy":   "task1_easy.yaml",
            "medium": "task2_medium.yaml",
            "hard":   "task3_hard.yaml",
        }
        cfg: dict = {}
        try:
            path = os.path.join(_root, "config", _task_files[task])
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except (KeyError, FileNotFoundError):
            pass

        object_ids = cfg.get("object_ids", [])
        targets_raw = cfg.get("targets", {})
        targets = {k: list(v) for k, v in targets_raw.items()} if isinstance(targets_raw, dict) else {}

        raw = self._scene.reset(
            task       = task,
            object_ids = object_ids,
            targets    = targets,
            seed       = seed,
        )
        return self._raw_to_observation(raw)

    def _physics_step(self, action: DeskAction) -> StepResult:
        """
        Delegate to DeskScene.step() and convert the result into a StepResult.
        """
        raw = self._scene.step(
            action_type=action.action_type,
            object_id=action.object_id,
            arm=action.arm or "right",
            target=action.target,
            direction=action.direction,
        )
        obs = self._raw_to_observation(raw)
        obs.step_count = self._step_count + 1  # pre-increment for this result

        reward_comps = raw.get("reward_components", {})
        cleanliness = float(reward_comps.get("cleanliness", 0.0))
        order = float(reward_comps.get("order", 1.0))
        safety = float(reward_comps.get("safety", 1.0))
        violations = reward_comps.get("violations", [])
        action_error = raw.get("violation")

        # task-aware weights
        _REWARD_WEIGHTS = {
            "easy":   {"cleanliness": 1.0, "order": 0.0, "safety": 0.0},
            "medium": {"cleanliness": 0.6, "order": 0.4, "safety": 0.0},
            "hard":   {"cleanliness": 0.5, "order": 0.3, "safety": 0.2},
        }
        weights = _REWARD_WEIGHTS.get(self._task or "easy", {"cleanliness": 1.0, "order": 0.0, "safety": 0.0})
        reward = (weights["cleanliness"] * cleanliness
                  + weights["order"] * order
                  + weights["safety"] * safety)

        info = StepInfo(
            cleanliness=cleanliness,
            order=order,
            safety=safety,
            violations=violations,
            action_error=action_error,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=bool(raw.get("done", False)),
            info=info,
        )

    # ------------------------------------------------------------------ #
    # Conversion helpers                                                   #
    # ------------------------------------------------------------------ #

    def _raw_to_observation(self, raw: dict) -> DeskObservation:
        """Convert a raw scene state dict into a DeskObservation."""
        objects = [
            ObjectState(
                id=obj["id"],
                position=obj["position"],
                held=obj.get("held", False),
                fragile=obj.get("fragile", False),
                material=obj.get("material", "plastic"),
            )
            for obj in raw.get("objects", [])
        ]
        targets = [
            TargetState(
                object_id=t["object_id"],
                position=t["position"],
            )
            for t in raw.get("targets", [])
        ]
        return DeskObservation(
            episode_id=self._episode_id or "",
            objects=objects,
            joint_states=raw.get("joint_states", [0.0] * 12),
            gripper_states=raw.get("gripper_states", [False, False]),
            targets=targets,
            step_count=self._step_count,
        )
