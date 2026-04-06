"""
Tests for deskbot/server.py using FastAPI TestClient.

A MockEnvironment overrides the physics hooks so tests don't need
Session 1's physics to be implemented.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from deskbot.environment import BaseEnvironment
from deskbot.models import (
    DeskAction,
    DeskObservation,
    ObjectState,
    StepInfo,
    StepResult,
    TargetState,
)
from deskbot.server import app


# ------------------------------------------------------------------ #
# Mock environment — overrides NotImplementedError physics stubs      #
# ------------------------------------------------------------------ #

class MockEnvironment(BaseEnvironment):
    """Minimal physics stub for testing."""

    def _make_obs(self) -> DeskObservation:
        return DeskObservation(
            episode_id=self._episode_id or "test_ep",
            objects=[
                ObjectState(id="cube_plastic", position=[0.3, 0.2, 0.03]),
                ObjectState(id="mug_ceramic", position=[0.1, 0.1, 0.03]),
            ],
            joint_states=[0.0] * 12,
            gripper_states=[False, False],
            targets=[
                TargetState(object_id="cube_plastic", position=[0.45, 0.30, 0.03]),
                TargetState(object_id="mug_ceramic",  position=[0.10, 0.30, 0.03]),
            ],
            step_count=self._step_count,
        )

    def _physics_reset(self, task: str, seed: int) -> DeskObservation:
        return self._make_obs()

    def _physics_step(self, action: DeskAction) -> StepResult:
        obs = self._make_obs()
        obs.step_count = self._step_count + 1  # will be incremented by caller
        return StepResult(
            observation=obs,
            reward=0.1,
            done=False,
            info=StepInfo(cleanliness=0.0, order=1.0, safety=1.0),
        )


# ------------------------------------------------------------------ #
# Fixture — patch server.env with MockEnvironment                     #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Replace the global env in server.py with a MockEnvironment."""
    import deskbot.server as server_module
    mock = MockEnvironment()
    monkeypatch.setattr(server_module, "_http_env", mock)
    return mock


@pytest.fixture
def client():
    return TestClient(app)


# ------------------------------------------------------------------ #
# Tests                                                               #
# ------------------------------------------------------------------ #

def test_health(client):
    """GET / returns 200 with status ok."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_reset(client):
    """POST /reset returns a valid DeskObservation shape."""
    resp = client.post("/reset", json={"task": "easy", "seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert "episode_id" in data
    assert "objects" in data
    assert isinstance(data["objects"], list)
    assert len(data["objects"]) > 0
    assert "step_count" in data
    assert data["step_count"] == 0


def test_step(client):
    """POST /step returns StepResult with a reward float."""
    # Reset first so there is an active episode
    client.post("/reset", json={"task": "easy", "seed": 0})

    resp = client.post("/step", json={
        "action_type": "pick",
        "object_id": "cube_plastic",
        "arm": "right",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "reward" in data
    assert isinstance(data["reward"], float)
    assert "done" in data
    assert "observation" in data
    assert "info" in data


def test_state(client):
    """GET /state returns episode_id and step_count."""
    client.post("/reset", json={"task": "easy", "seed": 1})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "episode_id" in data
    assert "step_count" in data
    assert isinstance(data["step_count"], int)


def test_tasks(client):
    """GET /tasks returns exactly 3 tasks."""
    resp = client.get("/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert "tasks" in data
    tasks = data["tasks"]
    assert len(tasks) == 3
    names = {t["name"] for t in tasks}
    assert names == {"easy", "medium", "hard"}


def test_step_invalid(client):
    """POST /step with an invalid action_type returns 422."""
    resp = client.post("/step", json={"action_type": "fly"})
    assert resp.status_code == 422


def test_step_count_increments(client):
    """step_count goes up after each step."""
    client.post("/reset", json={"task": "easy", "seed": 7})

    state0 = client.get("/state").json()
    count_before = state0["step_count"]

    client.post("/step", json={"action_type": "pick", "object_id": "cube_plastic"})
    client.post("/step", json={"action_type": "push", "object_id": "cube_plastic", "direction": [0.01, 0.0]})

    state2 = client.get("/state").json()
    count_after = state2["step_count"]

    assert count_after == count_before + 2
