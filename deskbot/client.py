"""
DeskBotEnv — typed Python client for the DeskBot OpenEnv environment.

Install from HF Space:
    pip install git+https://huggingface.co/spaces/rohitsuthar/deskbot

Usage (sync):
    from deskbot.client import DeskBotEnv, DeskAction

    with DeskBotEnv(base_url="http://localhost:8000") as env:
        obs  = env.reset(task="easy", seed=42)
        result = env.step(DeskAction(action_type="pick",
                                     object_id="mug_ceramic", arm="right"))
        print(result.reward)

Usage (async):
    import asyncio
    from deskbot.client import DeskBotEnv, DeskAction

    async def main():
        async with DeskBotEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset_async(task="easy", seed=42)
            result = await env.step_async(DeskAction(action_type="pick",
                                                     object_id="mug_ceramic"))
    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import websockets

from deskbot.models import DeskAction, DeskObservation, DeskState, StepResult


class DeskBotEnv:
    """
    WebSocket client for DeskBot OpenEnv.

    Each instance manages one persistent WebSocket connection → one isolated
    server-side environment session.  Multiple instances = multiple sessions.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # ── async interface ───────────────────────────────────────────────────────

    async def connect_async(self) -> "DeskBotEnv":
        self._ws = await websockets.connect(self._ws_url)
        return self

    async def close_async(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def reset_async(self, task: str = "easy", seed: int = 42) -> DeskObservation:
        await self._send({"type": "reset", "task": task, "seed": seed})
        data = await self._recv()
        return DeskObservation(**data)

    async def step_async(self, action: DeskAction) -> StepResult:
        payload = {"type": "step", **action.model_dump(exclude_none=True)}
        await self._send(payload)
        data = await self._recv()
        return StepResult(**data)

    async def state_async(self) -> DeskState:
        await self._send({"type": "state"})
        data = await self._recv()
        return DeskState(**data)

    async def _send(self, msg: dict) -> None:
        if not self._ws:
            raise RuntimeError("Not connected. Use 'async with DeskBotEnv(...) as env'")
        await self._ws.send(json.dumps(msg))

    async def _recv(self) -> dict:
        raw = await self._ws.recv()
        return json.loads(raw)

    # ── async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "DeskBotEnv":
        return await self.connect_async()

    async def __aexit__(self, *_) -> None:
        await self.close_async()

    # ── sync interface (runs its own event loop in a background thread) ───────

    def connect(self) -> "DeskBotEnv":
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self.connect_async(), self._loop)
        future.result(timeout=10)
        return self

    def close(self) -> None:
        if self._loop:
            future = asyncio.run_coroutine_threadsafe(self.close_async(), self._loop)
            future.result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop = None

    def reset(self, task: str = "easy", seed: int = 42) -> DeskObservation:
        return self._run(self.reset_async(task=task, seed=seed))

    def step(self, action: DeskAction) -> StepResult:
        return self._run(self.step_async(action))

    def state(self) -> DeskState:
        return self._run(self.state_async())

    def _run(self, coro):
        if not self._loop:
            raise RuntimeError("Not connected. Use 'with DeskBotEnv(...) as env'")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    # ── sync context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "DeskBotEnv":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    # ── factory: connect from HF Space repo id ────────────────────────────────

    @classmethod
    def from_hub(cls, repo_id: str) -> "DeskBotEnv":
        """
        Connect to a running HF Space.

        Example:
            env = DeskBotEnv.from_hub("rohitsuthar/deskbot")
        """
        space_name = repo_id.replace("/", "-")
        url = f"https://{space_name}.hf.space"
        return cls(base_url=url)
