"""
DeskBot server entry point for multi-mode deployment.

Usage:
    uv run --project . server
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""
import uvicorn
from deskbot.server import app  # noqa: F401  — re-exported for uvicorn


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
