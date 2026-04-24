"""
FastAPI server for OnCallEnv — OpenEnv-compliant REST API.

Endpoints:
  POST /reset         Reset environment (optionally with task_id)
  POST /step          Execute an action
  GET  /state         Get current internal state
  GET  /tasks         List available tasks
  GET  /              Health check
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from environment import OnCallEnvironment
from models import Action, Observation, StepResponse, EnvState

app = FastAPI(
    title="OnCallEnv",
    description=(
        "Production Incident Response environment for AI agent evaluation. "
        "Agents diagnose and remediate simulated infrastructure incidents."
    ),
    version="1.0.0",
)

# Single environment instance (stateful per session)
env = OnCallEnvironment()


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    command: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "environment": "OnCallEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> dict:
    """Reset the environment for a given task."""
    try:
        obs = env.reset(task_id=req.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> dict:
    """Execute one action in the environment."""
    try:
        action = Action(command=req.command)
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> dict:
    """Return full internal state for debugging."""
    try:
        return env.state().model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks() -> list[dict]:
    """List all available tasks."""
    return env.list_tasks()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
