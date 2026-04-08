"""Typed Pydantic models for OnCallEnv — OpenEnv spec compliant."""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Observation ──────────────────────────────────────────────────────────────

class Alert(BaseModel):
    """A single monitoring alert."""
    alert_id: str
    severity: str = Field(description="critical | warning | info")
    service: str
    message: str
    timestamp: str


class ServiceMetrics(BaseModel):
    """Metrics snapshot for a service."""
    service: str
    cpu_percent: float
    memory_percent: float
    request_latency_p50_ms: float
    request_latency_p99_ms: float
    error_rate_percent: float
    requests_per_second: float
    extra: dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """A single log line."""
    timestamp: str
    level: str
    service: str
    message: str


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    goal: str
    step: int
    max_steps: int
    current_time: str
    alerts: list[Alert] = Field(default_factory=list)
    last_action: Optional[str] = None
    last_action_result: Optional[str] = None
    last_action_error: bool = False
    available_commands: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


# ── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """An action the agent takes. Free-form command string parsed by env."""
    command: str = Field(
        description=(
            "One of: check_metrics <service>, check_logs <service>, "
            "check_config <service>, check_dependencies <service>, "
            "check_deploy_history <service>, restart_service <service>, "
            "rollback_deploy <service>, scale_service <service> <replicas>, "
            "update_config <service> <key> <value>, "
            "mark_resolved <root_cause_description>"
        )
    )


# ── Reward ───────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """Reward signal returned after each step."""
    total: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)


# ── Step response ────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ── State ────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Full internal state (for debugging / grading)."""
    task_id: str
    step: int
    done: bool
    actions_taken: list[str]
    investigation_log: list[str]
    root_cause_identified: bool
    remediation_applied: bool
    score: float
    reward_breakdown: dict[str, float]
