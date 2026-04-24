"""
Graders for OnCallEnv tasks.

Each grader takes the final EnvState and returns a float in [0.0, 1.0].
Grading is deterministic and based solely on the state record.

Graders compute scores independently from the environment's reward function
by examining the actions_taken and investigation_log directly.
"""

from __future__ import annotations
from models import EnvState


# ── Helpers ──────────────────────────────────────────────────────────────────

def _investigation_score(state: EnvState, targets: list[str],
                         root_cause_service: str, weight: float = 0.30) -> float:
    """Score investigation quality based on which services were checked."""
    if not targets:
        return 0.0
    checked = [s for s in state.investigation_log if s in targets]
    ratio = len(checked) / len(targets)
    # Bonus for checking root cause service
    if root_cause_service in state.investigation_log:
        ratio = min(1.0, ratio + 0.2)
    return round(weight * min(ratio, 1.0), 3)


def _root_cause_score(state: EnvState, root_cause_service: str,
                      keywords: list[str], weight: float = 0.25) -> float:
    """Score root cause identification from mark_resolved calls."""
    # Check if agent called mark_resolved with correct info
    for action in state.actions_taken:
        if action.lower().startswith("mark_resolved"):
            description = action.lower()
            matched = sum(1 for kw in keywords if kw in description)
            if matched >= 2 and root_cause_service.lower() in description:
                return weight  # Full credit
            if matched >= 1:
                return round(weight * 0.5, 3)  # Partial credit
    # Partial credit if at least investigated the right service
    if root_cause_service in state.investigation_log:
        return round(weight * 0.32, 3)  # 0.08 for 0.25 weight
    return 0.0


def _remediation_score(state: EnvState, correct_cmd: str,
                       correct_service: str, weight: float = 0.30,
                       correct_key: str | None = None) -> float:
    """Score remediation based on whether the correct fix was applied."""
    for action in state.actions_taken:
        parts = action.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        if cmd == correct_cmd and len(parts) >= 2 and parts[1] == correct_service:
            if correct_key is not None:
                # For update_config, also check the key
                if len(parts) >= 3 and parts[2] == correct_key:
                    return weight
            else:
                return weight
    # Partial credit if any remediation was attempted
    remediation_cmds = {"restart_service", "rollback_deploy", "scale_service", "update_config"}
    for action in state.actions_taken:
        parts = action.strip().split()
        if parts and parts[0].lower() in remediation_cmds:
            return round(weight * 0.17, 3)  # ~0.05
    return 0.0


def _efficiency_score(state: EnvState, max_steps: int,
                      resolved: bool, weight: float = 0.15) -> float:
    """Score efficiency based on steps taken."""
    if not resolved:
        return 0.0
    if state.step <= max_steps * 0.4:
        return weight
    if state.step <= max_steps * 0.7:
        return round(weight * 0.67, 3)
    return round(weight * 0.33, 3)


def _penalty_score(state: EnvState, penalty_services: list[str]) -> float:
    """Compute penalties for wrong destructive actions."""
    destructive = {"restart_service", "rollback_deploy", "scale_service"}
    wrong = 0
    for action in state.actions_taken:
        parts = action.strip().split()
        if len(parts) >= 2 and parts[0].lower() in destructive:
            if parts[1] in penalty_services:
                wrong += 1
    return round(-min(wrong * 0.05, 0.15), 3)


# ── Task-specific graders ────────────────────────────────────────────────────

def grade_easy_memory_leak(state: EnvState) -> float:
    """
    Easy task grading rubric:
      0.30 — Investigated payment-service and api-gateway
      0.25 — Identified root cause (OOM / memory) via mark_resolved
      0.30 — Restarted payment-service
      0.15 — Efficiency bonus (resolved in few steps)
      -0.05 — Penalty per wrong destructive action
    """
    investigation = _investigation_score(
        state, ["payment-service", "api-gateway"], "payment-service")
    root_cause = _root_cause_score(
        state, "payment-service",
        ["memory", "oom", "leak", "heap", "cache", "outofmemory"])
    remediation = _remediation_score(
        state, "restart_service", "payment-service")
    resolved = state.done and remediation >= 0.25
    efficiency = _efficiency_score(state, 10, resolved)
    penalty = _penalty_score(state, ["user-service", "postgres-primary"])

    return max(0.01, min(0.99, round(
        investigation + root_cause + remediation + efficiency + penalty, 3)))


def grade_medium_cascading_failure(state: EnvState) -> float:
    """
    Medium task grading rubric:
      0.30 — Investigated dependency chain (api-gateway → order-service → notification-service)
      0.25 — Identified db_pool_size config as root cause
      0.30 — Applied correct config update to order-service
      0.15 — Efficiency bonus
      -0.05 — Penalty per wrong destructive action
    """
    investigation = _investigation_score(
        state, ["api-gateway", "order-service", "notification-service"], "order-service")
    root_cause = _root_cause_score(
        state, "order-service",
        ["pool", "connection", "db_pool_size", "config", "5", "exhausted", "auto-scaler"])
    remediation = _remediation_score(
        state, "update_config", "order-service", correct_key="db_pool_size")
    resolved = state.done and remediation >= 0.25
    efficiency = _efficiency_score(state, 15, resolved)
    penalty = _penalty_score(state, ["user-service", "redis-cache"])

    return max(0.01, min(0.99, round(
        investigation + root_cause + remediation + efficiency + penalty, 3)))


def grade_hard_cache_degradation(state: EnvState) -> float:
    """
    Hard task grading rubric:
      0.30 — Investigated multiple services + cache-service
      0.25 — Identified cache deployment as root cause
      0.30 — Rolled back cache-service deployment
      0.15 — Efficiency bonus
    """
    investigation = _investigation_score(
        state,
        ["api-gateway", "order-service", "product-service",
         "cache-service", "postgres-primary", "search-service"],
        "cache-service")
    root_cause = _root_cause_score(
        state, "cache-service",
        ["cache", "hash", "hashing", "key", "murmur", "fnv", "v2.4",
         "deployment", "deploy", "miss", "format", "migration"])
    remediation = _remediation_score(
        state, "rollback_deploy", "cache-service")
    resolved = state.done and remediation >= 0.25
    efficiency = _efficiency_score(state, 20, resolved)
    penalty = _penalty_score(state, [])

    return max(0.01, min(0.99, round(
        investigation + root_cause + remediation + efficiency + penalty, 3)))


def grade_medium_dns_misconfiguration(state: EnvState) -> float:
    """
    Medium-hard task grading rubric:
      0.30 — Investigated order-service, inventory-service, api-gateway
      0.25 — Identified DNS / hostname misconfiguration as root cause
      0.30 — Applied correct config fix (inventory_host)
      0.15 — Efficiency bonus
      -0.05 — Penalty per wrong destructive action
    """
    investigation = _investigation_score(
        state, ["order-service", "inventory-service", "api-gateway"], "order-service")
    root_cause = _root_cause_score(
        state, "order-service",
        ["dns", "hostname", "inventory_host", "inventory-service-v2",
         "config", "misconfiguration", "wrong", "host", "resolution"])
    remediation = _remediation_score(
        state, "update_config", "order-service", correct_key="inventory_host")
    resolved = state.done and remediation >= 0.25
    efficiency = _efficiency_score(state, 15, resolved)
    penalty = _penalty_score(
        state, ["payment-service", "user-service", "postgres-primary", "redis-cache"])

    return max(0.01, min(0.99, round(
        investigation + root_cause + remediation + efficiency + penalty, 3)))


def grade_hard_replication_lag(state: EnvState) -> float:
    """
    Hard task grading rubric:
      0.30 — Investigated services showing stale data + postgres instances
      0.25 — Identified batch job / replication lag as root cause
      0.30 — Disabled batch_job_enabled on postgres-primary
      0.15 — Efficiency bonus
      -0.05 — Penalty per wrong destructive action
    """
    investigation = _investigation_score(
        state,
        ["user-service", "order-service", "product-service",
         "postgres-primary", "postgres-replica"],
        "postgres-primary")
    root_cause = _root_cause_score(
        state, "postgres-primary",
        ["batch", "job", "replication", "lag", "nightly", "aggregation",
         "write", "wal", "schedule", "cron", "batch_job_enabled"])
    remediation = _remediation_score(
        state, "update_config", "postgres-primary", correct_key="batch_job_enabled")
    resolved = state.done and remediation >= 0.25
    efficiency = _efficiency_score(state, 20, resolved)
    penalty = _penalty_score(state, ["api-gateway", "redis-cache", "cache-service"])

    return max(0.01, min(0.99, round(
        investigation + root_cause + remediation + efficiency + penalty, 3)))


def grade_expert_multi_root_cause(state: EnvState) -> float:
    """
    Expert task grading rubric (multi-root-cause):
      0.30 — Investigated search-service, order-service, api-gateway, elasticsearch
      0.25 — Identified both root causes (search deploy + order config)
      0.30 — Applied BOTH fixes (rollback search + update order config)
      0.15 — Efficiency bonus
      -0.05 — Penalty per wrong destructive action
    """
    investigation = _investigation_score(
        state,
        ["api-gateway", "search-service", "order-service", "elasticsearch"],
        "search-service")  # primary root cause service
    root_cause = _root_cause_score(
        state, "search-service",
        ["search", "deployment", "deploy", "elasticsearch", "query",
         "rollback", "order", "pool", "connection", "config",
         "db_pool_size", "both", "two", "multiple"])

    # Multi-remediation: check both fixes were applied
    fix1 = _remediation_score(state, "rollback_deploy", "search-service")
    fix2 = _remediation_score(state, "update_config", "order-service",
                              correct_key="db_pool_size")
    # Both fixes required for full remediation credit
    if fix1 >= 0.25 and fix2 >= 0.25:
        remediation = 0.30
    elif fix1 >= 0.25 or fix2 >= 0.25:
        remediation = 0.15  # Only fixed one of two issues
    elif fix1 > 0 or fix2 > 0:
        remediation = 0.05
    else:
        remediation = 0.0

    resolved = state.done and remediation >= 0.25
    efficiency = _efficiency_score(state, 25, resolved)
    penalty = _penalty_score(
        state, ["user-service", "product-service", "postgres-primary", "redis-cache"])

    return max(0.01, min(0.99, round(
        investigation + root_cause + remediation + efficiency + penalty, 3)))


# ── Registry ─────────────────────────────────────────────────────────────────

GRADERS = {
    "easy_memory_leak": grade_easy_memory_leak,
    "medium_cascading_failure": grade_medium_cascading_failure,
    "hard_cache_degradation": grade_hard_cache_degradation,
    "medium_dns_misconfiguration": grade_medium_dns_misconfiguration,
    "hard_replication_lag": grade_hard_replication_lag,
    "expert_multi_root_cause": grade_expert_multi_root_cause,
}


def grade_task(task_id: str, state: EnvState) -> float:
    """Run the grader for a specific task. Returns score in [0.0, 1.0]."""
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"No grader for task: {task_id}")
    score = grader(state)
    return max(0.01, min(0.99, round(score, 3)))
