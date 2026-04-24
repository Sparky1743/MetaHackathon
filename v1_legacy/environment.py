"""
Core environment logic for OnCallEnv.

Manages state transitions, command parsing, and reward computation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from models import (
    Action, Alert, EnvState, LogEntry, Observation,
    Reward, ServiceMetrics, StepResponse,
)
from scenarios import ALL_SCENARIOS, Scenario, ServiceDef


class OnCallEnvironment:
    """Simulates a production incident response scenario."""

    def __init__(self):
        self._scenario: Scenario | None = None
        self._step: int = 0
        self._done: bool = False
        self._actions_taken: list[str] = []
        self._investigation_log: list[str] = []   # services investigated
        self._root_cause_identified: bool = False
        self._remediation_applied: bool = False
        self._remediation_correct: bool = False
        self._resolved: bool = False
        self._wrong_actions: int = 0               # destructive actions on wrong services
        self._remediation_step: int | None = None  # step when correct remediation was applied
        self._multi_fixes_applied: list[dict] = [] # for multi-root-cause scenarios

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self, task_id: str | None = None) -> Observation:
        """Reset environment to initial state for the given task."""
        if task_id is None:
            task_id = "easy_memory_leak"
        if task_id not in ALL_SCENARIOS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(ALL_SCENARIOS)}")

        self._scenario = ALL_SCENARIOS[task_id]
        self._step = 0
        self._done = False
        self._actions_taken = []
        self._investigation_log = []
        self._root_cause_identified = False
        self._remediation_applied = False
        self._remediation_correct = False
        self._resolved = False
        self._wrong_actions = 0
        self._remediation_step = None
        self._multi_fixes_applied = []

        return self._build_observation(
            last_action=None,
            last_action_result="Environment reset. You are the on-call engineer. Read the alerts and begin investigating.",
            last_action_error=False,
        )

    def step(self, action: Action) -> StepResponse:
        """Execute one action and return the new observation + reward."""
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            return StepResponse(
                observation=self._build_observation(
                    last_action=action.command,
                    last_action_result="Episode already finished.",
                    last_action_error=True,
                ),
                reward=self._compute_reward(),
                done=True,
                info={"reason": "already_done"},
            )

        self._step += 1
        self._actions_taken.append(action.command)

        result, error = self._execute_command(action.command)

        # Check termination
        # After correct remediation, give agent 2 more steps to call mark_resolved
        if self._remediation_correct and self._remediation_step is None:
            self._remediation_step = self._step
        if self._resolved:
            self._done = True
        elif self._remediation_step and self._step >= self._remediation_step + 2:
            # Agent had 2 steps after remediation but didn't mark_resolved
            self._done = True
        elif self._step >= self._scenario.max_steps:
            self._done = True

        reward = self._compute_reward()
        obs = self._build_observation(
            last_action=action.command,
            last_action_result=result,
            last_action_error=error,
        )

        info: dict[str, Any] = {}
        if self._done:
            info["reason"] = "resolved" if self._resolved else "max_steps"
            info["final_score"] = reward.total

        return StepResponse(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return full internal state for debugging / grading."""
        return EnvState(
            task_id=self._scenario.task_id if self._scenario else "",
            step=self._step,
            done=self._done,
            actions_taken=list(self._actions_taken),
            investigation_log=list(self._investigation_log),
            root_cause_identified=self._root_cause_identified,
            remediation_applied=self._remediation_applied,
            score=self._compute_reward().total,
            reward_breakdown=self._compute_reward().breakdown,
        )

    def list_tasks(self) -> list[dict]:
        """Return metadata for all available tasks."""
        return [
            {
                "task_id": s.task_id,
                "task_name": s.task_name,
                "difficulty": s.difficulty,
                "description": s.description,
                "max_steps": s.max_steps,
            }
            for s in ALL_SCENARIOS.values()
        ]

    # ── Command execution ─────────────────────────────────────────────────

    def _execute_command(self, raw: str) -> tuple[str, bool]:
        """Parse and execute a command. Returns (result_text, is_error)."""
        raw = raw.strip()
        parts = raw.split(None, 2)
        if not parts:
            return "Empty command. Use one of: check_metrics, check_logs, check_config, check_dependencies, check_deploy_history, restart_service, rollback_deploy, scale_service, update_config, mark_resolved", True

        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        dispatch = {
            "check_metrics": self._cmd_check_metrics,
            "check_logs": self._cmd_check_logs,
            "check_config": self._cmd_check_config,
            "check_dependencies": self._cmd_check_dependencies,
            "check_deploy_history": self._cmd_check_deploy_history,
            "restart_service": self._cmd_restart_service,
            "rollback_deploy": self._cmd_rollback_deploy,
            "scale_service": self._cmd_scale_service,
            "update_config": self._cmd_update_config,
            "mark_resolved": self._cmd_mark_resolved,
        }

        handler = dispatch.get(cmd)
        if handler is None:
            return (
                f"Unknown command: '{cmd}'. Available commands: {', '.join(dispatch.keys())}",
                True,
            )
        return handler(args)

    def _get_service(self, args: list[str]) -> tuple[ServiceDef | None, str]:
        if not args:
            return None, "Missing service name."
        svc_name = args[0]
        svc = self._scenario.services.get(svc_name)
        if svc is None:
            available = ", ".join(self._scenario.services.keys())
            return None, f"Unknown service: '{svc_name}'. Available: {available}"
        return svc, ""

    def _record_investigation(self, service_name: str):
        if service_name not in self._investigation_log:
            self._investigation_log.append(service_name)

    def _check_multi_remediation(self, cmd: str, service: str,
                                  key: str | None = None) -> bool:
        """For multi-root-cause scenarios, track and check if this fix is one of the required ones."""
        sc = self._scenario
        if sc.correct_remediation != "multi" or not sc.correct_remediations:
            return False
        for req in sc.correct_remediations:
            if req["cmd"] == cmd and req["service"] == service:
                if "key" in req and key != req["key"]:
                    continue
                fix = {"cmd": cmd, "service": service}
                if fix not in self._multi_fixes_applied:
                    self._multi_fixes_applied.append(fix)
                # Check if ALL required fixes are now applied
                if len(self._multi_fixes_applied) >= len(sc.correct_remediations):
                    self._remediation_correct = True
                return True
        return False

    def _cmd_check_metrics(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True
        self._record_investigation(svc.name)

        # Dynamic metrics: degrade over time if unresolved, recover after fix
        cpu = svc.cpu
        memory = svc.memory
        latency_p50 = svc.latency_p50
        latency_p99 = svc.latency_p99
        error_rate = svc.error_rate
        rps = svc.rps
        extra = dict(svc.extra_metrics)

        if self._remediation_correct:
            # Post-fix: show healthy metrics
            if not svc.healthy or svc.name == self._scenario.root_cause_service:
                cpu = min(cpu, 30.0)
                memory = min(memory, 45.0)
                latency_p50 = min(latency_p50, 20.0)
                latency_p99 = min(latency_p99, 80.0)
                error_rate = 0.1
        elif not svc.healthy and self._step > 1:
            # Pre-fix degradation: metrics worsen each step for unhealthy services
            degrade = min(self._step * 0.03, 0.15)  # up to 15% worse
            cpu = min(99.0, cpu * (1 + degrade))
            memory = min(99.0, memory * (1 + degrade * 0.5))
            latency_p99 = latency_p99 * (1 + degrade)
            error_rate = min(99.0, error_rate * (1 + degrade))
            # Update extra metrics that are counts
            for k in extra:
                if isinstance(extra[k], (int, float)) and any(
                    w in k for w in ["fail", "error", "pending", "miss", "stale", "lag"]
                ):
                    extra[k] = round(extra[k] * (1 + degrade), 1) if isinstance(extra[k], float) else int(extra[k] * (1 + degrade))

        lines = [
            f"=== Metrics for {svc.name} (step {self._step}) ===",
            f"  CPU usage:       {cpu:.1f}%",
            f"  Memory usage:    {memory:.1f}%",
            f"  Latency p50:     {latency_p50:.0f}ms",
            f"  Latency p99:     {latency_p99:.0f}ms",
            f"  Error rate:      {error_rate:.1f}%",
            f"  Requests/sec:    {rps:.0f}",
        ]
        for k, v in extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines), False

    def _cmd_check_logs(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True
        self._record_investigation(svc.name)
        if not svc.logs:
            return f"No recent logs for {svc.name}.", False
        header = f"=== Recent logs for {svc.name} ==="
        return header + "\n" + "\n".join(svc.logs), False

    def _cmd_check_config(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True
        self._record_investigation(svc.name)
        lines = [f"=== Configuration for {svc.name} ==="]
        for k, v in svc.config.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines), False

    def _cmd_check_dependencies(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True
        self._record_investigation(svc.name)
        deps = svc.dependencies or ["(none)"]
        return f"=== Dependencies for {svc.name} ===\n  " + "\n  ".join(deps), False

    def _cmd_check_deploy_history(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True
        self._record_investigation(svc.name)
        if not svc.deploy_history:
            return f"No deployment history for {svc.name}.", False
        lines = [f"=== Deploy history for {svc.name} ==="]
        for d in svc.deploy_history:
            line = f"  [{d['date']}] {d['version']} — {d.get('status', 'unknown')}"
            if d.get("notes"):
                line += f"\n    Notes: {d['notes']}"
            lines.append(line)
        return "\n".join(lines), False

    def _cmd_restart_service(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True

        self._remediation_applied = True

        # Check if this is the correct remediation
        sc = self._scenario
        if self._check_multi_remediation("restart_service", svc.name):
            fixes_done = len(self._multi_fixes_applied)
            fixes_total = len(sc.correct_remediations)
            if fixes_done < fixes_total:
                return (
                    f"Service {svc.name} restarted successfully. Service recovering. "
                    f"[{fixes_done}/{fixes_total} issues fixed] — other issues remain. Continue investigating."
                ), False
            return (
                f"Service {svc.name} restarted successfully. All issues resolved! "
                f"Now call mark_resolved with a root cause description to close the incident."
            ), False
        if (sc.correct_remediation == "restart_service"
                and svc.name == sc.root_cause_service):
            self._remediation_correct = True
            return (
                f"Service {svc.name} restarted successfully. Service is now healthy. "
                f"Memory usage dropping to normal levels. Error rate returning to baseline. "
                f"Now call mark_resolved with a root cause description to close the incident."
            ), False

        if svc.name in sc.penalty_services:
            self._wrong_actions += 1
            return f"Service {svc.name} restarted, but this does not address the underlying issue. Service was already healthy.", False

        return f"Service {svc.name} restarted. Monitoring for changes...", False

    def _cmd_rollback_deploy(self, args: list[str]) -> tuple[str, bool]:
        svc, err = self._get_service(args)
        if svc is None:
            return err, True

        self._remediation_applied = True
        sc = self._scenario

        if self._check_multi_remediation("rollback_deploy", svc.name):
            prev = [d for d in svc.deploy_history if d.get("status") == "previous"]
            prev_ver = prev[0]["version"] if prev else "previous"
            fixes_done = len(self._multi_fixes_applied)
            fixes_total = len(sc.correct_remediations)
            if fixes_done < fixes_total:
                return (
                    f"Rolled back {svc.name} to {prev_ver}. Service recovering. "
                    f"[{fixes_done}/{fixes_total} issues fixed] — other issues remain. Continue investigating."
                ), False
            return (
                f"Rolled back {svc.name} to {prev_ver}. All issues resolved! "
                f"Now call mark_resolved with a root cause description to close the incident."
            ), False
        if (sc.correct_remediation == "rollback_deploy"
                and svc.name == sc.root_cause_service):
            self._remediation_correct = True
            prev = [d for d in svc.deploy_history if d.get("status") == "previous"]
            prev_ver = prev[0]["version"] if prev else "previous"
            return (
                f"Rolled back {svc.name} to {prev_ver}. "
                f"Service restarting with previous version... Service healthy. Metrics normalizing. "
                f"Now call mark_resolved with a root cause description to close the incident."
            ), False

        if svc.name in sc.penalty_services:
            self._wrong_actions += 1
        return f"Rolled back {svc.name} to previous version. No improvement observed.", False

    def _cmd_scale_service(self, args: list[str]) -> tuple[str, bool]:
        if len(args) < 2:
            return "Usage: scale_service <service> <replicas>", True
        svc, err = self._get_service(args[:1])
        if svc is None:
            return err, True
        try:
            replicas = int(args[1])
        except ValueError:
            return f"Invalid replica count: {args[1]}", True

        if svc.name in self._scenario.penalty_services:
            self._wrong_actions += 1
        return f"Scaled {svc.name} to {replicas} replicas. This may help with load but does not address root cause.", False

    def _cmd_update_config(self, args: list[str]) -> tuple[str, bool]:
        # Expected: update_config <service> <key> <value>
        if len(args) < 2:
            return "Usage: update_config <service> <key> <value>", True

        svc_name = args[0]
        svc = self._scenario.services.get(svc_name)
        if svc is None:
            available = ", ".join(self._scenario.services.keys())
            return f"Unknown service: '{svc_name}'. Available: {available}", True

        # Parse key and value from remaining args
        # args[1:] could be ["db_pool_size", "50"] or ["db_pool_size 50"]
        sub_parts = " ".join(args[1:]).split(None, 1)
        if len(sub_parts) < 2:
            return "Usage: update_config <service> <key> <value>", True
        key, value = sub_parts[0], sub_parts[1]

        self._remediation_applied = True
        sc = self._scenario

        if self._check_multi_remediation("update_config", svc_name, key=key):
            fixes_done = len(self._multi_fixes_applied)
            fixes_total = len(sc.correct_remediations)
            if fixes_done < fixes_total:
                return (
                    f"Configuration updated: {svc_name}.{key} = {value}. Service recovering. "
                    f"[{fixes_done}/{fixes_total} issues fixed] — other issues remain. Continue investigating."
                ), False
            return (
                f"Configuration updated: {svc_name}.{key} = {value}. All issues resolved! "
                f"Now call mark_resolved with a root cause description to close the incident."
            ), False
        if (sc.correct_remediation == "update_config"
                and svc_name == sc.root_cause_service
                and key == sc.correct_remediation_args.get("key")):
            self._remediation_correct = True
            return (
                f"Configuration updated: {svc_name}.{key} = {value}. "
                f"Service reloading config... Metrics stabilizing. Issue resolved. "
                f"Now call mark_resolved with a root cause description to close the incident."
            ), False

        if svc_name in sc.penalty_services:
            self._wrong_actions += 1
        return f"Configuration updated: {svc_name}.{key} = {value}. Monitoring for effect...", False

    def _cmd_mark_resolved(self, args: list[str]) -> tuple[str, bool]:
        if not args:
            return "Usage: mark_resolved <root_cause_description>", True

        description = " ".join(args).lower()
        sc = self._scenario

        # Check if root cause description matches keywords
        matched = sum(1 for kw in sc.root_cause_keywords if kw in description)
        has_service = sc.root_cause_service.lower() in description
        if matched >= 1 and has_service:
            self._root_cause_identified = True
            if self._remediation_correct:
                self._resolved = True
                return "Incident marked as resolved. Root cause correctly identified. Well done!", False
            return (
                "Root cause description acknowledged, but remediation has not been applied yet. "
                "Please fix the issue before marking as resolved."
            ), False
        elif matched >= 1 or has_service:
            self._root_cause_identified = True  # partial
            return "Partial root cause identified. Consider being more specific about the service and issue.", False
        else:
            return "Root cause description does not match the actual issue. Continue investigating.", False

    # ── Observation builder ───────────────────────────────────────────────

    def _build_observation(self, last_action: str | None,
                           last_action_result: str | None,
                           last_action_error: bool) -> Observation:
        sc = self._scenario

        # Build alerts with escalation if incident is unresolved
        alerts_data = list(sc.alerts)
        if not self._remediation_correct and self._step >= sc.max_steps * 0.5:
            # Escalate: add urgency alerts after 50% of steps
            alerts_data = alerts_data + [
                {"alert_id": f"ALT-ESC-{self._step}", "severity": "critical",
                 "service": sc.root_cause_service,
                 "message": f"ESCALATION: Incident unresolved after {self._step} steps. Impact increasing.",
                 "timestamp": f"2025-04-01T{10 + self._step}:{(self._step * 3) % 60:02d}:00Z"},
            ]
        elif self._remediation_correct:
            # Post-fix: show resolution in alerts
            alerts_data = alerts_data + [
                {"alert_id": "ALT-RESOLVED", "severity": "info",
                 "service": sc.root_cause_service,
                 "message": "Remediation applied. Metrics stabilizing. Call mark_resolved to close incident.",
                 "timestamp": f"2025-04-01T{10 + self._step}:{(self._step * 3) % 60:02d}:00Z"},
            ]

        # More realistic time progression
        hour = 10 + (self._step * 2) // 60
        minute = (self._step * 2) % 60

        return Observation(
            task_id=sc.task_id,
            goal=sc.goal,
            step=self._step,
            max_steps=sc.max_steps,
            current_time=f"2025-04-01T{hour:02d}:{minute:02d}:00Z",
            alerts=[Alert(**a) for a in alerts_data],
            last_action=last_action,
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            available_commands=[
                "check_metrics <service>",
                "check_logs <service>",
                "check_config <service>",
                "check_dependencies <service>",
                "check_deploy_history <service>",
                "restart_service <service>",
                "rollback_deploy <service>",
                "scale_service <service> <replicas>",
                "update_config <service> <key> <value>",
                "mark_resolved <root_cause_description>",
            ],
            services=list(sc.services.keys()),
        )

    # ── Reward computation ────────────────────────────────────────────────

    def _compute_reward(self) -> Reward:
        """
        Reward breakdown (0.0 — 1.0):
          investigation:  0.30  — checking relevant services
          root_cause:     0.25  — correctly identifying root cause
          remediation:    0.30  — applying correct fix
          efficiency:     0.15  — fewer steps = better
          penalties:      -0.05 each for wrong destructive actions
        """
        sc = self._scenario
        breakdown: dict[str, float] = {}

        # 1) Investigation quality (0.30)
        # Per-step incremental: each relevant service gives proportional credit
        relevant_checked = [s for s in self._investigation_log
                            if s in sc.investigation_targets]
        if sc.investigation_targets:
            investigation_ratio = len(relevant_checked) / len(sc.investigation_targets)
        else:
            investigation_ratio = 0.0
        # Bonus for checking root cause service
        if sc.root_cause_service in self._investigation_log:
            investigation_ratio = min(1.0, investigation_ratio + 0.2)
        # Bonus for deep investigation (config or deploy history of root cause)
        deep_cmds = [a for a in self._actions_taken
                     if (a.startswith("check_config " + sc.root_cause_service)
                         or a.startswith("check_deploy_history " + sc.root_cause_service))]
        if deep_cmds:
            investigation_ratio = min(1.0, investigation_ratio + 0.1)
        breakdown["investigation"] = round(0.30 * min(investigation_ratio, 1.0), 3)

        # 2) Root cause identification (0.25)
        if self._root_cause_identified:
            breakdown["root_cause"] = 0.25
        else:
            # Partial credit if they at least investigated the right service
            if sc.root_cause_service in self._investigation_log:
                breakdown["root_cause"] = 0.08
            else:
                breakdown["root_cause"] = 0.0

        # 3) Remediation (0.30)
        if self._remediation_correct:
            breakdown["remediation"] = 0.30
        elif self._remediation_applied:
            breakdown["remediation"] = 0.05  # tried but wrong
        else:
            breakdown["remediation"] = 0.0

        # 4) Efficiency (0.15)
        if self._done and self._resolved:
            steps_used = self._step
            max_s = sc.max_steps
            # Full efficiency bonus if resolved in <= 40% of max steps
            if steps_used <= max_s * 0.4:
                breakdown["efficiency"] = 0.15
            elif steps_used <= max_s * 0.7:
                breakdown["efficiency"] = 0.10
            else:
                breakdown["efficiency"] = 0.05
        elif self._resolved:
            breakdown["efficiency"] = 0.05
        else:
            breakdown["efficiency"] = 0.0

        # 5) Penalties
        penalty = self._wrong_actions * 0.05
        breakdown["penalty"] = round(-min(penalty, 0.15), 3)

        total = sum(breakdown.values())
        total = round(max(0.01, min(0.99, total)), 3)

        return Reward(total=total, breakdown=breakdown)
