"""
Interactive simulation runner — demonstrates OnCallEnv with optimal agent strategies.
Shows step-by-step investigation and remediation for all 6 tasks.
"""

import sys
sys.path.insert(0, ".")

from environment import OnCallEnvironment
from models import Action
from graders import grade_task


def banner(text: str):
    w = 70
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def run_task(env: OnCallEnvironment, task_id: str, actions: list[str]):
    """Run a task with a predefined action sequence and print each step."""
    obs = env.reset(task_id)
    print(f"\n  Task:       {obs.task_id}")
    print(f"  Goal:       {obs.goal}")
    print(f"  Max steps:  {obs.max_steps}")
    print(f"  Services:   {', '.join(obs.services)}")
    print(f"  Alerts:")
    for a in obs.alerts:
        print(f"    [{a.severity.upper()}] {a.service}: {a.message}")
    print()

    for i, cmd in enumerate(actions, 1):
        resp = env.step(Action(command=cmd))
        obs = resp.observation
        status = "ERROR" if obs.last_action_error else "OK"
        print(f"  Step {i}: {cmd}")
        lines = (obs.last_action_result or "").split("\n")
        for line in lines[:5]:
            print(f"    | {line}")
        if len(lines) > 5:
            print(f"    | ... ({len(lines) - 5} more lines)")
        print(f"    [{status}] Running score: {resp.reward.total}")
        print()

        if resp.done:
            print(f"  >>> Episode finished: {resp.info.get('reason', '?')}")
            break

    state = env.state()
    grader_score = grade_task(task_id, state)
    print(f"\n  -- Final Results --")
    print(f"  Env Score:     {state.score}")
    print(f"  Grader Score:  {grader_score}")
    print(f"  Breakdown:")
    for k, v in state.reward_breakdown.items():
        print(f"    {k:20s} {v:+.3f}")
    print(f"  Steps used:     {state.step}")
    print(f"  Root cause ID:  {state.root_cause_identified}")
    print(f"  Remediation:    {state.remediation_applied}")
    return state.score


def main():
    env = OnCallEnvironment()
    scores = {}

    # ── EASY ──────────────────────────────────────────────────────────
    banner("EASY: Memory Leak in Payment Service")
    scores["easy_memory_leak"] = run_task(env, "easy_memory_leak", [
        "check_logs payment-service",
        "check_metrics payment-service",
        "check_metrics api-gateway",
        "restart_service payment-service",
        "mark_resolved payment-service memory leak OOM out of memory causing repeated kills",
    ])

    # ── MEDIUM ────────────────────────────────────────────────────────
    banner("MEDIUM: Cascading Connection Pool Exhaustion")
    scores["medium_cascading_failure"] = run_task(env, "medium_cascading_failure", [
        "check_metrics api-gateway",
        "check_logs api-gateway",
        "check_dependencies api-gateway",
        "check_metrics order-service",
        "check_logs order-service",
        "check_config order-service",
        "update_config order-service db_pool_size 50",
        "mark_resolved order-service connection pool exhausted db_pool_size config changed to 5 by auto-scaler",
    ])

    # ── HARD (Cache) ──────────────────────────────────────────────────
    banner("HARD: Subtle Cache Bug Causing Cross-Service Degradation")
    scores["hard_cache_degradation"] = run_task(env, "hard_cache_degradation", [
        "check_metrics api-gateway",
        "check_metrics order-service",
        "check_metrics product-service",
        "check_metrics cache-service",
        "check_logs cache-service",
        "check_deploy_history cache-service",
        "check_metrics postgres-primary",
        "rollback_deploy cache-service",
        "mark_resolved cache-service deployment changed key hashing algorithm causing 60% cache miss rate",
    ])

    # ── MEDIUM (DNS) ──────────────────────────────────────────────────
    banner("MEDIUM: DNS Misconfiguration Causing Intermittent Failures")
    scores["medium_dns_misconfiguration"] = run_task(env, "medium_dns_misconfiguration", [
        "check_metrics order-service",
        "check_logs order-service",
        "check_config order-service",
        "check_metrics inventory-service",
        "check_metrics api-gateway",
        "update_config order-service inventory_host inventory-service.internal",
        "mark_resolved order-service dns hostname misconfiguration inventory_host pointed to decommissioned host",
    ])

    # ── HARD (Replication) ────────────────────────────────────────────
    banner("HARD: Database Replication Lag from Runaway Batch Job")
    scores["hard_replication_lag"] = run_task(env, "hard_replication_lag", [
        "check_metrics user-service",
        "check_logs user-service",
        "check_metrics order-service",
        "check_logs order-service",
        "check_metrics postgres-primary",
        "check_logs postgres-primary",
        "check_config postgres-primary",
        "check_metrics postgres-replica",
        "update_config postgres-primary batch_job_enabled false",
        "mark_resolved postgres-primary batch job nightly_aggregation running during peak hours causing replication lag",
    ])

    # ── EXPERT (Multi-Root-Cause) ────────────────────────────────────
    banner("EXPERT: Simultaneous Bad Deployment and Config Drift")
    scores["expert_multi_root_cause"] = run_task(env, "expert_multi_root_cause", [
        "check_metrics api-gateway",
        "check_logs api-gateway",
        "check_metrics search-service",
        "check_logs search-service",
        "check_deploy_history search-service",
        "check_metrics order-service",
        "check_logs order-service",
        "check_config order-service",
        "check_metrics elasticsearch",
        "rollback_deploy search-service",
        "update_config order-service db_pool_size 50",
        "mark_resolved search-service bad deployment v3.1.0 broke elasticsearch query AND order-service db_pool_size config reduced to 3 by capacity-planner both issues fixed",
    ])

    # ── SUMMARY ───────────────────────────────────────────────────────
    banner("SIMULATION SUMMARY")
    total = 0.0
    for tid, score in scores.items():
        print(f"  {tid:35s}  {score:.3f}")
        total += score
    avg = total / len(scores)
    print(f"  {'':35s}  -----")
    print(f"  {'Average':35s}  {avg:.3f}")
    print()


if __name__ == "__main__":
    main()
