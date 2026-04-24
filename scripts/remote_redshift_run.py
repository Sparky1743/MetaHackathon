"""Run a lightweight Red Shift simulator experiment.

This is intentionally model-free: it validates the environment and reward
surface on a remote server before spending GPU hours on GRPO.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action


TASKS = [
    "seed_easy_memory_leak",
    "seed_dns_misconfiguration",
    "seed_cert_expiry",
    "seed_cache_stampede",
    "seed_replica_lag",
    "seed_http_503_loop",
]

FIX = {
    "oom_kill": "kubectl_rollout_restart",
    "cpu_hog": "kubectl_scale",
    "network_partition": "traffic_split_update",
    "dns_misconfig": "kubectl_apply_config",
    "replica_lag": "feature_flag_toggle",
    "cache_stampede": "feature_flag_toggle",
    "http_503_loop": "kubectl_rollout_undo",
    "deadlock": "kubectl_rollout_restart",
    "disk_full": "kubectl_apply_config",
    "cert_expiry": "kubectl_apply_config",
    "clock_skew": "kubectl_rollout_restart",
    "gc_pause": "kubectl_rollout_restart",
}


def rca_json(service: str, category: str) -> str:
    return json.dumps(
        {
            "root_cause_service": service,
            "root_cause_category": category,
            "timeline": [
                {
                    "timestamp": "2026-04-24T09:00:00Z",
                    "service": service,
                    "description": f"{category} detected in telemetry",
                }
            ],
            "five_whys": [
                f"{service} emitted direct {category} telemetry during the incident",
                "The upstream symptom was caused by downstream dependency failure",
                "The missing guardrail allowed customer-facing errors to continue",
            ],
            "action_items": [f"Add alerting and regression tests for {service}"],
            "evidence_citations": [
                {"source": "log", "ref": f"kubectl_logs {service}", "excerpt": category}
            ],
            "blast_radius_description": "Customer-facing requests saw elevated error rate until remediation.",
        }
    )


def run_scripted(task: str) -> float:
    env = OnCallRedShiftEnv()
    env.reset(task_id=task)
    graph = env._runtime.graph  # experiment-only introspection, not client code
    service = graph.root_cause_service
    category = graph.root_cause_category
    commands = [
        "kubectl_get_pods",
        f"kubectl_logs {service}",
        f"kubectl_top {service}",
        f"check_deploy_history {service}",
        f"{FIX[category]} {service}",
        "declare_resolved",
        f"submit_rca {rca_json(service, category)}",
    ]
    obs = None
    for command in commands:
        obs = env.step(Action(command=command))
    return float(obs.reward)


def run_baseline(task: str, rng: random.Random) -> float:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task)
    probes = ["kubectl_get_pods", "kubectl_logs", "kubectl_top", "promql_query", "curl_service", "dns_lookup"]
    for _ in range(6):
        tool = rng.choice(probes)
        command = tool if tool == "kubectl_get_pods" else f"{tool} {rng.choice(obs.services)}"
        obs = env.step(Action(command=command))
    obs = env.step(Action(command="declare_resolved"))
    return float(obs.reward or 0.0)


def main() -> None:
    out = Path("remote_results")
    plots = Path("docs/plots")
    out.mkdir(exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260424)
    start = time.time()
    baseline = {task: [run_baseline(task, rng) for _ in range(20)] for task in TASKS}
    scripted = {task: [run_scripted(task) for _ in range(5)] for task in TASKS}
    summary = {
        "started_at": start,
        "duration_sec": time.time() - start,
        "baseline_mean": {task: sum(values) / len(values) for task, values in baseline.items()},
        "scripted_mean": {task: sum(values) / len(values) for task, values in scripted.items()},
        "baseline_raw": baseline,
        "scripted_raw": scripted,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = [task.replace("seed_", "").replace("_", "\n") for task in TASKS]
    x_values = range(len(TASKS))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([x - 0.18 for x in x_values], [summary["baseline_mean"][task] for task in TASKS], width=0.36, label="random baseline")
    ax.bar([x + 0.18 for x in x_values], [summary["scripted_mean"][task] for task in TASKS], width=0.36, label="scripted trained-target")
    ax.set_xticks(list(x_values), labels)
    ax.set_ylabel("Reward")
    ax.set_title("Remote Red Shift Run: Baseline vs Trained-Target")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "remote_baseline_vs_scripted.png", dpi=160)
    plt.close(fig)

    print(
        json.dumps(
            {
                "duration_sec": summary["duration_sec"],
                "baseline_avg": sum(summary["baseline_mean"].values()) / len(TASKS),
                "scripted_avg": sum(summary["scripted_mean"].values()) / len(TASKS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

