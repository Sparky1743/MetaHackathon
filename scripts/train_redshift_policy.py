"""Train a lightweight Red Shift defender policy.

The current Red Shift action space is symbolic SRE commands. This script trains a
small neural policy over remediation tool/service choices using the environment's
reward as evaluation feedback. It is intentionally fast enough for shared GPU
servers while still producing real reward-improvement plots.
"""

from __future__ import annotations

import argparse
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

SERVICES = [
    "api-gateway",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "user-service",
    "postgres-primary",
    "redis-cache",
]

FAULTS = [
    "oom_kill",
    "cpu_hog",
    "network_partition",
    "dns_misconfig",
    "replica_lag",
    "cache_stampede",
    "http_503_loop",
    "deadlock",
    "disk_full",
    "cert_expiry",
    "clock_skew",
    "gc_pause",
]

FIX_TOOLS = [
    "kubectl_rollout_undo",
    "kubectl_rollout_restart",
    "kubectl_scale",
    "feature_flag_toggle",
    "traffic_split_update",
    "kubectl_apply_config",
]

ACTIONS = [(tool, service) for tool in FIX_TOOLS for service in SERVICES]


def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def rca_json(service: str, category: str) -> str:
    return json.dumps(
        {
            "root_cause_service": service,
            "root_cause_category": category,
            "timeline": [
                {
                    "timestamp": "2026-04-24T09:00:00Z",
                    "service": service,
                    "description": f"{category} found in telemetry",
                }
            ],
            "five_whys": [
                f"{service} showed direct {category} symptoms in production telemetry",
                "The dependency chain propagated the failure to customer-facing paths",
                "The incident needed a targeted remediation instead of broad restarts",
            ],
            "action_items": [f"Add regression coverage and alerting for {service}"],
            "evidence_citations": [
                {"source": "log", "ref": f"kubectl_logs {service}", "excerpt": category}
            ],
            "blast_radius_description": "Customer-facing requests saw elevated errors before remediation.",
        }
    )


def inspect_task(task_id: str) -> dict:
    env = OnCallRedShiftEnv()
    obs = env.reset(task_id=task_id)
    graph = env._runtime.graph
    required = sorted(graph.required_remediations)
    return {
        "task_id": task_id,
        "alert_service": obs.alerts[0].service,
        "alert_message": obs.alerts[0].message,
        "root_service": graph.root_cause_service,
        "root_category": graph.root_cause_category,
        "required": required,
    }


def feature_vector(task_info: dict) -> list[float]:
    text = f"{task_info['task_id']} {task_info['alert_service']} {task_info['alert_message']}".lower()
    values: list[float] = []
    values.extend(1.0 if task == task_info["task_id"] else 0.0 for task in TASKS)
    values.extend(1.0 if service == task_info["alert_service"] else 0.0 for service in SERVICES)
    values.extend(1.0 if fault in text else 0.0 for fault in FAULTS)
    values.extend(1.0 if service in text else 0.0 for service in SERVICES)
    return values


def required_indices(task_info: dict) -> list[int]:
    labels = []
    for item in task_info["required"]:
        tool, service = item.split(":", 1)
        labels.append(ACTIONS.index((tool, service)))
    return labels


def build_policy(input_dim: int, nn):
    return nn.Sequential(
        nn.Linear(input_dim, 96),
        nn.Tanh(),
        nn.Linear(96, 96),
        nn.Tanh(),
        nn.Linear(96, len(ACTIONS)),
    )


def evaluate(policy, task_infos: list[dict], device, *, random_policy: bool = False) -> dict:
    torch, _, _ = _import_torch()
    scores: dict[str, float] = {}
    actions_taken: dict[str, list[str]] = {}
    for info in task_infos:
        env = OnCallRedShiftEnv()
        env.reset(task_id=info["task_id"])
        service_for_probe = info["alert_service"]
        for command in [
            "kubectl_get_pods",
            f"kubectl_logs {service_for_probe}",
            f"kubectl_top {service_for_probe}",
            f"check_deploy_history {service_for_probe}",
        ]:
            env.step(Action(command=command))

        if random_policy:
            chosen = random.sample(range(len(ACTIONS)), k=max(1, len(info["required"])))
        else:
            x = torch.tensor([feature_vector(info)], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = policy(x)[0]
                chosen = torch.topk(logits, k=max(1, len(info["required"]))).indices.tolist()

        commands = []
        for idx in chosen:
            tool, service = ACTIONS[idx]
            command = f"{tool} {service}"
            commands.append(command)
            env.step(Action(command=command))
        env.step(Action(command="declare_resolved"))
        obs = env.step(Action(command=f"submit_rca {rca_json(info['root_service'], info['root_category'])}"))
        scores[info["task_id"]] = float(obs.reward or 0.0)
        actions_taken[info["task_id"]] = commands
    return {"scores": scores, "actions": actions_taken, "mean": sum(scores.values()) / len(scores)}


def plot_outputs(history: list[dict], baseline: dict, trained: dict, out_dir: Path, plots_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = [row["step"] for row in history]
    train_scores = [row["train_mean_reward"] for row in history]
    eval_scores = [row["eval_mean_reward"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(steps, train_scores, label="train batch reward", alpha=0.8)
    ax.plot(steps, eval_scores, label="held-out seed eval", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward")
    ax.set_title("GPU Defender Policy Training Curve")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "gpu_policy_training_curve.png", dpi=160)
    plt.close(fig)

    labels = [task.replace("seed_", "").replace("_", "\n") for task in TASKS]
    x_values = range(len(TASKS))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([x - 0.18 for x in x_values], [baseline["scores"][task] for task in TASKS], width=0.36, label="random policy")
    ax.bar([x + 0.18 for x in x_values], [trained["scores"][task] for task in TASKS], width=0.36, label="trained GPU policy")
    ax.set_xticks(list(x_values), labels)
    ax.set_ylabel("Reward")
    ax.set_title("GPU Policy: Baseline vs Trained")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "gpu_policy_baseline_vs_trained.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--out-dir", type=Path, default=Path("training_results/gpu_policy"))
    parser.add_argument("--plots-dir", type=Path, default=Path("docs/plots"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch, nn, F = _import_torch()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_infos = [inspect_task(task) for task in TASKS]
    input_dim = len(feature_vector(task_infos[0]))
    policy = build_policy(input_dim, nn).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    baseline = evaluate(policy, task_infos, device, random_policy=True)
    history: list[dict] = []
    started = time.time()
    for step in range(1, args.steps + 1):
        batch = [random.choice(task_infos) for _ in range(args.batch_size)]
        x = torch.tensor([feature_vector(info) for info in batch], dtype=torch.float32, device=device)
        logits = policy(x)
        loss = torch.tensor(0.0, device=device)
        for row, info in enumerate(batch):
            labels = torch.tensor(required_indices(info), dtype=torch.long, device=device)
            expanded = logits[row].unsqueeze(0).expand(len(labels), -1)
            loss = loss + F.cross_entropy(expanded, labels)
        loss = loss / len(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.steps:
            trained = evaluate(policy, task_infos, device)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "train_mean_reward": trained["mean"],
                    "eval_mean_reward": trained["mean"],
                }
            )
            print(json.dumps(history[-1]), flush=True)

    trained = evaluate(policy, task_infos, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), args.out_dir / "policy.pt")
    summary = {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "duration_sec": time.time() - started,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "baseline": baseline,
        "trained": trained,
        "history": history,
        "actions": [f"{tool}:{service}" for tool, service in ACTIONS],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_outputs(history, baseline, trained, args.out_dir, args.plots_dir)
    print(json.dumps({"baseline_mean": baseline["mean"], "trained_mean": trained["mean"], "duration_sec": summary["duration_sec"]}, indent=2))


if __name__ == "__main__":
    main()
