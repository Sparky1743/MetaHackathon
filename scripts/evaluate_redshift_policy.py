"""Evaluate a saved Red Shift defender policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_redshift_policy import build_policy, evaluate, feature_vector, load_task_infos, plot_outputs, split_task_infos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("training_results/gpu_policy/policy.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/gpu_policy"))
    parser.add_argument("--plots-dir", type=Path, default=Path("docs/plots"))
    parser.add_argument("--curriculum-buffer", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--plot-prefix", default="gpu_policy")
    parser.add_argument("--feature-mode", choices=["text", "spec"], default="text")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_infos = load_task_infos(args.curriculum_buffer)
    _, eval_infos = split_task_infos(task_infos, args.train_fraction, seed=20260424)
    policy = build_policy(len(feature_vector(eval_infos[0], feature_mode=args.feature_mode)), nn).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    baseline = evaluate(policy, eval_infos, device, random_policy=True, feature_mode=args.feature_mode)
    trained = evaluate(policy, eval_infos, device, feature_mode=args.feature_mode)
    history_path = args.checkpoint.parent / "summary.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
    else:
        history = [{"step": 0, "train_mean_reward": baseline["mean"], "eval_mean_reward": baseline["mean"]}, {"step": 1, "train_mean_reward": trained["mean"], "eval_mean_reward": trained["mean"]}]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "curriculum_buffer": str(args.curriculum_buffer) if args.curriculum_buffer else None,
        "feature_mode": args.feature_mode,
        "eval_size": len(eval_infos),
        "baseline": baseline,
        "trained": trained,
        "history": history,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_outputs(history, baseline, trained, args.out_dir, args.plots_dir, plot_prefix=args.plot_prefix)
    print(json.dumps({"baseline_mean": baseline["mean"], "trained_mean": trained["mean"]}, indent=2))


if __name__ == "__main__":
    main()
