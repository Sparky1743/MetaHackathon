"""
inference.py — Baseline agent for OnCallEnv.

Uses the OpenAI API client to run an LLM against all tasks.
Supports OpenAI, Gemini, and any OpenAI-compatible API.

Required env vars:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key (used as OPENAI_API_KEY)

Usage:
  export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  export MODEL_NAME=gemini-2.0-flash
  export HF_TOKEN=your-api-key
  python inference.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import textwrap
import requests
from typing import Any

from openai import OpenAI


# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

TASKS = [
    "easy_memory_leak",
    "medium_cascading_failure",
    "hard_cache_degradation",
    "medium_dns_misconfiguration",
    "hard_replication_lag",
    "expert_multi_root_cause",
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Site Reliability Engineer (SRE) responding to a production incident.
You must diagnose the root cause and fix the issue as efficiently as possible.

INVESTIGATION COMMANDS:
  check_metrics <service>         — View CPU, memory, latency, error rates, custom metrics
  check_logs <service>            — View recent log entries
  check_config <service>          — View service configuration
  check_dependencies <service>    — View service dependency graph
  check_deploy_history <service>  — View recent deployments and version changes

REMEDIATION COMMANDS:
  restart_service <service>                    — Restart a service (use for OOM/crash issues)
  rollback_deploy <service>                    — Roll back to previous deployment version
  scale_service <service> <replicas>           — Scale replicas up/down
  update_config <service> <key> <value>        — Update a config parameter
  mark_resolved <root_cause_description>       — REQUIRED: Mark incident resolved with root cause

STRATEGY:
1. Read the alerts carefully to identify ALL affected services
2. CRITICAL: Before applying ANY remediation, you MUST investigate at least ALL alerted
   services and their direct dependencies. Check a minimum of 3-4 different services.
3. For each alerted service: check_metrics → check_logs → check_config or check_deploy_history
4. Follow the dependency chain — symptoms often appear upstream of the root cause
5. Look for recent changes (deploys, config changes) that correlate with the incident
6. There may be MULTIPLE independent root causes — fix ALL of them before marking resolved
7. Apply the correct remediation:
   - OOM/memory issues → restart_service
   - Bad deployment → rollback_deploy
   - Wrong config value → update_config
8. AFTER fixing ALL issues, you MUST call mark_resolved with a description that includes:
   - The name(s) of the root cause service(s)
   - What went wrong (e.g., "memory leak", "config change", "bad deployment")
   This step is CRITICAL for full credit.

IMPORTANT RULES:
- Respond with EXACTLY ONE command per turn
- No explanation, no markdown, no extra text — just the command string
- Do NOT restart or rollback healthy services — you will be penalized
- After a successful remediation, ALWAYS call mark_resolved immediately
- If there are multiple issues, fix each one before calling mark_resolved

Example responses:
  check_logs payment-service
  restart_service payment-service
  update_config order-service db_pool_size 50
  rollback_deploy cache-service
  mark_resolved payment-service memory leak due to unbounded transaction cache causing OOM
""")


# ── Environment client ────────────────────────────────────────────────────────

class EnvClient:
    """HTTP client for the OnCallEnv API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str) -> dict:
        resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, command: str) -> dict:
        resp = requests.post(f"{self.base_url}/step", json={"command": command})
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
    """Format the current observation into a user prompt for the LLM."""
    alerts_text = ""
    for a in obs.get("alerts", []):
        alerts_text += f"  [{a['severity'].upper()}] {a['service']}: {a['message']}\n"

    prompt = f"Step {obs['step']}/{obs['max_steps']}\n\n"
    prompt += f"GOAL: {obs['goal']}\n\n"
    prompt += f"ACTIVE ALERTS:\n{alerts_text}\n"
    prompt += f"AVAILABLE SERVICES: {', '.join(obs.get('services', []))}\n\n"

    if obs.get('last_action'):
        prompt += f"LAST ACTION: {obs['last_action']}\n"
        prompt += f"RESULT:\n{obs.get('last_action_result', 'N/A')}\n"
        if obs.get('last_action_error'):
            prompt += "STATUS: ERROR — command failed\n"
        prompt += "\n"

    prompt += "What is your next command? Reply with EXACTLY ONE command."
    return prompt


def parse_action(response_text: str) -> str:
    """Extract a single command from the LLM response."""
    if not response_text:
        return "check_metrics api-gateway"

    valid_cmds = [
        "check_metrics", "check_logs", "check_config", "check_dependencies",
        "check_deploy_history", "restart_service", "rollback_deploy",
        "scale_service", "update_config", "mark_resolved",
    ]

    # Try each line for a valid command
    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove markdown backticks, leading symbols, quotes
        line = line.strip("`").strip("- ").strip("> ").strip("'\"").strip()
        for cmd in valid_cmds:
            if line.lower().startswith(cmd):
                return line

    # Fallback: return the first non-empty line cleaned up
    for line in response_text.strip().splitlines():
        line = line.strip().strip("`").strip("- ").strip("> ").strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            return line

    return "check_metrics api-gateway"


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, env: EnvClient, task_id: str) -> float:
    """Run the agent on a single task and return the final score."""
    print(f"[START] task={task_id} env=OnCallEnv model={MODEL_NAME}")

    obs = env.reset(task_id)
    done = False
    step_num = 0
    rewards_list = []

    # Conversation history for multi-turn reasoning
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while not done:
        step_num += 1
        user_prompt = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        # Call LLM with retry
        raw_action = ""
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw_action = response.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raw_action = "check_metrics api-gateway"

        action = parse_action(raw_action)
        messages.append({"role": "assistant", "content": action})

        # Step environment
        result = env.step(action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        
        step_reward = reward["total"] if isinstance(reward, dict) and "total" in reward else float(reward)
        rewards_list.append(f"{step_reward:.2f}")
        
        last_error = obs.get("last_action_error")
        error_str = str(last_error) if last_error else "null"
        print(f"[STEP] step={step_num} action={action} reward={step_reward:.2f} done={str(done).lower()} error={error_str}")

        if done:
            final_score = step_reward
            success = str(final_score > 0.0).lower()
            rewards_str = ",".join(rewards_list)
            print(f"[END] success={success} steps={step_num} score={final_score:.2f} rewards={rewards_str}")
            return final_score

        # Trim conversation history if getting too long (keep system + last 16 turns)
        if len(messages) > 34:  # system + 16 pairs
            messages = [messages[0]] + messages[-32:]

    # Shouldn't reach here, but just in case
    st = env.state()
    return st.get("score", 0.0)


def main():
    if not API_KEY:
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    scores: dict[str, float] = {}

    for task_id in TASKS:
        try:
            score = run_task(client, env, task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
            scores[task_id] = 0.0

if __name__ == "__main__":
    main()
