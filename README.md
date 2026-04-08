# OnCallEnv — Production Incident Response Environment

**An OpenEnv-compliant environment where AI agents diagnose and remediate real-world production infrastructure incidents.**

> **Status:** 17/17 tests passing | 6 scenarios (easy → expert) | Dynamic metrics | Alert escalation | Multi-root-cause | Docker-ready

---

## Table of Contents

- [Motivation](#motivation)
- [Environment Overview](#environment-overview)
- [Tasks](#tasks)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Reward Function](#reward-function)
- [Baseline Scores](#baseline-scores)
- [Setup & Usage](#setup--usage)
- [Example Agent Transcript](#example-agent-transcript)
- [Evaluation Criteria](#evaluation-criteria)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Motivation

Every technology company runs on-call rotations. When production breaks at 3 AM, an engineer must rapidly triage alerts, trace dependency chains, correlate signals across services, identify root causes, and apply correct fixes — all under time pressure.

OnCallEnv simulates this exact workflow. It models a realistic microservice architecture with monitoring, logging, and configuration systems. Agents receive alerts and must interactively investigate and fix the incident using the same toolkit a human SRE would use.

**Why this matters:**

1. **Real operational skill gap** — Most AI benchmarks test coding or QA. OnCallEnv tests *operational reasoning* — the ability to diagnose complex systems from partial, noisy signals across multiple components.

2. **Partial observability** — Unlike many environments, the agent doesn't see everything upfront. It must actively investigate, making it a true exploration problem.

3. **Multi-hop reasoning** — Hard tasks require correlating signals across 6+ services to find a root cause that's invisible at the symptom level.

4. **Multi-root-cause diagnosis** — The expert task features two independent failures happening simultaneously, requiring the agent to identify and fix both.

5. **Dynamic, evolving incidents** — Metrics degrade over time for unhealthy services, and alert escalation injects urgency after prolonged inaction — just like real incidents.

6. **Immediate practical value** — Companies could use this to evaluate AI copilots for on-call, reducing MTTR (Mean Time To Resolution) for production incidents.

---

## Environment Overview

### Architecture

```
                         ┌─────────────┐
                         │ API Gateway │
                         └──────┬──────┘
                    ┌───────────┼───────────┐
              ┌─────▼─────┐ ┌──▼───┐ ┌─────▼──────┐
              │  Order    │ │ User │ │  Product   │
              │  Service  │ │ Svc  │ │  Service   │
              └─────┬─────┘ └──┬───┘ └─────┬──────┘
                    │          │            │
              ┌─────▼──────────▼────────────▼──────┐
              │        Infrastructure Layer         │
              │  PostgreSQL │ Redis │ Cache Service  │
              └────────────────────────────────────┘
```

The environment simulates a microservice backend with:
- **API Gateway** — Routes requests to downstream services
- **Business services** — Order, Payment, Product, User, Search, Notification, Inventory
- **Infrastructure** — PostgreSQL (primary + replica), Redis, cache service
- **Monitoring** — Metrics (CPU, memory, latency, error rate), structured logs, alert system
- **Configuration** — Service configs, deployment history, dependency maps

### How It Works

1. Agent receives initial **alerts** describing the symptoms
2. Agent uses **investigation commands** to gather data (metrics, logs, configs, dependencies, deploy history)
3. **Metrics degrade dynamically** — unhealthy services get worse each step until the fix is applied
4. **Alert escalation** — if the agent hasn't fixed the issue after 50% of max steps, ESCALATION alerts appear with increasing urgency
5. Agent reasons about root cause by correlating signals across services
6. Agent applies **remediation** (restart, rollback, config change, scaling)
7. For **multi-root-cause** scenarios, the agent must fix ALL issues — progress is tracked incrementally (`[1/2 issues fixed]`)
8. Agent calls **mark_resolved** with a root cause description to close the incident (2-step grace period after fix)

---

## Tasks

### Task 1: Memory Leak in Payment Service (Easy)
- **Difficulty:** Easy | **Max steps:** 10
- **Scenario:** `payment-service` is experiencing repeated OOM kills. Alerts are clear and point directly to the affected service.
- **Expected approach:** Check logs → see OOM errors → restart service → mark_resolved
- **Challenge:** Minimal — tests basic investigation and remediation skills
- **Key signal:** payment-service at 94.7% memory, 7 OOM kills/hour

### Task 2: Cascading Connection Pool Exhaustion (Medium)
- **Difficulty:** Medium | **Max steps:** 15
- **Scenario:** `api-gateway` reports timeouts, but the root cause is a config change that reduced the DB connection pool size in `order-service` from 50 to 5, causing starvation that cascades upstream.
- **Expected approach:** Check gateway → trace to order-service → find config change → update config → mark_resolved
- **Challenge:** Requires following the dependency chain and identifying a configuration error
- **Key signal:** `db_pool_size changed: 50 -> 5` in order-service logs

### Task 3: Subtle Cache Bug Causing Cross-Service Degradation (Hard)
- **Difficulty:** Hard | **Max steps:** 20
- **Scenario:** Multiple services show slightly elevated latency with no single critical failure. A recent `cache-service` deployment changed the key hashing algorithm (FNV1a → MurmurHash3), causing ~60% cache miss rate.
- **Expected approach:** Notice cross-service pattern → correlate cache miss rates → check cache deploy history → rollback → mark_resolved
- **Challenge:** No single service is "down." The cache-service itself reports healthy. Requires correlating subtle signals across 6+ services.
- **Key signal:** 60%+ cache miss rate across all services, cache-service deploy notes mention key format change

### Task 4: DNS Misconfiguration Causing Intermittent Failures (Medium)
- **Difficulty:** Medium | **Max steps:** 15
- **Scenario:** `order-service` intermittently fails to reach `inventory-service`. A config change pointed it at a decommissioned DNS name. DNS caching causes some requests to succeed while others fail.
- **Expected approach:** Check order-service logs → see DNS resolution failures → check config → fix hostname → mark_resolved
- **Challenge:** Intermittent nature makes it harder to diagnose — failures appear random at first glance
- **Key signal:** `inventory_host: inventory-service-v2.internal` in config (should be `inventory-service.internal`)

### Task 5: Database Replication Lag from Runaway Batch Job (Hard)
- **Difficulty:** Hard | **Max steps:** 20
- **Scenario:** Multiple services return stale data. A batch analytics job running during peak hours causes massive write amplification on postgres-primary, pushing replica lag to 45+ seconds.
- **Expected approach:** Notice stale-read pattern → check DB metrics → discover batch job → check config → disable batch job → mark_resolved
- **Challenge:** No errors, no downtime, no latency increase — data is just stale. Requires understanding database replication and correlating write load with read staleness.
- **Key signal:** `replication_lag_sec: 47.0`, `batch_job_write_ops: 4400` vs baseline 800

### Task 6: Simultaneous Bad Deployment and Config Drift (Expert)
- **Difficulty:** Expert | **Max steps:** 25
- **Scenario:** Two independent failures at once: (1) `search-service` deployed v3.1.0 with a broken Elasticsearch query builder (v2 DSL on a v7.x cluster), causing 95% error rate, AND (2) a capacity-planner bot reduced `order-service` DB pool from 50 to 3, causing connection timeouts.
- **Expected approach:** Investigate both failure chains independently → rollback search-service → update order-service config → mark_resolved with both root causes
- **Challenge:** The agent must recognize these are **two separate issues**, not one cascading failure. Fixing only one leaves the system partially broken. Progress is tracked incrementally (`[1/2 issues fixed]`).
- **Key signals:** `search-service` 95% error rate + `ElasticsearchParseException`, `order-service` `db_pool_size: 3` + 245 pending requests

---

## Action Space

| Command | Description |
|---------|-------------|
| `check_metrics <service>` | View CPU, memory, latency, error rates, custom metrics |
| `check_logs <service>` | View recent structured log entries |
| `check_config <service>` | View current service configuration |
| `check_dependencies <service>` | View upstream/downstream dependency graph |
| `check_deploy_history <service>` | View recent deployments and changes |
| `restart_service <service>` | Restart a service process |
| `rollback_deploy <service>` | Roll back to previous deployment version |
| `scale_service <service> <n>` | Scale service to n replicas |
| `update_config <service> <key> <value>` | Update a configuration parameter |
| `mark_resolved <description>` | Mark incident resolved with root cause (REQUIRED for full score) |

---

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `goal` | string | Description of what needs to be fixed |
| `step` | int | Current step number |
| `max_steps` | int | Maximum allowed steps |
| `alerts` | list[Alert] | Active monitoring alerts with severity/service/message |
| `services` | list[string] | All available service names |
| `last_action` | string | The command the agent just executed |
| `last_action_result` | string | Output from that command |
| `last_action_error` | bool | Whether the command failed |
| `available_commands` | list[string] | All valid commands |

---

## Reward Function

The reward provides continuous signal (not just binary success/failure):

| Component | Weight | Description |
|-----------|--------|-------------|
| Investigation | 0.30 | Checking relevant services (bonus for root cause service + deep investigation) |
| Root Cause | 0.25 | Correctly identifying what went wrong via `mark_resolved` |
| Remediation | 0.30 | Applying the correct fix (multi-fix scenarios require ALL fixes) |
| Efficiency | 0.15 | Fewer steps = higher bonus (full at ≤40% of max_steps) |
| Penalties | -0.05/ea | Wrong destructive actions on healthy services (capped at -0.15) |

**Per-step reward signals:** The environment provides incremental feedback — each new service investigated adds to the investigation score, enabling RL-style training with dense rewards rather than sparse terminal signals.

**Scoring examples:**
- Perfect run (investigate → fix → mark_resolved efficiently): **0.95-1.0**
- Good run (fix applied, no mark_resolved): **0.60-0.75**
- Partial (investigated correctly but wrong fix): **0.30-0.40**
- Random actions: **0.05-0.15**

---

## Baseline Scores

### Optimal Agent (scripted, deterministic)

| Task | Score | Steps | Investigation | Root Cause | Remediation | Efficiency |
|------|-------|-------|---------------|------------|-------------|------------|
| easy_memory_leak | **0.950** | 5/10 | 0.300 | 0.250 | 0.300 | 0.100 |
| medium_cascading_failure | **0.940** | 8/15 | 0.290 | 0.250 | 0.300 | 0.100 |
| hard_cache_degradation | **0.950** | 9/20 | 0.300 | 0.250 | 0.300 | 0.100 |
| medium_dns_misconfiguration | **0.950** | 7/15 | 0.300 | 0.250 | 0.300 | 0.100 |
| hard_replication_lag | **0.950** | 10/20 | 0.300 | 0.250 | 0.300 | 0.100 |
| expert_multi_root_cause | **0.950** | 12/25 | 0.300 | 0.250 | 0.300 | 0.100 |
| **Average** | **0.948** | | | | | |

### LLM Baseline (Gemini 2.0 Flash)

| Task | Score | Notes |
|------|-------|-------|
| easy_memory_leak | 0.910 | Correct diagnosis and fix with mark_resolved |
| medium_cascading_failure | 0.790 | Found correct fix; sometimes misses `mark_resolved` |
| hard_cache_degradation | 0.730 | Solved late; cache correlation is hardest |
| medium_dns_misconfiguration | 0.880 | Config fix found reliably |
| hard_replication_lag | 0.840 | Batch job identified after DB investigation |
| **Average (5 tasks)** | **0.854** | |

The improved inference.py adds multi-turn conversation history, explicit `mark_resolved` guidance, stronger investigation prompting, and multi-root-cause awareness.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (optional)
- OpenAI-compatible API key (OpenAI, Gemini, etc.)

### Local Development

```bash
# Clone the repository
git clone https://github.com/srimanreddy4/MetaHackathon.git
cd MetaHackathon

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# Server runs on http://localhost:7860

# Run tests
python test_env.py

# Run simulation (all 6 tasks with optimal agent)
python run_simulation.py
```

### Docker

```bash
docker build -t oncall-env .
docker run -p 7860:7860 oncall-env
```

### Running Inference

```bash
# Gemini
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash
export HF_TOKEN=your-gemini-api-key
export ENV_URL=http://localhost:7860
python inference.py

# OpenAI
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=sk-your-key-here
export ENV_URL=http://localhost:7860
python inference.py
```

### API Examples

```bash
# Health check
curl http://localhost:7860/

# List all 6 tasks
curl http://localhost:7860/tasks

# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_memory_leak"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"command": "check_logs payment-service"}'

# Check state
curl http://localhost:7860/state
```

---

## Example Agent Transcript

**Task: medium_cascading_failure** (Cascading Connection Pool Exhaustion)

```
Step 1: check_metrics api-gateway
  → CPU: 55%, Error rate: 35%, Latency p99: 8500ms
  → Observation: Gateway is clearly overloaded, need to trace upstream

Step 2: check_logs api-gateway
  → "Circuit breaker OPEN for order-service"
  → "Upstream order-service timeout after 5000ms"
  → Observation: order-service is the bottleneck

Step 3: check_metrics order-service
  → db_pool_active: 5/5, db_pool_pending: 312
  → CPU: 80%, Error rate: 42%
  → Observation: DB pool completely saturated

Step 4: check_logs order-service
  → "db_pool_size changed: 50 -> 5" (config reload)
  → "Connection pool exhausted — 312 requests queued"
  → ROOT CAUSE FOUND: config change reduced pool size

Step 5: check_config order-service
  → db_pool_size: 5 (confirms the issue)
  → config_modified_by: deploy-bot (auto-scaler rule #47)

Step 6: update_config order-service db_pool_size 50
  → "Configuration updated. Service reloading... Metrics stabilizing."
  → FIX APPLIED

Step 7: mark_resolved order-service connection pool exhausted due to config change reducing db_pool_size from 50 to 5
  → "Incident marked as resolved. Root cause correctly identified."
  → INCIDENT CLOSED

Final Score: 0.91 (investigation: 0.26, root_cause: 0.25, remediation: 0.30, efficiency: 0.10)
```

---

## Evaluation Criteria

OnCallEnv is designed to test agents on:

| Skill | How It's Tested |
|-------|----------------|
| **Alert triage** | Parsing alerts to identify which services to investigate first |
| **Systematic investigation** | Using check_metrics/logs/config to gather evidence |
| **Dependency reasoning** | Following service dependency chains to find root cause |
| **Pattern recognition** | Correlating signals across multiple services (especially hard tasks) |
| **Correct remediation** | Choosing the right fix (restart vs rollback vs config change) |
| **Root cause articulation** | Describing the issue accurately via mark_resolved |
| **Efficiency** | Solving with minimal steps (not over-investigating or trying random fixes) |

**Common failure modes:**
- Never calling `mark_resolved` (loses 0.25 per task)
- Restarting healthy services instead of tracing the dependency chain
- Identifying symptoms but not the root cause (e.g., "DB is slow" instead of "cache deployment broke key hashing")
- Over-investigating (checking every service wastes steps and hurts efficiency)

---

## Project Structure

```
MetaHackathon/
├── app.py              # FastAPI server (REST API endpoints)
├── environment.py      # Core environment logic (step/reset/state/reward)
├── models.py           # Pydantic typed models (Observation, Action, Reward)
├── scenarios.py        # 6 task scenario definitions (easy → expert)
├── graders.py          # Independent deterministic grading functions (1 per task)
├── inference.py        # Baseline inference script (OpenAI/Gemini client)
├── openenv.yaml        # OpenEnv metadata specification
├── pyproject.toml      # Python project metadata (openenv validate compatible)
├── server/             # Server entry point module for pyproject.toml
├── test_env.py         # 17 unit tests (scenarios, mark_resolved, edge cases)
├── run_simulation.py   # Simulation runner for all 6 tasks
├── Dockerfile          # Container definition for HF Spaces
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Troubleshooting

**Server won't start:**
- Ensure port 7860 is free: `lsof -i :7860`
- Install dependencies: `pip install -r requirements.txt`

**Inference script fails:**
- Check API key is set: `echo $HF_TOKEN`
- Verify server is running: `curl http://localhost:7860/`
- For Gemini, use base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`

**Low scores:**
- Most common issue: agent never calls `mark_resolved` (loses 0.25/task)
- Check if agent is using conversation history (single-turn = poor reasoning)
- Increase max_tokens (200 is too low for reasoning models)

**Tests fail:**
- Run `python test_env.py` — all 17 tests should pass
- If grader tests fail, check that `graders.py` has entries for all 6 tasks
