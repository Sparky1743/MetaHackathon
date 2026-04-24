#!/usr/bin/env bash
set -euo pipefail

openenv validate .
python -m pytest tests
python - <<'PY'
from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action

env = OnCallRedShiftEnv()
obs = env.reset(task_id="seed_easy_memory_leak")
assert obs.task_id == "seed_easy_memory_leak"
obs = env.step(Action(command="kubectl_get_pods"))
assert "payment-service" in obs.last_action_result
print("local smoke ok")
PY
