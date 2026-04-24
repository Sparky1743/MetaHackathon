#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-redshift_unsloth_grpo}"
GPU_IDS="${GPU_IDS:-1}"
RUN_ROOT="${RUN_ROOT:-/mnt/himanshu/oncallenv-redshift-runs}"
RUN_ID="${RUN_ID:-unsloth-grpo-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
REPO_SRC="$(pwd)"

mkdir -p "${RUN_DIR}"
rsync -a \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  "${REPO_SRC}/" "${RUN_DIR}/repo/"

cat > "${RUN_DIR}/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/repo"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:scripts
export HF_HOME="${HF_HOME:-/mnt/himanshu/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/himanshu/hf-cache}"

echo "=== Red Shift Unsloth GRPO run ==="
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-llm.txt

python -m pytest tests -q
openenv validate .

python scripts/train_unsloth_grpo.py \
  --model-name "${MODEL_NAME:-unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit}" \
  --curriculum-buffer curriculum_results/buffer.json \
  --out-dir training_results/unsloth_grpo \
  --max-tasks "${MAX_TASKS:-206}" \
  --max-steps "${MAX_STEPS:-1200}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-1536}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-1024}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-256}" \
  --eval-tasks "${EVAL_TASKS:-48}" \
  --lr "${LR:-5e-6}" \
  --save-steps "${SAVE_STEPS:-100}" \
  --logging-steps "${LOGGING_STEPS:-5}"

tar -czf "../unsloth_grpo_results.tar.gz" training_results/unsloth_grpo
echo "=== completed ==="
date
EOF

chmod +x "${RUN_DIR}/run.sh"

if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "screen session '${SESSION_NAME}' already exists. Attach with: screen -r ${SESSION_NAME}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" screen -dmS "${SESSION_NAME}" bash -lc "cd '${RUN_DIR}' && ./run.sh > train.log 2>&1"

cat <<EOF
Started Unsloth GRPO training in screen.

Session: ${SESSION_NAME}
Run dir: ${RUN_DIR}
GPU ids: ${GPU_IDS}

Monitor:
  screen -r ${SESSION_NAME}
  tail -f ${RUN_DIR}/train.log
  watch -n 5 nvidia-smi

Detach from screen:
  Ctrl-a then d

Expected result archive:
  ${RUN_DIR}/unsloth_grpo_results.tar.gz
EOF
