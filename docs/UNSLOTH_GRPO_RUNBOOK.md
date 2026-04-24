# Unsloth GRPO Training Runbook

This is the practical runbook for launching the real LLM training track while campus GPU access is temporarily unavailable.

The training code is already in the repo:

- Trainer: `scripts/train_unsloth_grpo.py`
- LLM dependencies: `requirements-llm.txt`
- Campus `screen` launcher: `scripts/run_unsloth_grpo_screen.sh`

## What This Run Trains

This run fine-tunes an instruction model with **Unsloth + TRL GRPO**.

The model is prompted with an SRE incident and must emit simulator commands inside:

```text
<actions>
kubectl_logs payment-service
kubectl_rollout_restart payment-service
declare_resolved
</actions>
```

Each completion is parsed into real `OnCallRedShiftEnv` actions. The environment executes those commands and returns the OpenEnv reward. GRPO then optimizes the model toward higher-reward incident-response action plans.

Default model:

```text
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

Why this model:

- Small enough for V100/T4-class GPUs.
- Already 4-bit quantized for memory.
- Strong enough to follow command-format instructions.

## Expected Runtime

Approximate estimates:

| Hardware | Suggested settings | Expected time |
| --- | --- | ---: |
| Campus V100 32GB | 1200 steps, 206 tasks, 4 generations | 3-8 hours |
| Kaggle T4 x2 | 400-800 steps, 120-160 tasks, 2-4 generations | 4-9 hours |
| Kaggle P100 | 300-600 steps, 80-120 tasks, 2 generations | 5-10 hours |

These are estimates. The first run can spend extra time installing packages and downloading the model.

## Option A: Kaggle Notebook

Use this if campus VPN is unavailable.

### 1. Create Notebook

In Kaggle:

1. Create a new notebook.
2. Turn on GPU in notebook settings.
3. If possible, choose T4 x2. P100 is also usable with smaller settings.
4. Turn Internet on.

### 2. Get The Code Into Kaggle

Best path if the branch is pushed:

```bash
!git clone -b round2-redshift https://github.com/<your-org-or-user>/<your-repo>.git /kaggle/working/MetaHackathon-R2
%cd /kaggle/working/MetaHackathon-R2
```

If the branch is not pushed, upload a zip of the repo as a Kaggle Dataset, then use:

```bash
!cp -r /kaggle/input/<dataset-name>/MetaHackathon-R2 /kaggle/working/MetaHackathon-R2
%cd /kaggle/working/MetaHackathon-R2
```

### 3. Install Dependencies

Run this cell:

```bash
!python -m pip install -U pip setuptools wheel
!python -m pip install -r requirements.txt
!python -m pip install -r requirements-llm.txt
```

If Kaggle has a CUDA/PyTorch conflict, try installing the LLM stack manually:

```bash
!python -m pip install -U transformers datasets accelerate trl peft bitsandbytes unsloth
```

### 4. Verify Environment

Run:

```bash
!export PYTHONPATH=src:scripts && python -m pytest tests -q
!export PYTHONPATH=src && openenv validate .
```

Expected:

```text
21 passed
[OK] : Ready for multi-mode deployment
```

### 5. Start A Kaggle-Sized GRPO Run

For a Kaggle T4, start with this:

```bash
!export PYTHONPATH=src:scripts && \
export WANDB_DISABLED=true && \
export TOKENIZERS_PARALLELISM=false && \
python scripts/train_unsloth_grpo.py \
  --model-name unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
  --curriculum-buffer curriculum_results/buffer.json \
  --out-dir training_results/unsloth_grpo_kaggle \
  --max-tasks 120 \
  --max-steps 500 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-generations 2 \
  --max-seq-length 1024 \
  --max-prompt-length 768 \
  --max-completion-length 192 \
  --eval-tasks 24 \
  --lr 5e-6 \
  --save-steps 100 \
  --logging-steps 5
```

If memory is fine on T4 x2, increase:

```text
--max-steps 800
--max-tasks 160
--num-generations 4
--max-seq-length 1536
```

If P100 or memory is tight, reduce:

```text
--max-steps 300
--max-tasks 80
--num-generations 2
--max-seq-length 768
--max-prompt-length 512
--max-completion-length 128
```

### 6. Save Kaggle Outputs

Run after training:

```bash
!tar -czf /kaggle/working/unsloth_grpo_kaggle_results.tar.gz training_results/unsloth_grpo_kaggle
!ls -lh /kaggle/working/unsloth_grpo_kaggle_results.tar.gz
```

Download this archive from Kaggle outputs.

Expected important files:

```text
training_results/unsloth_grpo_kaggle/summary.json
training_results/unsloth_grpo_kaggle/baseline_generations.json
training_results/unsloth_grpo_kaggle/trained_generations.json
training_results/unsloth_grpo_kaggle/adapter/
```

### 7. Interpret Results

Open:

```bash
!cat training_results/unsloth_grpo_kaggle/summary.json
```

Important fields:

- `baseline_mean_reward`
- `trained_mean_reward`
- `adapter_dir`
- `duration_sec`
- `num_train_tasks`
- `num_eval_tasks`

For the submission story, we want:

```text
trained_mean_reward > baseline_mean_reward
```

Even a modest LLM improvement is useful because the existing symbolic curriculum policy already provides the strong quantitative baseline.

## Option B: Campus GPU With Screen

Use this when VPN/server access returns.

### 1. Copy Repo To Server

From your laptop/local machine, once SSH works:

```bash
rsync -az --exclude .git --exclude .venv --exclude __pycache__ \
  /home/xiaofeng/projects/MetaHackthonR2/MetaHackathon-R2/ \
  himanshubeniwal@lingo-lexico.iitgn.ac.in:/mnt/himanshu/MetaHackathon-R2/
```

If the hostname still fails, use the correct IP or campus-resolvable hostname.

### 2. SSH And Check GPUs

```bash
ssh himanshubeniwal@lingo-lexico.iitgn.ac.in -p 2020
nvidia-smi
```

Pick free GPU IDs. Do not kill any existing processes.

### 3. Start Long Run In Screen

From the repo directory on the server:

```bash
cd /mnt/himanshu/MetaHackathon-R2
chmod +x scripts/run_unsloth_grpo_screen.sh
GPU_IDS=1 SESSION_NAME=redshift_unsloth_grpo ./scripts/run_unsloth_grpo_screen.sh
```

For multiple visible GPUs:

```bash
GPU_IDS=1,3 SESSION_NAME=redshift_unsloth_grpo ./scripts/run_unsloth_grpo_screen.sh
```

The default campus run uses:

```text
max steps: 1200
max tasks: 206
num generations: 4
model: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

### 4. Detach Safely

Inside screen:

```text
Ctrl-a then d
```

The job keeps running after laptop disconnect.

### 5. Monitor

```bash
screen -r redshift_unsloth_grpo
tail -f /mnt/himanshu/oncallenv-redshift-runs/<run-id>/train.log
watch -n 5 nvidia-smi
```

### 6. Find Results

The launcher prints the run directory. It will look like:

```text
/mnt/himanshu/oncallenv-redshift-runs/unsloth-grpo-YYYYMMDD-HHMMSS
```

Important files:

```text
train.log
repo/training_results/unsloth_grpo/summary.json
repo/training_results/unsloth_grpo/baseline_generations.json
repo/training_results/unsloth_grpo/trained_generations.json
repo/training_results/unsloth_grpo/adapter/
unsloth_grpo_results.tar.gz
```

To fetch results:

```bash
scp -P 2020 \
  himanshubeniwal@lingo-lexico.iitgn.ac.in:/mnt/himanshu/oncallenv-redshift-runs/<run-id>/unsloth_grpo_results.tar.gz \
  .
```

## Option C: Give This To Someone On Campus

Send them:

1. The repo branch or zip.
2. This runbook.
3. The campus command:

```bash
cd /mnt/himanshu/MetaHackathon-R2
chmod +x scripts/run_unsloth_grpo_screen.sh
GPU_IDS=1 SESSION_NAME=redshift_unsloth_grpo ./scripts/run_unsloth_grpo_screen.sh
```

Ask them to send back:

```text
/mnt/himanshu/oncallenv-redshift-runs/<run-id>/unsloth_grpo_results.tar.gz
/mnt/himanshu/oncallenv-redshift-runs/<run-id>/train.log
```

## Troubleshooting

### CUDA out of memory

Reduce:

```text
--per-device-train-batch-size 1
--num-generations 2
--max-seq-length 768
--max-completion-length 128
```

### Unsloth install fails

Try:

```bash
python -m pip install -U transformers datasets accelerate trl peft bitsandbytes
python -m pip install -U unsloth
```

### Model download fails

Check internet access and Hugging Face availability. For gated models, login:

```bash
huggingface-cli login
```

The default Qwen model should not require gated access.

### GRPOTrainer argument mismatch

The trainer script already checks whether the installed TRL version expects `processing_class` or `tokenizer`.

### Kaggle disconnect

Kaggle generally keeps the notebook session running for a while after browser disconnect, but it is not as durable as `screen` on the campus server. Save outputs frequently and prefer the campus `screen` run when access returns.

## Recommended Plan While VPN Is Down

1. Run the Kaggle-sized job first: 300-500 steps.
2. Confirm `summary.json` is created and rewards are nonzero.
3. If it works, run a longer Kaggle job.
4. When campus VPN returns, launch the full 1200-step screen job.
5. Pull the final archive and commit:
   - `training_results/unsloth_grpo/summary.json`
   - `baseline_generations.json`
   - `trained_generations.json`
   - LoRA adapter if size is acceptable
   - Updated plots/results in README

