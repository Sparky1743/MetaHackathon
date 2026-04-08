Round 1 — Problem Statement

The Task

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard  step() / reset() / state()  API.

Key Requirements at a Glance

Must simulate a real-world task (not games or toys)

Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml

Minimum 3 tasks with agent graders (easy → medium → hard, scores 0.0–1.0)

Meaningful reward function with partial progress signals

Baseline inference script with reproducible scores

Deploy to Hugging Face Spaces + working Dockerfile

README with environment description, action/observation spaces, setup instructions

Detailed Requirements
Functional Requirements

Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.

Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.

Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

Baseline inference script

Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.

Detailed Requirements

Non-Functional Requirements

Deploys to a Hugging Face Space

Environment must run as a containerized HF Space tagged with openenv.

Containerized execution

Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.

Documentation

README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.

Evaluation Criteria
Parameter

Weight

Description

Real-world utility

30%

Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?

Task & grader quality

25%

Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression?

Environment design

20%

Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.

Code quality & spec compliance

15%

Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.

Creativity & novelty

10%

Novel problem domain, interesting mechanics, clever reward design, original approach.

Scoring Breakdown

Real-world utility (30%)

•  0–5: Toy/artificial problem with no practical application

•  6–15: Valid domain but shallow modeling of the real task

•  16–25: Good domain modeling, would be useful for agent evaluation

•  26–30: Excellent — fills a real gap, immediate value for the RL/agent community

Task & grader quality (25%)

•  3+ tasks with difficulty range?

•  Graders produce scores between 0.0–1.0?

•  Graders deterministic and reproducible?

•  Hard task genuinely challenges frontier models?

Environment design (20%)

•  reset() produces clean state?

•  Action/observation types well-designed and documented?

•  Reward function provides useful varying signal (not just sparse)?

•  Episode boundaries sensible?

Code quality & spec compliance (15%)

•  openenv validate passes?

•  docker build && docker run works?

•  HF Space deploys and responds?

•  Baseline script runs and reproduces scores?

Creativity & novelty (10%)

•  Domain we haven’t seen in OpenEnv before?

•  Reward design has interesting properties?

•  Clever mechanics that make the environment engaging?

Judging:
Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

Disqualification Criteria

Environment does not deploy or respond

Plagiarized or trivially modified existing environments

Graders that always return the same score

No baseline inference script


Pre-Submission Checklist  — all must pass or you're disqualified

HF Space deploys

Automated ping to the Space URL — must return 200 and respond to reset()

OpenEnv spec compliance

Validate openenv.yaml, typed models, step()/reset()/state() endpoints

Dockerfile builds

Automated docker build on the submitted repo

Baseline reproduces

Run the submitted inference script — must complete without error and produce scores

3+ tasks with graders

Enumerate tasks, run each grader, verify scores in 0.0–1.0 range

Additional Instructions

Before submitting, ensure the following variables are defined in your environment configuration:  

API_BASE_URL   The API endpoint for the LLM.  

MODEL_NAME     The model identifier to use for inference. 

HF_TOKEN       Your Hugging Face / API key.

The inference script must be named `inference.py` and placed in the root directory of the project

Participants must use OpenAI Client for all LLM calls using above variables

Infra Restrictions

Runtime of inference script should be less than 20min 

Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

Validator

Run the pre-submission validation script before submitting

Sample Inference Script

"""

def build_user_prompt(step: int, observation, history: List[str]) -> str:
    goal = observation.goal or "(not provided)"
    url = observation.url or "(unknown)"
    error_note = "Yes" if observation.last_action_error else "No"

    clickables = extract_clickable_elements(observation)
    if clickables:
        actions_hint = "\n".join(
            f"    - {item['bid']} (bbox: {item['bbox']})" for item in clickables
        )
    else:
        actions_hint = "    (none detected)"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Current URL: {url}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}
        Available clickable element IDs: {actions_hint}
        Reply with exactly one BrowserGym action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    # Prefer the first line that looks like an action string
    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            # Collapse internal whitespace
            action = re.sub(r"\s+", " ", action)
            # If the model tried to click by natural-language description while we
            # only exposed numeric BrowserGym IDs, fallback to the single detected ID.
            return action

    # Fall back to searching the whole response
    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = BrowserGymEnv.from_docker_image(
        image="browsergym-env:latest",
Pre Validation Script

#!/usr/bin/env bash
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0

Prerequisites:
Install before April 1st.

Python 3.10+

Install 3.10, 3.11, or 3.12.

$
python --version
Copy
Git + GitHub account

Push your submission to GitHub or HF.

$
git --version
Copy
Hugging Face CLI

Deploy to HF Spaces.

$
pip install huggingface_hub --version
Copy
$
huggingface-cli login
Copy
OpenEnv

The framework.

$
pip install openenv-core
Copy
Google Colab

Prep course runs in Colab. Free tier works.

$
pip install openenv-core
Copy
OpenEnv

The framework.

→ colab.research.google.com
Copy
Docker

Isolated container testing.

docker --version
Copy
Recommended

VS Code

Best Python + Docker support
