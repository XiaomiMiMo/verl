#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${MODEL_PATH:?MODEL_PATH must point to a policy checkpoint}"
: "${TRAIN_DATA:?TRAIN_DATA must point to a training parquet}"
: "${VAL_DATA:?VAL_DATA must point to a validation parquet}"

export MIMOAGENT_SRC="${MIMOAGENT_SRC:-${REPO_ROOT}/third_party/mimoagent-osr}"
export MIMOAGENT_HARNESS_SPEC="${MIMOAGENT_HARNESS_SPEC:-${REPO_ROOT}/config/agent/code/mix-four-whitebox.yaml}"
export MIXED_HARNESS_MODE="${MIXED_HARNESS_MODE:-step-hash}"
export MIXED_HARNESS_SEED="${MIXED_HARNESS_SEED:-20260911}"

export N="${N:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
export MAXLEN="${MAXLEN:-262144}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
export HARNESS_TURN_MAX_TOKENS="${HARNESS_TURN_MAX_TOKENS:-32768}"
export MODEL_REQUEST_TIMEOUT="${MODEL_REQUEST_TIMEOUT:-3600}"
export MODEL_SDK_MAX_RETRIES="${MODEL_SDK_MAX_RETRIES:-0}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"

export LOSS_AGG_MODE="${LOSS_AGG_MODE:-prompt-mean}"
export NORM_ADV_BY_STD_IN_GRPO="${NORM_ADV_BY_STD_IN_GRPO:-False}"
export FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-True}"
export FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-reward}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
export ENTROPY_CHUNKING="${ENTROPY_CHUNKING:-True}"
export ENTROPY_CHUNK_SIZE="${ENTROPY_CHUNK_SIZE:-16384}"
export MICRO_BSZ_PER_GPU="${MICRO_BSZ_PER_GPU:-1}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-2}"
export MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-drop}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-error}"

export VAL_N="${VAL_N:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-500}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-${ROLLOUT_TEMPERATURE}}"
export VAL_TOP_P="${VAL_TOP_P:-${ROLLOUT_TOP_P}}"
export VAL_TOP_K="${VAL_TOP_K:-${ROLLOUT_TOP_K}}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"

export TRAIN_NNODES="${TRAIN_NNODES:-8}"
export TRAIN_NGPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE:-8}"
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-0}"
export ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-8}"
export ACTOR_TP="${ACTOR_TP:-8}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ACTOR_EP="${ACTOR_EP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export MEGATRON_OFFLOAD="${MEGATRON_OFFLOAD:-True}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.78}"

export REPETITION_DETECT_ENABLE="${REPETITION_DETECT_ENABLE:-true}"
export REPETITION_ZERO_REWARD="${REPETITION_ZERO_REWARD:-false}"
export REPETITION_PENALTY_ENABLE="${REPETITION_PENALTY_ENABLE:-false}"
export TOOL_CALL_ERROR_PENALTY_ENABLE="${TOOL_CALL_ERROR_PENALTY_ENABLE:-false}"
export DEEP_FAILURE_MASK_ENABLE="${DEEP_FAILURE_MASK_ENABLE:-false}"

export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-64}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
export MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-2048}"
export UNI_AGENT_RUNNER_TASK_NUM_CPUS="${UNI_AGENT_RUNNER_TASK_NUM_CPUS:-0.5}"
export ROLLOUT_MAX_RUNNING_REQUESTS="${ROLLOUT_MAX_RUNNING_REQUESTS:-128}"
export TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-4800}"
export AGENT_REQUEST_IDLE_TIMEOUT="${AGENT_REQUEST_IDLE_TIMEOUT:-720}"

export PROJECT_NAME="${PROJECT_NAME:-opensource-code}"
export EXP_NAME="${EXP_NAME:-four-whitebox}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}/${RUN_ID}}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:--1}"

if [[ "${MIMOAGENT_HARNESS_SPEC}" -ef "${REPO_ROOT}/config/agent/code/mix-four-whitebox.yaml" ]]; then
  mkdir -p "${RUN_DIR}"
  HARNESS_DIR="$(mktemp -d "${RUN_DIR}/harness.XXXXXX")"
  cp "${REPO_ROOT}/config/agent/code/"*.yaml "${HARNESS_DIR}/"
  for arm in mini-mimocode mini-bash mini-claude-code mini-codex; do
    token_key=max_tokens
    [[ "${arm}" != mini-codex ]] || token_key=max_output_tokens
    sed -i -E \
      -e '/^  model_kwargs:/,/^[^ ]/{ /^    (max_tokens|max_output_tokens|timeout|max_retries):/d; }' \
      -e "/^  model_kwargs:$/a\\    ${token_key}: ${HARNESS_TURN_MAX_TOKENS}\n    timeout: ${MODEL_REQUEST_TIMEOUT}\n    max_retries: ${MODEL_SDK_MAX_RETRIES}" \
      "${HARNESS_DIR}/${arm}.yaml"
  done
  export MIMOAGENT_HARNESS_SPEC="${HARNESS_DIR}/mix-four-whitebox.yaml"
fi

printf '%s\n' \
  "[code] repo          = ${REPO_ROOT}" \
  "[code] model         = ${MODEL_PATH}" \
  "[code] train / val   = ${TRAIN_DATA} / ${VAL_DATA}" \
  "[code] mimoagent     = ${MIMOAGENT_SRC}" \
  "[code] harness mix   = ${MIMOAGENT_HARNESS_SPEC} (${MIXED_HARNESS_MODE}, seed ${MIXED_HARNESS_SEED})" \
  "[code] topology      = train ${TRAIN_NNODES}x${TRAIN_NGPUS_PER_NODE}, rollout ${ROLLOUT_NNODES}x${ROLLOUT_NGPUS_PER_NODE}, tp ${ACTOR_TP} pp ${ACTOR_PP} cp ${ACTOR_CP}" \
  "[code] batch         = ${TRAIN_BATCH_SIZE} x n${N}, ${TOTAL_STEPS} steps"

exec bash "${REPO_ROOT}/recipes/code/run_train.sh" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_RUNNER_TASK_NUM_CPUS='${UNI_AGENT_RUNNER_TASK_NUM_CPUS}'" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_EMPTY_CACHE_BEFORE_OPTIM="1"' \
  trainer.resume_from_path=null \
  "$@"
