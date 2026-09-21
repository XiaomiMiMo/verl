#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export WEBDEV_MODE="${WEBDEV_MODE:-train}"
case "${WEBDEV_MODE}" in
  train|eval) ;;
  *) echo "WEBDEV_MODE must be 'train' or 'eval', got '${WEBDEV_MODE}'" >&2; exit 1 ;;
esac

MISSING=()
for v in MODEL_PATH TRAIN_DATA VAL_DATA KUBECONFIG POD_PROXY; do
  [ -n "${!v:-}" ] || MISSING+=("$v")
done
if [ "${WEBDEV_MODE}" = "train" ]; then
  for v in DESIGN_GRADER_URL LLM_JUDGE_API_KEY; do
    [ -n "${!v:-}" ] || MISSING+=("$v")
  done
else
  for v in WEBDEV_EVAL_JUDGE_BASE_URL WEBDEV_EVAL_JUDGE_API_KEY WEBDEV_EVAL_JUDGE_MODEL; do
    [ -n "${!v:-}" ] || MISSING+=("$v")
  done
fi
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "missing required environment variables: ${MISSING[*]}" >&2
  echo "usage is in the header of $0" >&2
  exit 1
fi
[ -f "${KUBECONFIG}" ] || { echo "KUBECONFIG does not exist: ${KUBECONFIG}" >&2; exit 1; }


export N="${N:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export MAXLEN="${MAXLEN:-262144}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"
export ROLLOUT_MAX_RUNNING_REQUESTS="${ROLLOUT_MAX_RUNNING_REQUESTS:-64}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"

export TRAIN_NNODES="${TRAIN_NNODES:-8}"
export TRAIN_NGPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE:-8}"
export ACTOR_TP="${ACTOR_TP:-8}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"

export TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-0}"
export ENV_SETUP_TIMEOUT="${ENV_SETUP_TIMEOUT:-900}"
export REWARD_TIMEOUT="${REWARD_TIMEOUT:-1500}"
export ENV_NUM_CPUS="${ENV_NUM_CPUS:-0.125}"
export FAIL_ON_ENV_SETUP_ERROR="${FAIL_ON_ENV_SETUP_ERROR:-False}"
export INVALID_REWARD_FOR_INFRA="${INVALID_REWARD_FOR_INFRA:-False}"

export WEBDEV_GRADE_HTTP="${WEBDEV_GRADE_HTTP:-1}"
export DROP_INFRA_FROM_GROUP="${DROP_INFRA_FROM_GROUP:-1}"
export WEBDEV_GRADE_MODE="${WEBDEV_GRADE_MODE:-${WEBDEV_MODE}}"

export PROJECT_NAME="${PROJECT_NAME:-opensource-design}"
export EXP_NAME="${EXP_NAME:-webdev-${WEBDEV_MODE}}"
case "${EXP_NAME}" in */*) echo "EXP_NAME cannot contain a slash: ${EXP_NAME}" >&2; exit 1 ;; esac
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-$([ "${WEBDEV_MODE}" = "train" ] && echo -1 || echo 1)}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-$([ "${WEBDEV_MODE}" = "train" ] && echo False || echo True)}"

printf '%s\n' \
  "[webdev] repo          = ${REPO_ROOT}" \
  "[webdev] mode          = ${WEBDEV_MODE}" \
  "[webdev] model         = ${MODEL_PATH}" \
  "[webdev] train / val   = ${TRAIN_DATA} / ${VAL_DATA}" \
  "[webdev] dumps         = ${WEBDEV_DEBUG_DIR:-<unset: no dumps, and the group pick cannot run>}" \
  "[webdev] topology      = train ${TRAIN_NNODES}x${TRAIN_NGPUS_PER_NODE}, tp ${ACTOR_TP} pp ${ACTOR_PP} cp ${ACTOR_CP}, rollout tp ${ROLLOUT_TP}" \
  "[webdev] batch         = ${TRAIN_BATCH_SIZE} x n${N}, ${TOTAL_EPOCHS} epochs" \
  "[webdev] sampling      = T ${ROLLOUT_TEMPERATURE} / top_p ${ROLLOUT_TOP_P} / top_k ${ROLLOUT_TOP_K}" \
  "[webdev] window        = ${PROMPT_LENGTH} prompt + $((MAXLEN - PROMPT_LENGTH)) response = ${MAXLEN}"

exec bash "${REPO_ROOT}/recipes/design/run_webdev.sh" "$@"
