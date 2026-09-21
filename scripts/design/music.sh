#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${MODEL_PATH:?MODEL_PATH must point to a policy checkpoint}"
: "${TRAIN_DATA:?TRAIN_DATA must point to a training parquet}"
: "${VAL_DATA:?VAL_DATA must point to a validation parquet}"

export N="${N:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
export MAXLEN="${MAXLEN:-116384}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.85}"
export ROLLOUT_MAX_RUNNING_REQUESTS="${ROLLOUT_MAX_RUNNING_REQUESTS:-128}"

export TRAIN_NNODES="${TRAIN_NNODES:-8}"
export TRAIN_NGPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE:-8}"
export ACTOR_TP="${ACTOR_TP:-8}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"

export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"

export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-16}"

export PROJECT_NAME="${PROJECT_NAME:-opensource-design}"
export EXP_NAME="${EXP_NAME:-music}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"

printf '%s\n' \
  "[music] repo          = ${REPO_ROOT}" \
  "[music] model         = ${MODEL_PATH}" \
  "[music] train / val   = ${TRAIN_DATA} / ${VAL_DATA}" \
  "[music] abc2midi      = ${ABC2MIDI_BIN:-<from PATH or image>}" \
  "[music] topology      = train ${TRAIN_NNODES}x${TRAIN_NGPUS_PER_NODE}, tp ${ACTOR_TP} pp ${ACTOR_PP} cp ${ACTOR_CP}, rollout tp ${ROLLOUT_TP}" \
  "[music] batch         = ${TRAIN_BATCH_SIZE} x n${N}, ${TOTAL_EPOCHS} epochs" \
  "[music] window        = ${PROMPT_LENGTH} prompt + $((MAXLEN - PROMPT_LENGTH)) response = ${MAXLEN}"

exec bash "${REPO_ROOT}/recipes/design/run_music.sh" "$@"
