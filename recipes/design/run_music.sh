#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_cc="/opt/cuda_compat.sh"
[ -f "$_cc" ] || _cc="${REPO_ROOT}/docker/cuda_compat.sh"
[ -f "$_cc" ] && . "$_cc"
unset _cc

CONFIG_PATH="${SCRIPT_DIR}/config"

: "${MODEL_PATH:?MODEL_PATH must point to the policy checkpoint}"
: "${TRAIN_DATA:?TRAIN_DATA must point to a training parquet}"
: "${VAL_DATA:?VAL_DATA must point to a validation parquet}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

if ! command -v abc2midi >/dev/null 2>&1; then
  if [ -n "${ABC2MIDI_BIN:-}" ] && [ -x "${ABC2MIDI_BIN}" ]; then
    export PATH="$(dirname "${ABC2MIDI_BIN}"):${PATH}"
    echo "[music] abc2midi from ABC2MIDI_BIN=${ABC2MIDI_BIN}"
  elif command -v apt-get >/dev/null 2>&1; then
    echo "[music] abc2midi missing; trying apt-get install abcmidi"
    apt-get update -y >/dev/null 2>&1 && apt-get install -y abcmidi >/dev/null 2>&1 || true
  fi
fi
if ! command -v abc2midi >/dev/null 2>&1; then
  echo "[music] FATAL: abc2midi is not available." >&2
  echo "        The scorer cannot produce any reward without it. Install the" >&2
  echo "        'abcmidi' package in the image, or set ABC2MIDI_BIN." >&2
  exit 2
fi

RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"

_hydra_file_list() {
  local raw="$1" item out="" first=1
  local -a items
  IFS=',' read -r -a items <<< "${raw}"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    item="${item#\"}"; item="${item%\"}"
    item="${item#\'}"; item="${item%\'}"
    [ -z "${item}" ] && continue
    if [ "${first}" -eq 1 ]; then
      out="'${item}'"
      first=0
    else
      out="${out},'${item}'"
    fi
  done
  printf '[%s]' "${out}"
}

TRAIN_DATA_HYDRA="$(_hydra_file_list "${TRAIN_DATA}")"
VAL_DATA_HYDRA="$(_hydra_file_list "${VAL_DATA}")"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/ckpt}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${RUN_DIR}/rollout}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${RUN_DIR}/validation}"
RESOLVED_CONFIG_PATH="${RESOLVED_CONFIG_PATH:-${RUN_DIR}/resolved_config.yaml}"

RESPONSE_LENGTH="${RESPONSE_LENGTH:-$((MAXLEN - PROMPT_LENGTH))}"

RAY_ENV=(
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
  +ray_kwargs.ray_init.runtime_env.env_vars.PATH="${PATH}"
  +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.EXP_NAME="${EXP_NAME}"
  +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}"
)

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo
  --config-name=music
  --config-path="${CONFIG_PATH}"
  hydra.searchpath=[pkg://verl.trainer.config]
  +ray_kwargs.ray_init.address="${RAY_INIT_ADDRESS}"
  actor_rollout_ref.model.path="${MODEL_PATH}"
  data.train_files="${TRAIN_DATA_HYDRA}"
  data.val_files="${VAL_DATA_HYDRA}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${PROMPT_LENGTH}"
  data.max_response_length="${RESPONSE_LENGTH}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-${MAXLEN}}"
  actor_rollout_ref.rollout.n="${N}"
  actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}"
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.max_model_len="${MAXLEN}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_RUNNING_REQUESTS}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
  actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}"
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}"
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
  actor_rollout_ref.ref.megatron.context_parallel_size="${ACTOR_CP}"
  reward.num_workers="${REWARD_NUM_WORKERS}"
  trainer.nnodes="${TRAIN_NNODES}"
  trainer.n_gpus_per_node="${TRAIN_NGPUS_PER_NODE}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
  "${RAY_ENV[@]}"
  "$@"
)

mkdir -p "${RUN_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}" "${CHECKPOINT_DIR}"

if ! "${MAIN_CMD[@]}" --cfg job --resolve >"${RESOLVED_CONFIG_PATH}"; then
  echo "failed to resolve effective Hydra config; refusing to launch" >&2
  exit 2
fi

if [ "${SKIP_SCORER_CHECK:-0}" != "1" ]; then
  python3 "${SCRIPT_DIR}/music/scorer_precheck.py" --address "${RAY_INIT_ADDRESS}"
fi

python3 "${SCRIPT_DIR}/../write_run_manifest.py" \
  --run-dir "${RUN_DIR}" \
  --verl-repo "${REPO_ROOT}" \
  --config-artifact "${SCRIPT_DIR}/config/music.yaml" \
  --resolved-config "${RESOLVED_CONFIG_PATH}" \
  --command "${MAIN_CMD[@]}"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "preflight passed; resolved config and provenance written to ${RUN_DIR}"
  exit 0
fi

exec "${MAIN_CMD[@]}"
