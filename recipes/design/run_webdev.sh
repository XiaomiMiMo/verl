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
: "${WEBDEV_MODE:?WEBDEV_MODE must be train or eval}"

if [ "${WEBDEV_MODE}" = "train" ]; then
  AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-recipes/design/config/webdev_agent_loop.yaml}"
else
  AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-recipes/design/config/webdev_eval_agent_loop.yaml}"
fi
[ -f "${REPO_ROOT}/${AGENT_LOOP_CONFIG}" ] || {
  echo "missing rollout registry: ${AGENT_LOOP_CONFIG}" >&2; exit 2; }

MIMOAGENT_SRC="${MIMOAGENT_SRC:-${REPO_ROOT}/third_party/mimoagent-osr/src}"
[ -d "${MIMOAGENT_SRC}/mimoagent" ] || {
  echo "MimoAgent source not found at ${MIMOAGENT_SRC}" >&2
  echo "  run: git submodule update --init third_party/mimoagent-osr" >&2
  exit 2; }
export PYTHONPATH="${REPO_ROOT}:${MIMOAGENT_SRC}:${PYTHONPATH:-}"

if [ "${PPO_MINI_BATCH_SIZE}" != "${TRAIN_BATCH_SIZE}" ]; then
  echo "[webdev] prompt-mean requires PPO_MINI_BATCH_SIZE == TRAIN_BATCH_SIZE," \
       "got ${PPO_MINI_BATCH_SIZE} vs ${TRAIN_BATCH_SIZE}." >&2
  echo "        Make them equal, or set loss_agg_mode=token-mean on the command line." >&2
  exit 1
fi

if [ "${WEBDEV_MODE}" = "train" ] && [ -z "${WEBDEV_DEBUG_DIR:-}" ]; then
  echo "[webdev] FATAL: WEBDEV_DEBUG_DIR is unset." >&2
  echo "        Training needs it: the per-rollout grader writes the judged screenshot" >&2
  echo "        there and the driver reads it back to rank the group. Without it every" >&2
  echo "        group is skipped and every reward stays 0.0, with no error at all." >&2
  echo "        It must be on a filesystem shared by the workers AND the driver." >&2
  exit 2
fi
if [ -n "${WEBDEV_DEBUG_DIR:-}" ]; then
  mkdir -p "${WEBDEV_DEBUG_DIR}"
fi

python3 - "${TRAIN_DATA}" "${REPO_ROOT}/${AGENT_LOOP_CONFIG}" <<'PY' || exit 1
import sys
import pandas as pd
import yaml

paths = [p.strip() for p in sys.argv[1].split(",") if p.strip()]
names = {e["name"] for e in yaml.safe_load(open(sys.argv[2]))}
seen, rows = set(), 0
for path in paths:
    d = pd.read_parquet(path, columns=["agent_name", "data_source"])
    seen |= set(d["agent_name"])
    rows += len(d)
print(f"[webdev]   data {rows} rows  agent_name={sorted(seen)}  registry={sorted(names)}")
bad = seen - names
if bad:
    sys.exit(f"[webdev]   FATAL: the data names harnesses the registry does not have: {bad}")
PY

if [ "${WEBDEV_MODE}" = "train" ] && [ "${SKIP_GRADER_HANDSHAKE:-0}" != "1" ]; then
  echo "[webdev] handshaking the group grader at ${DESIGN_GRADER_URL}"
  python3 - "${DESIGN_GRADER_URL}" <<'PY' || { echo "[webdev] FATAL: grader handshake failed" >&2; exit 2; }
import sys
from recipes.design.webdev.grader_client import check_capabilities

info = check_capabilities(sys.argv[1], need_group=True)
print(f"[webdev]   ok: version={info.get('grader_version')} model={info.get('model')} "
      f"semantics={info.get('reward_semantics')}")
PY
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
    if [ "${first}" -eq 1 ]; then out="'${item}'"; first=0; else out="${out},'${item}'"; fi
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
  +ray_kwargs.ray_init.runtime_env.env_vars.KUBECONFIG="${KUBECONFIG}"
  +ray_kwargs.ray_init.runtime_env.env_vars.POD_PROXY="'${POD_PROXY}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_GRADE_MODE="${WEBDEV_GRADE_MODE}"
  +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_GRADE_HTTP="\"${WEBDEV_GRADE_HTTP}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.DROP_INFRA_FROM_GROUP="\"${DROP_INFRA_FROM_GROUP}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_DEBUG_DIR="${WEBDEV_DEBUG_DIR:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_NODE_SELECTOR="'${WEBDEV_NODE_SELECTOR:-}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_TOLERATIONS="'${WEBDEV_TOLERATIONS:-}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_DNAT_PROXY_IP="'${WEBDEV_DNAT_PROXY_IP:-}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.TRAJECTORY_TIMEOUT="\"${TRAJECTORY_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENV_SETUP_TIMEOUT="\"${ENV_SETUP_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_TIMEOUT="\"${REWARD_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENV_NUM_CPUS="\"${ENV_NUM_CPUS}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.FAIL_ON_ENV_SETUP_ERROR="\"${FAIL_ON_ENV_SETUP_ERROR}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.INVALID_REWARD_FOR_INFRA="\"${INVALID_REWARD_FOR_INFRA}\""
)
if [ "${WEBDEV_MODE}" = "train" ]; then
  RAY_ENV+=(
    +ray_kwargs.ray_init.runtime_env.env_vars.DESIGN_GRADER_URL="${DESIGN_GRADER_URL}"
    +ray_kwargs.ray_init.runtime_env.env_vars.LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY}"
  )
else
  RAY_ENV+=(
    +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_EVAL_JUDGE_BASE_URL="${WEBDEV_EVAL_JUDGE_BASE_URL}"
    +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_EVAL_JUDGE_API_KEY="${WEBDEV_EVAL_JUDGE_API_KEY}"
    +ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_EVAL_JUDGE_MODEL="${WEBDEV_EVAL_JUDGE_MODEL}"
  )
fi

MODE_ARGS=()
if [ "${WEBDEV_MODE}" = "eval" ]; then
  MODE_ARGS=(
    trainer.val_only=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.val_kwargs.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.val_kwargs.top_k="${ROLLOUT_TOP_K}"
    actor_rollout_ref.rollout.val_kwargs.n=1
  )
fi

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo
  --config-name=webdev
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
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}"
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
  actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}"
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}"
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
  actor_rollout_ref.ref.megatron.context_parallel_size="${ACTOR_CP}"
  trainer.nnodes="${TRAIN_NNODES}"
  trainer.n_gpus_per_node="${TRAIN_NGPUS_PER_NODE}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.val_before_train="${VAL_BEFORE_TRAIN}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
  "${MODE_ARGS[@]+"${MODE_ARGS[@]}"}"
  "${RAY_ENV[@]}"
  "$@"
)

mkdir -p "${RUN_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}" "${CHECKPOINT_DIR}"

if ! "${MAIN_CMD[@]}" --cfg job --resolve >"${RESOLVED_CONFIG_PATH}"; then
  echo "failed to resolve effective Hydra config; refusing to launch" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/../write_run_manifest.py" \
  --run-dir "${RUN_DIR}" \
  --verl-repo "${REPO_ROOT}" \
  --component "mimoagent=${MIMOAGENT_SRC}" \
  --config-artifact "${REPO_ROOT}/${AGENT_LOOP_CONFIG}" \
  --resolved-config "${RESOLVED_CONFIG_PATH}" \
  --command "${MAIN_CMD[@]}"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "preflight passed; resolved config and provenance written to ${RUN_DIR}"
  exit 0
fi

cat <<EOF

=== once it is up, read these before anything else ===
  webdev_group/reward_mean            the reward. NOT critic/score/mean, which is
                                      identically 0 on this line: the driver rewrites
                                      token_level_rewards in memory and, with
                                      use_kl_in_reward off, never writes it back
  webdev_group/n_rows_judged          should track the number of rows in flight
  webdev_group/n_groups_too_small     persistently non-zero means the driver cannot read
                                      the shots -- almost always a node-local dump dir
  webdev_group/n_groups_pick_short    groups the judge could not finish; high means timeouts
  webdev_group/hook_failed            the rewrite raised; rewards are untouched placeholders
  training/infra/excluded_from_grpo   should track the batch's infra failures
  training/adv/dead_rows_non_infra    should stay near zero
  actor/grad_norm                     intermittent Inf has been seen here; clip_grad then
                                      discards that step's update

EOF

exec "${MAIN_CMD[@]}"
