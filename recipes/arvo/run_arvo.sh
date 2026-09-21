#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_cc="/opt/cuda_compat.sh"
[ -f "$_cc" ] || _cc="${REPO_ROOT}/docker/cuda_compat.sh"
[ -f "$_cc" ] && . "$_cc"
unset _cc
CONFIG_PATH="${SCRIPT_DIR}/config"

: "${MODEL_PATH:?set MODEL_PATH to an HF id or a local checkpoint}"
: "${TRAIN_DATA:?set TRAIN_DATA to a parquet path}"
: "${VAL_DATA:?set VAL_DATA to a parquet path}"

MIMOAGENT_SRC="${MIMOAGENT_SRC:-${REPO_ROOT}/third_party/mimoagent-osr/src}"
if [ ! -d "${MIMOAGENT_SRC}/mimoagent" ]; then
  echo "MIMOAGENT_SRC has no mimoagent package: ${MIMOAGENT_SRC}" >&2
  echo "run: git submodule update --init third_party/mimoagent-osr" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${MIMOAGENT_SRC}:${PYTHONPATH:-}"

if [ "${TRAIN_BATCH_SIZE}" != "${PPO_MINI_BATCH_SIZE}" ]; then
  echo "loss_agg_mode=prompt-mean needs TRAIN_BATCH_SIZE == PPO_MINI_BATCH_SIZE" >&2
  echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}" >&2
  exit 1
fi

if [ "${INVALID_REWARD_FOR_INFRA:-false}" = "true" ] && [ "${DROP_INFRA_FROM_GROUP:-0}" = "1" ]; then
  echo "INVALID_REWARD_FOR_INFRA=true conflicts with DROP_INFRA_FROM_GROUP=1" >&2
  echo "  ARVO uses DROP_INFRA_FROM_GROUP (plain 0.0, uid reassignment)" >&2
  echo "  pick one strategy, not both" >&2
  exit 1
fi

AGENT_LOOP_CONFIG="${SCRIPT_DIR}/config/arvo_agent_loop.yaml"
export AGENT_LOOP_CONFIG
python3 - "${TRAIN_DATA}" "${VAL_DATA}" "${AGENT_LOOP_CONFIG}" <<'PY' || exit 1
import sys, pandas as pd, yaml
train, val, registry_path = sys.argv[1], sys.argv[2], sys.argv[3]
known = {entry["name"] for entry in yaml.safe_load(open(registry_path))}
for path in dict.fromkeys(p for arg in (train, val) for p in arg.split(",") if p):
    names = set(pd.read_parquet(path, columns=["agent_name"])["agent_name"].unique())
    unknown = names - known
    if unknown:
        print(f"{path}: agent_name {sorted(unknown)} not in registry {sorted(known)}", file=sys.stderr)
        sys.exit(1)
print(f"[preflight] agent_name ok, registry knows {sorted(known)}")
PY

RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"

_hydra_file_list() {
  local raw="$1"
  if [[ "${raw}" == *,* ]]; then
    local joined="" part
    IFS=',' read -ra parts <<<"${raw}"
    for part in "${parts[@]}"; do
      [ -n "${part}" ] && joined+="${joined:+,}'${part}'"
    done
    printf '[%s]' "${joined}"
  else
    printf '%s' "${raw}"
  fi
}
TRAIN_DATA_HYDRA="$(_hydra_file_list "${TRAIN_DATA}")"
VAL_DATA_HYDRA="$(_hydra_file_list "${VAL_DATA}")"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/ckpt}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}"
TRAJ_DUMP_DIR="${TRAJ_DUMP_DIR:-${RUN_DIR}/traj}"
RESOLVED_CONFIG_PATH="${RESOLVED_CONFIG_PATH:-${RUN_DIR}/resolved_config.yaml}"
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${TRAJ_DUMP_DIR}"

export EXP_NAME TENSORBOARD_DIR
export KUBECONFIG="${KUBECONFIG:-}"

RAY_ENV=(
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
  +ray_kwargs.ray_init.runtime_env.env_vars.EXP_NAME="${EXP_NAME}"
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_DEBUG_DIR="${TRAJ_DUMP_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TENSORBOARD_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="\"1\""
  +ray_kwargs.ray_init.runtime_env.env_vars.DROP_INFRA_FROM_GROUP="\"${DROP_INFRA_FROM_GROUP}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.TRAJECTORY_TIMEOUT="\"${TRAJECTORY_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENV_SETUP_TIMEOUT="\"${ENV_SETUP_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENV_NUM_CPUS="\"${ENV_NUM_CPUS}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.FAIL_ON_ENV_SETUP_ERROR="\"${FAIL_ON_ENV_SETUP_ERROR}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.INVALID_REWARD_FOR_INFRA="\"${INVALID_REWARD_FOR_INFRA}\""
)
[ -n "${KUBECONFIG}" ] && RAY_ENV+=(
  +ray_kwargs.ray_init.runtime_env.env_vars.KUBECONFIG="${KUBECONFIG}"
  +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_GLOBAL_CONFIG_DIR="${AGENT_GLOBAL_CONFIG_DIR:-/opt/mimoagent-config}"
)

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo
  --config-name=arvo
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
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.megatron.param_offload="${MEGATRON_OFFLOAD}"
  actor_rollout_ref.actor.megatron.optimizer_offload="${MEGATRON_OFFLOAD}"
  actor_rollout_ref.actor.megatron.grad_offload="${MEGATRON_OFFLOAD}"
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
  actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}"
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}"
  actor_rollout_ref.ref.megatron.param_offload="${MEGATRON_OFFLOAD}"
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}"
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
  actor_rollout_ref.ref.megatron.context_parallel_size="${ACTOR_CP}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}"
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.max_model_len="${MAXLEN}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_RUNNING_REQUESTS}"
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}"
  actor_rollout_ref.rollout.multi_turn.format="${FORMAT}"
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}"
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}"
  algorithm.norm_adv_by_std_in_grpo=False
  algorithm.use_kl_in_reward=False
  actor_rollout_ref.actor.use_kl_loss=False
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=False
  trainer.nnodes="${NNODES}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}"
  trainer.resume_mode="${RESUME_MODE}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.val_before_train=False
  trainer.v1.trainer_mode="${TRAINER_MODE}"
  trainer.v1.colocate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}"
  trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}"
  trainer.v1.sampler.max_off_policy_strategy="${MAX_OFF_POLICY_STRATEGY}"
  "${RAY_ENV[@]}"
  "$@"
)

echo "[run_arvo] model       = ${MODEL_PATH}"
echo "[run_arvo] data        = $(basename "${TRAIN_DATA}") / $(basename "${VAL_DATA}")"
echo "[run_arvo] run dir     = ${RUN_DIR}"

if ! "${MAIN_CMD[@]}" --cfg job --resolve >"${RESOLVED_CONFIG_PATH}" 2>/dev/null; then
  echo "failed to resolve effective Hydra config; refusing to launch" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/../write_run_manifest.py" \
  --run-dir "${RUN_DIR}" \
  --verl-repo "${REPO_ROOT}" \
  --component "mimoagent=${REPO_ROOT}/third_party/mimoagent-osr" \
  --config-artifact "${REPO_ROOT}/config/agent/arvo/arvo.yaml" \
  --resolved-config "${RESOLVED_CONFIG_PATH}" \
  --command "${MAIN_CMD[@]}"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "[run_arvo] PREFLIGHT_ONLY=1, resolved config and provenance written to ${RUN_DIR}"
  exit 0
fi

exec "${MAIN_CMD[@]}"
