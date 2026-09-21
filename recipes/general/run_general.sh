#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config"

GENERAL_MODE="${GENERAL_MODE:-train}"
if [ "${GENERAL_MODE}" != "train" ] && [ "${GENERAL_MODE}" != "eval" ]; then
  echo "GENERAL_MODE must be train or eval, got '${GENERAL_MODE}'" >&2
  exit 1
fi

: "${MODEL_PATH:?set MODEL_PATH to an HF id or a local checkpoint directory}"
: "${TRAIN_DATA:?set TRAIN_DATA to a parquet path}"
: "${VAL_DATA:?set VAL_DATA to a parquet path}"
: "${KUBECONFIG:?set KUBECONFIG to a kubeconfig with pod-create permission}"
: "${GA_TASK_ROOT:?set GA_TASK_ROOT to the open_source_env bundle root}"

MIMOAGENT_SRC="${MIMOAGENT_SRC:-${REPO_ROOT}/third_party/mimoagent-osr/src}"
if [ ! -d "${MIMOAGENT_SRC}/mimoagent" ]; then
  echo "MIMOAGENT_SRC has no mimoagent package: ${MIMOAGENT_SRC}" >&2
  echo "run: git submodule update --init third_party/mimoagent-osr" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${MIMOAGENT_SRC}:${PYTHONPATH:-}"

if [ ! -d "${GA_TASK_ROOT}/envs" ]; then
  echo "GA_TASK_ROOT has no envs/ directory: ${GA_TASK_ROOT}" >&2
  exit 1
fi
cd "${GA_TASK_ROOT}"

: "${GA_JUDGE_URL:?set GA_JUDGE_URL to an OpenAI-compatible base URL}"
: "${GA_JUDGE_KEY:?set GA_JUDGE_KEY (use EMPTY for a local vLLM without auth)}"

if [ "${GENERAL_MODE}" = "train" ] && [ "${TRAIN_BATCH_SIZE}" != "${PPO_MINI_BATCH_SIZE}" ]; then
  echo "loss_agg_mode=prompt-mean needs TRAIN_BATCH_SIZE == PPO_MINI_BATCH_SIZE" >&2
  echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}" >&2
  exit 1
fi

if [ "${INVALID_REWARD_FOR_INFRA}" = "true" ] && [ "${INVALID_REWARD_VALUE}" = "null" ]; then
  echo "INVALID_REWARD_FOR_INFRA=true needs INVALID_REWARD_VALUE set (-999)" >&2
  exit 1
fi
if [ "${DROP_INFRA_FROM_GROUP:-0}" = "1" ]; then
  echo "DROP_INFRA_FROM_GROUP=1 conflicts with the sentinel path this arm uses" >&2
  echo "  the sentinel already zeroes an infra rollout's advantage; pick one" >&2
  exit 1
fi

python3 - "${TRAIN_DATA}" "${VAL_DATA}" "${REPO_ROOT}/recipes/general/config/general_agent_loop.yaml" <<'PY' || exit 1
import sys, pandas as pd, yaml
train, val, registry = sys.argv[1], sys.argv[2], sys.argv[3]
known = {entry["name"] for entry in yaml.safe_load(open(registry))}
for path in dict.fromkeys(p for arg in (train, val) for p in arg.split(",") if p):
    names = set(pd.read_parquet(path, columns=["agent_name"])["agent_name"].unique())
    unknown = names - known
    if unknown:
        print(f"{path}: agent_name {sorted(unknown)} not in the registry {sorted(known)}", file=sys.stderr)
        sys.exit(1)
print(f"[preflight] agent_name ok, registry knows {sorted(known)}")
PY

python3 - "${TRAIN_DATA},${VAL_DATA}" "${DOCKER_REGISTRY:-}" <<'PY' || exit 1
import json, sys, pandas as pd
paths, prefix = sys.argv[1], sys.argv[2].rstrip("/")
foreign = set()
for path in dict.fromkeys(p for p in paths.split(",") if p):
    for blob in pd.read_parquet(path, columns=["extra_info"])["extra_info"]:
        image = json.loads(dict(blob)["instance_json"]).get("docker_image", "")
        if "/" not in image:
            continue  # bare repo:tag -- the prefix applies cleanly
        head = image.split("/", 1)[0]
        if ("." in head or ":" in head) and not (prefix and image.startswith(prefix + "/")):
            foreign.add(image)
if foreign:
    print("these rows already name a registry, which would be prefixed again rather than", file=sys.stderr)
    print(f"replaced (DOCKER_REGISTRY={prefix or '<unset>'}):", file=sys.stderr)
    for image in sorted(foreign)[:3]:
        print(f"  {image}", file=sys.stderr)
    print("use the bundle's docker/retag_parquet.py output, which emits bare names", file=sys.stderr)
    sys.exit(1)
print("[preflight] docker_image ok, the registry prefix applies cleanly")
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
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${TRAJ_DUMP_DIR}"

RAY_ENV=(
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
  +ray_kwargs.ray_init.runtime_env.env_vars.PATH="${PATH}"
  +ray_kwargs.ray_init.runtime_env.env_vars.EXP_NAME="${EXP_NAME}"
  +ray_kwargs.ray_init.runtime_env.env_vars.KUBECONFIG="${KUBECONFIG}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_TASK_ROOT="${GA_TASK_ROOT}"
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_DEBUG_DIR="${TRAJ_DUMP_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TENSORBOARD_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_URL="${GA_JUDGE_URL}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_KEY="${GA_JUDGE_KEY}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_MODEL="${GA_JUDGE_MODEL}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_API="${GA_JUDGE_API}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_KEY_FILE="${GA_JUDGE_KEY_FILE:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
  +ray_kwargs.ray_init.runtime_env.env_vars.K8S_IMAGE_PULL_SECRET="${K8S_IMAGE_PULL_SECRET:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.K8S_TOLERATION_KEY="${K8S_TOLERATION_KEY:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.K8S_TOLERATION_VALUE="${K8S_TOLERATION_VALUE:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.K8S_NODE_LABEL_KEY="${K8S_NODE_LABEL_KEY:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.K8S_NODE_LABEL_VALUE="${K8S_NODE_LABEL_VALUE:-}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GENERAL_INFRA_METRICS="\"${GENERAL_INFRA_METRICS}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_BINARIZE="\"${REWARD_BINARIZE}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_BINARIZE_THRESHOLD="\"${REWARD_BINARIZE_THRESHOLD}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.MIMOAGENT_BASH_MAX_TIMEOUT_MS="\"${MIMOAGENT_BASH_MAX_TIMEOUT_MS}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.TRAJECTORY_TIMEOUT="\"${TRAJECTORY_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENV_SETUP_TIMEOUT="\"${ENV_SETUP_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_TIMEOUT="\"${REWARD_TIMEOUT}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENV_NUM_CPUS="\"${ENV_NUM_CPUS}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.FAIL_ON_ENV_SETUP_ERROR="\"${FAIL_ON_ENV_SETUP_ERROR}\""
  +ray_kwargs.ray_init.runtime_env.env_vars.INVALID_REWARD_FOR_INFRA="\"${INVALID_REWARD_FOR_INFRA}\""
)

MODE_ARGS=()
if [ "${GENERAL_MODE}" = "eval" ]; then
  MODE_ARGS=(
    trainer.val_only=True
    trainer.val_before_train=True
    trainer.total_epochs=1
    trainer.save_freq=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.val_kwargs.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.val_kwargs.top_k="${ROLLOUT_TOP_K}"
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_KWARGS_N}"
  )
fi

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo
  --config-name=general
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
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
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
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
  algorithm.invalid_reward_value="${INVALID_REWARD_VALUE}"
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}"
  algorithm.length_penalty.enable="${LENGTH_PENALTY_ENABLE}"
  trainer.nnodes="${NNODES}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXP_NAME}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  "${RAY_ENV[@]}"
  "${MODE_ARGS[@]}"
  "$@"
)

echo "[run_general] mode        = ${GENERAL_MODE}"
echo "[run_general] cwd         = $(pwd)  (relative env_task_dir resolves against this)"
echo "[run_general] model       = ${MODEL_PATH}"
echo "[run_general] data        = $(basename "${TRAIN_DATA}") / $(basename "${VAL_DATA}")"
echo "[run_general] judge       = ${GA_JUDGE_URL} (${GA_JUDGE_MODEL}, api=${GA_JUDGE_API})"
echo "[run_general] run dir     = ${RUN_DIR}"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "[run_general] PREFLIGHT_ONLY=1, resolving config without launching"
  exec "${MAIN_CMD[@]}" --cfg job --resolve
fi

exec "${MAIN_CMD[@]}"
