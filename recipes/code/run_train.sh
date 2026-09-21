#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UNI_AGENT_ROOT="${REPO_ROOT}/third_party/uni_agent"
MIMOAGENT_SRC="${MIMOAGENT_SRC:-${REPO_ROOT}/third_party/mimoagent-osr}"
MIMOAGENT_HARNESS_SPEC="${MIMOAGENT_HARNESS_SPEC:-${REPO_ROOT}/config/agent/code/mix-four-whitebox.yaml}"
if [ ! -f "${MIMOAGENT_HARNESS_SPEC}" ]; then
  echo "MIMOAGENT_HARNESS_SPEC is not a file: ${MIMOAGENT_HARNESS_SPEC}" >&2
  exit 2
fi

: "${MODEL_PATH:?MODEL_PATH must point to the policy checkpoint}"
: "${TRAIN_DATA:?TRAIN_DATA must point to a MimoAgent SWE parquet}"
: "${VAL_DATA:?VAL_DATA must point to a MimoAgent SWE validation parquet}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${UNI_AGENT_ROOT}:${MIMOAGENT_SRC}/src:${PYTHONPATH:-}"
export METRIC_PORT="${METRIC_PORT:-20000}"
export ENABLE_METRIC="${ENABLE_METRIC:-true}"

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

N="${N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"
MAXLEN="${MAXLEN:-262144}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-$((MAXLEN - PROMPT_LENGTH))}"
TRAIN_NNODES="${TRAIN_NNODES:-4}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-0}"
TRAIN_NGPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE:-8}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-8}"
ACTOR_TP="${ACTOR_TP:-8}"
ACTOR_PP="${ACTOR_PP:-1}"
ACTOR_CP="${ACTOR_CP:-2}"
ACTOR_EP="${ACTOR_EP:-1}"
if ! [[ "${ACTOR_CP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACTOR_CP must be a positive integer" >&2
  exit 2
fi
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAXLEN / ACTOR_CP))}"
MEGATRON_OFFLOAD="${MEGATRON_OFFLOAD:-True}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"
GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-512}"
ROLLOUT_MAX_RUNNING_REQUESTS="${ROLLOUT_MAX_RUNNING_REQUESTS:-64}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
ROLLOUT_MIN_P="${ROLLOUT_MIN_P:-0.0}"
ROLLOUT_PRESENCE_PENALTY="${ROLLOUT_PRESENCE_PENALTY:-0.0}"
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.0}"
ROLLOUT_REASONING_EFFORT="${ROLLOUT_REASONING_EFFORT:-}"
TOOL_CALL_ERROR_PENALTY_ENABLE="${TOOL_CALL_ERROR_PENALTY_ENABLE:-False}"
TOOL_CALL_ERROR_PENALTY_STRATEGY="${TOOL_CALL_ERROR_PENALTY_STRATEGY:-monitor}"
TOOL_CALL_ERROR_PENALTY_VALUE="${TOOL_CALL_ERROR_PENALTY_VALUE:-0.0}"
REPETITION_DETECT_ENABLE="${REPETITION_DETECT_ENABLE:-false}"
REPETITION_ZERO_REWARD="${REPETITION_ZERO_REWARD:-false}"
REPETITION_PENALTY_ENABLE="${REPETITION_PENALTY_ENABLE:-False}"
REPETITION_PENALTY_STRATEGY="${REPETITION_PENALTY_STRATEGY:-monitor}"
REPETITION_PENALTY_VALUE="${REPETITION_PENALTY_VALUE:-0.0}"
if [ "${REPETITION_PENALTY_STRATEGY}" = "early_stop" ] && [ "$(printf '%s' "${REPETITION_ZERO_REWARD}" | tr '[:upper:]' '[:lower:]')" != "true" ]; then
  echo "REPETITION_PENALTY_STRATEGY=early_stop requires REPETITION_ZERO_REWARD=true (the reference RL framework early_stop semantics)" >&2
  exit 2
fi
DEEP_FAILURE_MASK_ENABLE="${DEEP_FAILURE_MASK_ENABLE:-false}"
DEEP_FAILURE_MASK_STRATEGY="${DEEP_FAILURE_MASK_STRATEGY:-mask_failure}"
DEEP_FAILURE_MASK_ALPHA="${DEEP_FAILURE_MASK_ALPHA:-1.0}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-32768}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-32768}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-384}"
MTP_ENABLE="${MTP_ENABLE:-False}"
MTP_ENABLE_TRAIN="${MTP_ENABLE_TRAIN:-False}"
MTP_ENABLE_ROLLOUT="${MTP_ENABLE_ROLLOUT:-False}"
if [ "${MTP_ENABLE}" = "True" ]; then
  SGLANG_MAMBA_SCHEDULER_STRATEGY="${SGLANG_MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
  SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
else
  SGLANG_MAMBA_SCHEDULER_STRATEGY="${SGLANG_MAMBA_SCHEDULER_STRATEGY:-no_buffer}"
  SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-0}"
fi
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:--1}"
RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"
CONFIG_PATH="${SCRIPT_DIR}/config"
PROJECT_NAME="${PROJECT_NAME:-opensource-code}"
EXP_NAME="${EXP_NAME:-four-whitebox}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}/${RUN_ID}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
AGENT_DEBUG_DIR="${AGENT_DEBUG_DIR:-${RUN_DIR}/dumps}"
UNI_AGENT_LOG_DIR="${UNI_AGENT_LOG_DIR:-${RUN_DIR}/trajectories}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${RUN_DIR}/rollouts}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${RUN_DIR}/validation}"
RESOLVED_CONFIG_PATH="${RESOLVED_CONFIG_PATH:-${RUN_DIR}/resolved_config.yaml}"
KUBECONFIG="${KUBECONFIG:-}"

if ! [[ "${PPO_MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU must be a positive integer" >&2
  exit 2
fi
if (( PPO_MAX_TOKEN_LEN_PER_GPU * ACTOR_CP < MAXLEN )); then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU * ACTOR_CP must cover MAXLEN" >&2
  exit 2
fi

export EXP_NAME TENSORBOARD_DIR AGENT_DEBUG_DIR UNI_AGENT_LOG_DIR KUBECONFIG

WORKER_LD=""
if [ "${SKIP_CLUSTER_CHECK:-0}" != "1" ]; then
  IFS=: read -r -a PYTHONPATH_ENTRIES <<< "${PYTHONPATH}"
  PRECHECK_PYTHONPATH_ARGS=()
  for pythonpath_entry in "${PYTHONPATH_ENTRIES[@]}"; do
    [ -n "${pythonpath_entry}" ] && PRECHECK_PYTHONPATH_ARGS+=(--pythonpath "${pythonpath_entry}")
  done
  PRECHECK_PATH_ARGS=()
  IFS=',' read -r -a PRECHECK_PATHS <<< "${TRAIN_DATA}"
  for precheck_path in "${PRECHECK_PATHS[@]}"; do
    precheck_path="${precheck_path#"${precheck_path%%[![:space:]]*}"}"
    precheck_path="${precheck_path%"${precheck_path##*[![:space:]]}"}"
    [ -n "${precheck_path}" ] && PRECHECK_PATH_ARGS+=(--path "${precheck_path}")
  done
  WORKER_LD_OUTPUT=$(python3 "${SCRIPT_DIR}/cluster_precheck.py" \
    --address "${RAY_INIT_ADDRESS}" \
    --nnodes "${TRAIN_NNODES}" --gpus-per-node "${TRAIN_NGPUS_PER_NODE}" \
    --model "${MODEL_PATH}" "${PRECHECK_PATH_ARGS[@]}" --path "${KUBECONFIG}" \
    "${PRECHECK_PYTHONPATH_ARGS[@]}")
  WORKER_LD=$(printf '%s\n' "${WORKER_LD_OUTPUT}" | tail -n 1)
  if [ "${WORKER_LD_OUTPUT}" != "${WORKER_LD}" ]; then
    printf '%s\n' "${WORKER_LD_OUTPUT%$'\n'${WORKER_LD}}" >&2
  fi
else
  WORKER_LD="${WORKER_LD_LIBRARY_PATH:-}"
fi

RAY_ENV=(
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
  +ray_kwargs.ray_init.runtime_env.env_vars.KUBECONFIG="${KUBECONFIG}"
  +ray_kwargs.ray_init.runtime_env.env_vars.EXP_NAME="${EXP_NAME}"
  +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_DEBUG_DIR="${AGENT_DEBUG_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_LOG_DIR="${UNI_AGENT_LOG_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="\"1\""
  +ray_kwargs.ray_init.runtime_env.env_vars.ENABLE_METRIC="'${ENABLE_METRIC}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.METRIC_PORT="'${METRIC_PORT}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_ENABLE_SPEC_V2="'${SGLANG_ENABLE_SPEC_V2}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TENSORBOARD_DIR}"
  +ray_kwargs.ray_init.runtime_env.env_vars.MIXED_HARNESS_ENABLED="'True'"
  +ray_kwargs.ray_init.runtime_env.env_vars.MIXED_HARNESS_MODE="'${MIXED_HARNESS_MODE:-step-hash}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.MIXED_HARNESS_SEED="'${MIXED_HARNESS_SEED:-0}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.MIXED_HARNESS_SPEC="${MIMOAGENT_HARNESS_SPEC}"
  +ray_kwargs.ray_init.runtime_env.env_vars.TRAJECTORY_TIMEOUT="'${TRAJECTORY_TIMEOUT:-7200}'"
)
[ -n "${WORKER_LD}" ] && RAY_ENV+=(
  +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="'${WORKER_LD}'"
)

OPTIONAL_OVERRIDES=()
[ -n "${VAL_BATCH_SIZE}" ] && OPTIONAL_OVERRIDES+=(data.val_batch_size="${VAL_BATCH_SIZE}")
[ -n "${ROLLOUT_REASONING_EFFORT}" ] && OPTIONAL_OVERRIDES+=(data.apply_chat_template_kwargs.reasoning_effort="${ROLLOUT_REASONING_EFFORT}")

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo \
  --config-name=train \
  --config-path="${CONFIG_PATH}" \
  hydra.searchpath=[pkg://verl.trainer.config] \
  +ray_kwargs.ray_init.address="${RAY_INIT_ADDRESS}" \
  trainer.use_v1=True \
  transfer_queue.enable=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_DATA_HYDRA}" \
  data.val_files="${VAL_DATA_HYDRA}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.gen_batch_size=1 \
  data.filter_overlong_prompts=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=3.0 \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE:-prompt-mean}" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="${ENTROPY_CHUNKING:-True}" \
  actor_rollout_ref.actor.entropy_from_logits_chunk_size="${ENTROPY_CHUNK_SIZE:-16384}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BSZ_PER_GPU:-1}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BSZ_PER_GPU:-1}" \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO:-False}" \
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE:-True}" \
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC:-reward}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-${ROLLOUT_TEMPERATURE}}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-${ROLLOUT_TOP_P}}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:-${ROLLOUT_TOP_K}}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-False}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  data.max_prompt_length="${PROMPT_LENGTH}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${MAXLEN}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
  +actor_rollout_ref.rollout.min_p="${ROLLOUT_MIN_P}" \
  +actor_rollout_ref.rollout.presence_penalty="${ROLLOUT_PRESENCE_PENALTY}" \
  +actor_rollout_ref.rollout.repetition_penalty="${ROLLOUT_REPETITION_PENALTY}" \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.n="${N}" \
  actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
  actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
  actor_rollout_ref.rollout.custom.agent_framework.log_dir="${UNI_AGENT_LOG_DIR}" \
  actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mimoagent.trajectory_selection=longest \
  actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mimoagent.max_concurrent_sessions="${MAX_CONCURRENT_SESSIONS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_RUNNING_REQUESTS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.actor.megatron.param_offload="${MEGATRON_OFFLOAD}" \
  actor_rollout_ref.actor.megatron.optimizer_offload="${MEGATRON_OFFLOAD}" \
  actor_rollout_ref.actor.megatron.grad_offload="${MEGATRON_OFFLOAD}" \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}" \
  actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}" \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}" \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
  actor_rollout_ref.actor.megatron.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  actor_rollout_ref.ref.megatron.param_offload="${MEGATRON_OFFLOAD}" \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP}" \
  actor_rollout_ref.ref.megatron.context_parallel_size="${ACTOR_CP}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  trainer.nnodes="${TRAIN_NNODES}" \
  trainer.n_gpus_per_node="${TRAIN_NGPUS_PER_NODE}" \
  trainer.v1.trainer_mode="${TRAINER_MODE:-colocate_async}" \
  trainer.v1.colocate_async.num_warmup_batches="${NUM_WARMUP_BATCHES:-1}" \
  trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD:-2}" \
  trainer.v1.sampler.max_off_policy_strategy="${MAX_OFF_POLICY_STRATEGY:-drop}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.logger='["console","tensorboard"]' \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-6}" \
  trainer.log_val_generations=5 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.max_critic_ckpt_to_keep=null \
  actor_rollout_ref.rollout.prometheus.enable=True \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype="${SGLANG_KV_CACHE_DTYPE}" \
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${SGLANG_ATTENTION_BACKEND}" \
  actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size="${SGLANG_CHUNKED_PREFILL_SIZE}" \
  actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens="${SGLANG_MAX_PREFILL_TOKENS}" \
  actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy="${SGLANG_MAMBA_SCHEDULER_STRATEGY}" \
  actor_rollout_ref.rollout.engine_kwargs.sglang.log_level="${SGLANG_LOG_LEVEL:-error}" \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.max_mamba_cache_size="${MAX_MAMBA_CACHE_SIZE}" \
  actor_rollout_ref.rollout.engine_kwargs.sglang.enable_metrics_for_all_schedulers=True \
  actor_rollout_ref.model.mtp.enable="${MTP_ENABLE}" \
  actor_rollout_ref.model.mtp.enable_train="${MTP_ENABLE_TRAIN}" \
  actor_rollout_ref.model.mtp.enable_rollout="${MTP_ENABLE_ROLLOUT}" \
  algorithm.tool_call_error_penalty.enable="${TOOL_CALL_ERROR_PENALTY_ENABLE}" \
  algorithm.tool_call_error_penalty.strategy="${TOOL_CALL_ERROR_PENALTY_STRATEGY}" \
  algorithm.tool_call_error_penalty.penalty_value="${TOOL_CALL_ERROR_PENALTY_VALUE}" \
  actor_rollout_ref.rollout.custom.agent_framework.repetition_detect.enable="${REPETITION_DETECT_ENABLE}" \
  actor_rollout_ref.rollout.custom.agent_framework.repetition_detect.zero_reward="${REPETITION_ZERO_REWARD}" \
  algorithm.repetition_penalty.enable="${REPETITION_PENALTY_ENABLE}" \
  algorithm.repetition_penalty.strategy="${REPETITION_PENALTY_STRATEGY}" \
  algorithm.repetition_penalty.penalty_value="${REPETITION_PENALTY_VALUE}" \
  algorithm.deep_failure_mask.enable="${DEEP_FAILURE_MASK_ENABLE}" \
  algorithm.deep_failure_mask.strategy="${DEEP_FAILURE_MASK_STRATEGY}" \
  algorithm.deep_failure_mask.alpha="${DEEP_FAILURE_MASK_ALPHA}" \
  actor_rollout_ref.rollout.custom.agent_framework.ship_turn_index="${DEEP_FAILURE_MASK_ENABLE}" \
  algorithm.group_advantage_by_harness="${ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS:-false}" \
  "${RAY_ENV[@]}" \
  "${OPTIONAL_OVERRIDES[@]}" \
  "$@"
)

mkdir -p "${RUN_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}" "${CHECKPOINT_DIR}"

if ! "${MAIN_CMD[@]}" --cfg job --resolve >"${RESOLVED_CONFIG_PATH}"; then
  echo "failed to resolve effective Hydra config; refusing to launch" >&2
  exit 2
fi

VALIDATE_REASONING_ARGS=()
[ -n "${ROLLOUT_REASONING_EFFORT}" ] && VALIDATE_REASONING_ARGS+=(--reasoning-effort "${ROLLOUT_REASONING_EFFORT}")
python3 "${SCRIPT_DIR}/validate_resolved_config.py" "${RESOLVED_CONFIG_PATH}" \
  --save-freq "${SAVE_FREQ}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --rollout-data-dir "${ROLLOUT_DATA_DIR}" \
  --validation-data-dir "${VALIDATION_DATA_DIR}" \
  --temperature "${ROLLOUT_TEMPERATURE}" \
  --top-p "${ROLLOUT_TOP_P}" \
  --top-k "${ROLLOUT_TOP_K}" \
  "${VALIDATE_REASONING_ARGS[@]}" \
  --mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
  --mamba-scheduler "${SGLANG_MAMBA_SCHEDULER_STRATEGY}" \
  --entropy-coeff "${ENTROPY_COEFF:-0}" \
  --filter-groups-enabled "$(printf '%s' "${FILTER_GROUPS_ENABLE:-False}" | tr '[:upper:]' '[:lower:]')" \
  --tool-call-error-penalty-enabled "$(printf '%s' "${TOOL_CALL_ERROR_PENALTY_ENABLE}" | tr '[:upper:]' '[:lower:]')" \
  --tool-call-error-penalty-strategy "${TOOL_CALL_ERROR_PENALTY_STRATEGY}" \
  --tool-call-error-penalty-value "${TOOL_CALL_ERROR_PENALTY_VALUE}" \
  --repetition-detect-enabled "$(printf '%s' "${REPETITION_DETECT_ENABLE}" | tr '[:upper:]' '[:lower:]')" \
  --repetition-zero-reward "$(printf '%s' "${REPETITION_ZERO_REWARD}" | tr '[:upper:]' '[:lower:]')" \
  --repetition-penalty-enabled "$(printf '%s' "${REPETITION_PENALTY_ENABLE}" | tr '[:upper:]' '[:lower:]')" \
  --repetition-penalty-strategy "${REPETITION_PENALTY_STRATEGY}" \
  --repetition-penalty-value "${REPETITION_PENALTY_VALUE}"

python3 "${SCRIPT_DIR}/../write_run_manifest.py" \
  --run-dir "${RUN_DIR}" \
  --verl-repo "${REPO_ROOT}" \
  --component "mimoagent=${MIMOAGENT_SRC}" \
  --component "uni_agent=${UNI_AGENT_ROOT}" \
  --config-artifact "${MIMOAGENT_HARNESS_SPEC}" \
  --resolved-config "${RESOLVED_CONFIG_PATH}" \
  --command "${MAIN_CMD[@]}"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "preflight passed; resolved config and provenance written to ${RUN_DIR}"
  exit 0
fi

exec "${MAIN_CMD[@]}"
