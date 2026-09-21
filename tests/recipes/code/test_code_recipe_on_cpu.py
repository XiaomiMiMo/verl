# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[3]
RECIPE_ROOT = REPO_ROOT / "recipes" / "code"
AGENT_CONFIG_DIR = REPO_ROOT / "config" / "agent" / "code"


def _load_recipe_config():
    return yaml.safe_load((RECIPE_ROOT / "config" / "train.yaml").read_text())


def test_blackbox_recipe_selects_uni_agent_adapter_and_runner():
    config = (RECIPE_ROOT / "config" / "train.yaml").read_text()

    assert "uni_agent.framework.entry.AgentFrameworkRolloutAdapter" in config
    assert "recipes.code.mimoagent_runner" in config
    assert "recipes.code.dataset" in config
    assert "recipes.code.reward" in config
    assert "third_party.uni_agent.examples.blackbox_recipes" not in config
    assert "tool_image" not in config
    assert "trainer_mode: colocate_async" in config
    assert "name: sglang" in config
    assert "kv_cache_dtype: fp8_e4m3" in config
    assert "enable_metrics_for_all_schedulers: true" in config
    assert "transfer_queue:\n  enable: true" in config


def test_blackbox_training_defaults_match_reference_qwen38_run():
    config = _load_recipe_config()
    actor_rollout_ref = config["actor_rollout_ref"]
    actor = actor_rollout_ref["actor"]
    megatron = actor["megatron"]
    rollout = actor_rollout_ref["rollout"]
    trainer = config["trainer"]

    assert actor["ppo_max_token_len_per_gpu"] == 131072
    assert actor["megatron"]["context_parallel_size"] == 2
    assert actor_rollout_ref["ref"]["log_prob_max_token_len_per_gpu"] == 131072
    assert actor_rollout_ref["ref"]["megatron"]["context_parallel_size"] == 2
    assert actor["ppo_mini_batch_size"] == 8
    assert actor["clip_ratio"] == actor["clip_ratio_low"] == actor["clip_ratio_high"] == 0.2
    assert actor["clip_ratio_c"] == 3.0
    assert actor["optim"]["weight_decay"] == 0.01
    assert megatron["use_remove_padding"] is False
    assert megatron["override_transformer_config"] == {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "attention_backend": "auto",
    }
    assert rollout["nnodes"] == 0
    assert rollout["n"] == 16
    assert rollout["max_model_len"] == 262144
    assert rollout["log_prob_max_token_len_per_gpu"] == 131072
    assert rollout["custom"]["agent_framework"]["agent_runners"]["mimoagent"]["trajectory_selection"] == "longest"
    assert rollout["max_num_seqs"] == 512
    assert "reasoning_effort" not in config["data"]["apply_chat_template_kwargs"]
    assert trainer["test_freq"] == -1
    assert trainer["save_freq"] == 5
    assert trainer["max_actor_ckpt_to_keep"] is None
    assert trainer["max_critic_ckpt_to_keep"] is None
    assert "NCCL_P2P_DISABLE" not in config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
    assert "NCCL_SHM_DISABLE" not in config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]


def test_blackbox_specific_overrides_remain_explicit():
    config = _load_recipe_config()

    assert config["actor_rollout_ref"]["actor"]["use_rollout_log_probs"] is True
    assert config["algorithm"]["rollout_correction"]["bypass_mode"] is False
    assert config["transfer_queue"]["enable"] is True
    assert config["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]["gateway_count"] == 1


def test_blackbox_launcher_prioritizes_current_verl_checkout():
    launcher = (RECIPE_ROOT / "run_train.sh").read_text()
    expected = 'PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${UNI_AGENT_ROOT}:'

    assert expected in launcher
    assert "third_party/uni_agent" in launcher
    assert "MIMOAGENT_SRC" in launcher
    assert "CLAUDE_CODE_TOOL_IMAGE" not in launcher
    assert "MODEL_PATH" in launcher and "TRAIN_DATA" in launcher and "VAL_DATA" in launcher
    assert "AGENT_NUM_WORKERS" in launcher
    assert "ROLLOUT_MAX_RUNNING_REQUESTS" in launcher
    assert "actor_rollout_ref.rollout.name=sglang" in launcher
    assert "agent_runners.mimoagent.trajectory_selection=longest" in launcher
    assert "agent_runners.mimoagent.trajectory_selection=all" not in launcher
    assert 'data.train_files="${TRAIN_DATA_HYDRA}"' in launcher
    assert 'data.val_files="${VAL_DATA_HYDRA}"' in launcher
    assert "mamba_scheduler_strategy" in launcher
    assert 'export METRIC_PORT="${METRIC_PORT:-20000}"' in launcher
    assert 'ACTOR_CP="${ACTOR_CP:-2}"' in launcher
    assert 'PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAXLEN / ACTOR_CP))}"' in launcher
    assert 'USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"' in launcher
    assert 'actor_rollout_ref.actor.megatron.use_remove_padding="${USE_REMOVE_PADDING}"' in launcher
    assert "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full" in launcher
    assert "data.apply_chat_template_kwargs.reasoning_effort" in launcher
    assert "CUDA_DEVICE_MAX_CONNECTIONS" in launcher
    assert "trainer.max_actor_ckpt_to_keep=null" in launcher
    assert "trainer.max_critic_ckpt_to_keep=null" in launcher
    assert 'trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"' in launcher
    assert 'trainer.validation_data_dir="${VALIDATION_DATA_DIR}"' in launcher
    assert 'RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}/${RUN_ID}}"' in launcher
    assert 'CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"' in launcher
    assert 'if [ "${PREFLIGHT_ONLY:-0}" = "1" ]' in launcher
    assert 'SAVE_FREQ="${SAVE_FREQ:-5}"' in launcher
    assert "enable_metrics_for_all_schedulers=True" in launcher


def test_run_manifest_captures_small_untracked_files_and_excludes_outputs(tmp_path):
    import json
    import subprocess

    from recipes import write_run_manifest

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "source.py").write_text("value = 1\n")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "rollout.jsonl").write_text("generated\n")

    snapshot = tmp_path / "snapshot"
    records = write_run_manifest._snapshot_untracked(repo, snapshot)

    by_path = {record["path"]: record for record in records}
    assert by_path["source.py"]["captured"] is True
    assert (snapshot / "source.py").read_text() == "value = 1\n"
    assert by_path["outputs/rollout.jsonl"]["captured"] is False
    assert not (snapshot / "outputs" / "rollout.jsonl").exists()
    json.dumps(records)


def test_resolved_config_validator_rejects_hidden_override(tmp_path):
    from recipes.code import validate_resolved_config

    config = {
        "algorithm": {"rollout_correction": {"bypass_mode": False}},
        "transfer_queue": {"enable": True},
        "trainer": {
            "v1": {"trainer_mode": "sync"},
            "save_freq": 5,
            "max_actor_ckpt_to_keep": None,
            "max_critic_ckpt_to_keep": None,
            "default_local_dir": "/run/checkpoints",
            "rollout_data_dir": "/run/rollouts",
            "validation_data_dir": "/run/validation",
            "logger": ["console", "tensorboard"],
        },
        "actor_rollout_ref": {
            "rollout": {
                "prometheus": {"enable": True},
                "disable_log_stats": False,
                "engine_kwargs": {"sglang": {"enable_metrics_for_all_schedulers": True}},
            }
        },
    }

    with pytest.raises(ValueError, match="trainer.v1.trainer_mode"):
        validate_resolved_config._check(config, "trainer.v1.trainer_mode", "colocate_async")


def test_harness_helpers_are_importable_without_gateway_runtime():
    from recipes.code.mimoagent_runner import _build_model, _extract_task, _load_config

    assert _extract_task([{"role": "user", "content": "fix it"}], {}) == "fix it"
    config = _load_config(AGENT_CONFIG_DIR / "mini-mimocode.yaml")
    assert config["agent"]["type"] == "mimocode-agent"
    assert config["agent"]["step_limit"] == 500
    assert config["environment"]["environment_class"] == "kubernetes"
    model = _build_model(config, "http://gateway/session-1/v1", agent_type="mimocode-agent")
    assert model.config.model_kwargs["base_url"] == "http://gateway/session-1/v1"
    assert callable(model.query)
    assert callable(model.get_template_vars)


def test_claude_code_model_uses_anthropic_session_root():
    from recipes.code.mimoagent_runner import _build_model

    model = _build_model(
        {"model": {"model_name": "policy", "model_kwargs": {}}},
        "http://gateway/session-1/v1",
        agent_type="claude-code",
    )

    assert model.config.model_kwargs["base_url"] == "http://gateway/session-1"


def test_bashonly_agent_uses_the_gateway_model_adapter():
    from recipes.code.mimoagent_runner import _build_model

    model = _build_model(
        {"model": {"model_name": "policy", "model_kwargs": {"max_tokens": 32768}}},
        "http://gateway/session-1/v1",
        agent_type="bashonly-agent",
    )

    assert model.config.model_kwargs["base_url"] == "http://gateway/session-1/v1"
    assert callable(model.query)


def test_mimocode_agent_uses_the_gateway_model_adapter():
    from recipes.code.mimoagent_runner import _build_model

    model = _build_model(
        {"model": {"model_name": "policy", "model_kwargs": {"max_tokens": 32768}}},
        "http://gateway/session-1/v1",
        agent_type="mimocode-agent",
    )

    assert model.config.model_kwargs["base_url"] == "http://gateway/session-1/v1"
    assert callable(model.query)


def test_every_harness_arm_owns_its_environment_block():
    """No arm may rely on a shared overlay.

    The overlay used to be a separate file merged in at load time, which
    rewrote the environment of whichever arm a sample selected and made the
    arms incomparable. The replacement rule is structural: each profile
    declares its own block, and the loader has no merge path left.
    """
    spec = yaml.safe_load((AGENT_CONFIG_DIR / "mix-four-whitebox.yaml").read_text())
    assert [h["label"] for h in spec["harnesses"]] == [
        "mini-mimocode",
        "mini-bash",
        "mini-claude-code",
        "mini-codex",
    ]
    for entry in spec["harnesses"]:
        profile = yaml.safe_load((AGENT_CONFIG_DIR / entry["config"]).read_text())
        assert profile["use_dataset_env"] is True
        assert profile["environment"]["environment_class"] == "kubernetes"
        assert profile["environment"]["anti_hack_cleanup"] is True
        # One command budget across arms, or the comparison is meaningless.
        assert profile["environment"]["timeout"] == 300
        assert profile["agent"]["step_limit"] == 500
        assert "created-by" not in (profile["environment"].get("labels") or {})
    protocols = {
        yaml.safe_load((AGENT_CONFIG_DIR / e["config"]).read_text())["model"]["protocol"] for e in spec["harnesses"]
    }
    assert protocols == {"chat", "responses"}


def test_codex_model_catalog_override_does_not_change_policy_gateway():
    from recipes.code.mimoagent_runner import (
        _apply_agent_model_override,
        _build_model,
    )

    model = _build_model(
        {"model": {"model_kwargs": {}}},
        "http://gateway/session-2/v1",
        agent_type="codex",
    )
    agent_config = {"model_name": "gpt-5.6-sol"}
    _apply_agent_model_override("codex", agent_config, model)

    assert model.config.model_name == "gpt-5.6-sol"
    assert model.config.model_kwargs["base_url"] == "http://gateway/session-2/v1"
    assert "model_name" not in agent_config


def test_launcher_defaults_the_harness_spec_into_this_repository():
    """A fresh clone must be runnable without pointing anywhere outside it."""
    launcher = (RECIPE_ROOT / "run_train.sh").read_text()

    assert "MIMOAGENT_SRC:-${REPO_ROOT}/third_party/mimoagent-osr" in launcher
    assert "MIMOAGENT_HARNESS_SPEC:-${REPO_ROOT}/config/agent/code/mix-four-whitebox.yaml" in launcher
    # The five harness variables the old launcher carried are gone, not renamed
    # in one place and left behind in another.
    for dead in (
        "MINI_SWE_AGENT_SRC",
        "MINI_SWE_CONFIG_PATH",
        "MINI_SWE_ENVIRONMENT_CONFIG_PATH",
        "MINI_SWE_AGENT_COMMIT",
        "AGENT_GLOBAL_CONFIG_DIR",
        "SWEBENCH_BASE",
        "MIXED_HARNESS_MIMO_CONFIG",
        "MIXED_HARNESS_CLAUDE_CONFIG",
        "recipes/mimoagent_swe",
    ):
        assert dead not in launcher, dead


def test_paired_validation_selects_harness_by_rollout_index(monkeypatch, tmp_path):
    from recipes.code import mimoagent_runner as runner

    codex = tmp_path / "codex.yaml"
    claude = tmp_path / "claude.yaml"
    codex.write_text("agent:\n  type: codex\n")
    claude.write_text("agent:\n  type: claude-code\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        f"harnesses:\n  - label: codex\n    config: {codex}\n  - label: claude-code\n    config: {claude}\n"
    )
    monkeypatch.setenv("MIXED_HARNESS_ENABLED", "True")
    monkeypatch.setenv("MIXED_HARNESS_MODE", "paired-validation")
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))

    assert runner._select_config_path(
        sample_index=19,
        tools_kwargs={"dataset_index": 400, "rollout_index": 0},
    ) == (str(codex), "codex")
    assert runner._select_config_path(
        sample_index=19,
        tools_kwargs={"dataset_index": 400, "rollout_index": 1},
    ) == (str(claude), "claude-code")


def test_paired_subgroup_selects_harness_by_rollout_index(monkeypatch, tmp_path):
    from recipes.code import mimoagent_runner as runner

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("agent:\n  type: default\n")
    second.write_text("agent:\n  type: bashonly-agent\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"harnesses:\n  - label: first\n    config: {first}\n  - label: second\n    config: {second}\n")
    monkeypatch.setenv("MIXED_HARNESS_ENABLED", "True")
    monkeypatch.setenv("MIXED_HARNESS_MODE", "paired-subgroup")
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))

    selections = [
        runner._select_config_path(
            sample_index=7,
            tools_kwargs={"dataset_index": 400, "rollout_index": rollout_index},
        )[1]
        for rollout_index in range(8)
    ]
    assert selections == ["first", "second"] * 4


def test_step_hash_rotates_stable_sample_across_steps(monkeypatch, tmp_path):
    from recipes.code import mimoagent_runner as runner

    codex = tmp_path / "codex.yaml"
    claude = tmp_path / "claude.yaml"
    codex.write_text("agent:\n  type: codex\n")
    claude.write_text("agent:\n  type: claude-code\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        f"harnesses:\n  - label: codex\n    config: {codex}\n  - label: claude-code\n    config: {claude}\n"
    )
    monkeypatch.setenv("MIXED_HARNESS_ENABLED", "True")
    monkeypatch.setenv("MIXED_HARNESS_MODE", "step-hash")
    monkeypatch.setenv("MIXED_HARNESS_SEED", "0")
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))

    def select(step, rollout_index=0):
        return runner._select_config_path(
            sample_index=19,
            tools_kwargs={
                "dataset_index": 400,
                "rollout_index": rollout_index,
                "harness_round": step,
                "instance": {"instance_id": "stable-400"},
            },
        )

    assert select(0) == select(0, rollout_index=15)
    assert select(0) != select(3)


def test_step_hash_requires_submission_step(monkeypatch, tmp_path):
    from recipes.code import mimoagent_runner as runner

    codex = tmp_path / "codex.yaml"
    claude = tmp_path / "claude.yaml"
    codex.write_text("agent:\n  type: codex\n")
    claude.write_text("agent:\n  type: claude-code\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        f"harnesses:\n  - label: codex\n    config: {codex}\n  - label: claude-code\n    config: {claude}\n"
    )
    monkeypatch.setenv("MIXED_HARNESS_ENABLED", "True")
    monkeypatch.setenv("MIXED_HARNESS_MODE", "step-hash")
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))

    with pytest.raises(ValueError, match="harness_round"):
        runner._select_config_path(sample_index=0, tools_kwargs={"dataset_index": 0})


def test_default_agent_idle_status_is_a_success(monkeypatch):
    from recipes.code import mimoagent_runner as runner

    class FakeEnvironment:
        env = object()

        def setup_environment(self):
            return None

        def calculate_reward(self):
            return 1.0, "passed", {"verifier": "ok"}

        def cleanup(self):
            return None

    class FakeDefaultAgent:
        IDLE_STATUS = "Idle"

        def __init__(self, model, env, **kwargs):
            assert callable(model.query)
            self.tool_call_errors = [True, False, True]

        def run(self, task):
            return "Idle", f"fixed: {task}"

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: FakeDefaultAgent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", lambda *args, **kwargs: FakeEnvironment())

    result = runner._run_sync(
        raw_prompt="fix it",
        instance={"problem_statement": "fix it"},
        session=SimpleNamespace(base_url="http://gateway/session-1/v1"),
        config=_load_recipe_config()
        | {
            "agent": {"type": "default"},
            "environment": {"environment_class": "kubernetes"},
            "model": {
                "model_name": "policy",
                "protocol": "chat",
                "model_kwargs": {"max_tokens": 8},
            },
        },
        agent_overrides={},
        environment_overrides={},
    )

    assert result["reward"] == 1.0
    assert result["finished"] is True
    assert result["agent_type"] == "default"
    assert result["agent_status"] == "Idle"
    assert result["agent_completed"] is True
    assert result["termination_kind"] == "completed"
    assert result["verifier"] == "ok"
    assert result["tool_call_error_flags"] == [True, False, True]
    assert result["tool_call_error_flag_source"] == "agent_step"
    assert result["tool_call_error_count"] == 2


@pytest.mark.parametrize("status", ["LimitsExceeded", "ModelQueryError", "ClaudeCodeError", "CodexError"])
def test_noninfra_agent_status_is_graded_as_partial_rollout(monkeypatch, status):
    from recipes.code import mimoagent_runner as runner

    class FakeEnvironment:
        env = object()

        def setup_environment(self):
            return None

        def calculate_reward(self):
            return 1.0, "passed", {"verifier": "ok", "agent_completed": "must-not-win"}

        def cleanup(self):
            return None

    class FailedAgent:
        IDLE_STATUS = "Idle"

        def __init__(self, model, env, **kwargs):
            pass

        def run(self, task):
            return status, "partial rollout stopped"

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: FailedAgent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", lambda *args, **kwargs: FakeEnvironment())

    result = runner._run_sync(
        raw_prompt="fix it",
        instance={"problem_statement": "fix it"},
        session=SimpleNamespace(base_url="http://gateway/session-1/v1"),
        config={
            "agent": {"type": "default"},
            "environment": {"environment_class": "kubernetes"},
            "model": {"model_name": "policy", "protocol": "chat", "model_kwargs": {}},
        },
        agent_overrides={},
        environment_overrides={},
    )

    assert result["reward"] == 1.0
    assert result["finished"] is True
    assert result["agent_status"] == status
    assert result["agent_completed"] is False
    assert result["termination_kind"] == "truncated"
    assert result["verifier"] == "ok"


def test_infra_agent_status_still_fails_without_grading(monkeypatch):
    from recipes.code import mimoagent_runner as runner

    class FakeEnvironment:
        env = object()
        reward_called = False

        def setup_environment(self):
            return None

        def calculate_reward(self):
            self.reward_called = True
            return 1.0, "must not grade", {}

        def cleanup(self):
            return None

    environment = FakeEnvironment()

    class InfraFailedAgent:
        IDLE_STATUS = "Idle"

        def __init__(self, model, env, **kwargs):
            pass

        def run(self, task):
            return "InfraError", "pod transport failed"

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: InfraFailedAgent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", lambda *args, **kwargs: environment)

    with pytest.raises(RuntimeError, match="InfraError"):
        runner._run_sync(
            raw_prompt="fix it",
            instance={"problem_statement": "fix it"},
            session=SimpleNamespace(base_url="http://gateway/session-1/v1"),
            config={
                "agent": {"type": "default"},
                "environment": {"environment_class": "kubernetes"},
                "model": {"model_name": "policy", "protocol": "chat", "model_kwargs": {}},
            },
            agent_overrides={},
            environment_overrides={},
        )

    assert environment.reward_called is False


def test_runtime_dependencies_are_pinned_as_submodules():
    gitmodules = (REPO_ROOT / ".gitmodules").read_text()
    launcher = (RECIPE_ROOT / "run_train.sh").read_text()

    sections: dict[str, str] = {}
    current = None
    for line in gitmodules.splitlines():
        if line.startswith("[submodule "):
            current = line.split('"')[1]
            sections[current] = ""
        elif current is not None:
            sections[current] += line + "\n"

    # Both runtime dependencies must be submodules, so a run pins exact commits
    # and the manifest can record them. The assertion is on structure, not on a
    # host name: publishing changes the URLs and must not require editing tests.
    # It is checked per dependency rather than by counting every submodule in the
    # file, because other directions add their own and the code recipe's pinning
    # guarantee is not a statement about how many of those exist.
    for path in ("third_party/uni_agent", "third_party/mimoagent-osr"):
        assert path in sections, path
        assert f"path = {path}" in sections[path], path
        assert "url = " in sections[path], path
    assert "${REPO_ROOT}/third_party/mimoagent-osr" in launcher


def test_agent_finished_is_reported_to_gateway_before_grading(monkeypatch):
    """The request-idle watchdog must be disarmed while the verifier runs: the runner
    POSTs ``agent_finished`` to the session's reward_info_url before
    ``calculate_reward`` and the final reward_info POST is left to the async caller."""
    from recipes.code import mimoagent_runner as runner

    events: list = []

    class FakeEnvironment:
        env = object()

        def setup_environment(self):
            return None

        def calculate_reward(self):
            events.append("calculate_reward")
            return 1.0, "passed", {}

        def cleanup(self):
            return None

    class FakeDefaultAgent:
        IDLE_STATUS = "Idle"

        def __init__(self, model, env, **kwargs):
            self.tool_call_errors = []

        def run(self, task):
            return "Idle", "done"

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None):
            events.append(("post", url, json))
            return FakeResponse()

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: FakeDefaultAgent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", lambda *args, **kwargs: FakeEnvironment())
    monkeypatch.setattr(runner.httpx, "Client", FakeClient)

    runner._run_sync(
        raw_prompt="fix it",
        instance={"problem_statement": "fix it"},
        session=SimpleNamespace(
            session_id="s1",
            base_url="http://gateway/session-1/v1",
            reward_info_url="http://gateway/sessions/s1/reward_info",
        ),
        config=_load_recipe_config(),
        agent_overrides={},
        environment_overrides={},
    )

    assert events[0] == (
        "post",
        "http://gateway/sessions/s1/reward_info",
        {"reward_info": {"agent_finished": True, "agent_status": "Idle"}},
    )
    assert events[1] == "calculate_reward"
