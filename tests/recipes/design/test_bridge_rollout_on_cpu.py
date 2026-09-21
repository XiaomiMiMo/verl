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
"""End-to-end test of the web-dev bridge against a real tokenizer.

Where ``test_bridge_contract_on_cpu.py`` checks the output contract with fakes, this file
drives the *production* path: the real ``WebdevAgentLoop``, the real ``_VerlRolloutModel``
(including the ``run_coroutine_threadsafe`` hop from the agent thread back onto the event
loop), a real chat template, the real ``ToolParser``, and the real ``AgentLoopOutput``
assembly. Only two things are faked, and both are genuinely external:

* the rollout engine -- a stub ``server_manager.generate`` replaying canned token ids;
* the pod -- a Ray actor with the same method surface as ``DatasetEnvActor``, so ``ray.get``
  from the agent thread and ``ray.kill`` on teardown behave for real.

**Requires a checkpoint directory and skips without one.** The assertions here are about
token bookkeeping -- mask alignment, append-only growth, budget truncation -- and a fake
tokenizer would make them vacuous, so there is no fallback path.

    WEBDEV_TEST_MODEL=/path/to/ckpt \\
    PYTHONPATH=.:third_party/mimoagent-osr/src \\
      pytest tests/recipes/design/test_bridge_rollout_on_cpu.py -v

Needs MimoAgent, that tokenizer, and a local Ray instance (CPU only).
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

pytest.importorskip("mimoagent", reason="mimoagent is not importable")

import ray  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import verl  # noqa: E402

# Imported by canonical path so Ray workers can resolve the actor classes -- see fakes.py.
from tests.recipes.design.fakes import RECORDER_NAME, FailingEnv, Recorder, StubEnv  # noqa: E402
from verl.experimental.agent_loop.agent_loop import DictConfigWrap  # noqa: E402
from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

MODEL_PATH = os.environ.get("WEBDEV_TEST_MODEL", "")
CONFIG_PATH = "config/agent/design/webdev.yaml"

# Shaped like a row of the training parquet: the instance dict is what the loop reads out of
# extra_info.instance_json.
INSTANCE = {
    "instance_id": "webdev__study-bites-1",
    "task_id": "study-bites-1",
    "dataset_type": "webdev",
    "category": "website",
    "docker_image": "example.invalid/webdev-rl:v2",
    "problem_statement": "Build a static site for a tutoring cafe called Study Bites.",
    "cwd": "/workspace",
}


class _StubServerManager:
    """Replays canned generations; records the prompt it was handed on each turn."""

    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.seen_prompts: list[list[int]] = []
        self.seen_sampling_params: list[dict] = []
        self.request_ids: list[str] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.seen_prompts.append(list(prompt_ids))
        self.seen_sampling_params.append(dict(sampling_params))
        self.request_ids.append(request_id)
        text = self.texts.pop(0)
        return TokenOutput(token_ids=self.tokenizer.encode(text, add_special_tokens=False), log_probs=None)


@pytest.fixture
def fresh_recorder():
    """A recorder per test, under the fixed name the env stub looks up.

    Killed and recreated each time rather than shared: it is detached, so its event list
    would otherwise leak across tests.
    """
    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
            namespace="design-tests",
        )
    handle = Recorder.options(name=RECORDER_NAME, lifetime="detached", get_if_exists=True).remote()
    ray.get(handle.reset.remote())
    yield handle


@pytest.fixture(scope="module")
def trainer_config():
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(
            "set WEBDEV_TEST_MODEL to a checkpoint directory; these assertions are about real "
            "tokenization and are vacuous without one"
        )
    config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={MODEL_PATH}",
                "actor_rollout_ref.rollout.prompt_length=4096",
                "actor_rollout_ref.rollout.response_length=4096",
                # The canned generations below are hermes-shaped. Production runs
                # qwen3_coder (Qwen3.5 emits XML tool calls); the parser choice is the
                # launcher's and is covered by the config verification script, not here.
                "actor_rollout_ref.rollout.multi_turn.format=hermes",
            ],
        )


@pytest.fixture(scope="module")
def tokenizer(trainer_config):
    return AutoTokenizer.from_pretrained(MODEL_PATH)


def _make_agent_loop(trainer_config, tokenizer, server_manager, monkeypatch, env_cls=StubEnv, **kwargs):
    from recipes.design import agent_loop as agent_loop_module

    monkeypatch.setattr(agent_loop_module, "DatasetEnvActor", env_cls)
    return agent_loop_module.WebdevAgentLoop(
        trainer_config=DictConfigWrap(config=trainer_config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config=trainer_config.data),
        config_path=CONFIG_PATH,
        tools=None,
        **kwargs,
    )


def _hermes(command: str) -> str:
    return f'<tool_call>\n{{"name": "Bash", "arguments": {{"command": "{command}"}}}}\n</tool_call>'


def _run(loop, **extra_info):
    return asyncio.run(
        loop.run(
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            extra_info={"instance_json": json.dumps(INSTANCE), "index": 0, **extra_info},
        )
    )


def test_full_rollout_through_real_bridge(fresh_recorder, trainer_config, tokenizer, monkeypatch):
    server = _StubServerManager(
        tokenizer,
        [
            "Looking around.\n" + _hermes("ls /workspace"),
            "Writing the page.\n" + _hermes("mkdir -p /workspace/dist"),
            "Delivered: dist/index.html is in place.",
        ],
    )
    loop = _make_agent_loop(trainer_config, tokenizer, server, monkeypatch, per_turn_max_tokens=512)
    output = _run(loop)

    # --- metadata comes from the environment ---
    assert output.extra_fields["instance_id"] == "webdev__study-bites-1"
    assert output.extra_fields["exit_status"] == "Idle", "last turn had no tool call -> normal exit"
    assert output.extra_fields["n_model_calls"] == 3
    assert output.extra_fields["truncated"] is False
    # system + instance + 3 assistant + 2 tool
    assert output.num_turns == 7

    # --- the group path: the per-rollout reward is a placeholder, and the marker survives ---
    assert output.reward_score == 0.0, "group rows carry a placeholder, not a graded number"
    assert output.extra_fields["webdev_group_pending"] is True, (
        "without this marker the driver's group rewrite never fires and the row stays 0.0"
    )
    assert output.extra_fields["webdev_group_query_score"] == 0.6
    assert output.extra_fields["webdev_query"] == INSTANCE["problem_statement"]

    # --- the prompt is the profile's rendered templates, with the real tool schemas ---
    prompt_text = tokenizer.decode(output.prompt_ids)
    assert "CWD: /workspace" in prompt_text, "environment template vars must land"
    assert INSTANCE["problem_statement"] in prompt_text, "problem_statement is the task"
    for tool_name in ("Bash", "Read", "Write", "Edit", "Grep", "Glob"):
        assert tool_name in prompt_text, f"the {tool_name} schema must be rendered into the prompt"

    # --- mask: 1 exactly on generated text, 0 on injected observations ---
    trained = tokenizer.decode([t for t, m in zip(output.response_ids, output.response_mask, strict=True) if m == 1])
    injected = tokenizer.decode([t for t, m in zip(output.response_ids, output.response_mask, strict=True) if m == 0])
    assert "Looking around." in trained and "Delivered:" in trained
    assert "[Bash] ok" in injected
    assert "[Bash] ok" not in trained, "tool output must never be trained on"
    assert "Delivered:" not in injected

    # --- the tool parser actually drove the tools, and teardown happened ---
    tool_events = ray.get(fresh_recorder.events_of.remote("tool"))
    assert [e["name"] for e in tool_events] == ["Bash", "Bash"]
    assert tool_events[0]["params"] == {"command": "ls /workspace"}
    assert len(ray.get(fresh_recorder.events_of.remote("reward"))) == 1
    assert len(ray.get(fresh_recorder.events_of.remote("cleanup"))) == 1, "must clean up on the happy path"

    # --- sticky request id + shrinking per-turn budget ---
    assert len(set(server.request_ids)) == 1, "all turns must reuse one request_id for prefix caching"
    budgets = [p["max_tokens"] for p in server.seen_sampling_params]
    assert budgets[0] == 512 and all(b <= 512 for b in budgets)
    # Turn N's prompt is turn N-1's prompt plus that turn's generation and observation.
    assert len(server.seen_prompts[1]) > len(server.seen_prompts[0])
    assert server.seen_prompts[1][: len(server.seen_prompts[0])] == server.seen_prompts[0], (
        "the conversation must grow by append only -- a rewrite breaks prefix caching and on-policyness"
    )


def test_the_tokenizer_eos_is_added_to_the_stop_set(fresh_recorder, trainer_config, tokenizer, monkeypatch):
    """The engine's idea of eos can disagree with the chat template's, silently.

    Measured on Qwen3.5-9B: the template closes every turn with ``<|im_end|>`` while
    config.json's ``eos_token_id`` is ``<|endoftext|>``. Stopping only on the latter let one
    generate call run past the turn boundary and SELF-PLAY the rest of the conversation --
    one such call parsed as 289 tool calls, and 89% of rollouts ended on budget exhaustion.
    """
    server = _StubServerManager(tokenizer, ["done, no tool call"])
    loop = _make_agent_loop(trainer_config, tokenizer, server, monkeypatch)
    _run(loop)

    stop_ids = server.seen_sampling_params[0]["stop_token_ids"]
    eos = tokenizer.eos_token_id
    expected = set(eos) if isinstance(eos, list | tuple) else {eos}
    assert expected <= set(stop_ids), "the tokenizer's own eos must be in the stop set"


def test_env_setup_failure_still_yields_a_scored_sample(fresh_recorder, trainer_config, tokenizer, monkeypatch):
    """A dead environment must not produce reward_score=None: that drops the batch's rm_scores."""
    loop = _make_agent_loop(
        trainer_config, tokenizer, _StubServerManager(tokenizer, []), monkeypatch, env_cls=FailingEnv
    )
    output = _run(loop)

    assert output.reward_score == 0.0, "must be a float, not None"
    assert "pod stuck in Pending" in output.extra_fields["env_setup_error"]
    assert len(output.response_ids) == len(output.response_mask) == 1
    assert len(ray.get(fresh_recorder.events_of.remote("cleanup"))) == 1, "failed setup must still clean up"


def test_response_budget_truncates_instead_of_overflowing(fresh_recorder, trainer_config, tokenizer, monkeypatch):
    """When the trajectory outgrows response_length the rollout stops and reports it."""
    # Every turn calls a tool, so the agent would otherwise run to step_limit.
    server = _StubServerManager(tokenizer, ["step " + _hermes("ls")] * 70)
    loop = _make_agent_loop(trainer_config, tokenizer, server, monkeypatch, per_turn_max_tokens=64)
    loop.response_length = 300

    output = _run(loop)

    assert output.extra_fields["truncated"] is True
    assert len(output.response_mask) <= 300
    assert output.extra_fields["exit_status"] == "ModelQueryError", (
        "the budget exception surfaces through the agent's query wrapper"
    )
    assert output.extra_fields["webdev_group_pending"] is True, "a truncated trajectory is still graded"


def test_an_over_long_brief_shortens_the_task_not_the_system_prompt(
    fresh_recorder, trainer_config, tokenizer, monkeypatch
):
    """The prompt cap must cut the brief, never left-truncate.

    verl's own cap keeps the TAIL, which on this layout throws away the system prompt and all
    six tool schemas: the model is then asked to call tools it was never shown and scores 0
    by construction.
    """
    server = _StubServerManager(tokenizer, ["done, no tool call"])
    loop = _make_agent_loop(trainer_config, tokenizer, server, monkeypatch)
    loop.rollout_config.prompt_length = 1500

    huge = dict(INSTANCE, problem_statement="Build a site. " + ("filler sentence. " * 4000))
    output = asyncio.run(loop.run({"temperature": 0.6}, extra_info={"instance_json": json.dumps(huge), "index": 0}))

    prompt_text = tokenizer.decode(output.prompt_ids)
    assert len(output.prompt_ids) <= 1500
    assert "elided to fit the prompt budget" in prompt_text, "the brief must be middle-truncated"
    assert "CWD: /workspace" in prompt_text, "the system prompt must survive the cap"
    for tool_name in ("Bash", "Read", "Write", "Edit", "Grep", "Glob"):
        assert tool_name in prompt_text, f"{tool_name}'s schema must survive the cap"
