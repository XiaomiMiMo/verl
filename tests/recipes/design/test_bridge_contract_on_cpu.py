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
"""Contract tests for the web-dev bridge that need no checkpoint.

What this file covers is the part of the bridge whose failures are **silent and
batch-wide**: a rollout that never got off the ground still has to produce an output that
the rest of verl's batch assembly can consume. Each invariant here corresponds to a
measured incident in which ONE bad pod out of 32 trajectories killed a 64-GPU step --
after the rollout phase had fully succeeded, so the cost was a whole step's generation.

Deliberately no real tokenizer and no Ray. Real chat-template rendering, the token
bookkeeping and the tool-parser round trip are covered by
``test_bridge_rollout_on_cpu.py``, which needs a checkpoint and skips without one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mimoagent", reason="mimoagent is not importable")

from omegaconf import OmegaConf  # noqa: E402

from recipes.design.agent_loop import (  # noqa: E402
    INVALID_REWARD_VALUE,
    SPEC_DECODE_EXTRA_KEYS,
    WebdevAgentLoop,
    _as_bool,
)
from recipes.design.token_trace import TokenTrace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE = REPO_ROOT / "config/agent/design/webdev.yaml"
REGISTRY = REPO_ROOT / "recipes/design/config/webdev_agent_loop.yaml"


class _FakeTokenizer:
    """Deterministic ChatML-shaped tokenizer, sufficient for the template probes.

    ``AgentLoopBase.__init__`` derives the system prompt and the turn separator by rendering
    throwaway messages, and this recipe additionally renders an anchor turn. All three only
    need ``apply_chat_template``. This is NOT a stand-in for a real tokenizer's template --
    it exists so the invariants below can be checked without a checkpoint.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.eos_token_id = self._id("<|im_end|>")

    def _id(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab) + 1)

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=False, **kwargs):
        parts = []
        if tools:
            names = " ".join(t["function"]["name"] for t in tools)
            parts.append(f"<|im_start|>tools {names} <|im_end|>")
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"<|im_start|>{message['role']} {content} <|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant")
        text = " ".join(parts)
        return [self._id(piece) for piece in text.split()] if tokenize else text

    def decode(self, ids) -> str:
        reverse = {v: k for k, v in self._vocab.items()}
        return " ".join(reverse.get(i, "<unk>") for i in ids)


class _FakeAgent:
    def __init__(self, n_messages: int = 5) -> None:
        self.messages = [{"role": "assistant", "content": "ok"}] * n_messages


class _FakeModel:
    n_calls = 3
    n_generated_tokens = 42
    n_images_omitted = 0


def _make_loop(**kwargs) -> WebdevAgentLoop:
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    trainer_config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/nonexistent/ckpt-1400", "tokenizer_path": None},
                "rollout": {
                    "prompt_length": 4096,
                    "response_length": 4096,
                    "multi_turn": {"format": "hermes"},
                },
            }
        }
    )
    data_config = OmegaConf.create(
        {
            "apply_chat_template_kwargs": {},
            "mm_processor_kwargs": {},
            "continuous_token": {"enable": False, "model_family": None},
        }
    )
    return WebdevAgentLoop(
        trainer_config=DictConfigWrap(config=trainer_config),
        server_manager=None,
        tokenizer=_FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config=data_config),
        config_path=str(PROFILE),
        tools=None,
        **kwargs,
    )


def _build_output(loop: WebdevAgentLoop, *, reward: float, reward_extra: dict):
    trace = TokenTrace(response_length=4096)
    trace.append_prompt([1, 2, 3])
    trace.append_generated([4, 5], log_probs=[-0.1, -0.2])
    trace.append_observation([6])
    return loop._build_output(
        trace=trace,
        agent=_FakeAgent(),
        model=_FakeModel(),
        metrics={},
        instance={"problem_statement": "build a landing page"},
        instance_id="webdev-1",
        exit_status="Idle",
        exit_message="done",
        reward=reward,
        true_reward=reward,
        test_output="",
        reward_extra=reward_extra,
        engine_extra_fields={"min_global_steps": 11, "max_global_steps": 11},
        dump_dir=None,
    )


# ---------------------------------------------------------------------------
# The batch-wide invariants
# ---------------------------------------------------------------------------


def test_failure_and_success_publish_identical_reward_extra_info_keys():
    """The two paths must agree on the key SET, not merely both have one.

    verl's ``_postprocess`` takes ``reward_extra_keys`` from ``inputs[0]`` and then
    subscripts every other row with it. So a batch whose first row succeeded and whose
    fourth row failed dies on ``KeyError`` -- and because failures are rare, the crash waits
    until one happens not to be first. The reference this was ported from documented the
    requirement in a comment and still drifted: its success path grew a fourth key while its
    failure path kept three, and its own test hardcoded the stale literal so it passed.

    Asserting set equality rather than a literal is the point of this test.
    """
    loop = _make_loop()
    ok = _build_output(loop, reward=0.7, reward_extra={"model_patch": "x", "repetition_collapse": 0.0})
    failed = loop._failure_output("env_setup_error", "pod never became ready", {}, error_category="setup/failed")

    assert set(failed.extra_fields["reward_extra_info"]) == set(ok.extra_fields["reward_extra_info"])


def test_reward_extra_info_values_are_floatable():
    """The replay buffer does ``float(reward_extra_info[metric])`` on the filter metric."""
    loop = _make_loop()
    for output in (
        _build_output(loop, reward=0.7, reward_extra={"model_patch": "x"}),
        loop._failure_output("env_setup_error", "boom", {}, error_category="setup/failed"),
    ):
        extra = output.extra_fields["reward_extra_info"]
        assert "reward" in extra, "the metric the launcher filters on"
        assert float(extra["reward"]) == output.reward_score
        for value in extra.values():
            float(value)  # must not raise


def test_failure_output_is_well_formed_for_the_whole_batch():
    """A rollout that never generated must not poison the batch it lands in.

    * ``prompt_ids=[]`` -> verl's ``_pad_token_ids`` special-cases the empty list to a zero
      attention mask, so the sample's prompt length is 0 and ``no_padding_2_padding``
      asserts ``prompt_len > 0`` on the ordinary PPO-loss path.
    * a missing ``min_global_steps`` -> the metrics pass does ``np.array([...], dtype=int)``
      over a None.
    * ``reward_score=None`` -> ``rm_scores`` is built only when EVERY row has a score, so one
      None silently drops the whole batch's reward.
    * ``response_logprobs=None`` -> the column is emitted based on ``inputs[0]`` alone, so a
      failure at index 0 drops it for everyone.
    """
    loop = _make_loop()
    out = loop._failure_output(
        "env_setup_error", "pod never became ready", {}, error_category="setup/failed", global_steps=7
    )

    assert out.prompt_ids, "an empty prompt gives the sample attention_mask=0 and breaks ppo_loss"
    assert len(out.response_ids) == len(out.response_mask) >= 1
    assert out.response_logprobs is not None and len(out.response_logprobs) == len(out.response_ids)
    assert isinstance(out.reward_score, float)
    assert sum(out.response_mask) >= 1, "an all-zero mask makes the token-mean loss divide by zero"
    for key in ("min_global_steps", "max_global_steps"):
        assert out.extra_fields.get(key) == 7, f"{key} must carry the trainer's step, not None"
    for key in SPEC_DECODE_EXTRA_KEYS:
        assert out.extra_fields.get(key) == 0, f"{key} is subscripted when MTP rollout is on"


def test_failure_output_prefers_the_engine_reported_policy_version():
    """A trajectory that generated some turns and then failed keeps the engine's real span.

    Only a rollout with zero generate calls needs the synthesized value, so the engine's
    numbers must win when they exist -- otherwise a late failure would report staleness 0 for
    a trajectory that really did span several weight versions.
    """
    loop = _make_loop()
    out = loop._failure_output(
        "trajectory_timeout",
        "timed out",
        {},
        error_category="rollout/seq_timeout",
        global_steps=7,
        engine_extra_fields={"min_global_steps": 3, "max_global_steps": 5},
    )
    assert (out.extra_fields["min_global_steps"], out.extra_fields["max_global_steps"]) == (3, 5)


def test_is_infra_is_always_present_and_distinguishes_the_two_failure_kinds():
    """The per-step infra ratio reader subscripts this key, so every sample must carry it."""
    loop = _make_loop()
    infra = loop._failure_output("env_setup_error", "boom", {}, error_category="setup/failed")
    assert infra.extra_fields["is_infra"] == 1.0
    other = loop._failure_output("some_other_failure", "boom", {}, error_category="grader/refused")
    assert other.extra_fields["is_infra"] == 0.0, "a non-infra category must not count as infra"
    ok = _build_output(loop, reward=0.5, reward_extra={})
    assert ok.extra_fields["is_infra"] == 0.0


# ---------------------------------------------------------------------------
# Flags that arrive as strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("False", False),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("none", False),
        ("null", False),
        ("", False),
        (" FALSE ", False),
        ("True", True),
        ("true", True),
        ("1", True),
        ("3600", True),
        (False, False),
        (True, True),
        (0, False),
        (1, True),
    ],
)
def test_string_flags_from_oc_env_are_parsed_not_truthy(value, expected):
    """``${oc.env:...}`` yields STRINGS, and ``bool("False")`` is True.

    Both flags this guards are load-bearing: ``fail_on_env_setup_error`` left accidentally
    true raises on the first pod that fails to come up, killing the job; and
    ``invalid_reward_for_infra`` left accidentally true puts a -999 into a GRPO group, which
    collapsed the measured pass/fail advantage gap from 2.15 to 0.004.
    """
    assert _as_bool(value) is expected


def test_the_two_env_driven_flags_go_through_the_parser():
    loop = _make_loop(fail_on_env_setup_error="False", invalid_reward_for_infra="0")
    assert loop.fail_on_env_setup_error is False
    assert loop.invalid_reward_for_infra is False
    out = loop._failure_output("env_setup_error", "boom", {}, error_category="setup/failed", global_steps=1)
    assert out.reward_score == 0.0, "with the sentinel off an infra failure scores a plain 0.0"

    on = _make_loop(invalid_reward_for_infra="true")
    assert on.invalid_reward_for_infra is True
    out_on = on._failure_output("env_setup_error", "boom", {}, error_category="setup/failed", global_steps=1)
    assert out_on.reward_score == INVALID_REWARD_VALUE, "the eval path must still emit the sentinel"


def test_per_turn_budget_is_clamped_to_the_trajectory_budget():
    """A per-turn cap above response_length would let one turn claim the whole trajectory."""
    loop = _make_loop(per_turn_max_tokens=999_999)
    assert loop.per_turn_max_tokens == loop.response_length == 4096


def test_zero_timeouts_mean_no_wall_not_an_instant_wall():
    """0 is how the training registry spells "unbounded"; reading it as a deadline would
    cancel every rollout immediately."""
    loop = _make_loop(trajectory_timeout=0, env_setup_timeout=0, reward_timeout=0)
    assert loop._trajectory_timeout_or_none is None
    assert loop._env_setup_timeout_or_none is None
    assert loop._reward_timeout_or_none is None


# ---------------------------------------------------------------------------
# The agent the profile actually selects
# ---------------------------------------------------------------------------


def test_the_remote_agent_subclasses_the_profiles_agent_not_default_agent():
    """The catalogue, not the loop, is what distinguishes this arm.

    The reference profile named no ``agent.type`` and so ran on ``DefaultAgent`` with the six
    capitalised tools injected into one global catalogue. That injection point no longer
    exists, so the profile names ``webdev-agent`` and the remote-tool subclass must be built
    on top of it -- otherwise the tool schemas shipped to the policy come from a different
    catalogue than the implementations that run in the pod.
    """
    from recipes.design.agent_loop import _remote_tool_agent_class
    from recipes.design.webdev.agent import AGENT_TYPE, WebdevAgent

    loop = _make_loop()
    assert loop.agent_type == AGENT_TYPE

    cls = _remote_tool_agent_class(loop.agent_type)
    assert WebdevAgent in cls.__mro__
    assert cls._build_tool_registry is WebdevAgent._build_tool_registry

    # Only the two tool-dispatch seams are replaced; the loop itself must stay upstream's.
    for name in ("step", "query", "run", "add_message"):
        assert getattr(cls, name) is getattr(WebdevAgent, name), f"{name} must not be overridden"
    assert set(cls.__dict__) - {"__module__", "__qualname__", "__doc__", "__abstractmethods__", "_abc_impl"} == {
        "__init__",
        "execute_action",
        "_emit_outcome",
    }


def test_the_remote_agent_class_is_cached_per_type_not_globally():
    """One registry may route rows to several harnesses; a single cached class would hand the
    second one the first one's catalogue."""
    from recipes.design.agent_loop import _remote_tool_agent_class

    first = _remote_tool_agent_class("webdev-agent")
    assert _remote_tool_agent_class("webdev-agent") is first
    other = _remote_tool_agent_class("default")
    assert other is not first


def test_an_unknown_agent_type_is_refused():
    from recipes.design.agent_loop import _remote_tool_agent_class

    with pytest.raises(ValueError, match="Unknown agent type"):
        _remote_tool_agent_class("no-such-agent")


# ---------------------------------------------------------------------------
# The registry yaml and the profile it points at
# ---------------------------------------------------------------------------


def test_the_registry_resolves_to_this_recipe():
    """A registry entry is only reachable if every one of its three identifiers lines up."""
    entries = OmegaConf.load(REGISTRY)
    assert len(entries) == 1, "the web-dev line is single-harness"
    entry = entries[0]
    assert entry.name == "webdev", "must equal the dataset's agent_name and default_agent_loop"
    assert entry._target_ == "recipes.design.agent_loop.WebdevAgentLoop"
    assert (REPO_ROOT / entry.config_path).is_file()

    module_path, _, class_name = entry._target_.rpartition(".")
    import importlib

    assert getattr(importlib.import_module(module_path), class_name) is WebdevAgentLoop


def test_importing_the_agent_loop_module_does_not_clobber_the_registry_entry():
    """Regression: verl's ``@register`` decorator used to wipe the yaml's kwargs.

    ``AgentLoopWorker.__init__`` loads the yaml into ``_agent_loop_registry`` first; hydra
    imports this module later, on the first rollout. A ``@register`` here would overwrite the
    entry with a bare ``_target_``, so rollout #1 -- which already holds its config object --
    succeeded while rollout #2 in the same worker died on a missing ``config_path``. Small
    smoke runs hid it: 2 samples across 2 workers passed, 4 did not.

    Guards the ordering rather than the decorator's absence, so this still means something if
    registration is reintroduced some other way.
    """
    import importlib

    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

    for entry in OmegaConf.load(REGISTRY):
        _agent_loop_registry[entry.name] = entry

    importlib.reload(importlib.import_module("recipes.design.agent_loop"))

    survived = _agent_loop_registry["webdev"]
    assert "config_path" in survived, (
        "importing the agent loop stripped the yaml kwargs from the registry; the second "
        "rollout in a worker would fail to instantiate"
    )


def test_the_profile_selects_the_webdev_catalogue_and_the_group_grader():
    """The three profile fields the rest of the arm is wired to."""
    profile = OmegaConf.load(PROFILE)
    assert profile.agent.type == "webdev-agent"
    assert [t.tool for t in profile.agent.tools] == ["Bash", "Read", "Write", "Edit", "Grep", "Glob"]
    assert profile.traj_grader.correctness_mode == "design_group_v1"
    assert profile.environment.cwd == "/workspace"
    # Measured: at the 0.5-CPU default, 68% of full-page screenshots timed out while the
    # pages were healthy, so reward read 0 with no error anywhere.
    assert profile.environment.cpu_request == "2"
