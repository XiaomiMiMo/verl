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
"""verl ``AgentLoop`` that drives a MimoAgent web-dev agent as one rollout.

This is a **bridge**, and the asymmetry matters: MimoAgent owns the agent loop
(``BaseAgent.run`` / ``DefaultAgent.step``), the tools, the pod environment and the
reward. Only two things are replaced:

1. MimoAgent's HTTP chat model -> :class:`_VerlRolloutModel`. Instead of a
   chat-completion request, ``query()`` re-enters verl's rollout engine
   token-in/token-out and maintains the incremental token buffer training needs.
2. ``execute_action`` -> the remote counterpart built by
   :func:`_build_remote_tool_agent_class`, which forwards each call to the Ray actor
   holding the pod. Tools must run next to the pod; see :mod:`.env_actor`.

Everything else in the agent runs unmodified: parallel tool dispatch, observation
truncation, the ToolException-to-user-message mapping, ``step_limit``,
``tool_call_errors``.

The agent class is resolved from the harness profile's ``agent.type`` rather than
hardcoded, and for this arm that resolves to ``WebdevAgent`` -- see
:mod:`recipes.design.webdev.agent` for why the catalogue, not the loop, is what
distinguishes it.

MimoAgent contracts this file depends on (breaking any of these breaks the bridge;
the CPU tests pin the first two):

* ``BaseAgent.run(task)`` primes ``[system, instance]`` then loops ``step()``, and the
  message list is append-only with assistant turns added right after ``model.query()``.
* ``DefaultAgent.execute_action(action) -> dict`` with ``action = {"tool", "params"}``,
  raising ``NonTerminatingException`` / ``TerminatingException`` / ``InfraError``.
* The ``Model`` protocol: ``query(messages, **kwargs) -> dict`` and ``get_template_vars()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from .chat_delta import anchor_ids, render_ids, render_injected_turn, render_injected_turn_mm
from .env_actor import (
    KIND_LIMITS_EXCEEDED,
    KIND_OK,
    KIND_TOOL_EXCEPTION,
    KIND_TRANSPORT_ERROR,
    DatasetEnvActor,
    load_config,
)
from .repetition import first_degenerate_turn
from .token_trace import ResponseBudgetExhausted, TokenTrace, image_spans, select_delta_messages

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

INVALID_REWARD_VALUE: float = -999.0

_REPETITION_ZERO_REWARD = os.environ.get("WEBDEV_REPETITION_ZERO_REWARD", "1") not in ("0", "false", "False")
_REPETITION_REWARD_VALUE = float(os.environ.get("WEBDEV_REPETITION_REWARD", "0.0"))

_INFRA_ERROR_CATEGORIES = frozenset(
    {
        "setup/failed",  # env/pod creation failed or timed out
        "rollout/pod_conn_timeout",  # exec stream broke mid-trajectory (env_actor._infra_error)
        "rollout/seq_timeout",  # the trajectory wall elapsed
        "reward/exception",  # grading raised
        "reward/env_error",  # env actor side grading failure
    }
)

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)


def _as_bool(value: Any) -> bool:
    """Parse a flag that may arrive as a string from ``${oc.env:...}``.

    ``bool()`` is wrong for these: OmegaConf's ``oc.env`` resolver hands back the RAW
    STRING, so ``bool("False")`` is ``True`` and ``bool("0")`` is ``True``. The numeric
    knobs next to these survive only because they go through ``float(...)``. Getting this
    wrong is not a soft failure: ``fail_on_env_setup_error`` left accidentally-true raises
    on the first pod that fails to come up and kills the whole job.
    """
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off", "none", "null")
    return bool(value)


_AGENT_POOL: ThreadPoolExecutor | None = None


def _agent_pool(size: int) -> ThreadPoolExecutor:
    """Dedicated thread pool for the blocking ``agent.run()``.

    Deliberately NOT the event loop's default executor: verl runs ``apply_chat_template``
    through ``run_in_executor(None, ...)``, and a long-lived ``agent.run()`` parked on the
    default pool would starve it and deadlock the worker.
    """
    global _AGENT_POOL
    if _AGENT_POOL is None:
        _AGENT_POOL = ThreadPoolExecutor(max_workers=size, thread_name_prefix="webdev-agent")
    return _AGENT_POOL


def _is_image_part(item: Any) -> bool:
    return isinstance(item, dict) and ("image_url" in item or item.get("type") == "image")


def _count_image_parts(messages: list[dict[str, Any]]) -> int:
    """Image content parts across a message delta (list-of-parts content only)."""
    n = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            n += sum(1 for item in content if _is_image_part(item))
    return n


def _strip_image_parts(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """Copy of ``messages`` with every image part replaced by a text note.

    Used when an image-bearing observation does not fit the token budget: the text form
    keeps the turn structure (so the conversation stays coherent) without the thousands of
    placeholder tokens, and -- critically -- without shipping placeholder tokens that have
    no matching ``pixel_values``.
    """
    stripped: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list) and any(_is_image_part(item) for item in content):
            new_content = [{"type": "text", "text": note} if _is_image_part(item) else item for item in content]
            stripped.append({**m, "content": new_content})
        else:
            stripped.append(m)
    return stripped


class _RolloutStopped(Exception):
    """The outer coroutine gave up (cancel / teardown); stop generating for this rollout."""


@dataclass
class _TemplateVarsEnv:
    """Frozen stand-in for MimoAgent's ``Environment`` on the agent side.

    ``BaseAgent.render_template`` is the only consumer -- it calls
    ``env.get_template_vars()`` to fill ``{{cwd}}`` and friends. The real environment lives
    in the Ray actor; its template vars are fetched once via ``describe()``.
    """

    template_vars: dict[str, Any] = field(default_factory=dict)
    config: Any = None

    def get_template_vars(self) -> dict[str, Any]:
        return dict(self.template_vars)


class _VerlRolloutModel:
    """MimoAgent ``Model`` implementation backed by verl's rollout engine.

    ``query()`` is called from the agent thread and hops onto the AgentLoopWorker event
    loop to do the async work. The loop is free at that moment: the coroutine that started
    the agent is parked on ``run_in_executor``.
    """

    def __init__(
        self,
        *,
        agent_loop: WebdevAgentLoop,
        trace: TokenTrace,
        tool_schema_dicts: list[dict],
        tool_schema_objs: list[OpenAIFunctionToolSchema],
        sampling_params: dict[str, Any],
        request_id: str,
        per_turn_max_tokens: int,
        metrics: dict[str, Any],
        engine_extra_fields: dict[str, Any],
        raw_dump_path: str | None = None,
        max_images: int = 0,
        image_max_pixels: int = 1048576,
        image_dump_dir: str | None = None,
    ):
        self._agent_loop = agent_loop
        self._loop = agent_loop.loop
        self._trace = trace
        self._tool_schema_dicts = tool_schema_dicts
        self._tool_schema_objs = tool_schema_objs
        self._sampling_params = sampling_params
        self._request_id = request_id
        self._per_turn_max_tokens = per_turn_max_tokens
        self._metrics = metrics
        self._engine_extra_fields = engine_extra_fields
        self._raw_dump_path = raw_dump_path
        self._max_images = int(max_images)
        self._image_max_pixels = int(image_max_pixels)
        self._image_dump_dir = image_dump_dir

        self._cursor = 0
        self.stopped = False

        self._pending_images: list[Any] = []
        self.n_images_omitted = 0

        self.config = SimpleNamespace(model_name=agent_loop.model_name)

        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_generated_tokens = 0


    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Blocking call from the agent thread; ``kwargs`` (tools/tool_choice) is ignored.

        MimoAgent passes the OpenAI tool schemas here for the HTTP path. We already rendered
        the same schemas into the prompt's system block and hand them to the tool parser, so
        there is nothing to forward.
        """
        future = asyncio.run_coroutine_threadsafe(self._agenerate(messages), self._loop)
        return future.result()

    def get_template_vars(self) -> dict[str, Any]:
        return {
            "model_name": "verl-rollout",
            "n_model_calls": self.n_calls,
            "input_tokens": self.n_prompt_tokens,
            "output_tokens": self.n_generated_tokens,
            "total_tokens": self.n_prompt_tokens + self.n_generated_tokens,
        }

    def offer_images(self, data_urls: list[str]) -> list[str] | str:
        """Stage tool-returned images for the next render; called from the agent thread.

        Returns marker URLs (``webdev://image/K``) for the caller to place in a follow-up
        user message -- the markers keep the base64 out of ``agent.messages`` (and therefore
        out of traj.json and the streamed message log) while the chat template still renders
        one ``<|image_pad|>`` per ``image_url`` part regardless of URL content. Returns a
        plain string instead when the images are rejected (no processor / per-rollout cap /
        undecodable): the caller posts that as a text note. Never raises -- a bad image must
        degrade the observation, not kill the rollout mid-trajectory.
        """
        if self._max_images <= 0 or self._agent_loop.processor is None or self._agent_loop.image_token_id is None:
            self.n_images_omitted += len(data_urls)
            return "(image omitted: this run has image reading disabled)"
        n_have = len(self._trace.images) + len(self._pending_images)
        if n_have + len(data_urls) > self._max_images:
            self.n_images_omitted += len(data_urls)
            return (
                f"(screenshot omitted: the {self._max_images}-images-per-task limit is reached — "
                "rely on your earlier screenshots and the text signals)"
            )

        import base64
        import io
        import math

        from PIL import Image

        pils: list[Any] = []
        for url in data_urls:
            try:
                _, _, b64 = url.partition(",")
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                img = img.convert("RGB")
            except Exception as e:  # noqa: BLE001 -- any decode failure degrades, never raises
                self.n_images_omitted += len(data_urls)
                return f"(image omitted: could not be decoded: {type(e).__name__})"
            w, h = img.size
            if w * h > self._image_max_pixels:
                scale = math.sqrt(self._image_max_pixels / (w * h))
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
            pils.append(img)

        markers = []
        for img in pils:
            k = len(self._trace.images) + len(self._pending_images)
            self._pending_images.append(img)
            markers.append(f"webdev://image/{k}")
            if self._image_dump_dir:
                try:
                    d = Path(self._image_dump_dir) / "agent_images"
                    d.mkdir(parents=True, exist_ok=True)
                    img.save(d / f"img_{k}.png")
                except OSError:
                    pass  # dump filesystem hiccups must never fail the rollout
        return markers


    async def _agenerate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.stopped:
            raise _RolloutStopped("rollout torn down")

        await self._absorb_new_messages(messages)

        remaining = self._trace.remaining_budget()
        if remaining <= 1:
            self._trace.truncated = True
            raise ResponseBudgetExhausted(f"only {remaining} response tokens left, stopping")

        sampling_params = dict(self._sampling_params)
        sampling_params["max_tokens"] = min(self._per_turn_max_tokens, remaining)

        stop_ids = set(sampling_params.get("stop_token_ids") or [])
        tokenizer_eos = getattr(self._agent_loop.tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, list | tuple):
            stop_ids.update(int(x) for x in tokenizer_eos)
        elif tokenizer_eos is not None:
            stop_ids.add(int(tokenizer_eos))
        if self._agent_loop.tool_parser.stop_token_ids:
            stop_ids.update(self._agent_loop.tool_parser.stop_token_ids)
        if stop_ids:
            sampling_params["stop_token_ids"] = sorted(stop_ids)

        self.n_prompt_tokens += len(self._trace.token_ids)
        with simple_timer("generate_sequences", self._metrics):
            output: TokenOutput = await self._agent_loop.server_manager.generate(
                request_id=self._request_id,
                prompt_ids=self._trace.token_ids,
                sampling_params=sampling_params,
                image_data=list(self._trace.images) if self._trace.images else None,
            )

        if self._metrics.get("num_preempted") is None:
            self._metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        elif output.num_preempted is not None:
            self._metrics["num_preempted"] += output.num_preempted

        self._trace.append_generated(output.token_ids, output.log_probs)
        self.n_calls += 1
        self.n_generated_tokens += len(output.token_ids)
        self._absorb_engine_extra_fields(output.extra_fields)

        content, tool_calls = await self._agent_loop.tool_parser.extract_tool_calls(
            output.token_ids, self._tool_schema_objs
        )
        self._dump_raw_generation(output.token_ids, tool_calls)
        tool_calls = self._cap_tool_calls(tool_calls)
        return self._build_assistant_payload(content, tool_calls)

    def _cap_tool_calls(self, tool_calls: list) -> list:
        """Keep at most ``max_tool_calls_per_turn`` calls from one generation.

        Diagnostic lever for a measured pathology, off by default. On Qwen3.5-9B the
        ``qwen3_coder`` parser turns a batched multi-call turn into a long run of junk: one
        observed assistant turn parsed as 47 calls -- 3 real reads followed by 44 named
        ``command`` with EMPTY arguments, i.e. ``<parameter=command>`` being read as
        ``<function=command>``. Across 64 trajectories that made 76.6% of all tool calls
        nonexistent-tool calls (9659/12616), each answered with "Unknown tool", which the
        model retried until the 128K budget was gone (89.1% ended on budget exhaustion).

        Capping at 1 tests that story cheaply: if the junk rate collapses, the batch parse is
        the cause. It is NOT a fix -- it also drops legitimate parallel calls -- so the real
        repair is in the parser, for which ``_dump_raw_generation`` collects the evidence.
        """
        cap = self._agent_loop.max_tool_calls_per_turn
        if cap is None or len(tool_calls) <= cap:
            return tool_calls
        self._metrics["tool_calls_dropped"] = self._metrics.get("tool_calls_dropped", 0) + (len(tool_calls) - cap)
        return tool_calls[:cap]

    def _dump_raw_generation(self, token_ids: list[int], tool_calls: list) -> None:
        """Append the raw decoded generation, before the parser strips the tool-call blocks.

        Needed because the trajectory dump only keeps the parser's *output*: ``content`` has
        the tool-call blocks removed, so the XML the model actually emitted is unrecoverable
        from it -- which is exactly what is needed to fix the parser rather than work around
        it.
        """
        path = self._raw_dump_path
        if not path:
            return
        try:
            text = self._agent_loop.tokenizer.decode(token_ids)
            record = {
                "turn": self.n_calls,
                "n_parsed_calls": len(tool_calls),
                "parsed_names": [c.name for c in tool_calls],
                "raw": text,
            }
            with open(path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 -- diagnostics must never break a rollout
            logger.warning("[webdev] raw generation dump failed: %s", e)

    def _absorb_engine_extra_fields(self, extra_fields: dict[str, Any]) -> None:
        """Carry the engine's per-generate metadata across turns.

        ``min_global_steps`` / ``max_global_steps`` say which policy version produced each
        turn; the async trainers use the spread to compute sample staleness. A multi-turn
        trajectory spans several generate calls, so seed from the first and then keep the
        widest range -- the same rule verl's own ToolAgentLoop uses. Spec-decode counters
        accumulate instead.
        """
        if not extra_fields:
            return
        if not self._engine_extra_fields:
            self._engine_extra_fields.update(extra_fields)
            return
        if extra_fields.get("max_global_steps"):
            self._engine_extra_fields["max_global_steps"] = extra_fields["max_global_steps"]
        for key in SPEC_DECODE_EXTRA_KEYS:
            if key in extra_fields and key in self._engine_extra_fields:
                self._engine_extra_fields[key] = int(self._engine_extra_fields[key]) + int(extra_fields[key])

    async def _absorb_new_messages(self, messages: list[dict[str, Any]]) -> None:
        """Tokenize whatever the environment appended since the last query."""
        delta, self._cursor = select_delta_messages(messages, self._cursor)

        if self._trace.prompt_len == 0:
            assert delta, "first query got no messages"
            prompt_ids = await self._agent_loop.render_first_prompt(delta, self._tool_schema_dicts)
            self._trace.append_prompt(prompt_ids)
            return

        if not delta:
            return

        n_image_parts = _count_image_parts(delta)
        if n_image_parts:
            assert n_image_parts <= len(self._pending_images), (
                f"delta has {n_image_parts} image parts but only {len(self._pending_images)} "
                "staged PILs — offer_images / marker bookkeeping desynced"
            )
            pils = self._pending_images[:n_image_parts]
            self._pending_images = self._pending_images[n_image_parts:]
            observation_ids, _run_lengths = await self._agent_loop.render_injected_turn_mm(delta, pils)
            if not self._trace.fits(len(observation_ids)):
                self.n_images_omitted += n_image_parts
                delta = _strip_image_parts(delta, "(screenshot omitted: token budget exhausted)")
                pils = []
                observation_ids = await self._agent_loop.render_injected_turn(delta)
            self._trace.check_budget(len(observation_ids))
            spans = (
                image_spans(observation_ids, self._agent_loop.image_token_id, offset=len(self._trace.token_ids))
                if pils
                else None
            )
            if pils:
                assert spans is not None and len(spans) == len(pils), (
                    f"expected {len(pils)} image placeholder runs in the rendered delta, found "
                    f"{len(spans or [])} — template/processor expansion desynced"
                )
            self._trace.append_observation(observation_ids, images=pils, spans=spans)
            return

        observation_ids = await self._agent_loop.render_injected_turn(delta)
        self._trace.check_budget(len(observation_ids))
        self._trace.append_observation(observation_ids)

    def _build_assistant_payload(self, content: str, tool_calls: list) -> dict[str, Any]:
        """Shape the parsed generation like an OpenAI assistant message.

        ``DefaultAgent._collect_tool_calls`` wants ``response["tool_calls"]`` to be dicts
        with ``function.name`` / ``function.arguments`` (a JSON string, which
        ``_parse_arguments`` then loads).
        """
        payload: dict[str, Any] = {"content": content}
        if not tool_calls:
            return payload
        payload["tool_calls"] = [
            {
                "id": call.tool_call_id or f"call_{self.n_calls}_{index}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for index, call in enumerate(tool_calls)
        ]
        return payload


_REMOTE_AGENT_CLASSES: dict[str, type] = {}


def _remote_tool_agent_class(agent_type: str) -> type:
    """The profile's agent class, subclassed to execute its tools in the pod's Ray actor.

    Built lazily so importing this module needs no MimoAgent: verl imports every module
    under ``verl/experimental/agent_loop`` eagerly and the config validator imports recipes,
    so a top-level ``import mimoagent`` would make those fail on a machine without it. The
    class is created the first time a rollout actually runs.

    Cached per ``agent_type`` rather than in a single slot: one registry may route different
    rows to different harnesses, and a single cached class would hand the second harness the
    first one's tool catalogue -- schemas and implementations from different agents, in a
    different process, with nothing raising.
    """
    cached = _REMOTE_AGENT_CLASSES.get(agent_type)
    if cached is not None:
        return cached

    from mimoagent.agents.base import InfraError, NonTerminatingException, TerminatingException
    from mimoagent.agents.factory import get_agent_class

    base = get_agent_class(agent_type)

    class _RemoteToolAgent(base):  # type: ignore[misc, valid-type]
        """The agent class for this profile, with tool calls executed next to the pod."""

        def __init__(self, *args, env_actor: ray.actor.ActorHandle, **kwargs):
            super().__init__(*args, **kwargs)
            self._env_actor = env_actor

        def execute_action(self, action: dict) -> dict:
            """Remote counterpart of ``DefaultAgent.execute_action``.

            ``execute_tool`` returns a tagged dict rather than raising, because a MimoAgent
            exception crossing the Ray boundary would arrive as a ``RayTaskError`` and lose
            its type. Re-raise the right type here so the base loop's control flow (retry as
            a user message vs terminate) is the original behaviour.
            """
            tool_name = action["tool"]
            try:
                outcome = ray.get(self._env_actor.execute_tool.remote(tool_name, action.get("params", {})))
            except ray.exceptions.RayError as e:
                raise InfraError(f"env actor unavailable while executing '{tool_name}': {e}") from e

            kind = outcome["kind"]
            if kind == KIND_OK:
                return outcome["result"]
            if kind == KIND_TRANSPORT_ERROR:
                raise InfraError(outcome["message"])
            if kind == KIND_LIMITS_EXCEEDED:
                raise TerminatingException(outcome["message"])
            if kind == KIND_TOOL_EXCEPTION:
                raise NonTerminatingException(outcome["message"])
            raise NonTerminatingException(outcome["message"])

        def _emit_outcome(self, call: dict, outcome: Any) -> None:
            """Tool outcome -> messages, with images split into a follow-up user turn.

            Tools that return images stash data URLs under ``metadata["_images"]``, because
            MimoAgent's ``ToolOutput`` has no media field. Pop them BEFORE the base emit --
            ``show_tool_metadata`` would otherwise print the base64 into the tool message --
            then post a separate user message, the one position vision chat templates handle
            uniformly. That message carries only ``webdev://image/K`` markers; the actual
            pixels go to the model bridge via ``offer_images``, so no base64 ever enters the
            message list and traj.json stays small.
            """
            images = None
            if isinstance(outcome, dict):
                metadata = outcome.get("metadata")
                if isinstance(metadata, dict):
                    images = metadata.pop("_images", None)
            super()._emit_outcome(call, outcome)
            if not images:
                return
            markers = self.model.offer_images(list(images))
            if isinstance(markers, str):
                self.add_message("user", markers)
            else:
                self.add_message(
                    "user",
                    [{"type": "image_url", "image_url": {"url": marker}} for marker in markers],
                )

    _REMOTE_AGENT_CLASSES[agent_type] = _RemoteToolAgent
    return _RemoteToolAgent


class WebdevAgentLoop(AgentLoopBase):
    """Run a MimoAgent web-dev agent as one verl rollout, reward included."""

    def __init__(
        self,
        *args,
        config_path: str,
        tools: Optional[Any] = None,
        tool_call_format: Optional[str] = None,
        per_turn_max_tokens: int = 8192,
        reward_timeout: float = 1800.0,
        env_setup_timeout: float = 600.0,
        fail_on_env_setup_error: bool = False,
        invalid_reward_for_infra: bool = False,
        trajectory_timeout: float = 1800.0,
        env_num_cpus: float = 1,
        env_scheduling_strategy: str = "SPREAD",
        agent_thread_pool_size: int = 64,
        max_tool_calls_per_turn: Optional[int] = None,
        max_images_per_rollout: int = 0,
        image_max_pixels: int = 1048576,
        **kwargs,
    ):
        """``tools`` is accepted and ignored: ``AgentLoopWorker`` passes verl's own tool list
        to every agent loop unconditionally, while our tools come from the harness profile.
        """
        super().__init__(*args, **kwargs)
        del tools

        self.config_path = self._resolve_config_path(config_path)
        self.agent_config = load_config(self.config_path)

        from .webdev.agent import register_webdev_agent

        register_webdev_agent()
        self.agent_type = str((self.agent_config.get("agent") or {}).get("type") or "default")
        _remote_tool_agent_class(self.agent_type)

        self.response_length = self.rollout_config.response_length
        self.per_turn_max_tokens = min(int(per_turn_max_tokens), self.response_length)
        self.reward_timeout = float(reward_timeout)
        self._reward_timeout_or_none = self.reward_timeout if self.reward_timeout > 0 else None
        self.env_setup_timeout = float(env_setup_timeout)
        self.fail_on_env_setup_error = _as_bool(fail_on_env_setup_error)
        self.invalid_reward_for_infra = _as_bool(invalid_reward_for_infra)
        self.trajectory_timeout = float(trajectory_timeout)
        self._env_setup_timeout_or_none = self.env_setup_timeout if self.env_setup_timeout > 0 else None
        self._trajectory_timeout_or_none = self.trajectory_timeout if self.trajectory_timeout > 0 else None
        self.max_images_per_rollout = int(max_images_per_rollout)
        self.image_max_pixels = int(image_max_pixels)
        self.image_token_id: int | None = None
        if self.processor is not None:
            token_id = getattr(self.processor, "image_token_id", None)
            if token_id is None:
                token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            self.image_token_id = int(token_id) if token_id is not None and int(token_id) >= 0 else None
        if self.max_images_per_rollout > 0 and (self.processor is None or self.image_token_id is None):
            logger.warning(
                "[webdev] max_images_per_rollout=%s but no multimodal processor/image token is "
                "available for this checkpoint -- image reading disabled, tools returning "
                "images will degrade to text notes.",
                self.max_images_per_rollout,
            )
        self.max_tool_calls_per_turn = int(max_tool_calls_per_turn) if max_tool_calls_per_turn else None
        self.env_num_cpus = float(env_num_cpus)
        self.env_scheduling_strategy = env_scheduling_strategy
        self.agent_thread_pool_size = int(agent_thread_pool_size)

        tool_format = tool_call_format or self.rollout_config.multi_turn.format
        self.tool_parser = ToolParser.get_tool_parser(tool_format, self.tokenizer)

        self._processing_class = self.tokenizer
        self._anchor_ids = anchor_ids(self._processing_class, **self.apply_chat_template_kwargs)

        self.dump_root = os.environ.get("WEBDEV_DEBUG_DIR")

        self.model_name = os.path.basename(str(self.config.actor_rollout_ref.model.path).rstrip("/"))

    async def render_first_prompt(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
        """Token ids for the opening [system, instance] render, tool schemas included.

        Deliberately not ``AgentLoopBase.apply_chat_template``: that one prefers
        ``self.processor``, which on a multimodal checkpoint rejects MimoAgent's plain-string
        message content. Same tokenizer, same chat template, same kwargs as every other
        render in this class.

        Bypassing it also bypasses verl's own prompt cap, so the cap is applied here -- but by
        shortening the *task text*, not by verl's left-truncation. Left-truncation keeps the
        tail, which on this prompt layout throws away the system prompt and the tool schemas:
        the model is then asked to call tools it was never shown, and scores 0 by
        construction. Over-long prompts would otherwise reach ``_pad_token_ids(padding=
        "max_length")``, which does not truncate, and then crash the whole step in
        ``_postprocess``'s ``torch.cat`` on a shape mismatch -- taking every other prompt's
        finished rollouts down with them.
        """
        budget = self._agent_loop_prompt_budget()
        messages = self._shorten_task_to_fit(messages, tools, budget)
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: render_ids(
                self._processing_class,
                messages,
                add_generation_prompt=True,
                tools=tools,
                **self.apply_chat_template_kwargs,
            ),
        )

    def _agent_loop_prompt_budget(self) -> int:
        return int(self.rollout_config.prompt_length)

    def _shorten_task_to_fit(
        self, messages: list[dict[str, Any]], tools: list[dict] | None, budget: int
    ) -> list[dict[str, Any]]:
        """Middle-truncate the last user message until the rendered prompt fits ``budget``.

        Keeps the system prompt, the tool schemas and both ends of the brief -- the headline
        requirement is at the start and the acceptance details are often at the end. Returns
        ``messages`` unchanged in the common case, so the normal path pays one render.
        """

        def n_tokens(msgs: list[dict[str, Any]]) -> int:
            return len(
                render_ids(
                    self._processing_class,
                    msgs,
                    add_generation_prompt=True,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )
            )

        total = n_tokens(messages)
        if total <= budget:
            return messages

        idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        body = messages[idx]["content"]
        marker = "\n\n[... {dropped} characters of the brief elided to fit the prompt budget ...]\n\n"
        keep = len(body)
        for _ in range(12):
            over = total - budget
            if over <= 0:
                break
            keep = max(200, keep - max(200, int(over * len(body) / max(1, total))))
            head, tail = keep // 2, keep - keep // 2
            shortened = body[:head] + marker.format(dropped=len(body) - keep) + body[-tail:]
            trial = list(messages)
            trial[idx] = {**messages[idx], "content": shortened}
            total = n_tokens(trial)
            messages = trial
        logger.warning(
            "instance %s: rendered prompt exceeded rollout.prompt_length=%d; middle-truncated "
            "the brief to %d chars (now %d tokens). Raise PROMPT_LENGTH to avoid losing context.",
            getattr(self, "_current_instance_id", "?"),
            budget,
            keep,
            total,
        )
        return messages

    async def render_injected_turn(self, messages: list[dict[str, Any]]) -> list[int]:
        """Token ids for an environment-injected turn (tool results, error text)."""
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: render_injected_turn(
                self._processing_class,
                messages,
                turn_separator=self.turn_separator,
                anchor=self._anchor_ids,
                **self.apply_chat_template_kwargs,
            ),
        )

    async def render_injected_turn_mm(
        self, messages: list[dict[str, Any]], images: list[Any]
    ) -> tuple[list[int], list[int]]:
        """Token ids for an image-bearing injected turn, placeholder runs expanded.

        The tokenizer still renders the template text (``_processing_class`` stays pinned, so
        text renders never touch the processor); the processor only expands and tokenizes this
        one delta. See ``chat_delta.render_injected_turn_mm``.
        """
        assert self.processor is not None, "render_injected_turn_mm without a multimodal processor"
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: render_injected_turn_mm(
                self._processing_class,
                self.processor,
                messages,
                turn_separator=self.turn_separator,
                anchor=self._anchor_ids,
                images=images,
                mm_processor_kwargs=self.mm_processor_kwargs,
                **self.apply_chat_template_kwargs,
            ),
        )

    @staticmethod
    def _resolve_config_path(path: str) -> str:
        """Resolve the harness profile the same way verl resolves its own config paths.

        Ray workers on other nodes may have a different cwd, so a relative path is also
        tried against the verl project root.
        """
        from verl.experimental.agent_loop.utils import resolve_config_path

        return resolve_config_path(path)


    @staticmethod
    def _reward_extra_info(
        *, reward: float, true_reward: float, model_patch_len: float, repetition_collapse: float
    ) -> dict[str, float]:
        """The ``reward_extra_info`` dict, built in exactly one place.

        Not a convenience. verl's ``_postprocess`` takes the key set from ``inputs[0]`` and
        then SUBSCRIPTS every other row with it, so the success path and the failure path
        must emit identical keys: a batch whose first row succeeded and whose fourth row
        failed dies on ``KeyError`` and takes the training step with it. Emitting the union
        from one constructor makes that class of drift impossible rather than merely
        documented -- the reference this was ported from carried the documentation and the
        drift at the same time.

        Values must be ``float()``-able: the replay buffer calls ``float()`` on the
        ``filter_groups`` metric.
        """
        return {
            "reward": float(reward),
            "true_reward": float(true_reward),
            "model_patch_len": float(model_patch_len),
            "repetition_collapse": float(repetition_collapse),
        }

    def _dump_dir(self, instance_id: str, rollout_uid: str) -> Optional[str]:
        """One directory per rollout under ``$WEBDEV_DEBUG_DIR`` (unset = no dumps).

        Not grouped by training step: the step is only in verl's trace attributes, which are
        populated exclusively when a trace backend is configured. Point ``WEBDEV_DEBUG_DIR``
        at a per-run directory instead.

        The group grading path depends on these dumps existing -- the screenshot the judge
        saw is written here and re-sent by the driver -- so this is not an optional debug
        feature for a training run.
        """
        if not self.dump_root:
            return None
        return str(Path(self.dump_root) / str(instance_id) / rollout_uid)

    def _eos_token_id(self) -> int:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, list | tuple):
            eos = eos[0] if eos else None
        return int(eos) if eos is not None else 0

    def _failure_output(
        self,
        reason_key: str,
        message: str,
        metrics: dict,
        error_category: str | None = None,
        global_steps: int | None = None,
        engine_extra_fields: dict | None = None,
    ) -> AgentLoopOutput:
        """Minimal well-formed output for a rollout that never got off the ground.

        With ``invalid_reward_for_infra`` on, an infra category scores
        ``INVALID_REWARD_VALUE`` instead of 0: infra failures are visibly invalid, not
        silently a model failure.

        Four fields here are load-bearing for the *whole batch*, not just this sample,
        because ``_postprocess`` builds optional columns all-or-nothing:

        * ``reward_score`` must be a float, not None: ``rm_scores`` is only built when
          *every* sample in the batch has one, so a single None silently drops the whole
          batch's reward.
        * ``response_logprobs`` must not be None either. ``_postprocess`` decides whether to
          emit ``rollout_log_probs`` by looking at **inputs[0] only**. A failed rollout
          landing at index 0 therefore drops the column for the batch, and with
          ``rollout.calculate_log_probs=True`` the next debug-metrics call dies on
          ``KeyError: rollout_log_probs``. Failures are rare, so the crash waits until one
          happens to be first: a latent landmine, not a deterministic bug.

          The single 0.0 logprob is a placeholder for a token that was never sampled. It
          stays unmasked (``response_mask=[1]``) because an all-failure batch with an
          all-zero mask makes the token-mean loss divide by zero; the cost is that
          ``rollout_probs_diff_max`` can be inflated by that one token, so read
          ``diff_mean`` / ``pearson_corr`` instead on any step whose batch contains a failed
          rollout.
        * ``prompt_ids`` must not be empty. verl's ``_pad_token_ids`` special-cases an empty
          list to an all-pad tensor with a zero attention mask, so the sample's prompt length
          is 0, and ``no_padding_2_padding`` -- on the ordinary PPO-loss path -- asserts
          ``not prompt_lens.eq(0).any()``. MEASURED: one pod that never reached "environment
          ready" out of 32 trajectories was enough to kill a 64-GPU step in ``update_actor``.
          One real token makes ``tokenizer.pad`` emit mask=1 for it, so the prompt length is
          1 and the sequence is degenerate but well formed.
        * ``min_global_steps`` / ``max_global_steps`` normally come from the engine, which
          stamps every generate with the weight version it used -- so a rollout that never
          generated has neither, and the metrics pass does ``np.array([...], dtype=int)`` on
          a None. MEASURED on the very next 64-GPU smoke, again from a single pod that failed
          to come up. The honest value is the trainer's current step (span 1, staleness 0):
          this sample carries no tokens, so it was neither produced by an older policy nor
          spans versions. A sentinel such as 0 or -1 would report it as maximally stale.
        """
        score = 0.0
        if self.invalid_reward_for_infra and error_category is not None and error_category in _INFRA_ERROR_CATEGORIES:
            score = INVALID_REWARD_VALUE
        extra: dict = {
            reason_key: message,
            "true_reward": score,
            "reward_extra_info": self._reward_extra_info(
                reward=score, true_reward=score, model_patch_len=0.0, repetition_collapse=0.0
            ),
            "turn_scores": [],
            "tool_rewards": [],
        }
        if error_category is not None:
            extra["error_category"] = error_category
        extra["is_infra"] = 1.0 if error_category in _INFRA_ERROR_CATEGORIES else 0.0
        for key in ("min_global_steps", "max_global_steps"):
            value = (engine_extra_fields or {}).get(key)
            if value is None:
                value = global_steps
            if value is not None:
                extra[key] = value
        for key in SPEC_DECODE_EXTRA_KEYS:
            extra.setdefault(key, (engine_extra_fields or {}).get(key, 0))
        return AgentLoopOutput(
            prompt_ids=[self._eos_token_id()],
            response_ids=[self._eos_token_id()],
            response_mask=[1],
            response_logprobs=[0.0],
            reward_score=score,
            num_turns=0,
            metrics=metrics,
            extra_fields=extra,
        )

    def _tool_schemas(self, tool_definitions: list[dict]) -> tuple[list[dict], list[OpenAIFunctionToolSchema]]:
        """Tool definitions -> (dicts for the chat template, objects for the parser).

        The XML parser needs the typed pydantic form to coerce parameter values. Non-OpenAI
        JSON-Schema keys such as ``items`` / ``minItems`` are dropped by that model; that only
        degrades array coercion to ``literal_eval``.
        """
        objs = [OpenAIFunctionToolSchema.model_validate(d) for d in tool_definitions]
        return tool_definitions, objs

    @staticmethod
    def _load_instance(extra_info: dict) -> dict:
        """Pull the MimoAgent instance dict out of the dataset row.

        ``instance_json`` (a JSON string) is the on-disk format rather than a nested dict
        column, because HuggingFace ``datasets`` coerces a dict column into a fixed Arrow
        struct -- which unions the key sets of differing instances and reshapes nested
        fields. A raw ``instance`` dict is also accepted so tests can build rows in-process.
        """
        if extra_info.get("instance_json"):
            return json.loads(extra_info["instance_json"])
        instance = extra_info.get("instance")
        if isinstance(instance, dict) and instance:
            return dict(instance)
        raise ValueError(
            "the web-dev agent loop requires extra_info.instance_json (or extra_info.instance) "
            "on every dataset row, carrying at least instance_id, problem_statement, cwd and "
            "docker_image"
        )


    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        self.loop = asyncio.get_running_loop()

        metrics: dict[str, Any] = {}
        current_global_steps = kwargs.get("global_steps")
        extra_info = kwargs.get("extra_info") or {}
        instance = self._load_instance(extra_info)
        instance_id = instance.get("instance_id") or extra_info.get("instance_id") or "unknown"
        rollout_uid = uuid.uuid4().hex
        dump_dir = self._dump_dir(instance_id, rollout_uid[:8])
        if dump_dir:
            try:
                Path(dump_dir).mkdir(parents=True, exist_ok=True)
                (Path(dump_dir) / "rollout_meta.json").write_text(
                    json.dumps(
                        {
                            "global_steps": current_global_steps,
                            "instance_id": instance_id,
                            "rollout_uid": rollout_uid,
                        }
                    )
                )
            except OSError:
                pass  # dump filesystem hiccups must never fail the rollout

        env_actor = DatasetEnvActor.options(
            num_cpus=self.env_num_cpus,
            scheduling_strategy=self.env_scheduling_strategy,
        ).remote(
            instance=instance,
            instance_id=instance_id,
            config_path=self.config_path,
            dump_dir=dump_dir,
        )

        model: _VerlRolloutModel | None = None
        try:
            try:
                ok, error = await asyncio.wait_for(env_actor.setup.remote(), timeout=self._env_setup_timeout_or_none)
            except TimeoutError as e:
                message = f"environment setup timed out after {self.env_setup_timeout:.0f}s"
                if self.fail_on_env_setup_error and not self.invalid_reward_for_infra:
                    raise RuntimeError(f"{instance_id}: {message}; refusing to score an infra failure") from e
                logger.warning("[webdev] %s: %s", instance_id, message)
                return self._failure_output(
                    "env_setup_timeout",
                    message,
                    metrics,
                    error_category="setup/failed",
                    global_steps=current_global_steps,
                )
            if not ok:
                if self.fail_on_env_setup_error and not self.invalid_reward_for_infra:
                    raise RuntimeError(
                        f"{instance_id}: environment setup failed: {error}; refusing to score an infra failure"
                    )
                logger.warning("[webdev] %s: env setup failed: %s", instance_id, error)
                return self._failure_output(
                    "env_setup_error",
                    str(error),
                    metrics,
                    error_category="setup/failed",
                    global_steps=current_global_steps,
                )

            described = await env_actor.describe.remote()
            schema_dicts, schema_objs = self._tool_schemas(described["tool_definitions"])

            trace = TokenTrace(response_length=self.response_length)
            engine_extra_fields: dict[str, Any] = {}
            model = _VerlRolloutModel(
                agent_loop=self,
                trace=trace,
                tool_schema_dicts=schema_dicts,
                tool_schema_objs=schema_objs,
                sampling_params=sampling_params,
                request_id=rollout_uid,
                per_turn_max_tokens=self.per_turn_max_tokens,
                metrics=metrics,
                engine_extra_fields=engine_extra_fields,
                raw_dump_path=(str(Path(dump_dir) / "raw_generations.jsonl") if dump_dir else None),
                max_images=self.max_images_per_rollout,
                image_max_pixels=self.image_max_pixels,
                image_dump_dir=dump_dir,
            )

            agent_kwargs = dict(self.agent_config.get("agent") or {})
            agent_kwargs.pop("type", None)
            if dump_dir:
                msg_dir = Path(dump_dir) / "agent_msgs"
                msg_dir.mkdir(parents=True, exist_ok=True)
                agent_kwargs["msg_path"] = str(msg_dir / "main.log")

            agent = _remote_tool_agent_class(self.agent_type)(
                model=model,
                env=_TemplateVarsEnv(template_vars=described["template_vars"]),
                env_actor=env_actor,
                **agent_kwargs,
            )

            task = instance.get("problem_statement") or ""
            try:
                exit_status, exit_message = await asyncio.wait_for(
                    self.loop.run_in_executor(
                        _agent_pool(self.agent_thread_pool_size),
                        lambda: agent.run(task=task),
                    ),
                    timeout=self._trajectory_timeout_or_none,
                )
            except TimeoutError:
                logger.warning(
                    "[webdev] %s: trajectory timed out after %.0fs",
                    instance_id,
                    self._trajectory_timeout_or_none or 0.0,
                )
                return self._failure_output(
                    "trajectory_timeout",
                    "agent trajectory timed out",
                    metrics,
                    error_category="rollout/seq_timeout",
                    global_steps=current_global_steps,
                    engine_extra_fields=engine_extra_fields,
                )
            logger.info(
                "[webdev] %s: exit_status=%s turns=%d tokens=%d",
                instance_id,
                exit_status,
                len(agent.messages),
                len(trace.response_mask),
            )

            with simple_timer("compute_score", metrics):
                reward, test_output, reward_extra = await env_actor.calculate_reward.remote(
                    self._reward_timeout_or_none
                )

            if (
                self.invalid_reward_for_infra
                and isinstance(reward_extra, dict)
                and reward_extra.get("error_category") in _INFRA_ERROR_CATEGORIES
            ):
                logger.warning(
                    "[webdev] %s: infra-invalid rollout (%s), scoring %s",
                    instance_id,
                    reward_extra["error_category"],
                    INVALID_REWARD_VALUE,
                )
                reward = INVALID_REWARD_VALUE

            true_reward = reward
            degenerate_turn = first_degenerate_turn(agent.messages)
            if isinstance(reward_extra, dict):
                reward_extra["repetition_collapse"] = 1.0 if degenerate_turn is not None else 0.0
            if degenerate_turn is not None and _REPETITION_ZERO_REWARD:
                logger.warning(
                    "[webdev] %s: repetition collapse at message %d, setting reward to %.2f (was %.4f)",
                    instance_id,
                    degenerate_turn,
                    _REPETITION_REWARD_VALUE,
                    reward,
                )
                reward = _REPETITION_REWARD_VALUE

            self._dump_trajectory(dump_dir, agent, exit_status, exit_message, reward)
            return self._build_output(
                trace=trace,
                agent=agent,
                model=model,
                metrics=metrics,
                instance=instance,
                instance_id=instance_id,
                exit_status=exit_status,
                exit_message=exit_message,
                reward=reward,
                true_reward=true_reward,
                test_output=test_output,
                reward_extra=reward_extra,
                engine_extra_fields=engine_extra_fields,
                dump_dir=dump_dir,
            )
        finally:
            if model is not None:
                model.stopped = True
            try:
                await env_actor.cleanup.remote()
            except Exception as e:
                logger.warning("[webdev] %s: cleanup failed: %s", instance_id, e)
            ray.kill(env_actor, no_restart=True)

    def _dump_trajectory(self, dump_dir, agent, exit_status, exit_message, reward) -> None:
        if not dump_dir:
            return
        try:
            from mimoagent.run.utils.save import save_traj

            save_traj(
                agent,
                Path(dump_dir) / "traj.json",
                print_path=False,
                exit_status=exit_status,
                result=exit_message,
                extra_info={"reward": reward},
            )
        except Exception as e:
            logger.warning("[webdev] save_traj failed: %s", e)

    def _build_output(
        self,
        *,
        trace: TokenTrace,
        agent,
        model: _VerlRolloutModel,
        metrics: dict,
        instance: dict,
        instance_id: str,
        exit_status: str | None,
        exit_message: str | None,
        reward: float,
        true_reward: float,
        test_output: str,
        reward_extra: dict,
        engine_extra_fields: dict,
        dump_dir: str | None = None,
    ) -> AgentLoopOutput:
        from mimoagent.utils.tool_call_errors import collect_tool_call_errors

        prompt_ids, response_ids, response_mask, response_logprobs = trace.finalize()

        try:
            tool_call_errors = collect_tool_call_errors([agent])
        except Exception:
            tool_call_errors = None

        model_patch_len = len(reward_extra.get("model_patch") or "")
        repetition_collapse = float(reward_extra.get("repetition_collapse") or 0.0)
        shot_path = os.path.join(str(dump_dir), "webdev_shot.jpg") if dump_dir else None

        extra_fields: dict[str, Any] = dict(engine_extra_fields)
        extra_fields.update(
            {
                "instance_id": instance_id,
                "exit_status": exit_status,
                "exit_message": (exit_message or "")[:2000],
                "n_model_calls": model.n_calls,
                "n_generated_tokens": model.n_generated_tokens,
                "truncated": trace.truncated,
                "tool_call_errors": tool_call_errors,
                "error_category": reward_extra.get("error_category"),
                "is_infra": 1.0 if reward_extra.get("error_category") in _INFRA_ERROR_CATEGORIES else 0.0,
                "repetition_collapse": repetition_collapse,
                "true_reward": true_reward,
                "model_patch_len": model_patch_len,
                "test_output_tail": (test_output or "")[-2000:],
                "reward_extra_info": self._reward_extra_info(
                    reward=reward,
                    true_reward=true_reward,
                    model_patch_len=model_patch_len,
                    repetition_collapse=repetition_collapse,
                ),
                "turn_scores": [],
                "tool_rewards": [],
                "n_images": len(trace.images),
                "n_images_omitted": model.n_images_omitted,
                "webdev_function_score": reward_extra.get("webdev_function_score"),
                "webdev_r_query": reward_extra.get("webdev_r_query"),
                "webdev_shot_path": shot_path if shot_path and os.path.exists(shot_path) else None,
                "webdev_query": instance.get("problem_statement"),
                "webdev_group_pending": reward_extra.get("webdev_group_pending"),
                "webdev_group_query_score": reward_extra.get("webdev_group_query_score"),
                "webdev_group_runtime_factor": reward_extra.get("webdev_group_runtime_factor"),
                "webdev_group_runtime_why": reward_extra.get("webdev_group_runtime_why"),
            }
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={"images": list(trace.images)} if trace.images else None,
            reward_score=float(reward),
            num_turns=len(agent.messages),
            metrics=metrics,
            extra_fields=extra_fields,
        )


__all__ = ["INVALID_REWARD_VALUE", "WebdevAgentLoop"]
