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
"""Incremental token bookkeeping for the mimoagent rollout bridge.

mimoagent's ``BaseAgent`` keeps the conversation as an append-only ``list[dict]`` of
messages and re-sends the whole list to ``Model.query()`` every turn. verl needs the
opposite: one flat token sequence plus a ``response_mask`` marking which tokens the
policy generated (1) versus which the environment injected (0).

Re-rendering the full message list each turn would break RL training -- the token ids
from ``apply_chat_template(all_messages)`` do not equal ``concat(prompt_ids, response_ids)``
of the individual turns (tool-call text gets rewritten, thinking gets stripped, decode-
encode is not a round trip). See verl's ``docs/advance/agent_loop.rst``, section
"Chat completion vs Token in token out".

So we build the sequence incrementally and never re-render an assistant turn: its exact
generated token ids are already in the buffer. Only the messages the *environment* adds
(tool results, and the user-facing text of a ``NonTerminatingException``) get tokenized.

This module is deliberately free of ray / tokenizer / verl imports so the offset
arithmetic can be unit-tested on CPU (see tests/test_token_trace_on_cpu.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ResponseBudgetExhausted(Exception):
    """The trajectory would exceed ``rollout.response_length``.

    Raised from the Model bridge, so mimoagent's ``BaseAgent.query()`` wraps it into a
    ``ModelQueryError`` (a TerminatingException) and the loop exits cleanly with the
    partial trajectory intact. See mimoagent/agents/base.py:186.
    """


def select_delta_messages(messages: list[dict], cursor: int) -> tuple[list[dict], int]:
    """Messages appended since ``cursor`` that still need tokenizing.

    Assistant turns are dropped: the buffer already holds the exact token ids the policy
    emitted for them, and re-rendering would produce different ids (the chat template
    reformats ``tool_calls`` and may strip reasoning content from earlier turns).

    Args:
        messages: mimoagent's full ``agent.messages`` list, append-only.
        cursor: how many messages have already been accounted for.

    Returns:
        ``(messages_to_tokenize, new_cursor)``. ``messages_to_tokenize`` preserves order
        and must be rendered in a **single** ``apply_chat_template`` call -- chat
        templates group consecutive ``role="tool"`` messages into one turn (Qwen3 wraps
        them all in a single ``<|im_start|>user`` block), so rendering them one at a time
        would emit spurious turn headers.
    """
    assert 0 <= cursor <= len(messages), f"cursor {cursor} out of range for {len(messages)} messages"
    new_messages = messages[cursor:]
    return [m for m in new_messages if m.get("role") != "assistant"], len(messages)


@dataclass
class TokenTrace:
    """The flat token sequence of one rollout, plus its response mask.

    Layout mirrors what ``AgentLoopOutput`` expects: ``token_ids`` is the whole
    conversation, and the trailing ``len(response_mask)`` tokens of it are the response.
    Everything before that is the prompt (the first rendered system+instance turn).
    """

    response_length: int
    """Token budget for the response region (``rollout.response_length``)."""

    token_ids: list[int] = field(default_factory=list)
    """Prompt + response token ids, in order."""

    response_mask: list[int] = field(default_factory=list)
    """1 for policy-generated tokens, 0 for environment-injected ones."""

    response_logprobs: list[float] = field(default_factory=list)
    """Logprobs aligned with ``response_mask``; empty when the engine returns none."""

    prompt_len: int = 0
    """Length of the leading prompt region. Set once by ``append_prompt``."""

    truncated: bool = False
    """True once the token budget stopped the rollout (as opposed to the agent going idle)."""

    _logprobs_enabled: bool = False
    """Whether the engine is returning logprobs. Latched on the first generated turn."""

    def append_prompt(self, token_ids: list[int]) -> None:
        """Install the initial prompt. Must be called exactly once, before anything else."""
        assert not self.token_ids, "append_prompt called twice"
        self.token_ids = list(token_ids)
        self.prompt_len = len(token_ids)

    def append_observation(self, token_ids: list[int]) -> None:
        """Append environment-injected tokens (tool results, error text). Masked out."""
        assert self.prompt_len > 0, "append_observation before append_prompt"
        self.token_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        if self._logprobs_enabled:
            self.response_logprobs.extend([0.0] * len(token_ids))

    def append_generated(self, token_ids: list[int], log_probs: list[float] | None = None) -> None:
        """Append policy-generated tokens. Masked in (these are what we train on)."""
        assert self.prompt_len > 0, "append_generated before append_prompt"
        if log_probs and not self._logprobs_enabled:
            self._logprobs_enabled = True
            self.response_logprobs = [0.0] * len(self.response_mask)
        self.token_ids.extend(token_ids)
        self.response_mask.extend([1] * len(token_ids))
        if self._logprobs_enabled:
            probs = list(log_probs or [])
            if len(probs) < len(token_ids):
                probs.extend([0.0] * (len(token_ids) - len(probs)))
            self.response_logprobs.extend(probs[: len(token_ids)])

    def remaining_budget(self) -> int:
        """Tokens still available in the response region. Never negative."""
        remaining = self.response_length - len(self.response_mask)
        return max(0, remaining)

    def check_budget(self, incoming: int) -> None:
        """Raise if ``incoming`` more response tokens would blow the budget.

        Called before rendering a tool observation and before each generation, so the
        rollout stops at a message boundary instead of mid-turn.
        """
        if len(self.response_mask) + incoming >= self.response_length:
            self.truncated = True
            raise ResponseBudgetExhausted(
                f"response budget exhausted: {len(self.response_mask)} used + {incoming} incoming "
                f">= response_length {self.response_length}"
            )

    @property
    def prompt_ids(self) -> list[int]:
        return self.token_ids[: self.prompt_len]

    @property
    def response_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]

    def finalize(self) -> tuple[list[int], list[int], list[int], list[float] | None]:
        """Clip to the response budget and return ``(prompt, response, mask, logprobs)``.

        ``AgentLoopOutput`` requires response/mask/logprobs to be the same length, so all
        three are cut at the same point.
        """
        response_ids = self.response_ids
        mask = self.response_mask
        assert len(response_ids) == len(mask), f"response/mask desync: {len(response_ids)} vs {len(mask)}"
        logprobs = self.response_logprobs if self._logprobs_enabled else None
        if logprobs is not None:
            assert len(logprobs) == len(mask), f"logprob/mask desync: {len(logprobs)} vs {len(mask)}"
            logprobs = logprobs[: self.response_length]
        return (
            self.prompt_ids,
            response_ids[: self.response_length],
            mask[: self.response_length],
            logprobs,
        )
