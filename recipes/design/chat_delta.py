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
"""Rendering the environment-injected turns of a multi-turn rollout, exactly.

The rollout builds its token sequence incrementally, so each time the environment appends
messages we must produce *precisely* the tokens a chat template would emit for those turns
in the middle of the conversation. Rendering them on their own does not work: chat
templates decide a turn's opening header from its neighbours.

Two concrete traps, both found by rendering the same turns in both positions
and diffing the token ids:

* Qwen3-Coder / Qwen3.5 open a tool turn with ``<|im_start|>user`` only when the *previous*
  message exists and is not itself a tool message. Rendered alone, a tool message is first,
  so the header is silently dropped and the training sequence loses a turn boundary.
* Qwen3 renders an assistant turn differently depending on whether it is the last message
  (it injects an empty ``<think></think>`` block), so you cannot locate a turn boundary by
  rendering the prefix separately and slicing at its length.

The fix is to render the injected messages *after a fixed anchor turn* and slice off the
anchor. A plain user turn is used as the anchor because it renders identically in every
position in the templates we support, and because "previous message is not a tool message"
holds for it -- which is all the tool branch inspects.

The rollout (``recipes/design/agent_loop.py``) renders every injected turn through
``render_injected_turn``, so there is one implementation of this rule rather than one
per call site.
"""

from __future__ import annotations

from typing import Any

from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.utils.tokenizer.tokenizer import normalize_token_ids

ANCHOR_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": ""}]


def render_ids(
    processing_class,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict] | None = None,
    **chat_template_kwargs,
) -> list[int]:
    """``apply_chat_template`` returning a flat list of token ids."""
    return normalize_token_ids(
        apply_chat_template(
            processing_class,
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **chat_template_kwargs,
        )
    )


def anchor_ids(processing_class, **chat_template_kwargs) -> list[int]:
    """Token ids of the anchor turn alone. Cache this: it never changes for a tokenizer."""
    return render_ids(processing_class, ANCHOR_MESSAGES, add_generation_prompt=False, **chat_template_kwargs)


def render_injected_turn(
    processing_class,
    messages: list[dict[str, Any]],
    *,
    turn_separator: list[int],
    anchor: list[int],
    **chat_template_kwargs,
) -> list[int]:
    """Token ids for ``messages`` as a mid-conversation turn, plus the next generation prompt.

    Args:
        processing_class: tokenizer or processor.
        messages: the environment-appended messages, in order. Pass them **together** in one
            call: templates merge consecutive ``role="tool"`` messages into a single turn.
        turn_separator: from ``verl...initialize_turn_separator``. The policy stops at the
            turn close token and never emits the template's trailing separator, and slicing
            off the anchor removes it too, so it is restored here.
        anchor: result of :func:`anchor_ids` for the same tokenizer/kwargs.

    Returns:
        ``turn_separator + <the injected turn> + <generation prompt>``.
    """
    assert messages, "render_injected_turn called with no messages"
    full = render_ids(
        processing_class,
        ANCHOR_MESSAGES + messages,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    assert full[: len(anchor)] == anchor, (
        "anchor turn did not render as a prefix when followed by injected messages; "
        "this chat template needs an anchor that renders identically in every position."
    )
    return list(turn_separator) + full[len(anchor) :]


def render_injected_turn_mm(
    tokenizer,
    processor,
    messages: list[dict[str, Any]],
    *,
    turn_separator: list[int],
    anchor: list[int],
    images: list[Any],
    mm_processor_kwargs: dict | None = None,
    **chat_template_kwargs,
) -> tuple[list[int], list[int]]:
    """Like :func:`render_injected_turn`, for a delta that carries images.

    The chat template is rendered to *text* by the TOKENIZER (whose template accepts both
    plain-string and typed-part content, so mixed deltas like ``[tool(str), user([image])]``
    are safe -- this deliberately avoids the multimodal processor's ``apply_chat_template``,
    which rejects plain-string content; see the agent loop's __init__). The PROCESSOR then
    tokenizes that text and expands each ``<|image_pad|>`` into the grid-sized placeholder
    run matched to the image's ``pixel_values``. The anchor turn contains no image, so its
    prefix ids are untouched by the expansion and the usual slice applies.

    The per-image run lengths are pure arithmetic on the image dimensions
    (``grid.prod() / merge_size**2``), so a later re-run of the same processor on the same
    PIL objects -- which is exactly what verl's ``_compute_multi_modal_inputs`` does when
    building the training batch -- reproduces identical grids. Pinned by
    tests/recipes/design/test_mm_delta_on_cpu.py.

    Returns:
        ``(token_ids, run_lengths)`` where ``token_ids`` is
        ``turn_separator + <the injected turn with expanded placeholder runs> + <generation
        prompt>`` and ``run_lengths`` has one expanded-run length per image, in order.
    """
    assert messages, "render_injected_turn_mm called with no messages"
    assert images, "render_injected_turn_mm called with no images -- use render_injected_turn"
    from verl.utils.tokenizer.tokenizer import build_multimodal_processor_inputs

    text = apply_chat_template(
        tokenizer,
        ANCHOR_MESSAGES + messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[text],
        images=list(images),
        mm_processor_kwargs=dict(mm_processor_kwargs or {}),
    )
    full = normalize_token_ids(model_inputs["input_ids"])
    assert full[: len(anchor)] == anchor, (
        "anchor turn did not render as a prefix when followed by an image-bearing delta; "
        "this chat template needs an anchor that renders identically in every position."
    )

    merge = int(getattr(processor.image_processor, "merge_size", 2)) ** 2
    run_lengths = [int(grid.prod()) // merge for grid in model_inputs["image_grid_thw"]]
    ids = list(turn_separator) + full[len(anchor) :]
    image_token_id = int(processor.image_token_id)
    n_pads = sum(1 for t in ids if t == image_token_id)
    assert n_pads == sum(run_lengths), (
        f"placeholder expansion desynced from image grids: {n_pads} <|image_pad|> tokens in the "
        f"rendered delta vs {sum(run_lengths)} expected from image_grid_thw {run_lengths}"
    )
    return ids, run_lengths
