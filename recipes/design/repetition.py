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
"""Repetition-collapse detector for the web-dev rollout.

Its own module rather than part of the agent loop so the calibration below can be
tested without importing verl.
"""

from __future__ import annotations

from typing import Any


def degenerate_repetition(text: str, *, min_words: int = 500, uniq4_threshold: float = 0.15) -> bool:
    """Did generation collapse into a repetition loop?

    4-gram uniqueness of the model's own text. Measured over the 0828b run (16672
    rollouts) the distribution is strongly bimodal: healthy messages sit at median
    0.974, collapsed ones at 0.002, and the hit count barely moves between
    thresholds 0.05 and 0.30 -- 0.15 is a plateau, not a tuned knob. Catches word
    loops ("WAIT WAIT ...") AND phrase/sentence loops (".faq-item contains
    .faq-item...") that a top-token-frequency test misses (1398 such rollouts).

    ``min_words`` is free insurance rather than a tuned parameter: over those 16672
    rollouts the SHORTEST hit was 1865 words (p5 = 9867, median 30818 ~ the 32K
    per-turn cap), i.e. this never fires on normal-length reasoning. Worst case
    (32K words) costs ~6.5 ms.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return False
    return len(set(grams)) / len(grams) < uniq4_threshold


def first_degenerate_turn(messages: list[Any]) -> int | None:
    """Index of the first assistant message that collapsed into repetition.

    Judges only the model's OWN text: tool observations (file dumps, long listings)
    are legitimately repetitive and are not the policy's output.
    """
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if degenerate_repetition(str(message.get("content") or "")):
            return index
    return None


__all__ = ["degenerate_repetition", "first_degenerate_turn"]
