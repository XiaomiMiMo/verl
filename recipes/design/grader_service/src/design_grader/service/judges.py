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
"""The query judge — requirement coverage, 0-1.

This service serves the GROUP reward (group_v1) and nothing else, so exactly
one LLM judge lives here:

    query      requirement coverage, 0-1                (per rollout)

The other half of the group reward — the relative aesthetic pick — assembles
its own prompt in group_pick.py (prompts/group_pick.md), and the runtime gate
(runtime_gate.py) needs no model at all. Together those three are every input
`group_pick.reward_one(pick_norm, query_score, runtime_factor)` reads.

What used to live here — the pointwise rubric's layout / ugly / bland /
aesthetic judges, their human-labelled demo library and the three per-rubric
score formulas — is gone: the training path never called it (every `/grade`
request carries `judge_scope="query"`, which short-circuited the whole rubric),
and evaluation goes through recipes/design/webdev/eval_mode.py inside the pod
rather than through this service.

`parse_query` returns None for anything out of band. That is deliberate and it
is what the retry loop in vision.py keys on: an unparseable verdict and an
illegal verdict are the same failure ("the model did not hand back a usable
judgment"), so they share the attempt counter and both become a drop only after
it is spent.
"""

from __future__ import annotations

import math
from pathlib import Path

from .vision import call_for, image_part, json_object

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"


def unit_interval(x, name: str) -> float | None:
    """Validate a 0-1 score. None when out of band — the caller retries.

    bool is rejected explicitly: ``True in (0.0, 1.0)`` is True in Python, so a
    JSON ``true`` would otherwise slip through as 1.0.

    (Lived in final_score.py with the rest of the pointwise domain checks; the
    query judge is the only survivor that needs one, so it moved here rather
    than keeping a module alive for a single function.)
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    if not math.isfinite(x) or not (0.0 <= x <= 1.0):
        return None
    return x


def parse_query(text: str) -> dict | None:
    d = json_object(text)
    if d is None:
        return None
    s = unit_interval(d.get("query_score"), "query_score")
    if s is None:
        return None
    return {"query_score": s, "reason": str(d.get("reason", ""))[:400]}


def _rubric(name: str) -> str:
    return (PROMPTS / name).read_text()


def check_query(page_b64: str, query: str):
    if not (query or "").strip():
        return None, "no query text"
    parts = [
        f"【用户 query】\n{query.strip()}\n\n【该 query 的产出页面截图】：",
        image_part(page_b64),
        _rubric("query_fit.md"),
    ]
    return call_for("query", parts, parse_query, temperature=1.0)
