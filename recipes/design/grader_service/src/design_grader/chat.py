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
"""Chat-mode (single-turn) reward adapters — stdlib-only.

One shared core (`_grade`: ship the generation verbatim, <think> included, to
the grader service) behind two framework-facing entry points, because the two
training frameworks express "don't score this sample" differently:

  compute_score       the reference RL framework `default_compute_score` design* dispatch.
                      Returns a float; infra drop -> INVALID_REWARD (-999.0),
                      the reference RL framework masks it (group-mean replacement, adv 0).

  compute_score_dict  verl / verl-osr `custom_reward_function` + RewardLoop.
                      Returns {"score", "design_drop", "design_band"}; that repo
                      has no invalid-reward masking, so a drop scores 0.0 with
                      design_drop=1 riding into reward_extra_info metrics
                      (same convention as its agent-path webdev_drop flag).

The training side needs exactly one setting: DESIGN_GRADER_URL (env) or
``extra_info["grader_url"]``. A missing url is a config error and raises.
"""

from __future__ import annotations

import os

from .client import grade_remote

INVALID_REWARD = -999.0  # the reference RL framework's verl.constants.INVALID_REWARD_VALUE


def _grade(solution_str: str, extra_info: dict | None) -> dict:
    extra_info = extra_info or {}
    url = extra_info.get("grader_url") or os.getenv("DESIGN_GRADER_URL")
    if not url:
        raise RuntimeError("design chat reward: DESIGN_GRADER_URL unset and no extra_info['grader_url']")

    return grade_remote(
        url,
        task_id=str(extra_info.get("instance_id") or extra_info.get("index") or ""),
        query=str(extra_info.get("query") or ""),
        response=solution_str or "",
        global_step=extra_info.get("global_step"),
    )


def compute_score(
    solution_str: str, ground_truth: str = "", extra_info: dict | None = None, data_source: str | None = None, **_
) -> float:
    """the reference RL framework contract: float reward, drop -> INVALID_REWARD (masked).

    ⚠️ POINTWISE. This service's reward is a rank over the n siblings of one
    prompt, so /grade returns `reward: null` and a plain `reward` read would
    mask EVERY rollout of every step — an outage that looks like a judge
    outage. Use `compute_score_dict` (which reports the query score under its
    own name) or the trainer's group reward manager.
    """
    out = _grade(solution_str, extra_info)
    if out.get("status") != "ok":
        return INVALID_REWARD
    if out.get("reward") is None:
        raise RuntimeError(
            "this grader serves the webdev GROUP reward: /grade has no per-rollout "
            "reward, only query_score + the runtime gate, and the rank comes from "
            "POST /grade_group once the group is complete. A pointwise float "
            "contract cannot be satisfied — see compute_score_dict."
        )
    return float(out["reward"])


def compute_score_dict(
    data_source: str | None = None, solution_str: str = "", ground_truth: str = "", extra_info: dict | None = None, **_
) -> dict:
    """verl RewardLoop contract: dict with "score"; extra numeric keys land in
    reward_extra_info metrics. No masking on that side -> a drop scores 0.0,
    visible as design_drop (watch it: a climbing rate poisons the batch).

    The per-dimension keys are the answer to "why is the score low" — an
    uncovered brief and dead page JS call for different fixes, so they have to
    be separable on TB.

    ⚠️ POINTWISE, same caveat as rl.py: `score` here is the query judge's
    coverage number, NOT the training reward. The reward is a rank over the n
    siblings of one prompt and does not exist until POST /grade_group. Reported
    under its own name so nobody reads it as the reward: this seam has no group
    to rank against, and returning 0.0 for every rollout (which is what reading
    `reward` would do now) would be a silent, total outage.
    """
    out = _grade(solution_str, extra_info)
    empty = {
        "design_drop": 1,
        "design_query": 0.0,
        "design_runtime": 0.0,
    }
    if out.get("status") != "ok" or out.get("query_score") is None:
        return {"score": 0.0, **empty}
    rt = out.get("runtime") or {}
    return {
        # The query score, not a reward. See the docstring.
        "score": float(out["query_score"]),
        "design_drop": 0,
        "design_query": float(out.get("query_score") or 0.0),
        "design_runtime": float(rt.get("factor") if rt.get("factor") is not None else 1.0),
    }
