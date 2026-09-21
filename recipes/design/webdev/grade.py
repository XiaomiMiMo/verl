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
"""Grading seam: route a finished rollout to the grader its profile named.

``grade()`` is the single entry point the environment calls. It returns
``{grading, grader_reward, grader_turn_scores, grader_cost}``, and a
``grader_reward`` of None means ungradeable.

Two graders, picked **per row** by the profile's ``traj_grader.correctness_mode``:

* **training** -- ``design_group_v1``: ships the pod's workspace to the external
  grading service, which renders it with its own browser and egress and returns
  a group-relative pick. Group-relative means a single rollout's number has no
  absolute meaning, so this cannot be used to evaluate.
* **evaluation** -- ``webdev_eval_v1``: renders in the pod itself and scores one
  vision call. Absolute and comparable across arms; needs a vision endpoint but
  not the grading service.

Per row rather than per run, because one run can route different data sources to
different graders.
"""

from __future__ import annotations

DESIGN_MODES = ("design_group_v1",)
EVAL_MODES = ("webdev_eval_v1",)
ALL_MODES = DESIGN_MODES + EVAL_MODES


def resolve_grade_cfg(harness_cfg: dict | None = None) -> dict:
    """Validate the profile's ``traj_grader:`` block and return it.

    **The mode must be declared; there is no default.** An earlier version fell
    back to an environment variable, which meant a profile with no
    ``traj_grader`` block still graded -- on whichever grader the launcher
    happened to export. The modes differ in what they send and to which service,
    so choosing the wrong one shows up as a shifted reward distribution rather
    than as an error. "Silently graded by something nobody chose" is the failure
    this function exists to prevent.

    Lives here rather than in the environment module because it is pure config
    validation: that module's import pulls in MimoAgent's whole dataset
    registry, which a test of this seam has no business needing.
    """
    mode = (harness_cfg or {}).get("correctness_mode")
    if mode not in ALL_MODES:
        raise ValueError(
            f"traj_grader.correctness_mode={mode!r} is not a known mode {ALL_MODES}. "
            "Every web-dev profile must declare it, e.g. for training\n"
            "  traj_grader:\n    correctness_mode: design_group_v1\n    service_timeout_s: 720\n"
            "or for evaluation\n"
            "  traj_grader:\n    correctness_mode: webdev_eval_v1"
        )
    return dict(harness_cfg)


def _drop(reason: str) -> dict:
    """Ungradeable: no reward, so the caller can account for it separately."""
    return {
        "grading": {"verdict": "drop", "drop_reason": reason},
        "grader_reward": None,
        "grader_turn_scores": [],
        "grader_cost": {},
    }


def grade(env, deliver_dir: str, query: str, cfg: dict) -> dict:
    """Dispatch to the training grader or the evaluation grader."""
    mode = (cfg or {}).get("correctness_mode")
    if mode not in ALL_MODES:
        return _drop(
            f"correctness_mode={mode!r} is not a known mode {ALL_MODES}; "
            "the profile must declare traj_grader.correctness_mode"
        )
    try:
        if mode in EVAL_MODES:
            from .eval_mode import grade_eval

            return grade_eval(env, deliver_dir, query, cfg)
        from .design_mode import grade_design

        return grade_design(
            env,
            deliver_dir,
            query,
            cfg,
            messages=(cfg or {}).get("messages"),
            dump_dir=(cfg or {}).get("dump_dir"),
        )
    except Exception as e:
        import traceback

        return _drop(f"{mode} grade crashed: {type(e).__name__}: {e}\n{traceback.format_exc()[:400]}")


__all__ = ["ALL_MODES", "DESIGN_MODES", "EVAL_MODES", "grade", "resolve_grade_cfg"]
