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
"""Per-rollout half of the training grader, mode ``design_group_v1``.

Selected per ROW by the harness profile's ``traj_grader.correctness_mode`` -- see
``grade.resolve_grade_cfg``. Evaluation uses a different mode entirely
(``webdev_eval_v1`` in ``eval_mode.py``), because a group-relative pick cannot evaluate a
single arm: the number only means anything next to its siblings.

**The service renders, not the pod.** We ship the pod's working directory as a tar.gz and the
service rebuilds the site, localizes remote assets, screenshots it and runs the judges.
Nothing is rendered in the pod on this path. The evaluation path does need in-pod rendering,
which is why it has its own module.

What this call produces is deliberately incomplete: the rollout gets its query fit judged and
its rendered shot landed, and the reward is a **0.0 PLACEHOLDER** carrying
``webdev_group_pending=True``. The driver's group rewriter replaces it once every sibling of
the group has finished.

Three row classes never enter the group pick, and the distinction is the whole point:

===================  ==========================================================
no delivery          a real hard 0.0 the policy learns from (``pending=False``)
render/shot failure  a drop, masked, because we never saw the page
everything else      pending
===================  ==========================================================
"""

from __future__ import annotations

import base64
import os
import shlex
import time
import uuid

from . import grader_client as gc

_WORKSPACE_EXCLUDES = ("node_modules", ".git", ".cache", "__pycache__", ".npm")
_WORKSPACE_MAX_FILE = "2M"  # find -size: a single html/asset above this is not a page
_WORKSPACE_MAX_IMAGE = "500k"  # site assets that large are CDN-linked in practice
_WORKSPACE_MAX_B64 = 28 * 1024 * 1024  # ~20 MB of tar.gz once base64 is undone

GROUP_MODE = "design_group_v1"
MODES = (GROUP_MODE,)


def capture_workspace(env, cwd: str, *, timeout: int = 120) -> tuple[str | None, str | None]:
    """tar.gz the agent's cwd out of the pod as base64: ``(blob, None)`` or ``(None, why)``.

    The cwd is taken WHOLE rather than just ``dist/``: the profile's prompt only asks the
    agent to build under it, not in a fixed subdirectory, and the service wants a workspace
    it can rebuild from.
    """
    cwd = (cwd or "/workspace").rstrip("/") or "/"
    excludes = " ".join(f"--exclude={shlex.quote(e)}" for e in _WORKSPACE_EXCLUDES)
    cmd = (
        f"cd {shlex.quote(cwd)} && "
        f"find . -type f \\( -size +{_WORKSPACE_MAX_FILE} -o "
        f"\\( -size +{_WORKSPACE_MAX_IMAGE} -a \\( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \\) \\) \\) "
        f"> /tmp/.design_skip 2>/dev/null; "
        f"tar -czf - {excludes} --exclude-from=/tmp/.design_skip . 2>/dev/null | base64 -w 0"
    )
    try:
        r = env.execute(cmd, "/", timeout)
    except Exception as e:  # noqa: BLE001 - pod transport is the infra boundary: report, never raise
        return None, f"exec failed: {type(e).__name__}: {e}"[:200]
    if r.get("reason") not in (None, "ok"):
        return None, f"exec reason={r.get('reason')!r}"
    blob = (r.get("output") or "").strip().splitlines()
    blob = blob[-1].strip() if blob else ""
    if not blob:
        return None, "empty tar (cwd missing or unreadable)"
    if len(blob) > _WORKSPACE_MAX_B64:
        return None, f"workspace too large ({len(blob) * 3 // 4 >> 20} MB compressed)"
    return blob, None


def _drop(grading: dict, reason: str, log) -> dict:
    grading["drop_reason"] = reason[:300]
    log(f"[grader] drop: {grading['drop_reason']}")
    return {"grading": grading, "grader_reward": None, "grader_turn_scores": [], "grader_cost": {}}


def grade_design(
    env, deliver_dir: str, query: str, cfg: dict, *, messages=None, dump_dir: str | None = None, log=print
) -> dict:
    """``grade()``'s training path. See the module docstring for what it does and does not do.

    ``deliver_dir`` is only used to derive the pod cwd -- the service decides for itself what
    the deliverable is inside the workspace it receives.
    """
    cfg = cfg or {}
    mode = cfg.get("correctness_mode")
    if mode != GROUP_MODE:
        return _drop({"mode": mode}, f"this module only serves {GROUP_MODE!r}", log)
    grading: dict = {"mode": mode}
    t0 = time.perf_counter()

    cwd = str(cfg.get("cwd") or os.path.dirname(deliver_dir.rstrip("/")) or "/workspace")
    tgz, cap_err = capture_workspace(env, cwd, timeout=int(cfg.get("capture_timeout_s", 120)))
    if tgz is None:
        return _drop(grading, f"workspace capture failed: {cap_err}", log)

    verdict = gc.grade_remote(
        gc.resolve_service_url(cfg, env_var=str(cfg.get("service_url_env") or "DESIGN_GRADER_URL")),
        task_id=uuid.uuid4().hex,
        query=query or "",
        workspace_tgz_b64=tgz,
        messages=gc.slim_messages(messages or []),
        judge_scope="query",
        global_step=cfg.get("global_step"),
        timeout_s=float(cfg.get("service_timeout_s", gc.DEFAULT_TIMEOUT_S)),
        log=log,
    )

    grading.update(
        {
            "visual": verdict.get("visual"),
            "query_score": verdict.get("query_score"),
            "aes_tier": verdict.get("aes_tier"),
            "breakdown": verdict.get("breakdown"),
            "dims": verdict.get("dims"),
            "reasoning": verdict.get("reasoning") or verdict.get("reason") or "",
            "grader_version": verdict.get("grader_version"),
            "capture_kb": len(tgz) * 3 // 4 >> 10,
            "grade_total_s": round(time.perf_counter() - t0, 2),
        }
    )
    if verdict.get("status") != "ok":
        return _drop(grading, str(verdict.get("drop_reason", "service drop")), log)

    shot_b64 = verdict.get("shot_jpg_b64")
    shot_path = None
    if dump_dir and shot_b64:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            shot_path = os.path.join(str(dump_dir), "webdev_shot.jpg")
            with open(shot_path, "wb") as f:
                f.write(base64.b64decode(shot_b64))
        except OSError as e:
            log(f"[grader] shot dump failed: {e}")
            shot_path = None

    if verdict.get("no_delivery"):
        log(f"[webdev grade] reward=0.0 grading={{'mode': {mode!r}, 'no_delivery': True}}")
        return {
            "grading": grading,
            "grader_reward": 0.0,
            "grader_turn_scores": [],
            "grader_cost": {},
            "webdev_function_score": 0.0,
            "webdev_group_pending": False,
            "webdev_shot_path": shot_path,
            "shot_jpg_b64": shot_b64,
        }
    qs = verdict.get("query_score")
    if qs is None or shot_path is None:
        return _drop(
            grading,
            f"group verdict incomplete: query_score={qs} shot_landed={shot_path is not None}",
            log,
        )
    rt_factor, rt_why = gc.runtime_gate(verdict)
    slim = {k: v for k, v in grading.items() if k != "dims"}
    slim["runtime"] = rt_factor
    log(f"[webdev grade] reward=0.0 grading={slim!r}")
    return {
        "grading": grading,
        "grader_reward": 0.0,
        "grader_turn_scores": [],
        "grader_cost": {},
        "webdev_function_score": 0.0,
        "webdev_group_pending": True,
        "webdev_group_query_score": float(qs),
        "webdev_group_runtime_factor": rt_factor,
        "webdev_group_runtime_why": rt_why,
        "webdev_query": query or "",
        "webdev_shot_path": shot_path,
        "shot_jpg_b64": shot_b64,
    }


__all__ = ["GROUP_MODE", "MODES", "capture_workspace", "grade_design"]
