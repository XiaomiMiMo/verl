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
"""Grader entrypoint: grade one design rollout via the grader service.

Wired through the trainer's generic grader `entrypoint:` seam — the trainer calls
`grade(traj, query, dump_dir, cfg, log)` and knows nothing else about design RL.
The ROLLOUT side does no rendering and no scoring: the raw trajectory goes to
the service, which replays the file writes, renders, and judges. The pod is not
touched at all (reward_needs_env stays false in the yaml — it can be reclaimed
the moment the rollout ends).

Return contract (the grader callback): {grading, grader_reward,
grader_turn_scores, grader_cost}. grader_reward=None means the page was never
actually seen (render infra / judge / service down) — the caller masks the
trajectory (-999 → adv 0) instead of scoring blind. A FAILED DELIVERY is not a
drop: the service floors its score and the policy learns from it.

⚠️ POINTWISE — this seam wants one absolute reward per rollout, which the
service no longer produces. It serves the webdev GROUP reward: a rank over the
n siblings of one prompt, so `reward` is null on every /grade response and only
exists after POST /grade_group. Every call here would therefore mask its
trajectory, silently, at a 100% rate — so it refuses at the seam instead.

The group path is the trainer's own webdev reward manager, not this file; it
calls /grade per rollout for query_score + the runtime gate, then /grade_group
once the group is complete.
"""

from __future__ import annotations

import base64
import os
import uuid

from .client import DEFAULT_TIMEOUT_S, grade_remote


def _main_messages(traj: dict | list) -> list[dict]:
    """Accept either a bare message list or the traj.json dict (trajs.main.messages)."""
    if isinstance(traj, list):
        return traj
    main = (traj or {}).get("trajs", {}).get("main", {})
    return main.get("messages", []) if isinstance(main, dict) else []


def grade(traj, query: str, dump_dir, cfg: dict, log=print) -> dict:
    """See module docstring. `cfg` is the yaml `traj_grader:` block."""
    service_url = cfg.get("service_url") or os.getenv("DESIGN_GRADER_URL", "")
    if not service_url:
        return {
            "grading": {
                "mode": "design_v1",
                "drop_reason": "no service_url (cfg.service_url / DESIGN_GRADER_URL unset)",
            },
            "grader_reward": None,
            "grader_turn_scores": [],
            "grader_cost": {},
        }

    verdict = grade_remote(
        service_url,
        task_id=uuid.uuid4().hex,
        query=query or "",
        messages=_main_messages(traj),
        timeout_s=float(cfg.get("service_timeout_s", DEFAULT_TIMEOUT_S)),
        log=log,
    )

    grading = {
        "mode": "pointwise-unavailable",
        "visual": verdict.get("visual"),
        "query_score": verdict.get("query_score"),
        "aes_tier": verdict.get("aes_tier"),
        "breakdown": verdict.get("breakdown"),
        "dims": verdict.get("dims"),
        "reasoning": verdict.get("reason", ""),
        "signals": verdict.get("signals"),
        "grader_version": verdict.get("grader_version"),
        "timing": verdict.get("timing"),
    }
    reward = verdict.get("reward")
    if verdict.get("status") != "ok" or reward is None:
        # See the module docstring: reward is null on EVERY ok response here, so
        # this branch is the only one a healthy service can reach. Raise rather
        # than return grader_reward=None — masking every trajectory of every
        # step reads on the dashboards as a grader outage, and the run burns a
        # full rollout budget before anyone works out the contract is wrong.
        if verdict.get("status") == "ok":
            raise RuntimeError(
                "this grader serves the webdev GROUP reward, which has no per-rollout "
                f"reward (semantics={verdict.get('reward_semantics') or 'group_v1'}). "
                "A pointwise `entrypoint:` cannot be scored by it — use the trainer's "
                "webdev group reward manager (POST /grade per rollout for query_score + "
                "the runtime gate, then POST /grade_group once the group is complete)."
            )
        grading["drop_reason"] = str(verdict.get("drop_reason", "service drop"))[:300]
        log(f"[design_grader] drop: {grading['drop_reason']}")
        return {"grading": grading, "grader_reward": None, "grader_turn_scores": [], "grader_cost": {}}

    shot_b64 = verdict.get("shot_jpg_b64")
    if dump_dir and shot_b64:
        try:
            with open(os.path.join(str(dump_dir), "webdev_shot.jpg"), "wb") as f:
                f.write(base64.b64decode(shot_b64))
        except Exception as e:  # noqa: BLE001 — observability artifact, never fatal
            log(f"[design_grader] shot dump failed (non-fatal): {e}")

    reward = float(reward)
    slim = {k: v for k, v in grading.items() if k != "signals"}
    log(f"[webdev grade] reward={reward} grading={slim!r}")
    return {"grading": grading, "grader_reward": reward, "grader_turn_scores": [], "grader_cost": {}}
