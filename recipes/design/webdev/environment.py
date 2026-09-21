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
"""The web-dev task environment: build a site in a pod, graded by what it renders.

The agent builds a website in the sandbox pod and delivers it to ``{cwd}/dist``.
At reward time the delivered tree is shipped to an external grading service,
which rebuilds the site, screenshots it and runs vision judges. The trajectory's
reward is what those judges return.

Expressing this as a *dataset environment* rather than as a grader hanging off
the agent loop is what keeps the bridge unchanged: reward arrives through
``DatasetEnvironment.calculate_reward()``, the same seam every other dataset
uses, so the only integration point is the registry hook below.

**Infrastructure failures are scored 0.0, not masked.** The grader returns no
reward when it never saw the page -- judge channel down, render failed. verl's
GRPO has no mask for that, so this returns a plain 0.0 and flags it in
``extra``. That is an accounting convention, not a judgement: ``webdev_drop``
climbing means the reward is being polluted by infrastructure rather than by bad
sites, and downstream consumers are expected to slice those rows out.

Instance schema:
    dataset_type: "webdev"      routes here via DATASET_REGISTRY
    instance_id / task_id: str
    problem_statement: str      the build request, and also the grader's query
    docker_image: str
    cwd: str = "/workspace"
"""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile

from mimoagent.environments.datasets.base import DatasetEnvironment


class WebdevEnvironment(DatasetEnvironment):
    """Build-a-site environment: reward is a vision grade of ``{cwd}/dist``."""

    grader_cfg: dict | None = None

    dump_dir: str | None = None

    _DIST_DUMP_CAP_BYTES = 30 * 1024 * 1024

    def _setup_dataset_specific(self) -> None:
        cwd = self._cwd()
        self.env.execute(f"mkdir -p {cwd}/dist", "/", 60)

    def _cwd(self) -> str:
        return (self.instance.get("cwd") or "/workspace").rstrip("/")

    def _capture_model_diff(self) -> tuple[str, str]:
        return "", ""

    def _dump_dist(self, cwd: str) -> None:
        """Pull dist/ out of the live pod, before grading rather than after.

        Grading's screenshot and judge calls can exhaust the pod's exec budget,
        and a later tar would then fail silently. dist/ does not change during
        grading -- the agent is already done.
        """
        if not self.dump_dir:
            return

        out = self.dump_dir
        os.makedirs(out, exist_ok=True)
        probe = self.env.execute(f"du -sb {cwd}/dist 2>/dev/null | cut -f1", "/", 60)
        try:
            size = int((probe.get("output") or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            size = 0
        if size <= 0 or size > self._DIST_DUMP_CAP_BYTES:
            if size:
                self.logger.warning(f"{self.instance_id}: dist {size}B exceeds dump cap, skipping")
            return
        result = self.env.execute(f"tar -C {cwd} -czf - dist | base64 -w0", "/", 120)
        if result.get("reason") not in (None, "ok"):
            self.logger.warning(f"{self.instance_id}: dist tar failed: {result.get('reason')}")
            return
        blob = (result.get("output") or "").strip().splitlines()
        if not blob:
            return
        raw = base64.b64decode(blob[-1])
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            members = [m for m in tf.getmembers() if not (m.name.startswith("/") or ".." in m.name.split("/"))]
            tf.extractall(out, members=members)

    def _dump_grade(self, reward, grading: dict, shot_b64: str | None) -> None:
        """Write the verdict and the judge's screenshot, after grading."""
        if not self.dump_dir:
            return

        out = self.dump_dir
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "instance.log"), "w", encoding="utf-8") as f:
            f.write(f"instance_id={self.instance_id}\n")
            f.write(f"[webdev grade] reward={reward} grading={grading!r}\n")
        if shot_b64:
            try:
                with open(os.path.join(out, "webdev_shot.jpg"), "wb") as f:
                    f.write(base64.b64decode(shot_b64))
            except Exception:
                pass

    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        from .grade import grade, resolve_grade_cfg

        cwd = self._cwd()
        deliver_dir = f"{cwd}/dist"
        query = self.instance.get("problem_statement") or self.instance.get("query") or ""

        mode = (self.grader_cfg or {}).get("correctness_mode")
        key_var = "WEBDEV_EVAL_JUDGE_API_KEY" if mode == "webdev_eval_v1" else "LLM_JUDGE_API_KEY"
        if not os.getenv(key_var):
            return (
                0.0,
                f"webdev: {key_var} not set in the env actor (mode={mode})",
                {
                    "error_category": "reward/env_error",
                    "webdev_drop": True,
                    "drop_reason": f"missing {key_var}",
                },
            )

        try:
            self._dump_dist(cwd)
        except Exception as e:  # noqa: BLE001 - an artifact dump must not fail a graded rollout
            self.logger.warning(f"{self.instance_id}: dist dump failed (non-fatal): {e}")

        cfg = resolve_grade_cfg(self.grader_cfg)
        cfg.setdefault("cwd", cwd)
        cfg["dump_dir"] = self.dump_dir
        result = grade(self.env, deliver_dir, query, cfg=cfg)

        grading = result.get("grading") or {}
        reward = result.get("grader_reward")

        try:
            self._dump_grade(reward, grading, result.get("shot_jpg_b64"))
        except Exception as e:  # noqa: BLE001 - same reason as the dist dump
            self.logger.warning(f"{self.instance_id}: grade dump failed (non-fatal): {e}")

        if reward is None:
            reason = str(grading.get("drop_reason", "grader returned None"))
            return (
                0.0,
                f"webdev grader drop: {reason[:800]}",
                {
                    "error_category": "reward/env_error",
                    "webdev_drop": True,
                    "drop_reason": reason[:800],
                },
            )

        test_output = json.dumps(
            {
                "mode": grading.get("mode"),
                "function_score": grading.get("function_score"),
                "r_query": grading.get("r_query"),
            },
            ensure_ascii=False,
        )
        extra = {
            "webdev_function_score": grading.get("function_score", reward),
            "webdev_r_query": grading.get("r_query"),
            "webdev_drop": False,
        }
        for key in (
            "webdev_group_pending",
            "webdev_group_query_score",
            "webdev_group_runtime_factor",
            "webdev_group_runtime_why",
            "webdev_query",
            "webdev_shot_path",
        ):
            if key in result:
                extra[key] = result[key]
        return float(reward), test_output, extra
