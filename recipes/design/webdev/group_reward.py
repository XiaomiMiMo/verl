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
"""Driver-side group reward: turn a group of placeholder rows into real rewards.

A pointwise grader scores every rollout alone. This mode instead judges a whole GRPO group at
once -- a relative aesthetic pick across the n sibling screenshots, minus an absolute
query-fit deduct -- so it can only run once every sibling of a uid has finished.

In this trainer that moment is ``_compute_advantage``: the driver assembles the entire batch
(every uid times ``rollout.n``) before advantages are computed, so a plain synchronous pass
over the uid groups is enough. A streaming trainer would need a task queue hooked into its
sampler's buffer flush; here the guarantee this needs -- all siblings present, nothing
downstream has read the reward yet -- comes for free.

The per-rollout half (``design_mode.py``) left a 0.0 PLACEHOLDER reward and
``webdev_group_pending=True`` on every row whose query fit was judged and whose shot landed.
Rows that are NOT pending keep their reward and stay out of the pick: no delivery is a real
hard 0, and a capture or render failure was already dropped.

**The shots must be on a filesystem the driver can read.** The environment actors write them
and this module reads them back by path. With a node-local dump directory every group comes
back smaller than ``MIN_PENDING_ROWS`` and is skipped -- no error anywhere, just rewards that
stay 0.0. ``webdev_group/n_groups_too_small`` is the metric that catches it.

Rewrite rules -- all four have to exist or the run learns silently wrong things:

==========================  ===================================================
normal                      ``reward = items[i].reward``, in [0, 1]
``items[i].missing``        INVALID (a judge dimension failed for that row)
``group_ok == false``       the service already zeroed the pick for the whole
                            group (too few pick rounds came back); the rows are
                            KEPT and counted
task failure or drop        every pending row of that group is INVALID
==========================  ===================================================

INVALID here means "replace with the group's mean valid reward". This trainer has no
out-of-range sentinel and adding one would touch the advantage path; GRPO only ever looks at
``reward - group_mean``, so writing the mean gives exactly what a sentinel buys -- the row
contributes no gradient -- without a trainer change. A group with no valid row at all is left
untouched, since its rows are already equal and every advantage is 0 anyway.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from . import grader_client as gc

logger = logging.getLogger(__name__)

GROUP_SLICE = int(os.getenv("WEBDEV_GROUP_SLICE", "8"))
MIN_PENDING_ROWS = int(os.getenv("WEBDEV_GROUP_MIN_ROWS", "4"))
BUDGET_S = float(os.getenv("WEBDEV_GROUP_BUDGET_S", "1500"))
REQUEST_TIMEOUT_S = float(os.getenv("WEBDEV_GROUP_TIMEOUT_S", "1200"))
MAX_INFLIGHT_GROUPS = int(os.getenv("WEBDEV_GROUP_MAX_INFLIGHT", "24"))


def _slices(idx: list[int], k: int = GROUP_SLICE) -> list[list[int]]:
    if not k or len(idx) <= k:
        return [idx]
    out = [idx[i : i + k] for i in range(0, len(idx), k)]
    if len(out) > 1 and len(out[-1]) < max(2, k // 2):
        tail = out.pop()  # pop first: out[-1] must be re-resolved after it
        out[-1] = out[-1] + tail
    return out


def _judge_slice(shots: list[str], query: str, qscores: list, rtf: list, task_id: str, global_step) -> dict:
    """One ``POST /grade_group``. Never raises: a transport failure is a synthetic drop."""
    t0 = time.perf_counter()
    try:
        res = gc.grade_group_remote(
            gc.resolve_service_url(),
            shots,
            query,
            qscores,
            runtime_factors=rtf,
            task_id=task_id,
            global_step=global_step,
            timeout_s=REQUEST_TIMEOUT_S,
            log=lambda m: logger.warning("[webdev_group] %s", m),
        )
    except Exception as e:  # noqa: BLE001 - a judge crash must not kill the step
        res = {"status": "drop", "drop_reason": f"{type(e).__name__}: {e}"[:300]}
    res["latency_s"] = round(time.perf_counter() - t0, 2)
    return res


def apply_group_reward(data, *, extra_fields_list: list[dict], global_step=None) -> dict:
    """Rewrite ``token_level_rewards`` for pending group rows in a batch.

    Args:
        data: the batch, carrying ``token_level_rewards``, ``response_mask`` and ``uid``.
        extra_fields_list: the ``extra_fields`` dicts, aligned to the batch rows.
        global_step: labels the service's own dump; never affects a score.

    Returns a ``webdev_group/*`` metrics dict.
    """
    metrics: dict = {}
    uid = data.non_tensor_batch["uid"]
    tlr = data.batch["token_level_rewards"]
    resp_mask = data.batch["response_mask"].bool()

    aux = [ef if isinstance(ef, dict) else {} for ef in extra_fields_list]
    groups: dict[str, list[int]] = defaultdict(list)
    for i in range(len(uid)):
        a = aux[i]
        shot = a.get("webdev_shot_path")
        if a.get("webdev_group_pending") and shot and os.path.isfile(str(shot)):
            groups[uid[i]].append(i)

    metrics["webdev_group/n_groups_total"] = len(groups)
    judgeable = {u: idxs for u, idxs in groups.items() if len(idxs) >= MIN_PENDING_ROWS}
    metrics["webdev_group/n_groups_too_small"] = len(groups) - len(judgeable)
    if not judgeable:
        return metrics

    jobs: list[tuple[str, list[int]]] = []  # (task_id, row indices)
    for u, idxs in judgeable.items():
        sl = _slices(idxs)
        for s, rows in enumerate(sl):
            jobs.append((f"{u}#{s}" if len(sl) > 1 else str(u), rows))

    def run(job):
        task_id, rows = job
        query = next((aux[i].get("webdev_query") for i in rows if aux[i].get("webdev_query")), "")
        return _judge_slice(
            [str(aux[i]["webdev_shot_path"]) for i in rows],
            str(query),
            [aux[i].get("webdev_group_query_score") for i in rows],
            [aux[i].get("webdev_group_runtime_factor") for i in rows],
            task_id,
            global_step,
        )

    results: dict[int, dict] = {}
    executor = ThreadPoolExecutor(max_workers=min(MAX_INFLIGHT_GROUPS, len(jobs)), thread_name_prefix="webdev-group")
    try:
        futures = {executor.submit(run, j): n for n, j in enumerate(jobs)}
        done, not_done = wait(futures, timeout=BUDGET_S)
        for fut in done:
            try:
                results[futures[fut]] = fut.result()
            except Exception as e:  # noqa: BLE001 - run() already catches; belt and braces
                results[futures[fut]] = {"status": "drop", "drop_reason": f"{type(e).__name__}: {e}"[:200]}
        if not_done:
            logger.warning(
                "[webdev_group] %d/%d slices missed the %ss budget -> INVALID",
                len(not_done),
                len(futures),
                BUDGET_S,
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    finals: dict[int, float] = {}  # row -> reward; absent means INVALID
    invalid: set[int] = set()
    n_slices_ok = n_slices_dropped = n_pick_short = n_ncd = 0
    latencies: list[float] = []
    rewards: list[float] = []
    n_query_floor = n_runtime_gated = 0
    dumps: dict[str, list] = defaultdict(list)

    for n, (task_id, rows) in enumerate(jobs):
        res = results.get(n) or {"status": "drop", "drop_reason": f"missed the {BUDGET_S}s budget"}
        if res.get("latency_s") is not None:
            latencies.append(float(res["latency_s"]))
        items = res.get("items")
        if not items or len(items) != len(rows):
            why = res.get("drop_reason") or f"expected {len(rows)} items, got {len(items or [])}"
            logger.warning("[webdev_group] %s dropped (%d rows -> INVALID): %s", task_id, len(rows), str(why)[:200])
            invalid.update(rows)
            n_slices_dropped += 1
            dumps[task_id.split("#")[0]].append({"task_id": task_id, "rows": rows, "error": str(why)[:300]})
            continue
        n_slices_ok += 1
        if not res.get("group_ok", False):
            n_pick_short += 1
        n_ncd += int(bool(res.get("ncd_zeroed")))
        for k, i in enumerate(rows):
            item = items[k]
            r = item.get("reward")
            if item.get("missing") or r is None:
                invalid.add(i)
                continue
            finals[i] = float(r)
            rewards.append(float(r))
            if item.get("query_deduct") == "floor" or item.get("query_floored"):
                n_query_floor += 1
            if item.get("runtime_gated"):
                n_runtime_gated += 1
        dumps[task_id.split("#")[0]].append({"task_id": task_id, "rows": rows, "result": res})

    n_invalid_written = 0
    for _u, idxs in judgeable.items():
        valid = [finals[i] for i in idxs if i in finals]
        if not valid:
            continue
        mean = float(np.mean(valid))
        for i in idxs:
            if i in invalid:
                finals[i] = mean
                n_invalid_written += 1

    n_rewritten = 0
    for i, final in finals.items():
        row_mask = resp_mask[i]
        if not row_mask.any():
            continue
        last_idx = int(row_mask.nonzero(as_tuple=True)[0][-1])
        tlr[i].zero_()
        tlr[i, last_idx] = final
        n_rewritten += 1

    version = _grader_version(dumps)
    _dump(dumps, aux, finals, global_step, version)

    metrics.update(
        {
            "webdev_group/n_groups_judged": len(judgeable),
            "webdev_group/n_slices_ok": n_slices_ok,
            "webdev_group/n_slices_dropped": n_slices_dropped,
            "webdev_group/n_rows_judged": len(finals),
            "webdev_group/n_rows_rewritten": n_rewritten,
            "webdev_group/n_rows_invalid": len(invalid),
            "webdev_group/n_rows_invalid_written": n_invalid_written,
            "webdev_group/n_rows_query_floor": n_query_floor,
            "webdev_group/n_rows_runtime_gated": n_runtime_gated,
            "webdev_group/n_groups_pick_short": n_pick_short,
            "webdev_group/n_groups_ncd_zeroed": n_ncd,
        }
    )
    if rewards:
        metrics["webdev_group/reward_mean"] = float(np.mean(rewards))
        metrics["webdev_group/reward_std"] = float(np.std(rewards))
    if latencies:
        metrics["webdev_group/latency_s_mean"] = float(np.mean(latencies))
        metrics["webdev_group/latency_s_max"] = float(np.max(latencies))
    qs = [aux[i].get("webdev_group_query_score") for i in finals if aux[i].get("webdev_group_query_score") is not None]
    if qs:
        metrics["webdev_group/query_score_mean"] = float(np.mean(qs))
    return metrics


_LOGGED_VERSION: str | None = None


def _grader_version(dumps: dict[str, list]) -> str | None:
    """Which build of the service actually served this step, logged ONCE.

    Taken from a slice response we already hold rather than a fresh ``GET /version``: zero
    extra calls, and -- more importantly -- it reports the build that answered, not the build
    that answers now. The two differ exactly when it matters, which is when someone restarted
    the service mid-run.

    Deliberately NOT asserted anywhere. Pinning ``grader_version`` in a startup handshake and
    then upgrading the service made every training retry fatal in the handshake;
    ``grader_client.check_capabilities`` pins capabilities instead. This is provenance, not a
    gate -- so when a reward curve shifts for no apparent reason, the dumps can answer "did
    the judge change?".
    """
    global _LOGGED_VERSION
    v = None
    for slices in dumps.values():
        for sl in slices:
            v = (sl.get("result") or {}).get("grader_version")
            if v:
                break
        if v:
            break
    if v and v != _LOGGED_VERSION:
        logger.info("[webdev_group] group grader version=%s url=%s", v, gc.resolve_service_url())
        _LOGGED_VERSION = v
    return v


def _dump(
    dumps: dict[str, list],
    aux: list[dict],
    finals: dict[int, float],
    global_step,
    grader_version: str | None = None,
) -> None:
    """``group_reward.json`` next to the rollout dumps, one file per instance.

    Same root as the rollout dumps, because the whole point is to sit beside the shots the
    pick compared.
    """
    root = os.environ.get("WEBDEV_DEBUG_DIR")
    if not root or not dumps:
        return
    for uid, slices in dumps.items():
        try:
            rows = [i for sl in slices for i in sl["rows"]]
            d = None
            for i in rows:
                shot = aux[i].get("webdev_shot_path")
                if shot:
                    d = os.path.dirname(os.path.dirname(str(shot)))
                    break
            if not d or not os.path.isdir(d):
                continue
            payload = {
                "uid": uid,
                "step": global_step,
                "grader_version": grader_version,
                "rows": [
                    {
                        "row": i,
                        "reward": finals.get(i),
                        "shot": aux[i].get("webdev_shot_path"),
                        "query_score": aux[i].get("webdev_group_query_score"),
                        "runtime_factor": aux[i].get("webdev_group_runtime_factor"),
                        "runtime_why": aux[i].get("webdev_group_runtime_why"),
                    }
                    for i in rows
                ],
                "slices": slices,
            }
            with open(os.path.join(d, "group_reward.json"), "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
        except Exception as e:  # noqa: BLE001 - the dump is best-effort observability
            logger.warning("[webdev_group] dump failed for uid=%s: %s", uid, e)


__all__ = ["BUDGET_S", "GROUP_SLICE", "MIN_PENDING_ROWS", "apply_group_reward"]
