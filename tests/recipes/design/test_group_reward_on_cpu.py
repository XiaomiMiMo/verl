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
"""CPU tests for the driver-side group reward rewrite and the infra-exclusion hook.

The rewrite happens in ``_compute_advantage``, the first moment every sibling of a uid
exists and before anything reads the reward. These tests pin the four writeback rules,
because every failure mode here is silent in training: a missed rewrite leaves the 0.0
placeholder, so the group looks uniformly worthless, and a wrong INVALID handling gives a
failed rollout a real gradient.

Nothing external is needed -- the grading service is a loopback HTTP server, so the real
client, the real retry path and the real JSON contract are all exercised.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recipes.design.webdev import group_reward as gr


class _GroupService:
    """``POST /grade_group`` -> whatever ``responder(body)`` returns."""

    def __init__(self, responder):
        self.responder, self.bodies = responder, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.bodies.append(body)
                out = json.dumps(outer.responder(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def _batch(n_rows: int, uids: list[str], resp_len: int = 4):
    """A batch stand-in carrying only what the rewrite touches."""
    return SimpleNamespace(
        batch={
            "token_level_rewards": torch.zeros(n_rows, resp_len),
            "response_mask": torch.ones(n_rows, resp_len),
        },
        non_tensor_batch={"uid": np.array(uids, dtype=object)},
    )


def _pending(tmp_path, row: int, *, uid="u0", qs=0.9, rt=1.0, query="build a shop"):
    shot = tmp_path / f"{uid}_{row}" / "webdev_shot.jpg"
    shot.parent.mkdir(parents=True, exist_ok=True)
    shot.write_bytes(b"\xff\xd8jpg")
    return {
        "webdev_group_pending": True,
        "webdev_shot_path": str(shot),
        "webdev_group_query_score": qs,
        "webdev_group_runtime_factor": rt,
        "webdev_query": query,
    }


def _final(data, i):
    """The reward the rewrite wrote on the row's last response token.

    ``token_level_rewards`` is float32, so callers compare with ``pytest.approx``.
    """
    return float(data.batch["token_level_rewards"][i].sum())


def _finals(data, n):
    return [pytest.approx(_final(data, i)) for i in range(n)]


def _items(rewards, **extra):
    return {
        "status": "ok",
        "group_ok": True,
        "rounds_ok": 8,
        "rounds_total": 8,
        "items": [
            {
                "shot": k,
                "reward": r,
                "pick_norm": 0.25,
                "query_score": 0.9,
                "query_deduct": 0.0,
                "runtime_factor": 1.0,
                "runtime_gated": False,
                "missing": [],
            }
            for k, r in enumerate(rewards)
        ],
        **extra,
    }


@pytest.fixture(autouse=True)
def _no_dumps(monkeypatch):
    """The dump root is global state; a stray one would write into the repo."""
    monkeypatch.delenv("WEBDEV_DEBUG_DIR", raising=False)


@pytest.fixture
def url(monkeypatch):
    def _set(u):
        monkeypatch.setenv("DESIGN_GRADER_URL", u)

    return _set


def test_one_call_per_group_and_rewards_written(tmp_path, url):
    """A uid's n rows produce exactly ONE group call, and every row is rewritten."""
    ef = [_pending(tmp_path, i) for i in range(8)]
    data = _batch(8, ["u0"] * 8)
    with _GroupService(lambda b: _items([0.1, 0.9, 0.5, 0.5, 0.6, 0.2, 0.8, 0.4])) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef, global_step=3)
        assert len(s.bodies) == 1
        body = s.bodies[0]
        assert len(body["shots_jpg_b64"]) == 8, "every sibling's shot must ship"
        assert body["query_scores"] == [0.9] * 8, "the pod's judgements are reused, not re-derived"
        assert body["runtime_factors"] == [1.0] * 8
        assert body["query"] == "build a shop"
    assert _finals(data, 8) == [0.1, 0.9, 0.5, 0.5, 0.6, 0.2, 0.8, 0.4]
    assert m["webdev_group/n_rows_rewritten"] == 8
    assert m["webdev_group/n_rows_invalid"] == 0
    assert m["webdev_group/reward_std"] > 0, "the pick must differentiate or there is no signal"


def test_the_group_seed_is_derived_from_the_shot_paths(tmp_path, url):
    """Re-running the same group must see the same relabeling, or a retry scores differently."""
    ef = [_pending(tmp_path, i) for i in range(4)]
    data = _batch(4, ["u0"] * 4)
    seeds = []
    with _GroupService(lambda b: _items([0.5] * 4)) as s:
        url(s.url)
        for _ in range(2):
            gr.apply_group_reward(_batch(4, ["u0"] * 4), extra_fields_list=ef)
        seeds = [b["seed"] for b in s.bodies]
    assert seeds[0] == seeds[1]
    del data


def test_reward_lands_on_the_last_response_token(tmp_path, url):
    ef = [_pending(tmp_path, i) for i in range(4)]
    data = _batch(4, ["u0"] * 4, resp_len=5)
    data.batch["response_mask"][:, 3:] = 0  # rows end at index 2
    with _GroupService(lambda b: _items([0.7, 0.2, 0.5, 0.5])) as s:
        url(s.url)
        gr.apply_group_reward(data, extra_fields_list=ef)
    row = data.batch["token_level_rewards"][0]
    assert float(row[2]) == pytest.approx(0.7)
    assert float(row.sum()) == pytest.approx(0.7), "nothing may be left on a masked token"


def test_missing_dim_row_becomes_the_group_mean(tmp_path, url):
    """INVALID is the group mean: GRPO subtracts that mean, so the row has no gradient."""
    ef = [_pending(tmp_path, i) for i in range(4)]
    data = _batch(4, ["u0"] * 4)

    def responder(body):
        res = _items([0.2, 0.8, 0.5, None])
        res["items"][3]["missing"] = ["query"]
        return res

    with _GroupService(responder) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef)
    assert _final(data, 3) == pytest.approx((0.2 + 0.8 + 0.5) / 3)
    assert m["webdev_group/n_rows_invalid"] == 1
    assert m["webdev_group/n_rows_invalid_written"] == 1


def test_pick_short_group_is_kept_and_counted(tmp_path, url):
    """``group_ok=false``: the service already zeroed the pick; the rows still train."""
    ef = [_pending(tmp_path, i) for i in range(4)]
    data = _batch(4, ["u0"] * 4)
    with _GroupService(lambda b: _items([0.6] * 4, group_ok=False, rounds_ok=3)) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef)
    assert _finals(data, 4) == [0.6] * 4
    assert m["webdev_group/n_groups_pick_short"] == 1
    assert m["webdev_group/n_rows_invalid"] == 0


def test_service_drop_invalidates_the_whole_group(tmp_path, url):
    """Nothing was judged, so no row may keep the 0.0 placeholder as if it were a score."""
    ef = [_pending(tmp_path, i) for i in range(4)]
    data = _batch(4, ["u0"] * 4)
    with _GroupService(lambda b: {"status": "drop", "reward": None, "drop_reason": "judge pool down"}) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef)
    assert m["webdev_group/n_rows_invalid"] == 4
    assert m["webdev_group/n_slices_dropped"] == 1
    # No valid sibling to average, so the placeholders stay equal and every advantage is 0.
    assert _finals(data, 4) == [0.0] * 4
    assert m["webdev_group/n_rows_rewritten"] == 0


def test_item_count_mismatch_is_a_drop(tmp_path, url):
    ef = [_pending(tmp_path, i) for i in range(4)]
    data = _batch(4, ["u0"] * 4)
    with _GroupService(lambda b: _items([0.5, 0.5])) as s:  # 2 items for 4 shots
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef)
    assert m["webdev_group/n_rows_invalid"] == 4


def test_groups_are_independent_and_concurrent(tmp_path, url):
    ef = [_pending(tmp_path, i, uid="u0") for i in range(4)] + [
        _pending(tmp_path, i, uid="u1", query="build a blog") for i in range(4)
    ]
    data = _batch(8, ["u0"] * 4 + ["u1"] * 4)

    def responder(body):
        base = 0.1 if body["query"] == "build a shop" else 0.9
        return _items([base] * 4)

    with _GroupService(responder) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef)
        assert len(s.bodies) == 2, "one call per uid"
    assert _finals(data, 8) == [0.1] * 4 + [0.9] * 4
    assert m["webdev_group/n_groups_judged"] == 2


def test_oversized_group_is_sliced(tmp_path, url):
    """The pick is calibrated at 8 shots; a 16-row group becomes two independent picks."""
    ef = [_pending(tmp_path, i) for i in range(16)]
    data = _batch(16, ["u0"] * 16)
    with _GroupService(lambda b: _items([0.5] * len(b["shots_jpg_b64"]))) as s:
        url(s.url)
        gr.apply_group_reward(data, extra_fields_list=ef)
        assert [len(b["shots_jpg_b64"]) for b in s.bodies] == [8, 8]
    # A remainder smaller than half a slice folds into the previous one: a pick over 2-3
    # shots is mostly position noise, while a slightly oversized one is not.
    assert [len(x) for x in gr._slices(list(range(11)))] == [11]
    assert [len(x) for x in gr._slices(list(range(12)))] == [8, 4]
    assert gr._slices(list(range(5))) == [list(range(5))]


def test_non_pending_rows_are_never_touched(tmp_path, url):
    """A no-delivery row is a real hard 0 and must not enter the relative pick."""
    ef = [
        _pending(tmp_path, 0),
        _pending(tmp_path, 1),
        _pending(tmp_path, 2),
        _pending(tmp_path, 3),
        {"webdev_group_pending": False, "webdev_function_score": 0.0},  # no delivery
        {"webdev_function_score": 0.7},  # not a group row at all
    ]
    data = _batch(6, ["u0"] * 6)
    data.batch["token_level_rewards"][4, -1] = 0.0
    data.batch["token_level_rewards"][5, -1] = 0.7
    with _GroupService(lambda b: _items([0.3] * 4)) as s:
        url(s.url)
        gr.apply_group_reward(data, extra_fields_list=ef)
        assert len(s.bodies[0]["shots_jpg_b64"]) == 4
    assert _final(data, 4) == 0.0
    assert _final(data, 5) == pytest.approx(0.7)


def test_too_small_group_is_skipped(tmp_path, url):
    """Below MIN_PENDING_ROWS a relative pick is mostly position noise.

    This is also the metric that catches a node-local dump directory: the shots the driver
    cannot read make every group look this small, with no error anywhere.
    """
    ef = [_pending(tmp_path, i) for i in range(2)]
    data = _batch(2, ["u0"] * 2)
    with _GroupService(lambda b: _items([0.9, 0.1])) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef)
        assert s.bodies == []
    assert m["webdev_group/n_groups_too_small"] == 1
    assert _finals(data, 2) == [0.0, 0.0]


def test_missing_shot_file_excludes_the_row(tmp_path, url):
    """A pending row whose shot never landed cannot be picked; the rest still are.

    Five rows so the survivors still clear MIN_PENDING_ROWS.
    """
    ef = [_pending(tmp_path, i) for i in range(5)]
    os.unlink(ef[2]["webdev_shot_path"])
    data = _batch(5, ["u0"] * 5)
    with _GroupService(lambda b: _items([0.5] * len(b["shots_jpg_b64"]))) as s:
        url(s.url)
        gr.apply_group_reward(data, extra_fields_list=ef)
        assert len(s.bodies[0]["shots_jpg_b64"]) == 4
    assert _final(data, 2) == 0.0, "untouched placeholder"
    assert _final(data, 0) == pytest.approx(0.5)


def test_the_bridge_puts_the_group_keys_where_this_module_reads_them():
    """Justifies reading ``extra_fields`` flat, with no nested fallback.

    The reference carried a branch that merged ``extra_fields["reward_extra_info"]`` on top,
    for a producer it no longer shipped. That branch is dropped here, which is only correct
    while the bridge keeps the group keys at the top level and out of
    ``reward_extra_info`` -- so assert both halves rather than trusting the reading.
    """
    from recipes.design.agent_loop import WebdevAgentLoop

    rei = WebdevAgentLoop._reward_extra_info(reward=0.5, true_reward=0.5, model_patch_len=0.0, repetition_collapse=0.0)
    assert not [k for k in rei if k.startswith("webdev")], (
        "a webdev_* key inside reward_extra_info would need the nested fallback back"
    )


# ---------------------------------------------------------------------------
# End-to-end: the rewrite must actually reach a NON-ZERO GRPO advantage.
#
# Everything above stops at token_level_rewards, one link short of the symptom that
# motivated these tests. In one measured step the group grader was healthy -- 63 groups
# judged, 363 rows rewritten, mean 0.578 -- and the step STILL produced 512/512 dead rows
# with advantages/max 0.0, because a second writer ran afterwards and overwrote every group
# row. So this runs the real advantage function the trainer calls, on a real batch.
# ---------------------------------------------------------------------------


def _dead_rows(data):
    """Dead iff the advantage is 0 across the whole response."""
    adv, rmask = data.batch["advantages"], data.batch["response_mask"]
    return (~((adv.abs() * rmask).amax(dim=-1) > 0)).sum().item()


def _grpo(data, keys, n_per_uid):
    # `verl.trainer.ppo.v1.__init__` eagerly imports the transfer-queue agent loop, so
    # reaching the real advantage function needs that dependency present. Skipping rather
    # than reimplementing GRPO here: a hand-rolled copy would pass while the real estimator
    # changed under it, which is the opposite of what this test is for.
    pytest.importorskip("transfer_queue", reason="needed to import the real advantage function")

    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

    return compute_advantage_for_multi_trajectories(
        data, batch_keys=keys, adv_estimator=AdvantageEstimator.GRPO, num_repeat=n_per_uid
    )


def _real_batch(uids, resp_len=4):
    from tensordict import TensorDict

    from verl import DataProto

    n = len(uids)
    return DataProto(
        batch=TensorDict(
            {"token_level_rewards": torch.zeros(n, resp_len), "response_mask": torch.ones(n, resp_len)},
            batch_size=n,
        ),
        non_tensor_batch={"uid": np.array(uids, dtype=object)},
    )


def test_group_rewards_reach_a_nonzero_grpo_advantage(tmp_path, url):
    n_per_uid, uid_names = 8, ("u0", "u1")
    ef, uids, keys = [], [], []
    for u in uid_names:
        ef += [_pending(tmp_path, i, uid=u) for i in range(n_per_uid)]
        uids += [u] * n_per_uid
        # {uid}_{session}_{index}: one session per row, as the web-dev loop emits.
        keys += [f"{u}_s{i}_0" for i in range(n_per_uid)]

    data = _real_batch(uids)
    # Deliberately no pick equal to the group mean of 0.5: GRPO subtracts that mean, so such
    # a row is legitimately dead and would mask what this test is checking.
    picks = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.35, 0.65]
    with _GroupService(lambda b: _items(picks)) as s:
        url(s.url)
        m = gr.apply_group_reward(data, extra_fields_list=ef, global_step=1)
    assert m["webdev_group/n_rows_rewritten"] == len(uids)

    data = _grpo(data, keys, n_per_uid)

    assert _dead_rows(data) == 0
    adv = data.batch["advantages"]
    assert float((adv * data.batch["response_mask"]).abs().max()) > 0.1
    # The sign follows the pick: the group's best row must beat its worst.
    assert float(adv[1, 0]) > float(adv[0, 0])


def test_a_unique_uid_makes_a_row_a_singleton_group_with_zero_advantage():
    """What the infra-exclusion hook relies on.

    The hook reassigns each infra-failed row to a unique uid. That only achieves "no
    gradient, no perturbation" if a singleton group really does get a zero advantage, and if
    the surviving group normalizes over its remaining members alone -- so pin both.
    """
    n = 5
    # Row 0 is the infra failure: reward 0.0 while its siblings scored well.
    rewards = [0.0, 0.8, 0.9, 0.7, 0.85]
    keys = [f"u0_s{i}_0" for i in range(n)]

    excluded = _real_batch(["__infra_excluded__deadbeef"] + ["u0"] * (n - 1))
    kept = _real_batch(["u0"] * n)
    for data in (excluded, kept):
        for i, r in enumerate(rewards):
            data.batch["token_level_rewards"][i, -1] = r

    excluded = _grpo(excluded, keys, n)
    kept = _grpo(kept, keys, n)

    assert float(excluded.batch["advantages"][0].abs().max()) == 0.0, (
        "a singleton group must produce no gradient at all"
    )
    assert float(kept.batch["advantages"][0].abs().max()) > 0.0, (
        "kept in its group, the infra row does get a gradient -- which is the bug the hook fixes"
    )
    # And it perturbs its siblings: their advantages differ between the two batches.
    sib_excluded = [float(excluded.batch["advantages"][i, -1]) for i in range(1, n)]
    sib_kept = [float(kept.batch["advantages"][i, -1]) for i in range(1, n)]
    assert sib_excluded != pytest.approx(sib_kept), (
        "leaving the infra row in drags the group mean and shifts every sibling's advantage"
    )
