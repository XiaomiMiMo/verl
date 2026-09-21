# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""CPU-only tests for the experimental ``algorithm.deep_failure_mask`` channel.

The channel is an A/B instrument: per prompt group it drops the loss on tokens whose
generation-turn index is at or beyond ``ceil(alpha * median(turns of successes))``,
either on failed trajectories only (``mask_failure``) or on every trajectory
(``mask_both``). These tests pin the default-off contract, the group-relative
threshold, the fail-closed behaviour, and the emitted metrics.
"""

import numpy as np
import pytest
import torch

from verl.trainer.config.algorithm import AlgoConfig, DeepFailureMaskConfig
from verl.trainer.ppo.v1.trainer_base import _apply_deep_failure_mask, _depth_bucket_mass_metrics


def _group(turn_lists, scores, mu):
    """Build padded response_mask / turn_index / advantages for one group.

    ``turn_lists[i]`` gives the turn index of every generated token of row ``i``.
    """
    length = max(len(t) for t in turn_lists)
    n = len(turn_lists)
    response_mask = torch.zeros(n, length)
    turn_index = torch.zeros(n, length, dtype=torch.long)
    advantages = torch.zeros(n, length)
    for i, turns in enumerate(turn_lists):
        response_mask[i, : len(turns)] = 1.0
        turn_index[i, : len(turns)] = torch.tensor(turns)
        advantages[i, : len(turns)] = scores[i] - mu
    return response_mask, turn_index, advantages, torch.tensor(scores, dtype=torch.float32)


def test_config_default_is_disabled_and_validates():
    assert AlgoConfig().deep_failure_mask.enable is False
    assert DeepFailureMaskConfig().strategy == "mask_failure"
    with pytest.raises(ValueError):
        DeepFailureMaskConfig(strategy="mask_success")
    with pytest.raises(ValueError):
        DeepFailureMaskConfig(alpha=0.0)


def test_mask_failure_drops_only_failed_tail_beyond_group_success_median():
    # successes have 2 and 4 turns -> median 3 -> k = 3 (alpha=1)
    turns = [
        [0, 0, 1],  # success, 2 turns
        [0, 1, 2, 3, 3],  # success, 4 turns: turn 3 is beyond k but success -> kept
        [0, 1, 2, 3, 4, 5],  # failure, 6 turns: turns 3,4,5 dropped
        [0, 1],  # failure, 2 turns: nothing dropped
    ]
    scores = [1.0, 1.0, 0.0, 0.0]
    rm, ti, adv, sc = _group(turns, scores, mu=0.5)
    metrics = {}
    out = _apply_deep_failure_mask(rm, adv, ti, sc, np.array(["g"] * 4, dtype=object), "mask_failure", 1.0, metrics)
    expected = rm.clone()
    expected[2, 3:6] = 0.0
    assert torch.equal(out, expected)
    assert metrics["penalty/deep_failure_mask_status"] == 1.0
    assert metrics["penalty/deep_failure_mask_tokens"] == 3.0
    assert metrics["penalty/deep_failure_mask_sequences"] == 1.0
    assert metrics["penalty/deep_failure_mask_k_mean"] == 3.0
    assert metrics["penalty/deep_failure_mask_failure_token_frac"] == pytest.approx(3 / 8)
    # advantages are not touched by the channel
    assert adv[2, 0].item() == pytest.approx(-0.5)


def test_mask_both_drops_success_tail_too():
    turns = [[0, 0, 1], [0, 1, 2, 3, 3], [0, 1, 2, 3, 4, 5], [0, 1]]
    scores = [1.0, 1.0, 0.0, 0.0]
    rm, ti, adv, sc = _group(turns, scores, mu=0.5)
    out = _apply_deep_failure_mask(rm, adv, ti, sc, np.array(["g"] * 4, dtype=object), "mask_both", 1.0, {})
    expected = rm.clone()
    expected[1, 3:5] = 0.0
    expected[2, 3:6] = 0.0
    assert torch.equal(out, expected)


def test_alpha_scales_threshold():
    turns = [[0, 0, 1], [0, 1, 2, 3, 3], [0, 1, 2, 3, 4, 5], [0, 1]]
    scores = [1.0, 1.0, 0.0, 0.0]
    rm, ti, adv, sc = _group(turns, scores, mu=0.5)
    out = _apply_deep_failure_mask(rm, adv, ti, sc, np.array(["g"] * 4, dtype=object), "mask_failure", 1.5, {})
    expected = rm.clone()
    expected[2, 5:6] = 0.0  # k = ceil(4.5) = 5 -> only turn 5 dropped
    assert torch.equal(out, expected)


def test_group_without_success_is_untouched_and_groups_are_independent():
    # group a: all failures -> no k -> untouched. group b: normal.
    turns = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 0], [0, 1, 2, 3, 4]]
    scores = [0.0, 0.0, 1.0, 0.0]
    rm, ti, adv, sc = _group(turns, scores, mu=0.0)
    adv[2] = torch.where(rm[2] > 0, torch.tensor(0.5), torch.tensor(0.0))
    adv[3] = torch.where(rm[3] > 0, torch.tensor(-0.5), torch.tensor(0.0))
    uids = np.array(["a", "a", "b", "b"], dtype=object)
    metrics = {}
    out = _apply_deep_failure_mask(rm, adv, ti, sc, uids, "mask_failure", 1.0, metrics)
    expected = rm.clone()
    expected[3, 1:5] = 0.0  # group b: success has 1 turn -> k=1 -> failure turns >=1 dropped
    assert torch.equal(out, expected)
    assert metrics["penalty/deep_failure_mask_groups"] == 1.0


def test_missing_or_misaligned_turn_index_fails_closed():
    turns = [[0, 1, 2], [0, 1, 2, 3, 4, 5]]
    rm, ti, adv, sc = _group(turns, [1.0, 0.0], mu=0.5)
    uids = np.array(["g", "g"], dtype=object)
    for bad in (None, ti[:, :2]):
        metrics = {}
        out = _apply_deep_failure_mask(rm, adv, bad, sc, uids, "mask_failure", 1.0, metrics)
        assert torch.equal(out, rm)
        assert metrics["penalty/deep_failure_mask_status"] == 0.0
        assert metrics["penalty/deep_failure_mask_tokens"] == 0.0


def test_previously_masked_tokens_stay_masked():
    turns = [[0, 1], [0, 1, 2, 3]]
    rm, ti, adv, sc = _group(turns, [1.0, 0.0], mu=0.5)
    rm[1, 0] = 0.0  # e.g. repetition mask already cleared this token
    out = _apply_deep_failure_mask(rm, adv, ti, sc, np.array(["g", "g"], dtype=object), "mask_failure", 1.0, {})
    assert out[1, 0].item() == 0.0
    assert out[1, 1].item() == 1.0  # turn 1 < k=2 kept
    assert out[1, 2:].sum().item() == 0.0


def test_depth_bucket_mass_metrics_prompt_mean_weights():
    # one group, two rows: success 3 tokens at turns 0,0,6 ; failure 5 tokens at turns 0,6,6,120,120
    turns = [[0, 0, 6], [0, 6, 6, 120, 120]]
    rm, ti, adv, sc = _group(turns, [1.0, 0.0], mu=0.5)
    metrics = {}
    _depth_bucket_mass_metrics(rm, adv, ti, np.array(["g", "g"], dtype=object), metrics, "dm")
    w = 1.0 / (1 * 8)
    assert metrics["dm/t01_05/pos_mass"] == pytest.approx(2 * 0.5 * w)
    assert metrics["dm/t01_05/neg_mass"] == pytest.approx(1 * 0.5 * w)
    assert metrics["dm/t06_20/neg_over_pos"] == pytest.approx(2.0)
    assert metrics["dm/t101p/pos_mass"] == 0.0
    assert metrics["dm/t101p/neg_over_pos"] == 0.0  # no positive mass -> reported as 0
    assert metrics["dm/t101p/tokens"] == 2.0
    assert metrics["dm/t21_50/tokens"] == 0.0
