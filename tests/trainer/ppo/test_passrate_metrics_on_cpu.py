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
"""Unit tests for the per-prompt passrate metrics.

``critic/rewards/mean`` is a per-ROLLOUT average, so a prompt with many rollouts weighs more
than one with few. Grouping by uid first is what makes the number comparable across steps,
and splitting by ``data_source`` is what makes a mixed run readable at all.

The first test is the one that matters for every other recipe: without ``uid`` in
non_tensor_batch the helper must return nothing rather than raise, because it is called
unconditionally from ``compute_data_metrics``.
"""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_passrate_metrics


def _batch(rewards, uids=None, sources=None):
    n = len(rewards)
    tensors = {"dummy": torch.zeros(n, 1)}
    non_tensor = {}
    if uids is not None:
        non_tensor["uid"] = np.array(uids, dtype=object)
    if sources is not None:
        non_tensor["data_source"] = np.array(sources, dtype=object)
    batch = (
        DataProto.from_dict(tensors=tensors, non_tensors=non_tensor)
        if non_tensor
        else DataProto.from_dict(tensors=tensors)
    )
    return (
        batch,
        torch.tensor(rewards, dtype=torch.float32),
        torch.full((n,), 100.0, dtype=torch.float32),
    )


def test_without_uid_returns_nothing():
    """The safety property every other recipe depends on: no uid, no metrics, no raise."""
    batch, reward, length = _batch([1.0, 0.0])
    assert _compute_passrate_metrics(batch=batch, sequence_reward=reward, response_length=length) == {}


def test_passrate_is_the_mean_over_prompts_not_over_rollouts():
    """Prompt A has 4 rollouts, prompt B has 1. Weighting by prompt is the whole point."""
    # A: 2 of 4 solved (0.5). B: 1 of 1 solved (1.0). Prompt mean = 0.75.
    # A rollout mean would be 3/5 = 0.6.
    batch, reward, length = _batch(
        [1.0, 1.0, 0.0, 0.0, 1.0],
        uids=["a", "a", "a", "a", "b"],
    )
    m = _compute_passrate_metrics(batch=batch, sequence_reward=reward, response_length=length)
    assert m["train/passrate/avg_passrate"] == pytest.approx(0.75)
    assert m["train/passrate/num_uids"] == 2


def test_the_three_buckets_partition_the_prompts():
    """all-fail / all-pass / the learnable middle. Only the middle carries a gradient."""
    batch, reward, length = _batch(
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        uids=["zero", "zero", "one", "one", "mid", "mid"],
    )
    m = _compute_passrate_metrics(batch=batch, sequence_reward=reward, response_length=length)
    assert m["train/passrate/passrate_0_ratio"] == pytest.approx(1 / 3)
    assert m["train/passrate/passrate_1_ratio"] == pytest.approx(1 / 3)
    assert m["train/passrate/passrate_mid_ratio"] == pytest.approx(1 / 3)


def test_per_source_split_and_dynsam_aliases():
    """A mixed run is unreadable without the split; the aliases keep existing tags working."""
    batch, reward, length = _batch(
        [1.0, 1.0, 0.0, 0.0],
        uids=["a", "a", "b", "b"],
        sources=["src_x", "src_x", "src_y", "src_y"],
    )
    m = _compute_passrate_metrics(batch=batch, sequence_reward=reward, response_length=length)
    assert m["train/passrate/src_x/avg_passrate"] == pytest.approx(1.0)
    assert m["train/passrate/src_y/avg_passrate"] == pytest.approx(0.0)
    assert m["dynsam/src_x/avg@n"] == pytest.approx(1.0)
    assert m["response_length/by_source/src_x/mean"] == pytest.approx(100.0)


def test_sentinel_rollouts_are_excluded_from_the_passrate():
    """An infra failure is not a failed attempt; counting it would depress the passrate."""
    # Prompt has 4 rollouts: 2 solved, 2 invalidated by infrastructure.
    batch, reward, length = _batch(
        [1.0, 1.0, -999.0, -999.0],
        uids=["a", "a", "a", "a"],
    )
    graded = _compute_passrate_metrics(
        batch=batch, sequence_reward=reward, response_length=length, invalid_reward_value=-999
    )
    # Measured over the 2 real rollouts, not diluted to 0.5 by the sentinels.
    assert graded["train/passrate/avg_passrate"] == pytest.approx(1.0)
    # Without the sentinel value the same batch reads as a partial failure.
    naive = _compute_passrate_metrics(batch=batch, sequence_reward=reward, response_length=length)
    assert naive["train/passrate/avg_passrate"] < 0.0
