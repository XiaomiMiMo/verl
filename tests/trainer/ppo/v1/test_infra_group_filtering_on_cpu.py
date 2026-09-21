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
"""Unit tests for DAPO group filtering in the presence of infra-failed rollouts.

A rollout can fail for two unrelated reasons: the model produced a wrong answer, or the
infrastructure broke (env pod never came up, exec stream died, reward harness crashed). Only the
first carries learning signal. Rollouts tag themselves via ``extra_fields["is_infra"]``, and the
DAPO all-same check must be judged on the non-infra subset -- otherwise a group of "10 solved +
6 infra zeros" looks like it has variance, passes the filter, and then contributes nothing once
the infra rows are neutralized.

``is_infra`` is optional: a rollout implementation that does not set it must keep the upstream
behaviour, which is what ``test_missing_is_infra_behaves_like_upstream`` pins down. The rule is a
static method so it is testable without a TransferQueue.
"""

import pytest

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

classify = ReplayBuffer._classify_group


def test_mixed_group_carries_gradient():
    """Genuine disagreement among non-infra rollouts => keep (None)."""
    assert classify([(1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]) is None


def test_all_same_group_is_evicted_with_its_value():
    assert classify([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]) == 1.0
    assert classify([(0.0, 0.0), (0.0, 0.0)]) == 0.0


def test_solved_plus_infra_zeros_is_no_signal():
    """The regression this change exists for.

    Raw std over all 16 would be non-zero, so the group would pass the filter, enter training,
    and then contribute nothing once the infra rows are neutralized -- leaving 10 identical
    rewards with zero GRPO advantage, burning a full forward/backward and never being refilled.
    """
    group = [(1.0, 0.0)] * 10 + [(0.0, 1.0)] * 6
    assert classify(group) == 1.0


def test_single_valid_rollout_is_no_signal():
    """One survivor has no within-group contrast, mirroring the reference RL framework's passrate handling."""
    assert classify([(1.0, 0.0)] + [(0.0, 1.0)] * 15) == 1.0
    assert classify([(0.0, 0.0), (0.0, 1.0)]) == 0.0


def test_all_infra_group_is_no_signal_at_zero():
    """Nothing was measured, so the group is evicted and refilled rather than trained on."""
    assert classify([(0.0, 1.0)] * 8) == 0.0


def test_missing_is_infra_behaves_like_upstream():
    """A rollout that never sets the flag reads as is_infra=0.0 and keeps old behaviour."""
    no_flag_mixed = [(1.0, 0.0), (0.0, 0.0)]
    no_flag_same = [(1.0, 0.0), (1.0, 0.0)]
    assert classify(no_flag_mixed) is None
    assert classify(no_flag_same) == 1.0


def test_flag_is_thresholded_not_compared_to_one():
    """The flag arrives as a float through extra_fields, so 0.5 is the boundary."""
    # 0.9 counts as infra, leaving one valid rollout -> no signal at its value
    assert classify([(1.0, 1.0), (0.0, 0.9)]) == pytest.approx(0.0)
    # 0.4 counts as non-infra, so the pair disagrees and keeps its gradient
    assert classify([(1.0, 0.4), (0.0, 0.0)]) is None


def test_non_binary_metric_values():
    """filter_groups.metric need not be a 0/1 reward (e.g. a continuous score)."""
    assert classify([(0.31, 0.0), (0.31, 0.0)]) == pytest.approx(0.31)
    assert classify([(0.31, 0.0), (0.32, 0.0)]) is None
