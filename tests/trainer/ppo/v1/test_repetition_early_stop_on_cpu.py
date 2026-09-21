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

"""CPU tests for the experimental ``repetition_penalty.strategy=early_stop`` (the reference RL framework port):
tokens before the first repetition hit are masked, hit spans go to adv_signed."""

import pytest
import torch

from verl.trainer.config.algorithm import RepetitionPenaltyConfig
from verl.trainer.ppo.v1.trainer_base import _split_early_stop_mask


def test_early_stop_config_requires_multiplier():
    RepetitionPenaltyConfig(enable=True, strategy="early_stop", penalty_value=2.0)
    with pytest.raises(ValueError):
        RepetitionPenaltyConfig(enable=True, strategy="early_stop", penalty_value=0.5)
    # other strategies still validated the old way
    RepetitionPenaltyConfig(enable=True, strategy="mask", penalty_value=0.0)


def test_split_prefix_before_first_hit_and_hit_spans():
    rm = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1.0]])
    rep = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1.0]])
    prefix, hit = _split_early_stop_mask(rm, rep)
    assert prefix.tolist()[0] == [True, True, True, False, False, False, False, False]
    assert hit.tolist()[0] == [False, False, False, True, True, False, False, False]
    assert not prefix[1].any() and not hit[1].any()  # no hit -> untouched
    assert not prefix[2].any()  # hit at position 0 -> empty prefix
    assert hit[2].tolist() == [True, True, False, False, False, False, True, True]


def test_split_fails_closed_on_missing_or_misaligned_mask():
    rm = torch.ones(2, 5)
    for bad in (None, torch.ones(2, 3)):
        prefix, hit = _split_early_stop_mask(rm, bad)
        assert not prefix.any() and not hit.any()


def test_split_respects_response_mask_padding():
    rm = torch.tensor([[1, 1, 1, 0, 0.0]])
    rep = torch.tensor([[0, 0, 0, 1, 1.0]])  # hit only on padding
    prefix, hit = _split_early_stop_mask(rm, rep)
    assert not prefix.any() and not hit.any()
