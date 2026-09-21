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

import numpy as np
import pytest

pytest.importorskip("ray")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics, extract_spec_decode_stats


def test_spec_decode_metrics_detect_drafts_with_zero_acceptance():
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([3, 3, 3]),
        spec_accepts=np.array([0, 0, 0]),
        spec_verifies=np.array([1, 1, 1]),
    )

    assert metrics["rollout/spec_accept_rate"] == 0.0
    assert metrics["rollout/spec_accept_length"] == 1.0


def test_spec_decode_metrics_report_nonzero_acceptance_after_recovery():
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([3, 3, 3]),
        spec_accepts=np.array([3, 2, 1]),
        spec_verifies=np.array([1, 1, 1]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0


def test_spec_decode_metrics_drop_padded_placeholders():
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([3, 3, 3]),
        spec_accepts=np.array([3, 0, 0]),
        spec_verifies=np.array([1, 1, 1]),
        non_padding_mask=np.array([True, False, False]),
    )

    assert metrics["rollout/spec_accept_rate"] == 1.0
    assert metrics["rollout/spec_accept_length"] == 4.0


def test_extract_spec_decode_stats_skips_missing_extra_fields():
    stats = extract_spec_decode_stats({})

    assert stats == (None, None, None)
    assert compute_spec_decode_metrics(*stats) == {}


def test_extract_spec_decode_stats_skips_incomplete_extra_fields():
    spec_data = {
        "extra_fields": np.array(
            [{"spec_num_draft_tokens": 3, "spec_num_accepted_tokens": 2}],
            dtype=object,
        )
    }

    stats = extract_spec_decode_stats(spec_data)

    assert stats == (None, None, None)
    assert compute_spec_decode_metrics(*stats) == {}


def test_extract_spec_decode_stats_preserves_complete_metadata():
    spec_data = {
        "extra_fields": np.array(
            [
                {
                    "spec_num_draft_tokens": 3,
                    "spec_num_accepted_tokens": 2,
                    "spec_num_verify_steps": 1,
                },
                {
                    "spec_num_draft_tokens": 6,
                    "spec_num_accepted_tokens": 3,
                    "spec_num_verify_steps": 2,
                },
            ],
            dtype=object,
        )
    }

    drafts, accepts, verifies = extract_spec_decode_stats(spec_data)

    assert drafts == [3, 6]
    assert accepts == [2, 3]
    assert verifies == [1, 2]


def test_extract_spec_decode_stats_accepts_list_like_transfer_queue_fields():
    spec_data = {
        "extra_fields": [
            {
                "spec_num_draft_tokens": 3,
                "spec_num_accepted_tokens": 2,
                "spec_num_verify_steps": 1,
            }
        ]
    }

    assert extract_spec_decode_stats(spec_data) == ([3], [2], [1])
