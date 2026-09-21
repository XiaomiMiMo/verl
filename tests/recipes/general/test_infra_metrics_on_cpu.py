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
"""Unit tests for the general arm's infra accounting and gradient-yield metrics.

The numbers these produce are the only way to tell two different regressions apart on a
training curve, so their arithmetic is pinned here rather than eyeballed on a dashboard.
``dead_rows_non_infra`` in particular is the alarm for the infra-aware group filter: it
must count rows that are dead *despite* not being infra failures.
"""

import numpy as np
import pytest
import torch

from recipes.general.infra_metrics import infra_metrics


def _batch(adv_rows, tokens_per_row=4):
    """Build ``advantages`` / ``response_mask`` from one scalar advantage per row."""
    n = len(adv_rows)
    mask = torch.ones(n, tokens_per_row)
    adv = torch.tensor(adv_rows, dtype=torch.float32).unsqueeze(-1) * mask
    return adv, mask


def test_infra_rate_and_category_breakdown():
    adv, mask = _batch([0.5, -0.5, 0.0, 0.0])
    m = infra_metrics(
        advantages=adv,
        response_mask=mask,
        is_infra=np.array([0.0, 0.0, 1.0, 1.0]),
        error_categories=["ok", "ok", "setup/failed", "reward/exception"],
    )
    assert m["training/infra/count"] == 2.0
    assert m["training/infra/ratio"] == pytest.approx(0.5)
    assert m["training/infra/error/setup/failed"] == 1.0
    assert m["training/infra/error/reward/exception"] == 1.0
    # "ok" is every healthy rollout; emitting it would swamp the breakdown.
    assert not any(k.endswith("/ok") for k in m)


def test_dead_rows_separates_infra_from_collapsed_groups():
    """Two rows sit at zero advantage: one is infra, one is not.

    The infra row is expected to be there. The other one means its group lost its spread
    and trained on nothing, which is what the metric exists to surface.
    """
    adv, mask = _batch([0.5, 0.0, 0.0])
    m = infra_metrics(
        advantages=adv,
        response_mask=mask,
        is_infra=np.array([0.0, 1.0, 0.0]),
    )
    assert m["training/adv/dead_rows"] == 2.0
    assert m["training/adv/dead_rows_ratio"] == pytest.approx(2 / 3)
    assert m["training/adv/dead_rows_non_infra"] == 1.0
    assert m["training/valid_rate"] == pytest.approx(2 / 3)


def test_dead_tokens_ratio_weights_by_length():
    """A dead long row dilutes more than a dead short one, so the ratio is token-weighted."""
    mask = torch.zeros(2, 10)
    mask[0, :2] = 1  # live, 2 tokens
    mask[1, :8] = 1  # dead, 8 tokens
    adv = torch.zeros(2, 10)
    adv[0, :2] = 1.0
    m = infra_metrics(advantages=adv, response_mask=mask, is_infra=np.array([0.0, 0.0]))
    assert m["training/adv/dead_tokens_ratio"] == pytest.approx(0.8)


def test_padding_rows_are_not_counted_as_dead():
    """Padding carries no advantage; counting it would report a permanently dead batch."""
    adv, mask = _batch([0.5, 0.0])
    m = infra_metrics(
        advantages=adv,
        response_mask=mask,
        is_infra=np.array([0.0, 0.0]),
        keep=np.array([True, False]),
    )
    assert m["training/adv/dead_rows"] == 0.0
    assert m["training/adv/dead_rows_ratio"] == 0.0
    assert m["training/valid_rate"] == 1.0


def test_no_infra_means_exclude_the_sentinel_rows():
    """This is the point of the no-infra variants: the sentinel must not dilute the mean."""
    adv, mask = _batch([0.5, -0.5, 0.0])
    scores = torch.zeros(3, 4)
    scores[0, 0] = 1.0
    scores[1, 0] = 0.0
    scores[2, 0] = -999.0  # infra sentinel
    m = infra_metrics(
        advantages=adv,
        response_mask=mask,
        token_level_scores=scores,
        is_infra=np.array([0.0, 0.0, 1.0]),
        num_turns=np.array([10, 20, 1]),
    )
    assert m["training/no_infra/score_mean"] == pytest.approx(0.5)
    assert m["training/no_infra/num_turns_mean"] == pytest.approx(15.0)
    assert m["training/no_infra/response_length_mean"] == pytest.approx(4.0)


def test_without_is_infra_only_undifferentiated_counts():
    """A recipe that never tags rows still gets the dead-row counts, and no infra keys."""
    adv, mask = _batch([0.5, 0.0])
    m = infra_metrics(advantages=adv, response_mask=mask)
    assert m["training/adv/dead_rows"] == 1.0
    assert "training/adv/dead_rows_non_infra" not in m
    assert "training/infra/count" not in m
    assert "training/valid_rate" not in m


def test_degrades_instead_of_raising():
    """A metric helper that can kill a training step is worse than a missing metric."""
    adv, mask = _batch([0.5, 0.0])
    # Fully padded batch.
    assert infra_metrics(advantages=adv, response_mask=mask, keep=np.array([False, False])) == {}
    # Mismatched lengths are dropped rather than blowing up mid-step.
    assert infra_metrics(advantages=adv, response_mask=mask, keep=np.array([True])) == {}
    m = infra_metrics(advantages=adv, response_mask=mask, is_infra=np.array([1.0]))
    assert "training/infra/count" not in m and "training/adv/dead_rows" in m


def test_all_infra_batch_reports_zero_valid_rate():
    adv, mask = _batch([0.0, 0.0])
    m = infra_metrics(advantages=adv, response_mask=mask, is_infra=np.array([1.0, 1.0]))
    assert m["training/infra/ratio"] == 1.0
    assert m["training/valid_rate"] == 0.0
    assert m["training/adv/dead_rows_non_infra"] == 0.0
    assert "training/no_infra/response_length_mean" not in m
