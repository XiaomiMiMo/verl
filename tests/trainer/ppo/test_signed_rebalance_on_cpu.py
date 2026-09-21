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
"""adv_signed (the reference RL framework signed_rebalance port): sign-aware penalty with per-sign mass conservation."""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.signed_rebalance import rebalance_dense, signed_rebalance, solve_signed_factors


def test_positive_hit_goes_to_zero_and_mass_is_handed_back():
    # 4 positive tokens (adv 1.0), one marked; 4 negative tokens (adv -1.0), none marked.
    adv = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32)
    hit = np.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    sign = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
    out, m = signed_rebalance(adv, hit, sign, min_scale=0.5, max_scale=2.0)
    assert out[0] == 0.0
    assert np.allclose(out[1:4], 4 / 3)  # 3 clean tokens carry the removed 1.0
    assert np.allclose(out[4:], -1.0)  # negative side untouched
    assert m["penalty/signed/pos_scale"] == pytest.approx(4 / 3)
    assert m["penalty/signed/adv_pos_sum_post"] == pytest.approx(m["penalty/signed/adv_pos_sum_pre"])


def test_negative_hit_is_scaled_by_kappa_and_relieved_from_clean_negatives():
    adv = np.array([1, 1, -1, -1, -1, -1], dtype=np.float32)
    hit = np.array([0, 0, 2, 0, 0, 0], dtype=np.float32)
    sign = np.array([1, 1, -1, -1, -1, -1], dtype=np.int8)
    out, m = signed_rebalance(adv, hit, sign, min_scale=0.5, max_scale=2.0)
    assert out[2] == -2.0
    assert np.allclose(out[3:], -2 / 3)  # added 1.0 taken back from 3 clean tokens
    assert np.allclose(out[:2], 1.0)
    assert m["penalty/signed/neg_scale"] == pytest.approx(2 / 3)
    assert m["penalty/signed/adv_neg_sum_post"] == pytest.approx(m["penalty/signed/adv_neg_sum_pre"])


def test_clamps_are_reported_and_break_conservation():
    # Every positive token marked: nothing to hand back to -> alpha clamped.
    adv = np.array([1, 1, -1, -1], dtype=np.float32)
    hit = np.array([2, 2, 3, 0], dtype=np.float32)
    sign = np.array([1, 1, -1, -1], dtype=np.int8)
    out, m = signed_rebalance(adv, hit, sign, min_scale=0.5, max_scale=2.0)
    assert m["penalty/signed/pos_scale_clamped"] == 1
    assert np.allclose(out[:2], 0.0)
    # beta would be 1 - 2/1 = -1 -> clamped at 0.5
    assert m["penalty/signed/neg_scale_clamped"] == 1 and m["penalty/signed/neg_scale"] == 0.5
    assert out[3] == pytest.approx(-0.5)


def test_zero_sign_rows_are_untouched_and_excluded():
    adv = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    hit = np.array([2, 0, 2, 0], dtype=np.float32)
    sign = np.array([0, 0, 1, 1], dtype=np.int8)
    out, _ = signed_rebalance(adv, hit, sign, min_scale=0.5, max_scale=2.0)
    assert np.allclose(out[:2], 0.0)
    assert out[2] == 0.0 and out[3] == pytest.approx(2.0)


def test_solve_factors_refuses_to_flip_sign():
    # removed < 0 (another channel drove the marked span negative inside a positive
    # sample): a hand-back that would make alpha negative is refused ...
    alpha, a_cl, beta, b_cl = solve_signed_factors(-3.0, 2.0, 0.0, 0.0, min_scale=0.5, max_scale=2.0)
    assert (alpha, a_cl, beta, b_cl) == (1.0, True, 1.0, False)
    # ... while a milder one just shrinks the clean positives (the reference RL framework semantics).
    alpha, a_cl, _, _ = solve_signed_factors(-1.0, 2.0, 0.0, 0.0, min_scale=0.5, max_scale=2.0)
    assert (alpha, a_cl) == (0.5, False)


def test_weighted_conservation_under_prompt_mean():
    # Two positive rows with different prompt-mean weights: conservation must hold on w*adv.
    adv = np.array([1, 1, 1, 1], dtype=np.float32)
    hit = np.array([2, 0, 0, 0], dtype=np.float32)
    sign = np.ones(4, dtype=np.int8)
    w = np.array([0.5, 0.5, 0.1, 0.1])
    out, m = signed_rebalance(adv, hit, sign, min_scale=0.5, max_scale=2.0, weight=w)
    assert (w * out).sum() == pytest.approx((w * adv).sum())
    assert m["penalty/signed/pos_scale"] == pytest.approx(1 + 0.5 / 0.7)
    # Unweighted call would hand back 1/3 instead.
    out_u, _ = signed_rebalance(adv, hit, sign, min_scale=0.5, max_scale=2.0)
    assert out_u[1] == pytest.approx(4 / 3)


def test_rebalance_dense_uses_row_median_sign_and_respects_mask():
    advantages = torch.tensor([[0.5, 0.5, 0.5, 0.0], [-0.5, -0.5, -0.5, 0.0]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    kappa = torch.tensor([[0.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    out, m = rebalance_dense(advantages, kappa, response_mask, min_scale=0.5, max_scale=2.0)
    assert out[0].tolist() == pytest.approx([0.75, 0.0, 0.75, 0.0])
    assert out[1].tolist() == pytest.approx([-1.0, -0.25, -0.25, 0.0])
    assert out.dtype == advantages.dtype
    assert m["penalty/signed/pos_hit_tokens"] == 1 and m["penalty/signed/neg_hit_tokens"] == 1


def test_rebalance_dense_with_row_weights():
    advantages = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    response_mask = torch.ones(2, 2, dtype=torch.long)
    kappa = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    weights = torch.tensor([0.5, 0.1], dtype=torch.float64)
    out, _ = rebalance_dense(advantages, kappa, response_mask, min_scale=0.5, max_scale=2.0, row_weights=weights)
    assert out[0, 0] == 0.0
    total_pre = (weights[:, None] * advantages).sum()
    total_post = (weights[:, None] * out.double()).sum()
    assert total_post.item() == pytest.approx(total_pre.item())
