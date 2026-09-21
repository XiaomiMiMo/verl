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
"""``adv_signed``: sign-aware token penalty with per-sign mass conservation.

Sign-aware advantage rebalance for marked tokens. The penalty a marked token receives
depends on the sign of its OWNING SEQUENCE's outcome advantage:

  positive sample, marked token -> advantage set to 0 (stop reinforcing the span)
  negative sample, marked token -> advantage *= kappa (kappa >= 1; punish the span
                                   harder than the rest of its own trajectory)

Each sign then conserves its total mass over the whole train batch: the mass removed
from positive marked tokens is handed back to the clean tokens of positive samples by
one factor ``alpha >= 1`` (clamped at ``max_scale``); the mass added on negative marked
tokens is taken back from the clean tokens of negative samples by one factor
``beta <= 1`` (clamped at ``min_scale``). A whole-turn ``adv_set -1`` instead injects
net negative mass into the batch, which with no KL / entropy bonus flattens the policy
and lengthens outputs — the documented reason the reference RL framework replaced it with this.

One extension over the reference RL framework: an optional per-row ``token_weights`` (the ``prompt-mean``
loss weights) so that "mass" means gradient mass under that aggregation. With unit
weights the math is identical to the reference RL framework's.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def solve_signed_factors(
    removed: float,
    pos_base: float,
    added: float,
    neg_base: float,
    *,
    min_scale: float,
    max_scale: float,
) -> tuple[float, bool, float, bool]:
    """``(alpha, alpha_clamped, beta, beta_clamped)`` from the four mass sums.

    ``removed``: mass the positive marked tokens carried (set to 0).
    ``pos_base``: mass of the clean positive tokens.
    ``added``: penalty mass put on negative marked tokens, ``sum((kappa-1)*|adv|)``.
    ``neg_base``: ``|mass|`` of the clean negative tokens.
    A clamped side is NOT conserved; a side with no clean tokens keeps factor 1 and is
    flagged clamped for the same reason.
    """
    assert 0 < min_scale <= 1.0, f"min_scale must be in (0, 1], got {min_scale}"
    assert max_scale >= 1.0, f"max_scale must be >= 1, got {max_scale}"
    alpha, alpha_clamped = 1.0, False
    if removed != 0.0:
        if pos_base > 0.0:
            alpha = 1.0 + removed / pos_base
            if alpha > max_scale:
                alpha, alpha_clamped = max_scale, True
            elif alpha < 0.0:
                alpha, alpha_clamped = 1.0, True
        else:
            alpha_clamped = True
    beta, beta_clamped = 1.0, False
    if added > 0.0:
        if neg_base > 0.0:
            beta = 1.0 - added / neg_base
            if beta < min_scale:
                beta, beta_clamped = min_scale, True
        else:
            beta_clamped = True
    if alpha_clamped or beta_clamped:
        logger.warning(
            "[adv_signed] rebalance clamped (alpha=%.4f clamped=%s, beta=%.4f clamped=%s): "
            "penalty mass is comparable to outcome mass, the batch stays imbalanced. "
            "removed=%.4f pos_base=%.4f added=%.4f neg_base=%.4f",
            alpha,
            alpha_clamped,
            beta,
            beta_clamped,
            removed,
            pos_base,
            added,
            neg_base,
        )
    return alpha, alpha_clamped, beta, beta_clamped


def signed_rebalance(
    adv: np.ndarray,
    hit: np.ndarray,
    seq_sign: np.ndarray,
    *,
    min_scale: float,
    max_scale: float,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Flat form over every VALID token of the batch.

    ``adv``: [T] float advantages. ``hit``: [T] kappa >= 1 of the strongest marking
    rule, 0 = not marked. ``seq_sign``: [T] in {-1, 0, +1}, sign of the owning
    sequence's outcome advantage; 0 = zero-adv sequence (degenerate group), left
    untouched and excluded from every sum. ``weight``: optional [T] per-token loss
    weight; mass sums are ``weight * adv`` so conservation holds under prompt-mean.
    """
    assert adv.ndim == 1 and hit.shape == adv.shape and seq_sign.shape == adv.shape, (
        f"flat 1-D inputs of one length expected, got adv={adv.shape} hit={hit.shape} sign={seq_sign.shape}"
    )
    assert np.all((hit == 0) | (hit >= 1.0)), "hit values must be 0 (no hit) or a multiplier >= 1"
    adv = adv.astype(np.float64, copy=True)
    w = np.ones_like(adv) if weight is None else np.asarray(weight, dtype=np.float64)
    assert w.shape == adv.shape, f"weight must match adv, got {w.shape} vs {adv.shape}"
    is_hit = hit > 0
    pos_rows = seq_sign > 0
    neg_rows = seq_sign < 0
    pos_hit, pos_clean = pos_rows & is_hit, pos_rows & ~is_hit
    neg_hit, neg_clean = neg_rows & is_hit, neg_rows & ~is_hit

    mass = w * adv
    pos_pre = float(mass[pos_rows].sum())
    neg_pre = float(mass[neg_rows].sum())
    removed = float(mass[pos_hit].sum())
    pos_base = float(mass[pos_clean].sum())
    added = float(-((hit[neg_hit] - 1.0) * mass[neg_hit]).sum())  # (kappa-1)*|A|, A < 0
    neg_base = float(-mass[neg_clean].sum())
    alpha, a_cl, beta, b_cl = solve_signed_factors(
        removed, pos_base, added, neg_base, min_scale=min_scale, max_scale=max_scale
    )

    adv[pos_hit] = 0.0
    adv[pos_clean] *= alpha
    adv[neg_hit] *= hit[neg_hit]
    adv[neg_clean] *= beta

    mass = w * adv
    pos_post = float(mass[pos_rows].sum())
    neg_post = float(mass[neg_rows].sum())
    if not a_cl:
        assert abs(pos_post - pos_pre) <= 1e-6 * max(1.0, abs(pos_pre)), (
            f"positive mass not conserved: {pos_pre} -> {pos_post}"
        )
    if not b_cl:
        assert abs(neg_post - neg_pre) <= 1e-6 * max(1.0, abs(neg_pre)), (
            f"negative mass not conserved: {neg_pre} -> {neg_post}"
        )
    metrics = {
        "penalty/signed/pos_hit_tokens": int(pos_hit.sum()),
        "penalty/signed/pos_mass_removed": removed,
        "penalty/signed/pos_scale": float(alpha),
        "penalty/signed/pos_scale_clamped": int(a_cl),
        "penalty/signed/neg_hit_tokens": int(neg_hit.sum()),
        "penalty/signed/neg_mass_added": added,
        "penalty/signed/neg_scale": float(beta),
        "penalty/signed/neg_scale_clamped": int(b_cl),
        "penalty/signed/adv_pos_sum_pre": pos_pre,
        "penalty/signed/adv_pos_sum_post": pos_post,
        "penalty/signed/adv_neg_sum_pre": neg_pre,
        "penalty/signed/adv_neg_sum_post": neg_post,
    }
    return adv.astype(np.float32), metrics


def rebalance_dense(
    advantages: torch.Tensor,
    hit: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    min_scale: float,
    max_scale: float,
    row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dense ``[B, L]`` adapter. ``hit`` holds kappa per token (0 = unmarked).

    The sequence sign is the median advantage over the row's valid tokens: exact for a
    GRPO broadcast (every token equal) and robust to an already-applied
    ``adv_set``/``adv_reduction`` span as long as that span is a minority of the row.
    ``row_weights`` ([B], e.g. prompt-mean loss weights) are broadcast over the row.
    """
    valid = response_mask.to(torch.bool)
    adv_np = advantages.detach().float().cpu().numpy()
    hit_np = hit.detach().float().cpu().numpy()
    valid_np = valid.cpu().numpy()
    sign_rows = np.zeros(adv_np.shape[0], dtype=np.int8)
    for r in range(adv_np.shape[0]):
        vals = adv_np[r][valid_np[r]]
        if len(vals):
            sign_rows[r] = int(np.sign(np.median(vals)))
    sign_np = np.broadcast_to(sign_rows[:, None], adv_np.shape)
    weight_flat = None
    if row_weights is not None:
        w_np = np.broadcast_to(row_weights.detach().double().cpu().numpy()[:, None], adv_np.shape)
        weight_flat = w_np[valid_np]
    new_flat, metrics = signed_rebalance(
        adv_np[valid_np],
        hit_np[valid_np],
        sign_np[valid_np],
        min_scale=min_scale,
        max_scale=max_scale,
        weight=weight_flat,
    )
    out = adv_np.astype(np.float32, copy=True)
    out[valid_np] = new_flat
    return torch.from_numpy(out).to(advantages.device, dtype=advantages.dtype), metrics
