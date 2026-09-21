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
"""How much of a step actually produced a gradient, and how much was cluster noise.

This arm scores an infra-failed rollout with the ``algorithm.invalid_reward_value``
sentinel instead of dropping it, so those rows travel all the way into the optimizer
batch: they occupy a slot, run a full forward/backward, and contribute their tokens to
the token-mean denominator. Nothing here changes that -- these are metrics only -- but
without them the cost is invisible and two different regressions look identical on
``critic/score/mean``.

Why each number exists:

``training/infra/*``
    The rate itself, plus a per-category breakdown, so a bad step can be attributed to
    pod setup, a dead exec stream or a verifier crash rather than to the policy.

``training/no_infra/*``
    ``critic/score/mean`` counts an infra row as a model-earned zero: the row survives
    metric_utils' abort filter because a failed rollout is still a 1-token stub, and the
    same deflation hits response_length and num_turns. Measured on an earlier SWE run,
    step 45 had infra=1 and score/mean 0.6387 while step 89 had infra=166 and 0.4316 --
    and 0.6387 * (512-166)/512 = 0.4317, i.e. the entire apparent decline was the infra
    rate.

``training/adv/dead_rows_non_infra``
    The regression alarm for the infra-aware group filter. An infra row is *supposed* to
    sit at zero advantage. A non-infra row at zero means its whole group lost its spread
    once the infra members were excluded -- the group passed the filter, trained on
    nothing, and was never refilled. Any non-zero value here means that filter is not
    doing its job.

Deliberately a plain function over tensors: it takes no trainer, no config and no
transfer-queue handle, so it is testable on CPU and cannot take a training step down.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


def infra_metrics(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor | None = None,
    is_infra: np.ndarray | None = None,
    error_categories: list[Any] | None = None,
    num_turns: np.ndarray | None = None,
    keep: np.ndarray | None = None,
) -> dict[str, float]:
    """One step's infra accounting and gradient yield.

    Args:
        advantages: ``[n_rows, response_len]``.
        response_mask: ``[n_rows, response_len]``, 1 on trained tokens.
        token_level_scores: ``[n_rows, response_len]``; enables the no-infra score mean.
        is_infra: per-row 0/1 flags from ``extra_fields["is_infra"]``. Without it only the
            undifferentiated dead-row counts are produced -- which are the ones that
            cannot be acted on.
        error_categories: per-row ``extra_fields["error_category"]`` for the breakdown.
        num_turns: per-row turn counts, for the no-infra turn mean.
        keep: per-row bool mask of rows that are real rather than batch padding. Padding
            rows carry no advantage and would otherwise all count as dead.

    Returns a metrics dict. Returns fewer keys rather than raising on a short or fully
    padded batch: a metric helper that can kill a training step is worse than a gap in a
    chart.
    """
    metrics: dict[str, float] = {}
    device = advantages.device
    n_rows_total = int(advantages.shape[0])

    keep_t = (
        torch.as_tensor(np.asarray(keep, dtype=bool), dtype=torch.bool, device=device)
        if keep is not None
        else torch.ones(n_rows_total, dtype=torch.bool, device=device)
    )
    if keep_t.numel() != n_rows_total:
        return metrics
    n_rows = int(keep_t.sum().item())
    if n_rows == 0:
        return metrics

    infra_t: torch.Tensor | None = None
    if is_infra is not None:
        flags = np.asarray(is_infra, dtype=float)
        if flags.shape[0] == n_rows_total:
            infra_t = torch.as_tensor(flags > 0.5, dtype=torch.bool, device=device)
            kept_flags = infra_t & keep_t
            metrics["training/infra/count"] = float(kept_flags.sum().item())
            metrics["training/infra/ratio"] = float(kept_flags.sum().item()) / n_rows

    if error_categories is not None and len(error_categories) == n_rows_total:
        kept = [c for c, k in zip(error_categories, keep_t.tolist(), strict=True) if k]
        for category, count in Counter(kept).items():
            if category and category != "ok":
                metrics[f"training/infra/error/{category}"] = float(count)

    tokens = response_mask.sum(dim=-1).float()
    active = (advantages.abs() * response_mask).amax(dim=-1) > 0
    dead = (~active) & keep_t
    metrics["training/adv/dead_rows"] = float(dead.sum().item())
    metrics["training/adv/dead_rows_ratio"] = float(dead.sum().item()) / n_rows
    token_total = float((tokens * keep_t).sum().item())
    if token_total > 0:
        metrics["training/adv/dead_tokens_ratio"] = float((tokens * dead).sum().item()) / token_total

    if infra_t is None:
        return metrics

    metrics["training/adv/dead_rows_non_infra"] = float((dead & ~infra_t).sum().item())

    valid = keep_t & ~infra_t
    n_valid = int(valid.sum().item())
    metrics["training/valid_rate"] = n_valid / n_rows
    if n_valid == 0:
        return metrics

    if token_level_scores is not None:
        metrics["training/no_infra/score_mean"] = float(token_level_scores.sum(dim=-1).float()[valid].mean().item())
    metrics["training/no_infra/response_length_mean"] = float(tokens[valid].mean().item())
    if num_turns is not None:
        turns = np.asarray(num_turns, dtype=np.float64).reshape(-1)
        if turns.shape[0] == n_rows_total:
            turns_t = torch.as_tensor(turns, dtype=torch.float32, device=device)
            metrics["training/no_infra/num_turns_mean"] = float(turns_t[valid].mean().item())

    return metrics
