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
"""How much of a step's compute actually produced a gradient.

A sample whose advantage is identically zero still runs a full forward and backward, and
still contributes its tokens to the token-mean denominator -- so it dilutes every other
sample's gradient rather than merely wasting its own slot. Counting those rows is the only
way to tell a healthy step from one that trained on almost nothing.

The split by infra failure is what makes the count actionable:

* An infra row is *supposed* to sit at zero. The exclusion hook gives it a singleton group
  precisely so it contributes nothing.
* A **non-infra** row at zero means its whole group lost its spread -- typically after its
  infra members were split out -- so the group passed the dynamic-sampling filter, trained on
  nothing, and was never refilled. That is the number worth reacting to.

The same split also corrects a score that otherwise reads like a policy regression. Infra
rows score 0.0 and are counted in ``score/mean``, so a step with a bad pod rate looks like a
worse policy. Measured on one run: step 45 had 1 infra failure and score/mean 0.6387, step 89
had 166 and 0.4316 -- and 0.6387 * (512-166)/512 = 0.4317. The entire "decline" was the pod
rate.

Pure tensor arithmetic with no verl or MimoAgent imports, so it is testable on a CPU without
a trainer.
"""

from __future__ import annotations

import numpy as np
import torch


def gradient_yield_metrics(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_scores: torch.Tensor | None = None,
    is_infra: np.ndarray | None = None,
    keep: np.ndarray | None = None,
) -> dict[str, float]:
    """``training/adv/*`` and ``training/no_infra/*`` for one step.

    Args:
        advantages: ``[n_rows, response_len]``.
        response_mask: ``[n_rows, response_len]``, 1 on trained tokens.
        token_level_scores: ``[n_rows, response_len]``; enables the no-infra score mean.
        is_infra: per-row 0/1 infra flags. Without it only the undifferentiated counts are
            produced -- which are the ones that cannot be acted on.
        keep: per-row bool mask of rows that are real rather than batch padding. Padding
            rows have no advantage and would otherwise all count as dead.

    Returns a metrics dict. Never raises on an empty or fully-padded batch: it returns fewer
    keys instead, because a metric helper that can kill a training step is worse than a
    missing metric.
    """
    metrics: dict[str, float] = {}
    n_rows_total = int(advantages.shape[0])
    if n_rows_total == 0:
        return metrics

    device = advantages.device
    if keep is None:
        keep_t = torch.ones(n_rows_total, dtype=torch.bool, device=device)
    else:
        keep_t = torch.as_tensor(np.asarray(keep, dtype=bool), dtype=torch.bool, device=device)

    active = (advantages.abs() * response_mask).amax(dim=-1) > 0
    dead = (~active) & keep_t

    n_rows = int(keep_t.sum().item())
    if n_rows == 0:
        return metrics

    tokens = response_mask.sum(dim=-1).float()
    tok_total = float((tokens * keep_t).sum().item())

    metrics["training/adv/dead_rows"] = float(dead.sum().item())
    metrics["training/adv/dead_rows_ratio"] = float(dead.sum().item()) / n_rows
    if tok_total > 0:
        metrics["training/adv/dead_tokens_ratio"] = float((tokens * dead).sum().item()) / tok_total

    if is_infra is None:
        return metrics

    infra_t = torch.as_tensor(np.asarray(is_infra, dtype=float) > 0.5, dtype=torch.bool, device=device)
    if infra_t.shape[0] != n_rows_total:
        return metrics

    metrics["training/adv/dead_rows_non_infra"] = float((dead & ~infra_t).sum().item())

    valid = keep_t & ~infra_t
    n_valid = int(valid.sum().item())
    metrics["training/valid_rate"] = n_valid / n_rows
    if n_valid > 0 and token_level_scores is not None:
        metrics["training/no_infra/score_mean"] = float(token_level_scores.sum(dim=-1).float()[valid].mean().item())
    return metrics


__all__ = ["gradient_yield_metrics"]
