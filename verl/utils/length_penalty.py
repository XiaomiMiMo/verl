# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""In-group length penalty for passed rollouts (meta-only, per-rid).

Anti GRPO length-bias: when a rollout is correct (reward >= pass_threshold),
penalize it linearly if its turn count / output tokens / input tokens exceed
the anchor (quantile) of the *passed* rollouts in the same prompt group (uid).

Design:
- Pure scalar functions over control-plane metadata (SequenceMeta-derived).
  No token tensors: the caller owns where/how the scalar delta lands
  (accept-time hook mutates meta.reward; downstream materialization scatters
  the final reward into token_level_scores).
- One call = one uid group. The accept boundary delivers exactly one full uid
  group, and both the anchor and the penalty are group-local, so per-group
  computation is equivalent to whole-batch computation.
- Length signals are aggregated per rid across its non-filtered contexts
  (aggregate_seq_length_signals) — the former agg_level="seq" semantics.
  The former "ctx" mode (per-context penalty) is gone: reward is a per-rid
  scalar and cannot express per-context deltas.
- Only passed rollouts are penalized; failed rollouts are untouched.
- Excess ratio per metric = max(0, (value - group_anchor) / group_anchor).
- Metrics combined into a single excess via "max" / "mean" / "weighted".
- A deadzone `excess_threshold` suppresses penalty until combined excess
  exceeds it; penalty then ramps from 0 at `excess_threshold` to max_penalty
  at `excess_saturate` along a convex curve `t**penalty_exponent`.

Stats protocol: compute_group_length_penalty returns additive raw counters
(one group's contribution). The accept hook accumulates them across groups
and calls finalize_length_penalty_metrics once per step to derive the
reported ratios — metric names stay identical to the legacy whole-batch
implementation so dashboards line up.
"""

from __future__ import annotations

import logging
from typing import Any, Collection, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)



VALID_METRICS = ("turns", "input_tokens", "output_tokens")
VALID_COMBINE = ("max", "mean", "weighted")


class LengthPenaltyConfig(BaseModel):
    """Configuration for in-group length penalty.

    Fields:
    - enabled: master switch.
    - max_penalty: upper bound of penalty subtracted from reward (per rollout).
    - excess_threshold: deadzone — combined excess at or below this gets zero
        penalty. e.g. 0.5 means "no penalty until 150% of group anchor". Must be
        non-negative and strictly less than `excess_saturate`.
    - excess_saturate: combined excess ratio at which penalty hits max_penalty.
        e.g. 1.0 means "200% of group anchor" (excess of 100%) reaches max.
    - penalty_exponent: convexity of the ramp. Penalty is
        `max_penalty * t**penalty_exponent` where t = normalised progress in
        [0, 1] across the [threshold, saturate] band. 1.0 = linear;
        2.0 = quadratic. Must be >= 1.0 so the curve is convex.
    - metrics: which length signals to compare against the group anchor.
        Subset of {"turns", "input_tokens", "output_tokens"}.
    - combine: how to fold per-metric excess into a single excess.
        "max"  – take the worst dimension (default; strictest).
        "mean" – arithmetic mean across enabled metrics.
        "weighted" – weighted sum; requires `weights`.
    - weights: dict of {metric: weight}; required iff combine == "weighted".
    - pass_threshold: rollouts with reward >= this are considered "passed".
    - anchor_quantile: quantile of the group's pass-rollout length signals used
        as the comparison anchor. 0.5 = median (default). Lower values
        (e.g. 0.25 = p25) compress harder. Must be in (0, 1).
    - min_pass_rate: skip the entire group when its pass-fraction
        len(passed) / len(group) is NOT strictly greater than this value.
        Default 0.0 = apply whenever there is at least one pass.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_penalty: float = Field(default=0.3, ge=0.0)
    excess_threshold: float = Field(default=0.0, ge=0.0)
    excess_saturate: float = Field(default=1.0, gt=0.0)
    penalty_exponent: float = Field(default=1.0, ge=1.0)
    metrics: tuple[str, ...] = VALID_METRICS
    combine: str = "max"
    weights: dict[str, float] | None = None
    pass_threshold: float = 0.5
    anchor_quantile: float = Field(default=0.5, gt=0.0, lt=1.0)
    min_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> LengthPenaltyConfig:
        bad = [m for m in self.metrics if m not in VALID_METRICS]
        if bad:
            raise ValueError(f"length_penalty.metrics has invalid entries: {bad}")
        if self.combine not in VALID_COMBINE:
            raise ValueError(f"length_penalty.combine={self.combine!r}; must be one of {VALID_COMBINE}")
        if self.combine == "weighted" and not self.weights:
            raise ValueError("length_penalty.combine='weighted' requires non-empty weights")
        if self.excess_threshold >= self.excess_saturate:
            raise ValueError(
                f"length_penalty.excess_threshold ({self.excess_threshold}) must be "
                f"strictly less than excess_saturate ({self.excess_saturate})"
            )
        return self



SEQ_LENGTH_KEYS = (
    "turn_count",
    "decode_length",
    "prefill_length",
    "prompt_length",
    "tool_length",
    "response_length",
)


def aggregate_seq_length_signals(
    context_metrics: Mapping[str, Mapping[str, Any]],
    filtered_cids: Collection[str],
) -> dict[str, int] | None:
    """Sum length signals across the training pool (non-filtered contexts).

    `context_metrics` is SequenceMeta.context_metrics: {cid: get_metric()},
    each value shaped {"length": {"turn_count": int, ...}, ...}. Returns a
    flat {signal: sum} dict, or None when no non-filtered context carries a
    length dict (the rollout then contributes no length signal at all).
    """
    filtered = set(filtered_cids)
    acc: dict[str, int] | None = None
    for cid, metric in context_metrics.items():
        if cid in filtered:
            continue
        length = metric.get("length", {}) if isinstance(metric, dict) else {}
        if not length:
            continue
        if acc is None:
            acc = {k: 0 for k in SEQ_LENGTH_KEYS}
        for k in SEQ_LENGTH_KEYS:
            v = length.get(k)
            if v is not None:
                acc[k] += int(v)
    return acc


def _extract_length_signal(signals: Mapping[str, float] | None, metric_name: str) -> float | None:
    """Read one of {turns, input_tokens, output_tokens} from a flat signal dict."""
    if not signals:
        return None
    if metric_name == "turns":
        v = signals.get("turn_count")
    elif metric_name == "input_tokens":
        v = signals.get("prefill_length")
        if v is None:
            v = signals.get("prompt_length")
    elif metric_name == "output_tokens":
        v = signals.get("decode_length")
    else:
        return None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _combine_excess(
    excess_per_metric: dict[str, float],
    combine: str,
    weights: dict[str, float] | None,
) -> float:
    """Fold per-metric excess ratios into a single excess scalar."""
    if not excess_per_metric:
        return 0.0
    values = list(excess_per_metric.values())
    if combine == "max":
        return max(values)
    if combine == "mean":
        return sum(values) / len(values)
    weighted_sum = 0.0
    total_w = 0.0
    for k, v in excess_per_metric.items():
        w = float(weights.get(k, 0.0))  # type: ignore[union-attr]
        weighted_sum += v * w
        total_w += w
    if total_w <= 0.0:
        return 0.0
    return weighted_sum / total_w




def _zero_stats(metrics: Sequence[str]) -> dict[str, float]:
    stats: dict[str, float] = {
        "groups_total": 0.0,
        "groups_with_pass": 0.0,
        "groups_skipped_low_pass_rate": 0.0,
        "passed": 0.0,
        "penalized": 0.0,
        "penalty_sum": 0.0,
        "excess_sum": 0.0,
    }
    for m in metrics:
        stats[f"excess_n/{m}"] = 0.0
        stats[f"excess_sum/{m}"] = 0.0
        stats[f"pass_var_groups/{m}"] = 0.0
        stats[f"pass_std_sum/{m}"] = 0.0
        stats[f"pass_cv_sum/{m}"] = 0.0
        stats[f"pass_mean_sum/{m}"] = 0.0
        stats[f"pass_range_ratio_sum/{m}"] = 0.0
    return stats


def compute_group_length_penalty(
    rewards: Sequence[float],
    signals: Sequence[Mapping[str, float] | None],
    cfg: LengthPenaltyConfig,
) -> tuple[list[float], dict[str, float]]:
    """Compute per-rid length-penalty deltas for ONE uid group.

    Args:
        rewards: scalar reward per rid in the group.
        signals: flat length-signal dict per rid (aggregate_seq_length_signals
            output; None when the rid carries no length data).
        cfg: LengthPenaltyConfig (cross-field validity enforced at construction).

    Returns:
        (deltas, stats)
        deltas: per-rid non-positive delta to ADD to the reward (0.0 = no
            penalty). Same order as `rewards`.
        stats: additive raw counters for this group — accumulate across groups
            and fold with finalize_length_penalty_metrics.
    """
    if len(rewards) != len(signals):
        raise ValueError(f"rewards/signals length mismatch: {len(rewards)} vs {len(signals)}")
    n = len(rewards)
    deltas = [0.0] * n
    stats = _zero_stats(cfg.metrics)

    if not cfg.enabled or n == 0:
        return deltas, stats

    stats["groups_total"] = 1.0

    mvs = [{m: _extract_length_signal(signals[i], m) for m in cfg.metrics} for i in range(n)]
    passed_idxs = [i for i in range(n) if rewards[i] >= cfg.pass_threshold]
    if not passed_idxs:
        return deltas, stats
    stats["groups_with_pass"] = 1.0
    stats["passed"] = float(len(passed_idxs))

    if len(passed_idxs) >= 2:
        for m in cfg.metrics:
            vals = [mvs[i][m] for i in passed_idxs if mvs[i][m] is not None]
            if len(vals) < 2:
                continue
            arr = np.asarray(vals, dtype=float)
            mean_v = float(arr.mean())
            std_v = float(arr.std(ddof=1))
            stats[f"pass_var_groups/{m}"] += 1.0
            stats[f"pass_mean_sum/{m}"] += mean_v
            stats[f"pass_std_sum/{m}"] += std_v
            stats[f"pass_cv_sum/{m}"] += std_v / mean_v if mean_v > 0.0 else 0.0
            stats[f"pass_range_ratio_sum/{m}"] += float(arr.max() - arr.min()) / mean_v if mean_v > 0.0 else 0.0

    if len(passed_idxs) / n <= cfg.min_pass_rate:
        stats["groups_skipped_low_pass_rate"] = 1.0
        return deltas, stats

    anchors: dict[str, float | None] = {}
    for m in cfg.metrics:
        vals = [mvs[i][m] for i in passed_idxs if mvs[i][m] is not None]
        anchors[m] = float(np.percentile(vals, cfg.anchor_quantile * 100.0)) if vals else None
    if all(v is None for v in anchors.values()):
        return deltas, stats

    for i in passed_idxs:
        excess_per_metric: dict[str, float] = {}
        for m in cfg.metrics:
            anchor = anchors[m]
            v = mvs[i][m]
            if anchor is None or v is None or anchor <= 0.0:
                continue
            e = max(0.0, (v - anchor) / anchor)
            excess_per_metric[m] = e
            if e > 0.0:
                stats[f"excess_sum/{m}"] += e
                stats[f"excess_n/{m}"] += 1.0

        if not excess_per_metric:
            continue

        combined = _combine_excess(excess_per_metric, cfg.combine, cfg.weights)
        if combined <= cfg.excess_threshold:
            continue

        ramp_span = cfg.excess_saturate - cfg.excess_threshold
        t = min((combined - cfg.excess_threshold) / ramp_span, 1.0)
        penalty = cfg.max_penalty * (t**cfg.penalty_exponent)
        if penalty <= 0.0:
            continue

        deltas[i] = -float(penalty)
        stats["penalized"] += 1.0
        stats["penalty_sum"] += float(penalty)
        stats["excess_sum"] += combined

    return deltas, stats




def finalize_length_penalty_metrics(acc: Mapping[str, float], metrics: Sequence[str]) -> dict[str, float]:
    """Fold accumulated additive stats into the reported metric dict.

    Metric names are identical to the legacy whole-batch implementation so
    existing dashboards keep working.
    """
    n_passed = acc.get("passed", 0.0)
    n_penalized = acc.get("penalized", 0.0)
    out: dict[str, float] = {
        "length_penalty/groups_total": acc.get("groups_total", 0.0),
        "length_penalty/groups_with_pass": acc.get("groups_with_pass", 0.0),
        "length_penalty/groups_skipped_low_pass_rate": acc.get("groups_skipped_low_pass_rate", 0.0),
        "length_penalty/passed_rollouts": n_passed,
        "length_penalty/penalized_rollouts": n_penalized,
        "length_penalty/penalize_rate": n_penalized / n_passed if n_passed else 0.0,
        "length_penalty/penalty_mean": (acc.get("penalty_sum", 0.0) / n_penalized if n_penalized else 0.0),
        "length_penalty/penalty_sum": acc.get("penalty_sum", 0.0),
        "length_penalty/excess_mean": (acc.get("excess_sum", 0.0) / n_penalized if n_penalized else 0.0),
    }
    for m in metrics:
        n = acc.get(f"excess_n/{m}", 0.0)
        s = acc.get(f"excess_sum/{m}", 0.0)
        out[f"length_penalty/excess_per_metric/{m}/over_median_rate"] = n / n_passed if n_passed else 0.0
        out[f"length_penalty/excess_per_metric/{m}/excess_mean"] = s / n if n else 0.0

        ng = acc.get(f"pass_var_groups/{m}", 0.0)
        out[f"length_penalty/passed_variance/{m}/groups_counted"] = ng
        out[f"length_penalty/passed_variance/{m}/std_mean"] = acc.get(f"pass_std_sum/{m}", 0.0) / ng if ng else 0.0
        out[f"length_penalty/passed_variance/{m}/cv_mean"] = acc.get(f"pass_cv_sum/{m}", 0.0) / ng if ng else 0.0
        out[f"length_penalty/passed_variance/{m}/range_ratio_mean"] = (
            acc.get(f"pass_range_ratio_sum/{m}", 0.0) / ng if ng else 0.0
        )
        out[f"length_penalty/passed_variance/{m}/mean_mean"] = acc.get(f"pass_mean_sum/{m}", 0.0) / ng if ng else 0.0
    return out
