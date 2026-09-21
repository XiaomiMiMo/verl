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
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY


class MetricsAggregator:
    """
    Combine per-iteration training metrics collected within a single ``parameter_sync_step`` cycle.
    Adapted from ``verl.experimental.fully_async_policy.detach_utils.MetricsAggregator`.
    """

    def __init__(self):
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        self.metric_weights: dict[str, list[int]] = defaultdict(list)
        self.dict_metrics: dict[str, Counter] = defaultdict(Counter)
        self.step_count = 0
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, list[str]]:
        return {
            "sum": [
                "training/off_policy/evicted_samples",
                "validation/off_policy/evicted_samples",
                "training/filter_groups/evicted_samples",
                "validation/filter_groups/evicted_samples",
                "training/filter_groups/discarded_surplus_samples",
                "validation/filter_groups/discarded_surplus_samples",
                "training/rollout_failure/evicted_samples",
                "validation/rollout_failure/evicted_samples",
            ],
            "last": [
                "training/global_step",
                "training/rollout_probs_diff_valid",
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int = 0):
        """Record one iteration's metrics."""
        self.step_count += 1
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if key == DAPO_FILTERED_REWARD_COUNTS_KEY and isinstance(value, dict):
                self.dict_metrics[key].update(value)
                continue
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
                self.metric_weights[key].append(self._get_metric_weight(key, metrics, sample_count))
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.metric_values[key].append(float(value.item()))
                self.metric_weights[key].append(self._get_metric_weight(key, metrics, sample_count))

    def _get_metric_weight(self, metric_name: str, metrics: dict[str, Any], sample_count: int) -> int:
        """Return the sample weight used when reducing per-iteration average metrics."""
        if metric_name.endswith("/off_policy/evicted_samples_staleness/mean"):
            prefix = metric_name.rsplit("_staleness/mean", 1)[0]
            evicted_samples = metrics.get(prefix, sample_count)
            if isinstance(evicted_samples, torch.Tensor):
                return int(evicted_samples.item()) if evicted_samples.numel() == 1 else sample_count
            if isinstance(evicted_samples, int | float | np.number):
                return int(evicted_samples)
        return sample_count

    def _get_aggregation_type(self, metric_name: str) -> str:
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if metric_lower.endswith("/lr") or metric_lower.endswith("_lr") or metric_lower == "lr":
            return "last"
        if "timing_s/" in metric_lower or "timing_per_token_ms/" in metric_lower:
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if "/filter_groups/group_count" in metric_lower or "/filter_groups/group_success_count/" in metric_lower:
            return "sum"
        if "/filter_groups/raw_" in metric_lower and metric_lower.endswith(("_sum", "_count")):
            return "sum"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg", "mean", "avg", "average"]):
            return "weighted_avg"
        return "weighted_avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)
        if agg_type == "last":
            return values[-1]
        if agg_type == "weighted_avg":
            weights = self.metric_weights[metric_name]
            if len(values) != len(weights) or sum(weights) == 0:
                return sum(values) / len(values)
            weighted_sum = sum(v * c for v, c in zip(values, weights, strict=False))
            return weighted_sum / sum(weights)
        if agg_type in ("sum", "time_sum"):
            return sum(values)
        if agg_type == "max":
            return max(values)
        if agg_type == "min":
            return min(values)
        return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        if self.step_count == 0:
            return {}
        aggregated = {name: self._aggregate_single_metric(name, values) for name, values in self.metric_values.items()}
        for name, counter in self.dict_metrics.items():
            if counter:
                aggregated[name] = dict(counter)
        return self._special_metrics_aggregate(aggregated)

    def _special_metrics_aggregate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """Recompute derived metrics that cannot be reduced from their per-iteration values."""
        if {"global_seqlen/minmax_diff", "global_seqlen/max", "global_seqlen/min"}.issubset(aggregated):
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        for count_key in [key for key in list(aggregated) if key.endswith("/filter_groups/group_count")]:
            prefix = count_key[: -len("/group_count")]
            total_groups = float(aggregated[count_key])

            raw_metric_keys = {
                f"{prefix}/raw_group_passrate_sum",
                f"{prefix}/raw_reward_sum",
                f"{prefix}/raw_rollout_count",
            }
            if raw_metric_keys & aggregated.keys():
                raw_group_passrate_sum = float(aggregated.pop(f"{prefix}/raw_group_passrate_sum", 0.0))
                raw_reward_sum = float(aggregated.pop(f"{prefix}/raw_reward_sum", 0.0))
                raw_rollout_count = float(aggregated.pop(f"{prefix}/raw_rollout_count", 0.0))
                if total_groups:
                    aggregated[f"{prefix}/raw/avg@n"] = raw_group_passrate_sum / total_groups
                if raw_rollout_count:
                    aggregated[f"{prefix}/raw/reward_mean"] = raw_reward_sum / raw_rollout_count
                    aggregated[f"{prefix}/raw/rollout_count"] = raw_rollout_count
                for bucket in ("zero", "one", "mid"):
                    count = float(aggregated.pop(f"{prefix}/raw_passrate_{bucket}_count", 0.0))
                    aggregated[f"{prefix}/raw/passrate/{bucket}"] = count / total_groups if total_groups else 0.0

            marker = f"{prefix}/group_success_count/"
            for key in list(aggregated):
                if not key.startswith(marker):
                    continue
                success_count = key[len(marker) :]
                ratio = float(aggregated[key]) / total_groups if total_groups else 0.0
                aggregated[f"{prefix}/group_success_ratio/{success_count}"] = ratio

        harness_marker = "/filter_groups/raw_harness/"
        for count_key in [key for key in list(aggregated) if harness_marker in key and key.endswith("/group_count")]:
            raw_prefix = count_key[: -len("/group_count")]
            output_prefix = raw_prefix.replace(
                "/filter_groups/raw_harness/",
                "/filter_groups/raw/harness/",
                1,
            )
            total_groups = float(aggregated.pop(count_key, 0.0))
            raw_group_passrate_sum = float(aggregated.pop(f"{raw_prefix}/group_passrate_sum", 0.0))
            raw_reward_sum = float(aggregated.pop(f"{raw_prefix}/reward_sum", 0.0))
            raw_rollout_count = float(aggregated.pop(f"{raw_prefix}/rollout_count", 0.0))
            aggregated[f"{output_prefix}/group_count"] = total_groups
            if total_groups:
                aggregated[f"{output_prefix}/avg@n"] = raw_group_passrate_sum / total_groups
            if raw_rollout_count:
                aggregated[f"{output_prefix}/reward_mean"] = raw_reward_sum / raw_rollout_count
                aggregated[f"{output_prefix}/rollout_count"] = raw_rollout_count
            for bucket in ("zero", "one", "mid"):
                count = float(aggregated.pop(f"{raw_prefix}/passrate_{bucket}_count", 0.0))
                aggregated[f"{output_prefix}/passrate/{bucket}"] = count / total_groups if total_groups else 0.0

        return aggregated

    def reset(self):
        self.metric_values.clear()
        self.metric_weights.clear()
        self.dict_metrics.clear()
        self.step_count = 0


def compute_advantage_for_multi_trajectories(
    data: DataProto,
    batch_keys: list[str],
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    group_suffixes: np.ndarray | list[str] | None = None,
) -> DataProto:
    """Compute GRPO advantages from each session's final output. For non-GRPO
    estimators, such as GAE, are delegated to the original compute_advantage() unchanged.

    For GRPO, only the final output in each ``{uid}_{session_id}`` group participates
    in advantage computation, and the result is broadcast to the other outputs in
    the same session. Sessions whose AgentLoop returns ``None`` simply do not appear
    in ``batch_keys``. Non-GRPO estimators, such as GAE, are delegated to the
    original ``compute_advantage()`` unchanged.
    """
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    # final session of each agent loop: {uid}_{session_id} => (index, row_index)
    final_sessions: dict[str, tuple[int, int]] = {}
    row_session_keys = []
    for i, key in enumerate(batch_keys):
        fields = key.rsplit("_", 2)
        assert len(fields) == 3, f"Unexpected key format: {key}"
        uid, session_id, index = fields[0], fields[1], int(fields[2])
        session_key = f"{uid}_{session_id}"
        row_session_keys.append(session_key)
        if session_key not in final_sessions or final_sessions[session_key][0] < index:
            final_sessions[session_key] = (index, i)

    if group_suffixes is not None:
        group_suffixes = np.asarray(group_suffixes, dtype=object).reshape(-1)
        if len(group_suffixes) != len(data):
            raise ValueError(f"group_suffixes has {len(group_suffixes)} rows, expected {len(data)}")

    # final session indices in batch data
    final_indices = []
    session_key_to_local_index = {}
    for session_key, (_, row_index) in final_sessions.items():
        final_indices.append(row_index)
        session_key_to_local_index[session_key] = len(final_indices) - 1
    row_to_local_index = [session_key_to_local_index[session_key] for session_key in row_session_keys]

    # select final sessions from batch data for group relative advantage computation
    final_data = data.select_idxs(final_indices)
    if group_suffixes is not None:
        base_uids = data.non_tensor_batch["uid"]
        final_data.non_tensor_batch["uid"] = np.asarray(
            [f"{base_uids[row_index]}::{group_suffixes[row_index]}" for row_index in final_indices],
            dtype=object,
        )
    final_data = compute_advantage(
        final_data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]

    # scatter final scores to all rows in batch data
    scores = final_scores[row_to_local_index]
    scores = scores.unsqueeze(-1) * data.batch["response_mask"]

    data.batch["advantages"] = scores
    data.batch["returns"] = scores
    return data
