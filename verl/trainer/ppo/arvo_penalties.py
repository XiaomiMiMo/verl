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
"""Tool-error and length penalty operators."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.signed_rebalance import rebalance_dense
from verl.utils.length_penalty import LengthPenaltyConfig, compute_group_length_penalty, finalize_length_penalty_metrics

_PROFILE_PATH = Path(__file__).resolve().parents[3] / "recipes/arvo/REFERENCE_PENALTIES.json"


def _metadata(values):
    return [getattr(value, "data", value) for value in values]


def _final_sessions(keys):
    final = {}
    sessions = []
    for row, key in enumerate(keys):
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid rollout key: {key}")
        uid, session_id, output_index = parts
        session = (uid, session_id)
        sessions.append(session)
        output_index = int(output_index)
        if session not in final or output_index > final[session][0]:
            final[session] = (output_index, row)
    return final, sessions


@dataclass(frozen=True)
class ReferencePenalties:
    length: LengthPenaltyConfig
    kappa: float
    min_scale: float
    max_scale: float

    @classmethod
    def reference(cls):
        profile = json.loads(_PROFILE_PATH.read_text())
        if profile["selected_rules"] != ["tool_call_error", "length_penalty"]:
            raise ValueError("This build only supports the two explicitly selected penalties")
        tool = profile["tool_call_error"]
        if tool["strategy"] != "adv_signed" or tool["level"] != "segment":
            raise ValueError("The selected reference uses segment-level adv_signed")
        return cls(
            LengthPenaltyConfig(**profile["length_penalty"]),
            tool["negative_multiplier"],
            tool["signed_min_scale"],
            tool["signed_max_scale"],
        )

    @classmethod
    def for_training(cls, config):
        if OmegaConf.select(config, "trainer.val_only", default=False):
            return None
        expected = cls.reference()
        required = {
            "algorithm.adv_estimator": "grpo",
            "algorithm.norm_adv_by_std_in_grpo": False,
            "algorithm.use_kl_in_reward": False,
            "algorithm.filter_groups.enable": True,
            "algorithm.filter_groups.metric": "reward",
            "actor_rollout_ref.actor.use_kl_loss": False,
            "actor_rollout_ref.actor.loss_agg_mode": "prompt-mean",
        }
        for key, value in required.items():
            if OmegaConf.select(config, key) != value:
                raise ValueError(f"The frozen reference profile requires {key}={value!r}")
        supplied = OmegaConf.select(config, "reward_model.length_penalty")
        values = expected.length.model_dump()
        if supplied is not None:
            values.update(OmegaConf.to_container(supplied, resolve=True))
        if LengthPenaltyConfig(**values).model_dump() != expected.length.model_dump():
            raise ValueError("The penalty-enabled build requires the fixed reference length configuration")
        OmegaConf.update(config, "reward_model.length_penalty", expected.length.model_dump(), force_add=True)
        return expected

    def shape_scores(self, raw_scores, keys, infos):
        """Return float64 scalar rewards, using one final output per rollout session."""
        infos = _metadata(infos)
        if len(raw_scores) != len(keys) or len(infos) != len(keys):
            raise ValueError("Rollout scores, keys and metadata must be aligned")
        finals, sessions = _final_sessions(keys)
        grouped = defaultdict(list)
        for session, (_, row) in finals.items():
            info = infos[row]
            if float(info.get("is_infra", 0.0)) > 0.5:
                continue
            if "length_signals" not in info:
                raise ValueError("Reference length penalty requires recorded length_signals")
            signals = info["length_signals"]
            for name in ("turn_count", "prefill_length", "decode_length"):
                if name not in signals:
                    raise ValueError(f"Missing reference length signal {name}")
            grouped[session[0]].append((session, row))

        per_session = {}
        accumulated = {}
        for rows in grouped.values():
            scores = [float(raw_scores[row]) for _, row in rows]
            signals = [infos[row]["length_signals"] for _, row in rows]
            deltas, stats = compute_group_length_penalty(scores, signals, self.length)
            for (session, _), score, delta in zip(rows, scores, deltas, strict=True):
                per_session[session] = score + delta
            for key, value in stats.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
        shaped = [per_session.get(session, float(raw_scores[row])) for row, session in enumerate(sessions)]
        return shaped, finalize_length_penalty_metrics(accumulated, self.length.metrics)

    def shape_training_rewards(self, raw_token_scores, generation_mask, keys, infos):
        raw = raw_token_scores.sum(dim=-1).detach().cpu().tolist()
        shaped, metrics = self.shape_scores(raw, keys, infos)
        output = raw_token_scores.clone()
        if not output.shape[1]:
            return output, metrics
        positions = torch.arange(output.shape[1], device=output.device)
        ends = torch.where(generation_mask.bool(), positions, -1).amax(dim=-1)
        for row, score in enumerate(shaped):
            if score != raw[row] and ends[row] >= 0:
                output[row] = 0.0
                output[row, ends[row]] = score
        return output, metrics

    def tool_error_hits(self, generation_mask, infos):
        infos = _metadata(infos)
        if len(infos) != len(generation_mask):
            raise ValueError("Tool-error metadata must align with the training rows")
        hit = torch.zeros_like(generation_mask, dtype=torch.float32)
        for row, info in enumerate(infos):
            if float(info.get("is_infra", 0.0)) > 0.5:
                continue
            if "llm_turn_spans" not in info or "tool_call_error_flags" not in info:
                raise ValueError("Tool-error penalty requires explicit model-turn spans and native error flags")
            spans, errors = info["llm_turn_spans"], info["tool_call_error_flags"]
            if len(spans) != len(errors):
                raise ValueError("Native tool-error flags are not aligned with model turns")
            if not spans:
                continue
            coverage = torch.zeros_like(generation_mask[row], dtype=torch.bool)
            previous_end = 0
            for (start, end), error in zip(spans, errors, strict=True):
                if not (0 <= previous_end <= start <= end <= generation_mask.shape[1]):
                    raise ValueError("Model-turn spans overlap or exceed the response bounds")
                coverage[start:end] = True
                if error:
                    hit[row, start:end] = self.kappa
                previous_end = end
            if not torch.equal(coverage, generation_mask[row].bool()):
                raise ValueError("Model-turn spans must cover exactly the generated assistant tokens")
        return hit

    def apply_tool_penalty(self, advantages, response_mask, generation_mask, infos):
        """Apply the batch-wide signed operator after GRPO."""
        infos = _metadata(infos)
        hit = self.tool_error_hits(generation_mask, infos)
        invalid = torch.tensor(
            [float(info.get("is_infra", 0.0)) > 0.5 for info in infos],
            dtype=torch.bool,
            device=advantages.device,
        ).unsqueeze(-1)
        return rebalance_dense(
            advantages,
            hit,
            response_mask,
            invalid,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
        )
