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
"""Fail fast when critical unified-run settings were overridden."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _get(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"resolved config is missing {dotted_path}")
        value = value[key]
    return value


def _check(config: dict[str, Any], dotted_path: str, expected: Any) -> None:
    actual = _get(config, dotted_path)
    if actual != expected:
        raise ValueError(f"{dotted_path} must resolve to {expected!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--rollout-data-dir", required=True)
    parser.add_argument("--validation-data-dir", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--mamba-cache-size", type=int, required=True)
    parser.add_argument("--mamba-scheduler", required=True)
    parser.add_argument("--entropy-coeff", type=float, required=True)
    parser.add_argument("--filter-groups-enabled", choices=("true", "false"), required=True)
    parser.add_argument("--tool-call-error-penalty-enabled", choices=("true", "false"), required=True)
    parser.add_argument("--tool-call-error-penalty-strategy", required=True)
    parser.add_argument("--tool-call-error-penalty-value", type=float, required=True)
    parser.add_argument("--repetition-detect-enabled", choices=("true", "false"), required=True)
    parser.add_argument("--repetition-zero-reward", choices=("true", "false"), required=True)
    parser.add_argument("--repetition-penalty-enabled", choices=("true", "false"), required=True)
    parser.add_argument("--repetition-penalty-strategy", required=True)
    parser.add_argument("--repetition-penalty-value", type=float, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("resolved config must be a mapping")

    expected = {
        "algorithm.rollout_correction.bypass_mode": False,
        "transfer_queue.enable": True,
        "trainer.v1.trainer_mode": "colocate_async",
        "trainer.save_freq": args.save_freq,
        "trainer.max_actor_ckpt_to_keep": None,
        "trainer.max_critic_ckpt_to_keep": None,
        "trainer.default_local_dir": args.checkpoint_dir,
        "trainer.rollout_data_dir": args.rollout_data_dir,
        "trainer.validation_data_dir": args.validation_data_dir,
        "actor_rollout_ref.rollout.prometheus.enable": True,
        "actor_rollout_ref.rollout.disable_log_stats": False,
        "actor_rollout_ref.rollout.engine_kwargs.sglang.enable_metrics_for_all_schedulers": True,
        "actor_rollout_ref.rollout.temperature": args.temperature,
        "actor_rollout_ref.rollout.top_p": args.top_p,
        "actor_rollout_ref.rollout.top_k": args.top_k,
        "actor_rollout_ref.actor.calculate_entropy": True,
        "actor_rollout_ref.actor.entropy_coeff": args.entropy_coeff,
        "actor_rollout_ref.rollout.engine_kwargs.sglang.max_mamba_cache_size": args.mamba_cache_size,
        "actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy": args.mamba_scheduler,
        "actor_rollout_ref.model.mtp.enable": False,
        "actor_rollout_ref.model.mtp.enable_train": False,
        "actor_rollout_ref.model.mtp.enable_rollout": False,
        "algorithm.filter_groups.enable": args.filter_groups_enabled == "true",
        "algorithm.tool_call_error_penalty.enable": args.tool_call_error_penalty_enabled == "true",
        "algorithm.tool_call_error_penalty.strategy": args.tool_call_error_penalty_strategy,
        "algorithm.tool_call_error_penalty.penalty_value": args.tool_call_error_penalty_value,
        "actor_rollout_ref.rollout.custom.agent_framework.repetition_detect.enable": (
            args.repetition_detect_enabled == "true"
        ),
        "actor_rollout_ref.rollout.custom.agent_framework.repetition_detect.zero_reward": (
            args.repetition_zero_reward == "true"
        ),
        "algorithm.repetition_penalty.enable": args.repetition_penalty_enabled == "true",
        "algorithm.repetition_penalty.strategy": args.repetition_penalty_strategy,
        "algorithm.repetition_penalty.penalty_value": args.repetition_penalty_value,
    }
    if args.reasoning_effort:
        expected["data.apply_chat_template_kwargs.reasoning_effort"] = args.reasoning_effort
    for dotted_path, value in expected.items():
        _check(config, dotted_path, value)

    loggers = _get(config, "trainer.logger")
    if "tensorboard" not in loggers:
        raise ValueError(f"trainer.logger must include 'tensorboard', got {loggers!r}")


if __name__ == "__main__":
    main()
