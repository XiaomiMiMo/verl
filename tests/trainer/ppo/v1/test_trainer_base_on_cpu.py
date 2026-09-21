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

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import (
    PPOTrainer,
    _apply_tool_call_error_strategy,
    _log_harness_rollout_metrics,
)


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_log_harness_rollout_metrics_groups_reward_length_and_entropy():
    metrics = {}
    rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    response_mask = torch.tensor([[1, 1], [1, 0], [1, 1]])
    entropys = torch.tensor([[2.0, 4.0], [6.0, 0.0], [1.0, 3.0]])
    num_turns = [4, 8, 10]

    _log_harness_rollout_metrics(
        metrics,
        ["mimoagent", "claude-code", "mimoagent"],
        rewards,
        response_mask,
        entropys,
        num_turns,
        tool_call_error_values={
            "tool_call_error_count": [2, 0, 1],
            "tool_call_error_invalid_arguments_count": [1, 0, 1],
        },
    )

    assert metrics["harness/mimoagent/reward_mean"] == pytest.approx(1.0)
    assert metrics["harness/mimoagent/response_length_mean"] == pytest.approx(2.0)
    assert metrics["harness/mimoagent/entropy_mean"] == pytest.approx(2.5)
    assert metrics["harness/mimoagent/num_turns/mean"] == pytest.approx(7.0)
    assert metrics["harness/mimoagent/num_turns/min"] == pytest.approx(4.0)
    assert metrics["harness/mimoagent/num_turns/max"] == pytest.approx(10.0)
    assert metrics["harness/claude-code/reward_mean"] == pytest.approx(0.0)
    assert metrics["harness/claude-code/response_length_mean"] == pytest.approx(1.0)
    assert metrics["harness/claude-code/entropy_mean"] == pytest.approx(6.0)
    assert metrics["harness/claude-code/num_turns/mean"] == pytest.approx(8.0)
    assert metrics["harness/claude-code/num_turns/min"] == pytest.approx(8.0)
    assert metrics["harness/claude-code/num_turns/max"] == pytest.approx(8.0)
    assert metrics["harness/mimoagent/tool_call_error_rate"] == pytest.approx(1.0)
    assert metrics["harness/mimoagent/tool_call_error_count/mean"] == pytest.approx(1.5)
    assert metrics["harness/mimoagent/tool_call_error_invalid_arguments_count/mean"] == pytest.approx(1.0)
    assert metrics["rollout/tool_call_error_rate"] == pytest.approx(2 / 3)


def test_tool_call_error_monitor_preserves_response_mask():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    error_mask = torch.tensor([[0, 1, 0], [0, 0, 0]])

    updated, applied, metrics = _apply_tool_call_error_strategy(response_mask, error_mask, "monitor", penalty_value=0.5)

    assert torch.equal(updated, response_mask)
    assert torch.equal(applied, torch.tensor([[False, True, False], [False, False, False]]))
    assert metrics["penalty/tool_call_error_tokens"] == 1.0
    assert metrics["penalty/tool_call_error_sequences"] == 1.0
    assert metrics["penalty/tool_call_error_rate"] == pytest.approx(1 / 5)


def test_tool_call_error_mask_only_removes_marked_tokens():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    error_mask = torch.tensor([[0, 1, 0], [1, 0, 1]])

    updated, applied, _ = _apply_tool_call_error_strategy(response_mask, error_mask, "mask", penalty_value=0.0)

    assert torch.equal(applied, torch.tensor([[False, True, False], [True, False, False]]))
    assert torch.equal(updated, torch.tensor([[1, 0, 1], [0, 1, 0]]))


def test_tool_call_error_invalid_shape_is_fail_closed():
    response_mask = torch.ones((2, 3), dtype=torch.int64)
    error_mask = torch.ones((2, 2), dtype=torch.int64)

    updated, applied, metrics = _apply_tool_call_error_strategy(
        response_mask, error_mask, "adv_reduction", penalty_value=1.0
    )

    assert torch.equal(updated, response_mask)
    assert not applied.any()
    assert metrics["penalty/tool_call_error_tokens"] == 0.0


def test_tool_call_error_adv_set_mask_identifies_only_valid_error_tokens():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    error_mask = torch.tensor([[0, 1, 0], [1, 0, 1]])

    updated, applied, _ = _apply_tool_call_error_strategy(response_mask, error_mask, "adv_set", penalty_value=-1.0)
    advantages = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    result = torch.where(applied, torch.tensor(-1.0), advantages)

    assert torch.equal(updated, response_mask)
    assert torch.equal(result, torch.tensor([[0.5, -1.0, 0.5], [-1.0, 0.0, 0.0]]))


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )
