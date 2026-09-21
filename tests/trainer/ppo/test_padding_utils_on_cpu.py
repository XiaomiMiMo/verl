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

import pytest
import torch

from verl.trainer.ppo.padding_utils import (
    construct_minimal_padding_template,
    get_megatron_sequence_length_multiple,
)


def test_megatron_sequence_length_multiple_for_cp2() -> None:
    assert get_megatron_sequence_length_multiple(tensor_parallel_size=8, context_parallel_size=2) == 32


@pytest.mark.parametrize("position_id_rank", [1, 3])
def test_construct_padding_template_is_cp2_aligned_and_has_zero_loss(position_id_rank: int) -> None:
    source_seq_len = 17
    if position_id_rank == 1:
        position_ids = torch.arange(source_seq_len, dtype=torch.int64)
    else:
        position_ids = torch.arange(source_seq_len, dtype=torch.int64).expand(3, -1).clone()
    source_td = {
        "position_ids": position_ids,
        "multi_modal_inputs": {"pixel_values": torch.ones(1)},
        "routed_experts": torch.ones(source_seq_len, 2, dtype=torch.int64),
    }

    sample, tag = construct_minimal_padding_template(
        source_td,
        {"prompt_len": 7, "response_len": 10, "seq_len": source_seq_len},
        eos_token_id=42,
        sequence_length_multiple=32,
    )

    assert sample["prompts"].shape == (31,)
    assert sample["responses"].shape == (1,)
    assert sample["input_ids"].shape == (32,)
    assert sample["attention_mask"].shape == (32,)
    assert sample["position_ids"].shape == (*position_ids.shape[:-1], 32)
    assert sample["routed_experts"].shape == (32, 2)
    assert sample["response_mask"].shape == (1,)
    assert sample["loss_mask"].shape == (1,)
    assert sample["rm_scores"].shape == (1,)
    assert sample["rollout_log_probs"].shape == (1,)
    assert torch.all(sample["input_ids"] == 42)
    assert torch.all(sample["attention_mask"] == 1)
    assert torch.all(sample["loss_mask"] == 0)
    assert sample["multi_modal_inputs"] == {}
    assert tag == {"is_padding": True, "prompt_len": 31, "response_len": 1, "seq_len": 32}
