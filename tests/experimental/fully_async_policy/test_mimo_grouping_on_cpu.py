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
import numpy as np
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import prepare_single_generation_data


def test_one_queued_prompt_keeps_all_grpo_siblings() -> None:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 16,
                    "multi_turn": {"enable": True},
                }
            }
        }
    )
    batch = prepare_single_generation_data(
        {
            "agent_name": np.array(["mimo_swe_agent"], dtype=object),
            "extra_info": np.array([{"index": 7}], dtype=object),
        },
        config,
    )

    assert len(batch) == 16
    assert batch.non_tensor_batch["agent_name"].tolist() == ["mimo_swe_agent"] * 16
    assert [item["index"] for item in batch.non_tensor_batch["extra_info"]] == [7] * 16


def test_multi_turn_must_be_enabled_to_preserve_custom_agent_route() -> None:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 2,
                    "multi_turn": {"enable": False},
                }
            }
        }
    )
    batch = prepare_single_generation_data(
        {"agent_name": np.array(["mimo_swe_agent"], dtype=object)},
        config,
    )

    assert batch.non_tensor_batch["agent_name"].tolist() == ["single_turn_agent"] * 2
