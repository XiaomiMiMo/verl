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
"""Dataset adapter for the MimoAgent harness runner.

The existing MimoAgent SWE parquet deliberately stores the task as a JSON
string under ``extra_info.instance_json``.  Keeping that representation avoids
Arrow struct-union corruption; this adapter expands it only at rollout time
into the Uni-Agent ``tools_kwargs`` transport field.
"""

from __future__ import annotations

import json

from verl.utils.dataset.rl_dataset import RLHFDataset


class MimoAgentSWEDataset(RLHFDataset):
    """Expose a MimoAgent instance through Uni-Agent's runner contract."""

    def __getitem__(self, item):
        row = super().__getitem__(item)
        extra_info = dict(row.get("extra_info") or {})
        raw_instance = extra_info.get("instance_json")
        if raw_instance:
            instance = json.loads(raw_instance) if isinstance(raw_instance, str) else raw_instance
        else:
            instance = extra_info.get("instance")
        if not isinstance(instance, dict) or not instance.get("docker_image"):
            raise ValueError("MimoAgentSWEDataset requires extra_info.instance_json with docker_image")

        tools_kwargs = dict(extra_info.get("tools_kwargs") or {})
        tools_kwargs["instance"] = dict(instance)
        tools_kwargs["dataset_index"] = int(item)
        row["tools_kwargs"] = tools_kwargs
        extra_info["tools_kwargs"] = tools_kwargs
        row["extra_info"] = extra_info
        row.setdefault("data_source", "mimoagent/swe")
        row.setdefault("reward_model", {"style": "rule", "ground_truth": ""})
        return row
