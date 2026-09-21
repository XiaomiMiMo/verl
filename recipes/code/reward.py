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
"""Fallback reward hook for the Uni-Agent recipe.

Normal rollouts use the runner's session ``reward`` payload directly.  This
function keeps verl's reward-loop initialization independent of the old
sidecar example and is useful for diagnostics or replayed trajectories.
"""

from __future__ import annotations


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    value = extra_info.get("reward", extra_info.get("reward_score", 0.0))
    return {"score": float(value)}
