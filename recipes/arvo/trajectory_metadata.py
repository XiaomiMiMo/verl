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
"""Token and turn metadata for the ARVO trajectory bridge.

These are observations recorded per rollout, consumed by
``ReferencePenalties.shape_scores`` and ``tool_error_hits`` when
``algorithm.arvo_penalties.enable=true``.
"""


def trajectory_metadata(prompt_ids, response_mask, llm_turn_spans, tool_call_error_flags):
    decode_length = sum(int(value) for value in response_mask)
    response_length = len(response_mask)
    leading_input = llm_turn_spans[0][0] if llm_turn_spans else response_length
    tool_length = response_length - leading_input - decode_length
    prompt_length = len(prompt_ids) + leading_input
    response_length -= leading_input
    return {
        "llm_turn_spans": [list(span) for span in llm_turn_spans],
        "tool_call_error_flags": list(tool_call_error_flags),
        "length_signals": {
            "prompt_length": prompt_length,
            "response_length": response_length,
            "decode_length": decode_length,
            "tool_length": tool_length,
            "prefill_length": prompt_length + tool_length,
            "turn_count": len(llm_turn_spans),
        },
    }
