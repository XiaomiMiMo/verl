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
"""The four-whitebox code recipe: verl -> Uni-Agent -> MimoAgent.

"Whitebox" is the distinction that matters: all four arms drive their tools
in-process through the Gateway, so the policy's own tokens are what call them.
The blackbox harnesses that run a real CLI inside the pod are a different
mechanism and are not part of this mix.

This package is the glue between three components that do not depend on each
other. See README.md for the two reflection hand-offs, and why both are name
strings resolved at rollout time rather than imports.
"""
