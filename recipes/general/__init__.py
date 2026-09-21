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
"""General recipe: the s3k enterprise-function agent arm.

Each task hands the agent a workspace of business documents (pptx/xlsx/docx/pdf) in the
main container and a sidecar running that task's MCP servers -- simulated SaaS systems
backed by sqlite -- plus the verifier. The agent reads the documents, queries the systems
through MCP, and writes its conclusion to ``answer.md``; an LLM judge grades it against
the task's rubric of pass/fail assertions (``verifier_meta.json``).

The two-container pod topology is what separates this arm from the others: the MCP servers
have to sit next to the same workspace the tools mutate, so they cannot be moved out into
an external grading service the way the design arm's graders were.
"""
