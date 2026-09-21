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
"""The web-dev arm's agent: MimoAgent's default loop over the web-dev catalogue.

The reference run's harness profile named no ``agent.type``, so it ran on
``DefaultAgent`` with the six capitalised tools injected into MimoAgent's single
global tool catalogue. That injection point is gone -- the packaged MimoAgent now
keeps one catalogue per agent type -- so this registers a named agent instead of
mutating a global table.

The pairing is the point, and it is not the same as picking ``cc-agent``.
``CCAgent`` carries two behaviours ``DefaultAgent`` does not, both visible to the
policy and neither present in the reference run:

* ``show_context_usage`` defaults on, appending a ``<context_usage>`` footer to
  every tool result once usage crosses 80% of a 100k window.
* ``step()`` adds stray-``<tool_call>`` detection, which changes how malformed
  output is handled.

So the loop comes from ``DefaultAgent`` and only the catalogue is replaced.
"""

from __future__ import annotations

from mimoagent.agents.default import DefaultAgent

from .tools import WebdevToolRegistry

AGENT_TYPE = "webdev-agent"


class WebdevAgent(DefaultAgent):
    """``DefaultAgent`` with the web-dev tool catalogue.

    ``_build_tool_registry`` is the hook catalogue variants are meant to
    override; everything else in the loop -- parallel tool dispatch, observation
    truncation, ToolException-to-user-message mapping, ``step_limit``,
    ``tool_call_errors`` -- runs unmodified.
    """

    def _build_tool_registry(self) -> WebdevToolRegistry:
        return WebdevToolRegistry.from_config(self.config.tools)


def register_webdev_agent() -> None:
    """Make ``agent.type: webdev-agent`` resolvable. Idempotent and additive.

    ``_AGENT_LOADERS`` is a plain module-level dict, so this needs no change to
    the vendored package -- the same property the dataset-environment
    registration relies on.
    """
    from mimoagent.agents import factory

    factory._AGENT_LOADERS.setdefault(AGENT_TYPE, lambda: WebdevAgent)


__all__ = ["AGENT_TYPE", "WebdevAgent", "register_webdev_agent"]
