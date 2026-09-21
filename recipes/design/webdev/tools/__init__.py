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
"""The web-dev arm's tool catalogue.

Six Claude-Code-named tools -- ``Bash / Read / Write / Edit / Grep / Glob`` --
ported as-is from the reference run's recipe-local copies, descriptions
included. The descriptions are prompt text the policy reads on every turn, so a
reworded one is a different arm, not a tidier one. That is why this catalogue
exists instead of reusing ``mimoagent.tools.cc``: the packaged versions have
since gained a concurrency note on Bash and Edit and made Read's image support
opt-in, all of which the reference run's rollouts never saw.

``Agent`` (subagent) and ``Compact`` are deliberately absent. Both fork or
rewrite the conversation, and the rollout keeps one append-only token buffer
that cannot represent either; they also need ``agent``/``model`` in the tool
context, which the environment actor cannot supply because the agent lives in a
different process. ``environment.load_config`` rejects them by name so a
profile naming one fails before a pod is created.
"""

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException
from mimoagent.tools.registry import ToolRegistry

from .bash import BashTool
from .edit import EditTool
from .glob import GlobTool
from .grep import GrepTool
from .read import ReadTool
from .write import WriteTool

WEBDEV_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "Bash": BashTool,
    "Read": ReadTool,
    "Write": WriteTool,
    "Edit": EditTool,
    "Grep": GrepTool,
    "Glob": GlobTool,
}

CONVERSATION_TOOLS = frozenset({"agent", "Agent", "actor", "task", "compact", "Compact"})


class WebdevToolRegistry(ToolRegistry):
    """ToolRegistry over the web-dev catalogue.

    Name lookup is exact. The reference run resolved these names out of a single
    global catalogue, so a typo surfaced as ``Unknown tool type`` there too;
    keeping it exact means a profile cannot silently get a different tool than
    the one it named.
    """

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "WebdevToolRegistry":
        registry = cls()
        for tool_spec in config:
            tool_name = tool_spec.get("tool")
            if not tool_name:
                raise ToolException(f"Tool specification missing 'tool' field: {tool_spec}")
            tool_class = WEBDEV_TOOL_CLASSES.get(tool_name)
            if tool_class is None:
                raise ToolException(f"Unknown tool type: {tool_name}. Web-dev catalogue: {sorted(WEBDEV_TOOL_CLASSES)}")
            registry.register(tool_class(tool_spec.get("config", {})))
        return registry


__all__ = [
    "CONVERSATION_TOOLS",
    "WEBDEV_TOOL_CLASSES",
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "WebdevToolRegistry",
    "WriteTool",
]
