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
"""Claude-Code-aligned tool catalogue, vendored for the recipe's EVAL path.

Verbatim copies of mimoagent's ``tools/cc/{bash,read,write,edit,grep,glob}.py``
— the ``Bash / Read / Write / Edit / Grep / Glob`` tools with Claude-Code names and
schemas — so a Claude-Code-distilled checkpoint's tool calls resolve here instead of
hitting ``Unknown tool`` (see the recipe README's "Real-reward diagnosis").

The subagent (``Agent``) and ``Compact`` tools are deliberately NOT vendored: both fork
or rewrite the conversation, which this recipe's append-only token buffer (token_trace.py)
cannot represent — the same reason ``env_actor.load_mimoagent_config`` rejects the
``agent`` tool.

``register_cc_tools()`` inserts these six under their capitalized names into mimoagent's
``tools.registry._TOOL_CLASSES`` so that BOTH ``ToolRegistry.from_config`` call sites
resolve CC names:
  * the **agent-side** ``DefaultAgent.__init__`` (``mimoagent/agents/default.py``), which
    builds a local registry to validate tool names, and
  * the **env-actor-side** registry that actually executes tools (``env_actor.py``).
It is additive (``setdefault``) and idempotent. Training uses lowercase names, which keep
hitting mimoagent's own entries, so registering these is inert for the training path.
"""

from mimoagent.tools.base import BaseTool

from .bash import BashTool
from .edit import EditTool
from .glob import GlobTool
from .grep import GrepTool
from .read import ReadTool
from .write import WriteTool

CC_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "Bash": BashTool,
    "Read": ReadTool,
    "Write": WriteTool,
    "Edit": EditTool,
    "Grep": GrepTool,
    "Glob": GlobTool,
}


def register_cc_tools() -> None:
    """Add the CC tools to mimoagent's ToolRegistry catalogue (idempotent, additive)."""
    from mimoagent.tools import registry

    for tool_name, tool_class in CC_TOOL_CLASSES.items():
        registry._TOOL_CLASSES.setdefault(tool_name, tool_class)


__all__ = [
    "CC_TOOL_CLASSES",
    "register_cc_tools",
    "BashTool",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "GrepTool",
    "GlobTool",
]
