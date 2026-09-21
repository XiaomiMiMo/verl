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
"""Glob tool: fast file-pattern matching.

Returns paths matching a glob pattern, sorted by mtime descending (Claude Code
semantics — the most-recently-modified files come first, which is usually what
the model wants when navigating an unfamiliar codebase).

Implementation uses bash globstar via `bash -O globstar -c 'ls -1d ...'`. Inside
SWE-bench `/testbed` images, bash with globstar support is universal, so no
ripgrep-style hard-dep probe is needed.
"""

import shlex
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput


class GlobTool(BaseTool):
    """Resolve a glob pattern, sorted by mtime (newest first)."""

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead"""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in. If not specified, the current working directory will be used.",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for glob execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        pattern = params.get("pattern")
        if not pattern:
            raise ToolException("Missing required parameter: pattern")

        path = params.get("path") or "."
        path_q = shlex.quote(str(path))

        if any(ch in pattern for ch in ("`", "$", ";", "&", "|", "\n", "\r")):
            raise ToolException(f"Glob pattern contains disallowed shell metacharacters: {pattern!r}")

        bash_script = (
            f"shopt -s globstar nullglob dotglob; "
            f"cd {path_q} && "
            f"for f in {pattern}; do "
            f'  if [ -e "$f" ]; then '
            f'    printf "%s\\t%s\\n" "$(stat -c %Y "$f" 2>/dev/null || echo 0)" "$(readlink -f "$f")"; '
            f"  fi; "
            f"done | sort -rn | cut -f2-"
        )

        result = env.execute(f"bash -c {shlex.quote(bash_script)}")
        output = result.get("output", "").strip()
        if not output:
            output = "(no matches)"

        return ToolOutput(output=output, success=True)
