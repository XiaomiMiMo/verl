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
"""Grep tool: ripgrep-backed content search across a tree.

`rg` must be available in the rollout environment. When it is missing, the
tool stages the static binary bundled at ``tools/bin/rg`` into the
environment on first use (lazy: pods whose rollouts never Grep pay nothing).
If neither works the call fails with a clear error — there is no silent
fallback to plain `grep`, because the two tools have materially different
output, ignore-file, and regex semantics, and the model would otherwise be
trained against a moving contract.
"""

import os
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput

_BUNDLED_RG = Path(os.environ.get("AGENT_RG_PATH") or (Path(__file__).resolve().parent / "bin" / "rg"))
_RG_DEST = "/usr/local/bin/rg"


@dataclass
class GrepToolConfig(ToolConfig):
    default_head_limit: int = 250  # Cap output lines/files when the model doesn't supply one


class GrepTool(BaseTool):
    """Search file contents with ripgrep."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._rg_lock = threading.Lock()
        self._rg_ready_envs: set[int] = set()

    def _create_config(self, config_dict: dict[str, Any]) -> GrepToolConfig:
        return GrepToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return """A powerful search tool built on ripgrep

Usage:
- ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
- Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")
- Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
- Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
- Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`
"""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than glob for standard file types.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": 'Output mode. Defaults to "files_with_matches".',
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search (rg -i)",
                },
                "-n": {
                    "type": "boolean",
                    "description": 'Show line numbers in output (rg -n). Requires output_mode: "content". Defaults to true.',
                },
                "-A": {
                    "type": "number",
                    "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content".',
                },
                "-B": {
                    "type": "number",
                    "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content".',
                },
                "-C": {
                    "type": "number",
                    "description": 'Number of lines to show before and after each match (rg -C). Requires output_mode: "content".',
                },
                "context": {
                    "type": "number",
                    "description": "Alias for -C.",
                },
                "head_limit": {
                    "type": "number",
                    "description": "Limit output to first N lines/entries. Defaults to 250 when unspecified. Pass 0 for unlimited.",
                },
                "offset": {
                    "type": "number",
                    "description": "Skip first N lines/entries before applying head_limit. Defaults to 0.",
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for grep execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        pattern = params.get("pattern")
        if not pattern:
            raise ToolException("Missing required parameter: pattern")

        self._ensure_rg(env)

        argv: list[str] = ["rg", "--color=never"]

        output_mode = params.get("output_mode") or "files_with_matches"
        if output_mode == "files_with_matches":
            argv.append("-l")
        elif output_mode == "count":
            argv.append("-c")
        elif output_mode == "content":
            show_line_numbers = params.get("-n")
            if show_line_numbers is None or bool(show_line_numbers):
                argv.append("-n")
            for flag in ("-A", "-B", "-C"):
                val = params.get(flag)
                if val is not None:
                    argv.extend([flag, str(int(val))])
            ctx = params.get("context")
            if ctx is not None and params.get("-C") is None:
                argv.extend(["-C", str(int(ctx))])
        else:
            raise ToolException(f"Invalid output_mode: {output_mode!r}")

        if bool(params.get("-i")):
            argv.append("-i")
        if bool(params.get("multiline")):
            argv.extend(["-U", "--multiline-dotall"])

        glob_filter = params.get("glob")
        if glob_filter:
            argv.extend(["--glob", str(glob_filter)])
        type_filter = params.get("type")
        if type_filter:
            argv.extend(["--type", str(type_filter)])

        argv.extend(["-e", str(pattern)])

        path = params.get("path")
        if path:
            argv.append(str(path))

        rg_cmd = " ".join(shlex.quote(a) for a in argv)

        head_limit_raw = params.get("head_limit")
        try:
            head_limit = int(head_limit_raw) if head_limit_raw is not None else self.config.default_head_limit
        except (TypeError, ValueError):
            head_limit = self.config.default_head_limit
        try:
            offset = int(params.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0

        full_cmd = rg_cmd
        if offset > 0:
            full_cmd += f" | tail -n +{offset + 1}"
        if head_limit > 0:
            full_cmd += f" | head -n {head_limit}"

        result = env.execute(f"({full_cmd}); rc=$?; if [ $rc -eq 0 ] || [ $rc -eq 1 ]; then exit 0; else exit $rc; fi")

        output = result.get("output", "")
        if not output.strip():
            output = "(no matches)"

        return ToolOutput(output=output, success=True, metadata={"output_mode": output_mode})

    def _ensure_rg(self, env: Any) -> None:
        """Make sure ``rg`` is runnable in ``env``, staging the bundled static
        binary on first use if the image doesn't ship one.

        Lazy by design (per user decision): environments whose rollouts never
        call Grep pay no copy cost. The per-env memo makes the probe a no-op
        after the first successful call.
        """
        env_key = id(env)
        if env_key in self._rg_ready_envs:
            return
        with self._rg_lock:
            if env_key in self._rg_ready_envs:
                return
            which = env.execute("command -v rg || true").get("output", "").strip()
            if which:
                self._rg_ready_envs.add(env_key)
                return
            if not _BUNDLED_RG.exists():
                raise ToolException(
                    "ripgrep (rg) is not installed in this environment and the bundled "
                    f"static binary is missing ({_BUNDLED_RG}). The Grep tool requires rg. "
                    "Provide a static rg at tools/bin/rg or set AGENT_RG_PATH."
                )
            try:
                env.copy_to(str(_BUNDLED_RG), _RG_DEST)
            except Exception as e:
                raise ToolException(f"Failed to stage bundled rg into the environment: {e}") from e
            res = env.execute(f"chmod +x {shlex.quote(_RG_DEST)} && command -v rg")
            if res.get("returncode", 1) != 0 or not res.get("output", "").strip():
                raise ToolException(f"Staged rg at {_RG_DEST} but it is not runnable: {res.get('output', '')[:300]}")
            self._rg_ready_envs.add(env_key)
