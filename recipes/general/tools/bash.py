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
"""Bash command execution tool."""

from dataclasses import dataclass
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput


@dataclass
class BashToolConfig(ToolConfig):
    """Configuration for the Bash tool.

    `timeout` is the per-call default (seconds) used when the model omits the
    `timeout` argument. `max_timeout` is a hard ceiling (seconds): a model may
    raise its per-call timeout up to this value but no further, so a single
    Bash call can't tie up a pod long enough to stall training throughput.
    """

    timeout: int = 30
    max_timeout: int = 300
    cwd: str = ""

    def __post_init__(self) -> None:
        if self.max_timeout < self.timeout:
            raise ValueError(f"Bash max_timeout ({self.max_timeout}s) must be >= default timeout ({self.timeout}s)")


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    def _create_config(self, config_dict: dict[str, Any]) -> BashToolConfig:
        return BashToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        default_ms = self.config.timeout * 1000
        max_ms = self.config.max_timeout * 1000
        return f"""Executes a given bash command and returns its output.

Each invocation runs in a fresh subshell — directory and environment changes do NOT persist between calls. Always use absolute paths (or prefix commands with `cd /path && ...`) instead of relying on a persisted working directory.

IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience:

 - File search: Use Glob (NOT find or ls)
 - Content search: Use Grep (NOT grep or rg)
 - Read files: Use Read (NOT cat/head/tail)
 - Edit files: Use Edit (NOT sed/awk)
 - Write files: Use Write (NOT echo >/cat <<EOF)

 - If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists and is the correct location.
 - Always quote file paths that contain spaces with double quotes in your command (e.g., cd "path with spaces/file.txt")
 - Use non-interactive flags (e.g. -y for apt). Avoid interactive tools like vi, nano, or anything requiring stdin.
 - You may specify an optional timeout in milliseconds (up to {max_ms}ms). By default, your command will timeout after {default_ms}ms; raise it only for genuinely slow commands (e.g. a cold compile). Requests above {max_ms}ms are capped at {max_ms}ms.
 - When issuing multiple commands:
  - If commands are independent and can run in parallel, make multiple Bash tool calls in a single message.
  - If commands depend on each other, chain them with `&&`.
  - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail.


Only create commits when requested by the user. If unclear, ask first.

Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless explicitly requested
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless explicitly requested
- When staging files, prefer adding specific files by name rather than `git add -A` / `git add .`, which can include sensitive files (.env, credentials) or large binaries
- NEVER commit changes unless the user explicitly asks you to."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "description": {
                    "type": "string",
                    "description": "Clear, concise description of what this command does in active voice (5-10 words for simple commands; more for piped/obscure ones).",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Optional timeout in milliseconds (capped at {self.config.max_timeout * 1000})",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Set to true to run this command in the background. Use this if you don't need the result immediately.",
                },
            },
            "required": ["command"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for bash execution")

        command = self._extract_command(params)
        timeout_s = self._resolve_timeout(
            params.get("timeout") if isinstance(params, dict) else None,
            self.config.timeout,
            self.config.max_timeout,
        )

        try:
            result = env.execute(command, cwd=self.config.cwd or "", timeout=timeout_s)
        except TransportError:
            raise  # Let infra errors propagate — execute_action handles them
        except Exception as e:
            raise ToolException(f"Failed to execute command: {e}") from e

        output = result.get("output", "")
        returncode = result.get("returncode")
        reason = result.get("reason", "ok")

        if reason == "pod_timeout":
            raise ToolException(f"Command timed out in pod after {timeout_s}s:\n{output}")
        if reason == "client_timeout":
            raise ToolException(f"Command timed out (client) after {timeout_s}s:\n{output}")
        if reason == "transport_error":
            return ToolOutput(
                output=f"Error: Transport error while executing command:\n{output}",
                success=False,
                metadata={"returncode": returncode, "reason": reason},
            )

        rc = returncode if returncode is not None else -1
        return ToolOutput(
            output=output,
            success=rc == 0,
            metadata={"returncode": rc},
        )

    @staticmethod
    def _extract_command(params: Any) -> str:
        if not isinstance(params, dict):
            raise ToolException(f"Bash tool params must be a dictionary, got {type(params).__name__}")
        command = (params.get("command") or "").strip()
        if not command:
            raise ToolException("Bash tool requires a non-empty 'command' parameter")
        return command

    @staticmethod
    def _resolve_timeout(raw_ms: Any, default_s: int, max_s: int) -> int:
        """Resolve the effective per-call timeout in seconds.

        The model supplies `timeout` in milliseconds (per the function schema);
        the config is in seconds. When the model omits it (or sends something
        non-numeric), fall back to the config default. Any explicit value is
        converted ms→s and clamped to [1, max_s] so a single call can neither
        round down to 0s nor exceed the training-throughput ceiling.
        """
        if raw_ms is None:
            return default_s
        try:
            requested_s = float(raw_ms) / 1000.0
        except (TypeError, ValueError):
            return default_s
        return max(1, min(int(requested_s), max_s))
