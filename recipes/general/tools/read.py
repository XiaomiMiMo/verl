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
"""Read tool: view a file's contents or list a directory."""

from dataclasses import dataclass
from typing import Any

from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput


@dataclass
class ReadToolConfig(ToolConfig):
    max_view_lines: int = 500  # Default lines to show when `limit` is not provided
    max_depth: int = 2  # Maximum depth for directory listing
    truncate_message: str = "<response clipped>"


class ReadTool(BaseTool):
    """View a file with line numbers, or list a directory as a shallow tree."""

    def _create_config(self, config_dict: dict[str, Any]) -> ReadToolConfig:
        return ReadToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return """Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 500 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Results are returned using `cat -n`-like format, with line numbers starting at 1
- This tool can only read files, not directories. If `file_path` is a directory, the output is a shallow tree listing.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-indexed). Only provide if the file is too large to read at once.",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["file_path"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for read execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        file_path = params.get("file_path")
        if not file_path:
            raise ToolException("Missing required parameter: file_path")
        if not file_path.startswith("/"):
            raise ToolException(f"file_path must be absolute (start with /), got: {file_path}")

        check_cmd = f'''
if [ -f "{file_path}" ]; then
    echo "FILE"
elif [ -d "{file_path}" ]; then
    echo "DIR"
else
    echo "NOTFOUND"
fi
        '''
        path_type = env.execute(check_cmd).get("output", "").strip()

        if path_type == "NOTFOUND":
            return ToolOutput(output=f"Error: Path does not exist: {file_path}", success=False)

        if path_type == "DIR":
            list_cmd = rf'''
echo "{file_path}/"
find "{file_path}" -maxdepth {self.config.max_depth} -not -path "{file_path}" -not -path "*/.*" | sort | sed 's|{file_path}/||' | awk '{{
    depth = gsub(/\//, "/")
    for(i=0; i<depth; i++) printf("  ")
    split($0, parts, "/")
    print parts[length(parts)]
}}'
            '''
            output = env.execute(list_cmd).get("output", "").strip()
            return ToolOutput(output=output, success=True)

        if path_type == "FILE":
            readable_cmd = f'[ -r "{file_path}" ] && echo "OK" || echo "NO"'
            if env.execute(readable_cmd).get("output", "").strip() != "OK":
                return ToolOutput(output=f"Error: File not readable: {file_path}", success=False)

            raw_offset = params.get("offset")
            raw_limit = params.get("limit")

            try:
                offset = int(raw_offset) if raw_offset is not None else 1
                if offset < 1:
                    offset = 1
                limit = int(raw_limit) if raw_limit is not None else self.config.max_view_lines
                if limit <= 0:
                    raise ValueError("limit must be positive")
            except (ValueError, TypeError):
                return ToolOutput(
                    output="Error: offset must be an integer >= 1 and limit must be a positive integer",
                    success=False,
                )

            partial_view = raw_offset is not None or raw_limit is not None

            if partial_view:
                cmd = f'tail -n +{offset} "{file_path}" | head -n {limit} | nl -ba -v{offset}'
            else:
                cmd = f'cat "{file_path}" | nl -ba | head -n {limit}'

            output = env.execute(cmd).get("output", "")

            total_lines_cmd = f'wc -l < "{file_path}"'
            total_lines = int(env.execute(total_lines_cmd).get("output", "0").strip() or 0)
            last_line_shown = offset + limit - 1 if partial_view else limit
            if total_lines > last_line_shown:
                output += f"\n{self.config.truncate_message}"

            return ToolOutput(output=output, success=True)

        return ToolOutput(output="Error: cannot handle this path", success=False)
