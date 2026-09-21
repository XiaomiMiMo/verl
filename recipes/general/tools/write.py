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
"""Write tool: create or overwrite a file with the given contents.

The contents travel via ``env.copy_to`` (tar stream) — never on a command
line — so arbitrarily large files work and no shell quoting is involved.
"""

import os
import tempfile
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput


class WriteTool(BaseTool):
    """Create or overwrite a file with the specified contents."""

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Parent directories are created as needed.
- `file_path` must be absolute (start with `/`)."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for write execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        file_path = params.get("file_path")
        if not file_path:
            raise ToolException("Missing required parameter: file_path")
        if not file_path.startswith("/"):
            raise ToolException(f"file_path must be absolute (start with /), got: {file_path}")

        content = params.get("content")
        if content is None:
            raise ToolException("Missing required parameter: content")

        existed_before = (
            env.execute(f'[ -f "{file_path}" ] && echo "EXISTS" || echo "OK"').get("output", "").strip() == "EXISTS"
        )

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False, suffix=".txt") as f:
            f.write(content)
            local_path = f.name
        try:
            env.copy_to(local_path, file_path)
        except Exception as e:
            raise ToolException(f"Failed to write file to environment: {e}") from e
        finally:
            os.unlink(local_path)

        action = "overwritten" if existed_before else "created"
        verify_cmd = f'''
if [ -f "{file_path}" ]; then
    lines=$(wc -l < "{file_path}")
    echo "File {action} successfully: {file_path} ($lines lines)"
    echo "First few lines:"
    nl -ba "{file_path}" | head -10
else
    echo "Failed to write file"
fi
'''
        output = env.execute(verify_cmd).get("output", "").strip()
        success = "Failed to write file" not in output
        return ToolOutput(output=output, success=success)
