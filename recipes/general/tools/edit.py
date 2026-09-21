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
"""Edit tool: exact-match string replacement in an existing file.

Implementation contract: behaves like ``content.replace(old_string, new_string, 1)``
on the file bytes (or replace-all when ``replace_all`` is set), but only after
verifying ``old_string`` occurs — and, unless replacing all, occurs exactly once.

Container-side dependencies are bash + awk + the POSIX basics (tr, wc) only —
no python in the image. The old/new strings travel via ``env.copy_to`` (tar
stream), never on a command line — so there is no ARG_MAX ceiling and no
shell-quoting hazard.

awk reads the strings and the target file as single whole records via
``RS="\\1"`` (the SOH control byte). ``RS="\\0"`` would be the natural choice
but is NOT portable: busybox awk (alpine images) stores strings as C strings,
so ``"\\0"`` collapses to ``""`` which is awk *paragraph mode* — files with
blank lines get split and the edit fails. ``\\1`` is a real byte in every awk
(verified against gawk, mawk, busybox awk). Files containing NUL bytes are
rejected as binary by a bash-level ``tr -d '\\0' | wc -c`` check *before* awk
runs (a C-string awk would silently truncate the body at the first NUL);
files containing ``\\1`` itself are caught by awk's second-record probe.

Total environment round-trips per edit: one ``copy_to`` + one ``execute``.
"""

import os
import shlex
import tempfile
import uuid
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput

_AWK_PROG = r"""
BEGIN {
    RS = "\1"; ORS = ""
    if ((getline old < OLDF) < 0) exit 6
    if ((getline x < OLDF) > 0) exit 5
    close(OLDF)
    if ((getline new < NEWF) <= 0) new = ""
    close(NEWF)
    if ((getline body < TARGET) < 0) exit 6
    if ((getline extra < TARGET) > 0) exit 5
    close(TARGET)

    count = 0; s = body
    while ((i = index(s, old)) > 0) { count++; s = substr(s, i + length(old)) }
    if (count == 0) exit 3
    if (count > 1 && REPLACE_ALL != "1") { print count; exit 4 }

    out = ""; s = body
    if (REPLACE_ALL == "1") {
        while ((i = index(s, old)) > 0) {
            out = out substr(s, 1, i - 1) new
            s = substr(s, i + length(old))
        }
        out = out s
    } else {
        i = index(body, old)
        out = substr(body, 1, i - 1) new substr(body, i + length(old))
    }
    print out > OUT
    close(OUT)
    exit 0
}
"""


class EditTool(BaseTool):
    """Replace an exact (possibly multi-line) string in an existing file."""

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

Notes:
- `file_path` must be absolute (start with `/`) and must point to an existing file.
- Pass an empty string for `new_string` to delete `old_string`.
- Ensure the edit results in idiomatic, correct code; do not leave the file in a broken state."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences of old_string (default false)",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for edit execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        file_path = params.get("file_path")
        if not file_path:
            raise ToolException("Missing required parameter: file_path")
        if not file_path.startswith("/"):
            raise ToolException(f"file_path must be absolute (start with /), got: {file_path}")

        old_string = params.get("old_string")
        if old_string is None:
            raise ToolException("Missing required parameter: old_string")
        new_string = params.get("new_string")
        if new_string is None:
            raise ToolException("Missing required parameter: new_string")
        replace_all = bool(params.get("replace_all", False))

        if old_string == "":
            return ToolOutput(
                output="Error: old_string must not be empty. To create or overwrite a file, use the Write tool.",
                success=False,
            )
        if old_string == new_string:
            return ToolOutput(
                output="Error: old_string and new_string are identical — nothing to change.",
                success=False,
            )
        if any(c in old_string or c in new_string for c in ("\0", "\x01")):
            return ToolOutput(
                output="Error: old_string/new_string must not contain NUL or \\x01 control bytes.", success=False
            )

        remote_dir = f"/tmp/mimo_edit_{uuid.uuid4().hex[:12]}"
        self._ship_strings(env, remote_dir, old_string, new_string)

        q_target = shlex.quote(file_path)
        q_dir = shlex.quote(remote_dir)
        replace_all_flag = "1" if replace_all else "0"
        script = f"""
target={q_target}
d={q_dir}
trap 'rm -rf "$d"' EXIT
if [ ! -f "$target" ]; then echo "MIMO_EDIT:NOFILE"; exit 0; fi
if [ ! -r "$target" ] || [ ! -w "$target" ]; then echo "MIMO_EDIT:NOACCESS"; exit 0; fi
if [ "$(wc -c < "$target")" -ne "$(tr -d '\\0' < "$target" | wc -c)" ]; then
  echo "MIMO_EDIT:BINARY"; exit 0
fi
count=$(awk -v OLDF="$d/old" -v NEWF="$d/new" -v TARGET="$target" -v OUT="$d/out" -v REPLACE_ALL="{replace_all_flag}" {shlex.quote(_AWK_PROG)})
rc=$?
case $rc in
  0) ;;
  3) echo "MIMO_EDIT:NOTFOUND"; exit 0 ;;
  4) echo "MIMO_EDIT:MULTIPLE:$count"; exit 0 ;;
  5) echo "MIMO_EDIT:BINARY"; exit 0 ;;
  *) echo "MIMO_EDIT:AWKFAIL:$rc"; exit 0 ;;
esac
echo "Replacement successful. Showing difference:"
if command -v git >/dev/null 2>&1; then
  git diff --no-index -- "$target" "$d/out" 2>/dev/null | head -50
elif command -v diff >/dev/null 2>&1; then
  diff -u "$target" "$d/out" 2>/dev/null | head -50
fi
cat "$d/out" > "$target" || {{ echo "MIMO_EDIT:WRITEFAIL"; exit 0; }}
echo "MIMO_EDIT:OK"
"""
        result = env.execute(script)
        output = result.get("output", "")

        if "MIMO_EDIT:OK" in output:
            body = output.replace("MIMO_EDIT:OK", "").strip()
            return ToolOutput(output=body or "String replaced successfully.")
        if "MIMO_EDIT:NOFILE" in output:
            return ToolOutput(output=f"Error: File not found: {file_path}", success=False)
        if "MIMO_EDIT:NOACCESS" in output:
            return ToolOutput(output=f"Error: File not readable/writable: {file_path}", success=False)
        if "MIMO_EDIT:NOTFOUND" in output:
            return ToolOutput(
                output="Error: The exact string was not found in the file. Make sure the old_string matches exactly including whitespace and newlines.",
                success=False,
            )
        if "MIMO_EDIT:MULTIPLE:" in output:
            count = output.split("MIMO_EDIT:MULTIPLE:")[1].split()[0]
            return ToolOutput(
                output=f"Error: String found {count} times, must be unique. Please include more context to make the string unique, or pass replace_all=true to change every occurrence.",
                success=False,
            )
        if "MIMO_EDIT:BINARY" in output:
            return ToolOutput(
                output=f"Error: {file_path} appears to be a binary file (contains NUL/control bytes); edit only supports text files.",
                success=False,
            )
        return ToolOutput(output=f"Error during replacement: {output.strip()[:2000]}", success=False)

    @staticmethod
    def _ship_strings(env, remote_dir: str, old_str: str, new_str: str) -> None:
        """Materialise old/new as files and copy them into the environment in
        one tar-stream transfer (a directory with two entries)."""
        with tempfile.TemporaryDirectory(prefix="mimo_edit_") as local_dir:
            oldp = os.path.join(local_dir, "old")
            newp = os.path.join(local_dir, "new")
            with open(oldp, "w", encoding="utf-8", newline="") as f:
                f.write(old_str)
            with open(newp, "w", encoding="utf-8", newline="") as f:
                f.write(new_str)
            try:
                env.copy_to(local_dir, remote_dir)
            except Exception as e:
                raise ToolException(f"Failed to transfer edit strings to environment: {e}") from e
