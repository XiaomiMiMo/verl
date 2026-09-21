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
"""Write reproducibility metadata for one training run, for any recipe here.

The components differ per direction -- the code line pins two submodules, the design line
pins two different ones -- so they arrive as repeatable ``--component NAME=PATH`` pairs
rather than as one flag each. A direction that forgets to pass one records nothing about it,
which is visible in provenance.json; a hardcoded flag list would instead have made the
design line's grading service silently unrecorded.

What it records, and why each part earns its place: the commit and dirty state of every
component (a run from an uncommitted tree is not reproducible, and that has to be on the
record rather than in someone's memory), the diff (so "dirty" is actionable), untracked
files that look like code or config (the usual way a change escapes both), the hash of the
config artifact the run was given, and the exact command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MAX_UNTRACKED_SNAPSHOT_BYTES = 1024 * 1024
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "ckpt",
    "eval_checkpoints",
    "logs",
    "models",
    "outputs",
    "wandb",
}
SENSITIVE_NAMES = {".env", ".netrc", "credentials", "id_rsa", "id_ed25519"}
SNAPSHOT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".jinja",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SNAPSHOT_NAMES = {"Dockerfile", "Makefile"}


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _repo_state(repo: Path) -> dict[str, object]:
    state: dict[str, object] = {"path": str(repo), "exists": repo.is_dir()}
    if not repo.is_dir():
        return state
    state["commit"] = _run_git(repo, "rev-parse", "HEAD")
    state["branch"] = _run_git(repo, "branch", "--show-current")
    state["status"] = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all") or ""
    state["remote"] = _run_git(repo, "remote", "get-url", "origin")
    return state


def _write_diff(repo: Path, destination: Path) -> None:
    diff = _run_git(repo, "diff", "HEAD") or ""
    cached = _run_git(repo, "diff", "--cached") or ""
    destination.write_text(diff + ("\n" if diff and cached else "") + cached)


def _snapshot_untracked(repo: Path, destination: Path) -> list[dict[str, object]]:
    output = _run_git(repo, "ls-files", "--others", "--exclude-standard", "-z") or ""
    records: list[dict[str, object]] = []
    destination.mkdir(parents=True, exist_ok=True)
    for relative_name in filter(None, output.split("\0")):
        relative_path = Path(relative_name)
        source = repo / relative_path
        record: dict[str, object] = {"path": relative_name}
        if not source.is_file():
            record["captured"] = False
            record["reason"] = "not a regular file"
        elif any(part in EXCLUDED_PARTS for part in relative_path.parts):
            record["captured"] = False
            record["reason"] = "generated artifact path"
        elif source.name in SENSITIVE_NAMES:
            record["captured"] = False
            record["reason"] = "potential credential file"
        else:
            data = source.read_bytes()
            record["size"] = len(data)
            record["sha256"] = hashlib.sha256(data).hexdigest()
            is_source = source.suffix.lower() in SNAPSHOT_SUFFIXES or source.name in SNAPSHOT_NAMES
            if not is_source:
                record["captured"] = False
                record["reason"] = "non-source file"
            elif len(data) <= MAX_UNTRACKED_SNAPSHOT_BYTES:
                target = destination / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                record["captured"] = True
            else:
                record["captured"] = False
                record["reason"] = "file exceeds snapshot size limit"
        records.append(record)
    return records


def parse_component(pair: str) -> tuple[str, Path]:
    """``NAME=PATH``. Rejects a bare path rather than inventing a name for it.

    The name becomes a key in provenance.json and a filename in the snapshot, so guessing it
    from the directory would make two runs of the same component disagree on what to call it.
    """
    name, sep, path = pair.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {pair!r}")
    return name.strip(), Path(path.strip()).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verl-repo", type=Path, required=True)
    parser.add_argument(
        "--component",
        type=parse_component,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a pinned dependency to record; repeatable",
    )
    parser.add_argument("--config-artifact", type=Path, default=None)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.run_dir / "code_snapshot"
    snapshot_dir.mkdir(exist_ok=True)

    components: list[tuple[str, Path]] = [("verl", args.verl_repo), *args.component]
    repositories: dict[str, object] = {}
    untracked: dict[str, object] = {}
    for name, path in components:
        repositories[name] = _repo_state(path)
        _write_diff(path, snapshot_dir / f"{name}.diff")
        untracked[name] = _snapshot_untracked(path, snapshot_dir / "untracked" / name)

    config_hash = None
    if args.config_artifact is not None and args.config_artifact.is_file():
        config_hash = hashlib.sha256(args.config_artifact.read_bytes()).hexdigest()

    selected_env = {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "ENABLE_METRIC",
            "METRIC_PORT",
            "SGLANG_ENABLE_SPEC_V2",
            "RAY_ADDRESS",
            "PYTHONPATH",
        )
        if key in os.environ
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "repositories": repositories,
        "untracked_files": untracked,
        "config_artifact": {
            "path": str(args.config_artifact) if args.config_artifact else None,
            "sha256": config_hash,
        },
        "resolved_config": str(args.resolved_config),
        "environment": selected_env,
        "command": args.command,
    }
    (args.run_dir / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.run_dir / "command.json").write_text(json.dumps(args.command, indent=2) + "\n")


if __name__ == "__main__":
    main()
