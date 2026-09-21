#!/usr/bin/env python3
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
"""Convert build-a-site instances into the parquet this arm consumes.

Input is one build task per row with the instance dict somewhere under
``extra_info`` -- three shapes are accepted, because the upstream layout has moved::

    {category, cwd, docker_image, problem_statement, task_id}

Output is the row shape ``RLHFDataset`` expects, with ``dataset_type="webdev"`` injected
into the instance so the environment actor routes it to this arm's environment rather than
to a code-task one.

    python3 -m recipes.design.webdev.build_parquet \\
        --input  /path/to/webdev_tasks.parquet \\
        --output /path/to/webdev_train.parquet

    python3 -m recipes.design.webdev.build_parquet --input ... --output ... --limit 4

Validation checks exactly what something downstream dereferences, and nothing else:

* ``task_id`` names the row in every log and dump.
* ``problem_statement`` is BOTH the agent's task and the judge's query, so an empty one
  produces a rollout that builds nothing and a judge that grades against nothing.
* ``docker_image`` is the pod the site is built and screenshotted in. It has to ship
  playwright and chromium; an image without them makes every reward a drop rather than a
  low score, which is harder to notice because the run keeps going.

A rejected row is written to ``<output>.rejected.jsonl`` with its reason rather than
dropped silently -- a build set that quietly lost a third of its rows still trains, just on
a different task distribution than the one that was asked for.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

AGENT_NAME = "webdev"
DATASET_TYPE = "webdev"


def extract_instance(row: dict[str, Any]) -> dict[str, Any] | None:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict):
        interaction_kwargs = extra_info.get("interaction_kwargs") or {}
        if isinstance(interaction_kwargs, dict) and isinstance(interaction_kwargs.get("instance"), dict):
            return interaction_kwargs["instance"]
        if isinstance(extra_info.get("instance"), dict):
            return extra_info["instance"]
    if isinstance(row.get("instance"), dict):
        return row["instance"]
    return None


def validate(instance: dict[str, Any]) -> str | None:
    """Return an error string, or None if the row is trainable."""
    if not (instance.get("task_id") or instance.get("instance_id")):
        return "missing/empty task_id"
    if not instance.get("problem_statement"):
        return "missing/empty problem_statement (the agent's task AND the judge's query)"
    image = instance.get("docker_image") or ""
    if not image:
        return "missing/empty docker_image"
    if "webdev" not in image:
        return f"docker_image {image!r} does not look like a web-dev image (needs playwright + chromium)"
    return None


def convert(rows: list[dict[str, Any]], limit: int | None) -> tuple[list[dict], list[dict], collections.Counter]:
    converted: list[dict] = []
    rejected: list[dict] = []
    stats: collections.Counter = collections.Counter()

    for source_index, row in enumerate(rows):
        instance = extract_instance(row)
        if instance is None:
            rejected.append({"source_index": source_index, "error": "no instance dict found in row"})
            stats["rejected"] += 1
            continue

        error = validate(instance)
        if error is not None:
            rejected.append(
                {
                    "source_index": source_index,
                    "task_id": instance.get("task_id"),
                    "docker_image": instance.get("docker_image"),
                    "error": error,
                }
            )
            stats["rejected"] += 1
            continue

        instance = dict(instance)
        instance["dataset_type"] = DATASET_TYPE
        instance.setdefault("instance_id", instance.get("task_id"))
        instance.setdefault("cwd", "/workspace")

        index = len(converted)
        converted.append(
            {
                "data_source": row.get("data_source") or "blackbox/webdev",
                "ability": row.get("ability") or "webdev",
                "agent_name": AGENT_NAME,
                "prompt": [{"role": "user", "content": instance["problem_statement"]}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "index": index,
                    "instance_id": instance["instance_id"],
                    "dataset_type": DATASET_TYPE,
                    "instance_json": json.dumps(instance, ensure_ascii=False),
                },
            }
        )
        stats["converted"] += 1
        if limit is not None and len(converted) >= limit:
            break

    return converted, rejected, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = pq.read_table(args.input).to_pylist()
    converted, rejected, stats = convert(rows, args.limit)

    if not converted:
        print(f"nothing converted; stats={dict(stats)}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df = pd.DataFrame(converted)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), args.output)
    print(f"wrote {len(converted)} rows -> {args.output}")

    if rejected:
        rejected_path = f"{args.output}.rejected.jsonl"
        with open(rejected_path, "w") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"rejected {len(rejected)} rows -> {rejected_path}", file=sys.stderr)

    print(f"stats: {dict(stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
