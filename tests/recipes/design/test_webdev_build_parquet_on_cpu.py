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
"""Round-trip the web-dev parquet builder, and pin the row shape it has to produce.

The row shape is a contract with four readers that fail differently when it is wrong: the
launcher's harness check reads ``agent_name``, the dataset registry reads
``instance.dataset_type``, the agent loop reads ``extra_info.instance_json``, and the grader
reads ``problem_statement`` as its query. Only the first of those fails loudly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "recipes/design/config/webdev_agent_loop.yaml"
PROFILE = REPO / "config/agent/design/webdev.yaml"

# The shape the upstream task set arrives in.
UPSTREAM = [
    {
        "data_source": "blackbox/webdev",
        "ability": "webdev",
        "extra_info": {
            "interaction_kwargs": {
                "instance": {
                    "task_id": "study-bites-1",
                    "category": "website",
                    "docker_image": "example.invalid/webdev-rl:v2",
                    "problem_statement": "Build a static site for a tutoring cafe.",
                }
            }
        },
    },
    {
        # Second accepted shape: the instance directly under extra_info.
        "extra_info": {
            "instance": {
                "task_id": "gallery-2",
                "docker_image": "example.invalid/webdev-rl:v2",
                "problem_statement": "Build a photo gallery.",
            }
        }
    },
]


def _build(tmp_path, rows, *extra_args):
    src = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    pq.write_table(pa.Table.from_pylist(rows), src)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "recipes.design.webdev.build_parquet",
            "--input",
            str(src),
            "--output",
            str(out),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return proc, out


def test_the_round_trip_produces_a_trainable_row(tmp_path):
    proc, out = _build(tmp_path, UPSTREAM)
    assert proc.returncode == 0, proc.stderr
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 2

    row = rows[0]
    assert set(row) == {"data_source", "ability", "agent_name", "prompt", "reward_model", "extra_info"}
    # The launcher's preflight compares this against the registry's harness names.
    assert row["agent_name"] == OmegaConf.load(REGISTRY)[0].name

    info = row["extra_info"]
    assert set(info) == {"index", "instance_id", "dataset_type", "instance_json"}
    assert info["index"] == 0 and rows[1]["extra_info"]["index"] == 1

    instance = json.loads(info["instance_json"])
    # dataset_type is what routes the row to this arm's environment rather than a code one.
    assert instance["dataset_type"] == "webdev" == info["dataset_type"]
    assert instance["instance_id"] == "study-bites-1", "defaulted from task_id"
    # Has to agree with the profile, or the prompt tells the policy to deliver somewhere the
    # grader does not look.
    assert instance["cwd"] == OmegaConf.load(PROFILE).environment.cwd


def test_instance_json_is_a_string_not_a_nested_struct(tmp_path):
    """A dict column becomes a fixed Arrow struct, which unions the key sets of differing
    instances and reshapes nested fields. The agent loop therefore reads a JSON string."""
    _, out = _build(tmp_path, UPSTREAM)
    schema = pq.read_schema(out)
    field = schema.field("extra_info")
    instance_json = field.type.field("instance_json")
    assert pa.types.is_string(instance_json.type), instance_json.type


@pytest.mark.parametrize(
    "broken,reason",
    [
        ({"task_id": "x", "docker_image": "example.invalid/webdev:v1"}, "problem_statement"),
        ({"problem_statement": "build", "docker_image": "example.invalid/webdev:v1"}, "task_id"),
        ({"task_id": "x", "problem_statement": "build"}, "docker_image"),
        # A pod with no browser makes every reward a DROP, not a low score, so the run keeps
        # going and looks merely unlucky.
        ({"task_id": "x", "problem_statement": "build", "docker_image": "ubuntu:22.04"}, "playwright"),
    ],
)
def test_a_row_something_downstream_dereferences_is_rejected_with_its_reason(tmp_path, broken, reason):
    proc, out = _build(tmp_path, [{"extra_info": {"instance": broken}}, *UPSTREAM])
    assert proc.returncode == 0, proc.stderr
    rejected = [json.loads(line) for line in Path(f"{out}.rejected.jsonl").read_text().splitlines()]
    assert len(rejected) == 1
    assert reason in rejected[0]["error"]
    # The good rows still convert, and reindex from 0 rather than keeping source positions.
    assert [r["extra_info"]["index"] for r in pq.read_table(out).to_pylist()] == [0, 1]


def test_a_row_with_no_instance_at_all_is_reported_not_crashed(tmp_path):
    # Appended rather than prepended: Arrow infers the parquet schema from the leading rows,
    # so a first row without `extra_info` drops the column for everything after it -- a
    # fixture artifact that would look like a builder bug.
    proc, out = _build(tmp_path, [*UPSTREAM, {"data_source": "x"}])
    assert proc.returncode == 0, proc.stderr
    rejected = Path(f"{out}.rejected.jsonl").read_text()
    assert "no instance dict found" in rejected


def test_nothing_convertible_is_a_failure_not_an_empty_parquet(tmp_path):
    """An empty training set would otherwise start a run that trains on nothing."""
    proc, out = _build(tmp_path, [{"data_source": "x"}])
    assert proc.returncode == 1
    assert not out.exists()


def test_limit_takes_a_smoke_subset(tmp_path):
    proc, out = _build(tmp_path, UPSTREAM, "--limit", "1")
    assert proc.returncode == 0, proc.stderr
    assert len(pq.read_table(out).to_pylist()) == 1


def test_the_output_matches_the_reference_runs_column_set():
    """Pinned against the shape the reference run's own parquet had.

    Hardcoded rather than read from that file: it lives in a colleague's checkout, so a test
    that depends on it would break when that directory moves -- which it already did once.
    """
    reference_columns = {"data_source", "ability", "agent_name", "prompt", "reward_model", "extra_info"}
    reference_extra_info = {"dataset_type", "index", "instance_id", "instance_json"}
    reference_instance = {
        "category",
        "cwd",
        "dataset_type",
        "docker_image",
        "instance_id",
        "problem_statement",
        "task_id",
    }

    from recipes.design.webdev.build_parquet import convert

    converted, _, _ = convert(UPSTREAM, None)
    assert set(converted[0]) == reference_columns
    assert set(converted[0]["extra_info"]) == reference_extra_info
    assert set(json.loads(converted[0]["extra_info"]["instance_json"])) == reference_instance
