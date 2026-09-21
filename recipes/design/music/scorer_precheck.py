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
"""Probe the music scorer on the Ray workers that will actually run it.

    python3 scorer_precheck.py --address auto --tasks 8

Exits non-zero with the reason if any worker cannot produce a usable score.

Why this runs on workers and not in the launcher: the scorer shells out to
``abc2midi`` by relative name, and Ray workers inherit the *raylet's*
environment, not the launcher shell's. A driver-side probe therefore passes on a
cluster where every worker is missing the binary -- which is exactly the
configuration that produced a full run of zero rewards, because a zero-variance
GRPO group yields zero advantage and nothing downstream looks broken.

Why the assertion is ``0 < score < 1`` rather than "it did not raise": both ends
are distinct failures that a bare call cannot tell apart from a policy that
writes bad music.

    score == 0   the ABC was not extracted, or the reject gate fired
    score == 1   the percentile baseline did not load, so every band reads as met

The fixture is a deliberately mediocre four-bar score: a healthy toolchain puts
it around 0.2, comfortably inside the open interval.
"""

from __future__ import annotations

import argparse
import os
import sys

FIXTURE_ABC = "X:1\nT:t\nM:4/4\nL:1/8\nK:C\nCDEF|GABc|c4|z4|"

_REMEDY = (
    "\n  Install the `abcmidi` package in the image, or set ABC2MIDI_BIN and let the\n"
    "  launcher forward PATH through ray_init.runtime_env.env_vars. An export in the\n"
    "  launcher shell reaches the driver only: workers inherit the raylet's PATH, and\n"
    "  the platform starts the raylet before the launcher runs."
)


def _probe() -> dict:
    """Score the fixture in this process and report where it ran."""
    import os
    import shutil
    import socket

    from recipes.design.music import scorer

    fenced = scorer.compute_score("music", f"```abc\n{FIXTURE_ABC}\n```")
    bare = scorer.compute_score("music", f"Write the final score.\n\n{FIXTURE_ABC}")
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "abc2midi": shutil.which("abc2midi"),
        "fenced": fenced,
        "bare": bare,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="auto")
    ap.add_argument(
        "--tasks",
        type=int,
        default=8,
        help="probe tasks to fan out; raise it to cover more of the cluster",
    )
    args = ap.parse_args()

    import ray

    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    ray.init(address=args.address, ignore_reinit_error=True, log_to_driver=False)
    try:
        remote_probe = ray.remote(num_cpus=1)(_probe)
        try:
            results = ray.get([remote_probe.remote() for _ in range(args.tasks)])
        except Exception as exc:
            root = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
            print(f"[music] FATAL: the scorer failed on a worker.\n  {root}", file=sys.stderr)
            return 1
    finally:
        ray.shutdown()

    failures: list[str] = []
    hosts = set()
    for result in results:
        hosts.add(result["host"])
        where = f"{result['host']}/{result['pid']}"
        for path_name in ("fenced", "bare"):
            score = result[path_name]
            if not 0.0 < score < 1.0:
                failures.append(
                    f"{where}: {path_name} score {score} is outside (0, 1); abc2midi={result['abc2midi']!r}"
                )

    print(
        f"[music] probed {len(results)} task(s) across {len(hosts)} host(s); "
        f"fenced={results[0]['fenced']:.4f} bare={results[0]['bare']:.4f} "
        f"abc2midi={results[0]['abc2midi']}",
        file=sys.stderr,
    )

    if failures:
        print("[music] FATAL: the scorer is not usable on every worker:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print(
            "\n  A score of 0.0 with abc2midi=None means the binary is not on the worker's PATH." + _REMEDY,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
