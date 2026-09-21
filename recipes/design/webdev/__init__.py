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
"""The web-dev arm: build a static site in a pod, graded by an external service.

Two things have to be registered into MimoAgent before a rollout can start, and
both are additive dict inserts rather than edits to the vendored package:

* ``webdev-agent`` -- MimoAgent's default loop over this arm's tool catalogue.
  See ``agent.py`` for why this is not ``cc-agent``.
* ``webdev`` as a dataset type, so ``make_dataset_env`` routes a row here
  instead of to a SWE environment.

Both are called lazily from the environment actor, so a run that never touches
this arm never imports it.

What the reward is, and is not: the per-rollout call returns a placeholder 0.0
plus a pending marker, and the real number is written back later from the driver
once a whole GRPO group can be judged together. A single rollout's reward is
therefore meaningless in isolation -- the arm's signal is group-relative, and
the only absolute number comes from the separate evaluation path.
"""

from __future__ import annotations

from .agent import AGENT_TYPE, WebdevAgent, register_webdev_agent

DATASET_TYPE = "webdev"


def register_webdev_env() -> None:
    """Make ``dataset_type: webdev`` routable. Idempotent and additive.

    ``DATASET_REGISTRY`` is a plain module-level dict and ``detect_dataset_type``
    honours a declared type that is already in it, so this needs no change to
    the vendored package. ``setdefault`` rather than assignment: a caller who
    registered their own web-dev environment keeps it.
    """
    from mimoagent.environments.datasets import DATASET_REGISTRY

    from .environment import WebdevEnvironment

    DATASET_REGISTRY.setdefault(DATASET_TYPE, WebdevEnvironment)


__all__ = [
    "AGENT_TYPE",
    "DATASET_TYPE",
    "WebdevAgent",
    "register_webdev_agent",
    "register_webdev_env",
]
