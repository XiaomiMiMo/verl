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
"""Ray-actor fakes for the web-dev rollout test.

Their own module, imported by its canonical ``tests.recipes.design.fakes`` path, because
Ray serialises an actor class by its ``__module__`` and the worker has to import it. pytest
imports test files under a rootdir-derived name that Ray workers cannot resolve, so an actor
class defined inside a test file dies with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import ray
import yaml

from recipes.design.env_actor import KIND_OK

RECORDER_NAME = "design-test-recorder"


@ray.remote(num_cpus=0)
class Recorder:
    """Named side channel: the env stub runs in its own process, so the test reads it here."""

    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def record(self, kind, payload=None):
        self.events.append((kind, payload))

    def events_of(self, kind):
        return [payload for k, payload in self.events if k == kind]

    def reset(self):
        self.events.clear()


class StubEnvImpl:
    """Same method surface as ``DatasetEnvActor``, with no pod behind it."""

    def __init__(self, instance, instance_id, config_path, dump_dir=None):
        from recipes.design.webdev.tools import WebdevToolRegistry

        self.instance_id = instance_id
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # The real catalogue, not a stub one: the prompt the policy sees must carry the
        # production tool schemas, which is half of what this test is checking.
        self.registry = WebdevToolRegistry.from_config(config["agent"]["tools"])
        self.recorder = ray.get_actor(RECORDER_NAME)

    async def setup(self, max_retries: int = 2):
        return True, None

    def describe(self):
        return {
            "tool_definitions": self.registry.get_function_definitions(),
            "tool_names": self.registry.list_tools(),
            "template_vars": {"cwd": "/workspace"},
            "pod_name": "stub-pod",
            "node_name": "stub-node",
        }

    async def execute_tool(self, name, params):
        # Awaited, not fire-and-forget: Ray gives no cross-actor ordering guarantee, so an
        # un-awaited record can still be in flight when the test inspects the recorder.
        await self.recorder.record.remote("tool", {"name": name, "params": params})
        return {"kind": KIND_OK, "result": {"output": f"[{name}] ok", "success": True, "metadata": {}}}

    async def calculate_reward(self, timeout=None):
        await self.recorder.record.remote("reward", {"timeout": timeout})
        # Shaped like the training grader's return: the per-rollout reward is a PLACEHOLDER
        # and the real number is written back by the driver once a whole group can be judged.
        return (
            0.0,
            "",
            {
                "webdev_group_pending": True,
                "webdev_group_query_score": 0.6,
                "webdev_group_runtime_factor": 1.0,
                "webdev_group_runtime_why": "rendered",
                "webdev_function_score": 0.0,
            },
        )

    async def cleanup(self):
        await self.recorder.record.remote("cleanup", None)


class FailingEnvImpl(StubEnvImpl):
    async def setup(self, max_retries: int = 2):
        return False, "pod stuck in Pending"


StubEnv = ray.remote(num_cpus=0)(StubEnvImpl)
FailingEnv = ray.remote(num_cpus=0)(FailingEnvImpl)
