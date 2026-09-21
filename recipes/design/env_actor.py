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
"""Ray actor owning one task environment (a pod) and the tools that reach into it.

Why the pod lives in its own actor rather than inside the agent-loop worker:

* Every tool call is a blocking exec round trip measured in seconds. Running
  them in the worker would need a thread per in-flight call anyway, and a
  hundred concurrent rollouts would put the whole Kubernetes client stack in
  one process on a GPU node.
* ``Write`` and ``Edit`` ship their payload through ``env.copy_to(local_temp,
  remote_path)``, so **the tools must run in the same process as the pod
  client**. That is why the tool registry lives here and not on the agent side.
* ``ray.kill`` gives the caller a way out when a rollout wedges on a dead pod.

The agent side never executes a tool: it receives the schemas from
:meth:`DatasetEnvActor.describe` and forwards each call here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import ray
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


KIND_OK = "ok"
KIND_TOOL_EXCEPTION = "tool_exception"
KIND_TRANSPORT_ERROR = "transport_error"
KIND_UNEXPECTED = "unexpected"
KIND_LIMITS_EXCEEDED = "limits_exceeded"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a harness profile, rejecting ones this recipe cannot honour."""
    from .webdev.tools import CONVERSATION_TOOLS

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    tools = (config.get("agent") or {}).get("tools") or []
    tool_names = [t.get("tool") for t in tools if isinstance(t, dict)]
    if not tool_names:
        raise ValueError(f"{path}: agent.tools is empty; the agent would have nothing to call.")

    rejected = sorted(set(tool_names) & CONVERSATION_TOOLS)
    if rejected:
        raise ValueError(
            f"{path}: tools {rejected} are not supported by this recipe. They need "
            "'agent' or 'model' in the tool context, and tools execute in this actor "
            "where only 'env' exists. They also fork or rewrite the conversation, while "
            "the rollout keeps a single append-only token buffer (see token_trace.py)."
        )
    return config


_PLACEMENT_ENV = (
    ("WEBDEV_NODE_SELECTOR", "node_selector", True),
    ("WEBDEV_TOLERATIONS", "tolerations", True),
    ("WEBDEV_DNAT_PROXY_IP", "dnat_proxy_ip", False),
)


def _apply_placement_overrides(env_kwargs: dict[str, Any]) -> None:
    """Overlay site-specific placement from the environment. Unset -> profile default."""
    for var, key, is_json in _PLACEMENT_ENV:
        raw = (os.environ.get(var) or "").strip()
        if not raw:
            continue
        if not is_json:
            env_kwargs[key] = raw
            continue
        try:
            env_kwargs[key] = json.loads(raw)
        except ValueError as e:
            raise ValueError(f"{var} is not valid JSON (it sets environment.{key}): {e}") from e


@ray.remote(num_cpus=1)
class DatasetEnvActor:
    """Owns the pod, the tool registry, and reward computation for one rollout."""

    def __init__(
        self,
        instance: dict[str, Any],
        instance_id: str,
        config_path: str,
        dump_dir: str | None = None,
    ):
        self.instance = instance
        self.instance_id = instance_id
        self.config = load_config(config_path)

        self.dataset_env = None
        self._infra_error: str | None = None
        self._cleanup_failures = 0

        self.tool_registry = self._build_tool_registry()

        self._dump_dir = Path(dump_dir) if dump_dir else None
        self._env_logger: logging.Logger | None = None
        if self._dump_dir is not None:
            from mimoagent.utils.log import make_file_logger

            self._dump_dir.mkdir(parents=True, exist_ok=True)
            self._env_logger = make_file_logger(f"design.env.{instance_id}", self._dump_dir / "env.log")

    def _build_tool_registry(self):
        """Build the catalogue this profile's ``agent.type`` names.

        Kept to the types this recipe ships rather than reading a global
        registry: a silent fallback to a different catalogue would hand the
        model schemas for one implementation and execute another.
        """
        from .webdev.agent import AGENT_TYPE as WEBDEV_AGENT_TYPE
        from .webdev.tools import WebdevToolRegistry

        agent_type = (self.config.get("agent") or {}).get("type")
        if agent_type != WEBDEV_AGENT_TYPE:
            raise ValueError(
                f"unsupported agent.type {agent_type!r}; this recipe ships "
                f"{WEBDEV_AGENT_TYPE!r}. The tool catalogue is chosen here, in a "
                "different process from the agent, so an unrecognised type cannot be "
                "resolved by guessing."
            )
        return WebdevToolRegistry.from_config(self.config["agent"]["tools"])


    def _log(self, msg: str) -> None:
        if self._env_logger is not None:
            self._env_logger.info(msg)
        else:
            logger.warning(msg)


    def _create(self) -> None:
        """Build the DatasetEnvironment and start the pod. Blocking; raises on failure."""
        from mimoagent.environments.utils import make_dataset_env

        from .webdev import register_webdev_env

        register_webdev_env()

        env_kwargs = dict(self.config.get("environment") or {})
        env_kwargs["labels"] = {"exp": os.getenv("EXP_NAME", "unknown"), **(env_kwargs.get("labels") or {})}
        if os.environ.get("KUBECONFIG") and "kubeconfig" not in env_kwargs:
            env_kwargs["kubeconfig"] = os.environ["KUBECONFIG"]
        _apply_placement_overrides(env_kwargs)

        self._log(f"creating environment, env_kwargs={env_kwargs}")
        self.dataset_env = make_dataset_env(self.instance, **env_kwargs)

        if self._dump_dir is not None and hasattr(self.dataset_env, "dump_dir"):
            self.dataset_env.dump_dir = str(self._dump_dir)
        grader_cfg = self.config.get("traj_grader")
        if grader_cfg and hasattr(self.dataset_env, "grader_cfg"):
            self.dataset_env.grader_cfg = dict(grader_cfg)

        self.dataset_env.setup_environment()
        self._log("environment ready")

    async def setup(self, max_retries: int = 2) -> tuple[bool, str | None]:
        """Create the environment, retrying after a full cleanup. Returns ``(ok, error)``.

        ``async`` so the actor's event loop stays responsive to a concurrent
        ``cleanup.remote()`` while a slow pod creation is in flight.
        """
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                self._log(f"setup attempt {attempt} after failure: {last_error}")
                await self.cleanup()
            try:
                await asyncio.to_thread(self._create)
                return True, None
            except Exception as e:
                last_error = f"{e}\n{traceback.format_exc()}"
                self._log(f"setup attempt {attempt} failed: {last_error}")
        await self.cleanup()
        return False, last_error

    def describe(self) -> dict[str, Any]:
        """Tool schemas plus template vars, both needed before the agent exists.

        ``template_vars`` is what fills ``{{cwd}}`` in the profile's
        ``system_template``. It comes from the environment, so it is only known
        once the pod does.
        """
        assert self.dataset_env is not None, "describe() before setup()"
        return {
            "tool_definitions": self.tool_registry.get_function_definitions(),
            "tool_names": self.tool_registry.list_tools(),
            "template_vars": self.dataset_env.get_template_vars(),
            "pod_name": getattr(self.dataset_env.env, "pod_name", ""),
            "node_name": getattr(self.dataset_env.env, "node_name", ""),
        }


    def _execute_tool_sync(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run one tool, mapping its exceptions onto tagged return values.

        The taxonomy mirrors MimoAgent's own ``DefaultAgent.execute_action`` so
        the agent side can reproduce it verbatim across the Ray boundary.
        """
        import time

        from mimoagent.agents.base import LimitsExceeded
        from mimoagent.environments import TransportError
        from mimoagent.tools import ToolException

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                tool = self.tool_registry.get(name)
                result = tool.execute(params, {"env": self.dataset_env.env})
            except LimitsExceeded as e:
                return {"kind": KIND_LIMITS_EXCEEDED, "message": str(e)}
            except TransportError as e:
                if "opening exec stream" in str(e) and attempt < max_attempts:
                    self._log(
                        f"exec stream open failed for tool '{name}' (attempt {attempt}/{max_attempts}), retrying: {e}"
                    )
                    time.sleep(2 * attempt)
                    continue
                self._infra_error = f"rollout/pod_conn_timeout: {e}"
                self._log(f"transport error during tool '{name}': {e}")
                return {"kind": KIND_TRANSPORT_ERROR, "message": str(e)}
            except ToolException as e:
                return {"kind": KIND_TOOL_EXCEPTION, "message": str(e)}
            except Exception as e:
                return {"kind": KIND_UNEXPECTED, "message": f"Unexpected error executing tool '{name}': {e}"}
            return {"kind": KIND_OK, "result": result.to_dict()}

    async def execute_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.dataset_env is not None, "execute_tool() before setup()"
        return await asyncio.to_thread(self._execute_tool_sync, name, params)

    async def execute(self, command: str, **kwargs) -> dict[str, Any]:
        """Raw pod exec. Not used by the rollout; kept for smoke scripts."""
        assert self.dataset_env is not None, "execute() before setup()"
        return await asyncio.to_thread(lambda: self.dataset_env.env.execute(command, **kwargs))


    async def calculate_reward(self, timeout: float | None = None) -> tuple[float, str, dict]:
        """Delegate to the dataset environment's grading. Never raises."""
        if self._infra_error:
            self._log(f"skipping reward, infra_error={self._infra_error}")
            return (
                0.0,
                f"infra: {self._infra_error}",
                {"error_category": "rollout/pod_conn_timeout", "infra_error": self._infra_error},
            )

        if self.dataset_env is None:
            return 0.0, "dataset env not set up", {}

        try:
            reward, test_output, extra = await asyncio.to_thread(
                lambda: self.dataset_env.calculate_reward(timeout=timeout)
            )
        except Exception as e:
            self._log(f"calculate_reward raised: {e}\n{traceback.format_exc()}")
            return 0.0, str(e), {"error_category": "reward/exception"}

        self._log(f"reward={reward}")
        self._log(f"test_output={test_output}")
        return reward, test_output, extra


    def get_stats(self) -> dict[str, Any]:
        return {"cleanup_failures": self._cleanup_failures, "infra_error": self._infra_error}

    async def cleanup(self) -> None:
        if self.dataset_env is not None:
            try:
                await asyncio.to_thread(self.dataset_env.cleanup)
            except Exception as e:
                self._cleanup_failures += 1
                self._log(f"cleanup error: {e}")
            self.dataset_env = None
        if self._env_logger is not None:
            for handler in list(self._env_logger.handlers):
                try:
                    handler.close()
                except Exception:
                    pass
            self._env_logger = None
