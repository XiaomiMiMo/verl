# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Ray actor owning one mimoagent ``DatasetEnvironment`` (k8s pod) plus its tools.

Stripped copy of ``recipes/general/env_actor.py`` for the ARVO arm:
no MCP, no sidecar, no Claude-Code tools, no answer.md, no model_patch dump.
Adds ``_UserRestrictedEnv`` to enforce ``exec_user`` from the harness config.
"""

from __future__ import annotations

import asyncio
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
REWARD_TESTBED_CORRUPTED = "reward/testbed_corrupted"


def load_mimoagent_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    tools = (config.get("agent") or {}).get("tools") or []
    tool_names = [t.get("tool") for t in tools if isinstance(t, dict)]
    if "agent" in tool_names:
        raise ValueError(f"{path}: the 'agent' (subagent) tool is not supported by recipes/arvo.")
    if not tool_names and not config.get("use_dataset_env"):
        raise ValueError(f"{path}: agent.tools is empty; the agent would have nothing to call.")
    return config


class _UserRestrictedEnv:
    """Wrap environment operations so pod filesystem access uses one user.

    Restricting only ``execute`` is insufficient: tools such as write/edit
    transfer payloads with ``copy_to``, whose Kubernetes implementation opens
    a separate exec stream.  Without forwarding ``as_user`` that stream runs
    as the container default user (normally root).
    """

    def __init__(self, env, user: str):
        self._env = env
        self._user = user

    def execute(self, command, **kwargs):
        kwargs.setdefault("as_user", self._user)
        return self._env.execute(command, **kwargs)

    def copy_to(self, src_path, dest_path, **kwargs):
        kwargs.setdefault("as_user", self._user)
        return self._env.copy_to(src_path, dest_path, **kwargs)

    def copy_out(self, src_path, dest_path, **kwargs):
        kwargs.setdefault("as_user", self._user)
        return self._env.copy_out(src_path, dest_path, **kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


@ray.remote(num_cpus=1)
class DatasetEnvActor:
    """Owns the k8s pod, the mimoagent tool registry, and reward computation."""

    def __init__(
        self,
        instance: dict[str, Any],
        instance_id: str,
        mimoagent_config_path: str,
        dump_dir: str | None = None,
    ):
        self.instance = instance
        self.instance_id = instance_id
        self.config = load_mimoagent_config(mimoagent_config_path)

        from mimoagent.tools import ToolRegistry

        self.tool_registry = ToolRegistry.from_config(self.config["agent"]["tools"])

        self._dump_dir = Path(dump_dir) if dump_dir else None
        self._env_logger: logging.Logger | None = None
        if self._dump_dir is not None:
            from mimoagent.utils.log import make_file_logger

            self._dump_dir.mkdir(parents=True, exist_ok=True)
            self._env_logger = make_file_logger(f"agent.env.{instance_id}", self._dump_dir / "env.log")

        self.dataset_env = None
        self._infra_error: str | None = None
        self._cleanup_failures = 0
        self._exec_user: str | None = None

    def _log(self, msg: str) -> None:
        if self._env_logger is not None:
            self._env_logger.info(msg)
        else:
            logger.warning(msg)


    def _create(self) -> None:
        from mimoagent.environments.utils import make_dataset_env

        _env_block = self.config.get("environment") or {}
        env_kwargs = dict(_env_block)
        env_kwargs["labels"] = {"exp": os.getenv("EXP_NAME", "unknown"), **(env_kwargs.get("labels") or {})}
        if os.environ.get("KUBECONFIG") and "kubeconfig" not in env_kwargs:
            env_kwargs["kubeconfig"] = os.environ["KUBECONFIG"]
        _registry = os.environ.get("DOCKER_REGISTRY")
        if _registry and "image_prefix" not in env_kwargs:
            env_kwargs["image_prefix"] = _registry

        self._log(f"creating environment, env_kwargs={env_kwargs}")
        self.dataset_env = make_dataset_env(self.instance, **env_kwargs)
        self.dataset_env.setup_environment()

        self._exec_user = getattr(getattr(self.dataset_env, "env", None), "config", None)
        if self._exec_user is not None:
            self._exec_user = getattr(self._exec_user, "exec_user", None)
        if self._exec_user:
            self._log(f"exec_user={self._exec_user!r}: tools will run as this user")

    async def setup(self, max_retries: int = 2) -> tuple[bool, str | None]:
        for attempt in range(1, max_retries + 1):
            try:
                await asyncio.to_thread(self._create)
                return True, None
            except Exception as e:
                err = f"setup attempt {attempt}/{max_retries} failed: {e}\n{traceback.format_exc()}"
                self._log(err)
                if attempt == max_retries:
                    self._infra_error = f"rollout/env_setup_failure: {e}"
                    return False, str(e)

    def describe(self) -> dict[str, Any]:
        assert self.dataset_env is not None, "describe() before setup()"
        return {
            "tool_definitions": self.tool_registry.get_function_definitions(),
            "tool_names": self.tool_registry.list_tools(),
            "template_vars": self.dataset_env.get_template_vars(),
            "pod_name": getattr(self.dataset_env.env, "pod_name", ""),
            "node_name": getattr(self.dataset_env.env, "node_name", ""),
        }


    def _tool_env(self):
        raw_env = self.dataset_env.env
        if self._exec_user:
            return _UserRestrictedEnv(raw_env, self._exec_user)
        return raw_env

    def _execute_tool_sync(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        import time

        from mimoagent.agents.base import LimitsExceeded
        from mimoagent.environments import TransportError
        from mimoagent.tools import ToolException

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                tool = self.tool_registry.get(name)
                result = tool.execute(params, {"env": self._tool_env()})
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
        assert self.dataset_env is not None, "execute() before setup()"
        return await asyncio.to_thread(lambda: self.dataset_env.env.execute(command, **kwargs))


    async def calculate_reward(self, timeout: float | None = None, final_message: str = "") -> tuple[float, str, dict]:
        if self._infra_error:
            self._log(f"skipping reward, infra_error={self._infra_error}")
            return (
                0.0,
                f"infra: {self._infra_error}",
                {"error_category": self._infra_error, "infra_error": self._infra_error},
            )

        if self.dataset_env is None:
            return 0.0, "dataset env not set up", {}

        try:
            reward, test_output, extra = await asyncio.to_thread(
                lambda: self.dataset_env.calculate_reward(timeout=timeout)
            )
        except Exception as e:
            from mimoagent.environments import TransportError

            if isinstance(e, TransportError):
                self._log(f"reward-phase TransportError: {e}")
                return 0.0, str(e), {"error_category": REWARD_TESTBED_CORRUPTED}
            self._log(f"calculate_reward raised: {e}\n{traceback.format_exc()}")
            return 0.0, str(e), {"error_category": "reward/exception"}

        if extra.get("transport_error"):
            extra["error_category"] = REWARD_TESTBED_CORRUPTED
            self._log(f"reward-phase TransportError (returned): {test_output}")
            return reward, test_output, extra

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
