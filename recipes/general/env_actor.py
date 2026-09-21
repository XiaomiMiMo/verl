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
"""Ray actor owning one mimoagent ``DatasetEnvironment`` (k8s pod) plus its tools.

Modelled on the reference RL stack's environment actor, minus the black-box router /
``/finalize`` plumbing, plus an ``execute_tool`` entry point.

Why the pod lives in a separate actor rather than inside ``AgentLoopWorker``:

* Every tool call is a blocking ``kubectl exec``-style round trip (~seconds). Running
  them in the worker process would need a thread per in-flight call anyway, and a
  hundred concurrent SWE rollouts would put the whole k8s client stack on the GPU node's
  single worker process.
* ``write``/``edit`` ship their payload via ``env.copy_to(local_tempfile, remote_path)``,
  so the **tools must run in the same process as the pod client**. Hence ``ToolRegistry``
  lives here, not on the agent-loop side.
* ``ray.kill`` gives the caller a way out when a rollout wedges on a dead pod.
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
    """Load the mimoagent yaml and reject configs this recipe cannot honour."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    tools = (config.get("agent") or {}).get("tools") or []
    tool_names = [t.get("tool") for t in tools if isinstance(t, dict)]
    if "agent" in tool_names:
        raise ValueError(
            f"{path}: the 'agent' (subagent) tool is not supported by recipes/general. "
            "A subagent forks the conversation into a tree, but the rollout keeps a single "
            "append-only token buffer (see token_trace.py). Use a config without the 'agent' "
            "tool, e.g. swe_antihack_nosubagent_train.yaml."
        )
    if not tool_names and not config.get("use_dataset_env"):
        raise ValueError(f"{path}: agent.tools is empty; the agent would have nothing to call.")
    return config


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

        _setup_proxy = self.config.get("setup_proxy")
        if _setup_proxy:
            for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ.setdefault(_key, str(_setup_proxy))
            for _key in ("no_proxy", "NO_PROXY"):
                os.environ.setdefault(_key, "localhost,127.0.0.1,.svc.cluster.local,.local")

        self.dataset_env = None
        self._infra_error: str | None = None
        self._cleanup_failures = 0

        self._reward_binarize = str(os.environ.get("REWARD_BINARIZE", "")).strip().lower() in {"1", "true", "yes", "on"}
        try:
            self._reward_binarize_threshold = float(os.environ.get("REWARD_BINARIZE_THRESHOLD", "1.0"))
        except (TypeError, ValueError):
            self._reward_binarize_threshold = 1.0

        from mimoagent.tools import ToolRegistry
        from mimoagent.tools.registry import _TOOL_CLASSES as _MIMO_TOOL_CLASSES

        _tool_names = [t.get("tool") for t in self.config["agent"]["tools"] if isinstance(t, dict)]
        if any(name not in _MIMO_TOOL_CLASSES for name in _tool_names):
            from .tools import register_cc_tools

            register_cc_tools()

        self.tool_registry = ToolRegistry.from_config(self.config["agent"]["tools"])

        self._dump_dir = Path(dump_dir) if dump_dir else None
        self._env_logger: logging.Logger | None = None
        if self._dump_dir is not None:
            from mimoagent.utils.log import make_file_logger

            self._dump_dir.mkdir(parents=True, exist_ok=True)
            self._env_logger = make_file_logger(f"agent.env.{instance_id}", self._dump_dir / "env.log")


    def _log(self, msg: str) -> None:
        if self._env_logger is not None:
            self._env_logger.info(msg)
        else:
            logger.warning(msg)


    def _create(self) -> None:
        """Build the DatasetEnvironment and start the pod. Blocking; raises on failure."""
        from mimoagent.environments.utils import make_dataset_env

        from .general_agent import register_general_agent_env

        register_general_agent_env()

        from mimoagent.environments.datasets import DATASET_REGISTRY

        _ds_type = self.instance.get("dataset_type")
        _supported = set(DATASET_REGISTRY)
        if _ds_type and _ds_type not in _supported:
            raise NotImplementedError(
                f"dataset_type '{_ds_type}' has no registered environment yet. "
                f"Supported: {sorted(_supported)}. To add support: implement the "
                f"corresponding Environment class in mimoagent.environments and "
                f"register it in mimoagent.environments.datasets."
            )

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
        _env = self.dataset_env.env
        _mcp_servers = getattr(_env, "mcp_servers", None)
        _bridge_script = getattr(_env, "mcp_bridge_script", None)
        _bridge_python = getattr(_env, "mcp_bridge_python", "python3")
        if _mcp_servers and _bridge_script:
            from .mcp_proxy import discover_mcp_tools

            _mcp_tools = discover_mcp_tools(_env, _mcp_servers, _bridge_python, _bridge_script)
            for _t in _mcp_tools:
                self.tool_registry.register(_t)
            self._log(f"registered {len(_mcp_tools)} MCP tool(s) from {len(_mcp_servers)} server(s)")
            self._ensure_bridge_in_sidecar(_bridge_script)
        elif _mcp_servers:
            self._log(
                f"WARNING: env has {len(_mcp_servers)} mcp_servers but no mcp_bridge_script; MCP tools not registered"
            )
        self._log("environment ready")

    def _ensure_bridge_in_sidecar(self, bridge_path: str) -> None:
        """Make the MCP bridge visible to the sidecar's verifier.

        automation_bench grades live-world state through ``/work/_setup/mcp_bridge.py``
        (verifier/evaluation/main.py: ``BRIDGE = Path(os.environ.get("MCP_BRIDGE_SCRIPT",
        "/work/_setup/mcp_bridge.py"))``), but its manifest.json routes that upload to
        ``main`` while the verifier command runs in ``sidecar``. So ``BRIDGE.is_file()``
        is False there, ``_score_live_world()`` returns None, and scoring silently
        degrades to ``state.json`` with partial_credit forced to 0 — measured at some point
        on v95r2: ``rubrics=7 passed=5 partial_credit=0.0000``, where the offline run of
        the same instance scores passed/rubrics = 0.25. s3k is unaffected (it grades from
        DB post-state + LLM judge, no live-world), which is why this only dragged
        automation_bench down to ~0.16.

        Best-effort: a failure here just leaves the pre-fix degraded scoring in place.
        """
        src = os.path.join(self.instance.get("env_task_dir") or "", "mcp_bridge.py")
        if not os.path.isfile(src):
            return
        try:
            env = self.dataset_env.env
            env.execute(f"mkdir -p {os.path.dirname(bridge_path)}", container="sidecar")
            env.copy_to(src, bridge_path, container="sidecar", dereference=True)
            check = env.execute(f"test -s {bridge_path} && echo OK", container="sidecar")
            if "OK" in (check.get("output") or ""):
                self._log(f"bridge copied into sidecar at {bridge_path}")
            else:
                self._log(f"WARNING: bridge copy to sidecar unverified at {bridge_path}")
        except Exception as e:
            self._log(f"WARNING: could not copy bridge into sidecar (live-world scoring stays degraded): {e}")

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
        """Tool schemas + jinja template vars, needed before the agent can be built.

        ``template_vars`` is what fills ``{{cwd}}`` in the yaml's ``system_template``; it
        comes from the environment, so it is only known after the pod exists.
        """
        assert self.dataset_env is not None, "describe() before setup()"
        _env = self.dataset_env.env
        return {
            "tool_definitions": self.tool_registry.get_function_definitions(),
            "tool_names": self.tool_registry.list_tools(),
            "template_vars": self.dataset_env.get_template_vars(),
            "pod_name": getattr(_env, "pod_name", ""),
            "node_name": getattr(_env, "node_name", ""),
            "mcp_servers": getattr(_env, "mcp_servers", None),
            "mcp_bridge_script": getattr(_env, "mcp_bridge_script", None),
            "mcp_bridge_python": getattr(_env, "mcp_bridge_python", "python3"),
        }


    def _execute_tool_sync(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run one mimoagent tool, mapping its exceptions onto tagged return values.

        The exception taxonomy mirrors ``mimoagent/agents/default.py:execute_action`` so
        the caller can reproduce it verbatim on the other side of the Ray boundary.
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
        """Raw pod exec. Not used by the rollout; kept for debugging / smoke scripts."""
        assert self.dataset_env is not None, "execute() before setup()"
        return await asyncio.to_thread(lambda: self.dataset_env.env.execute(command, **kwargs))


    async def calculate_reward(self, timeout: float | None = None, final_message: str = "") -> tuple[float, str, dict]:
        """Delegate to mimoagent's dataset-specific grading. Never raises."""
        if self._infra_error:
            self._log(f"skipping reward, infra_error={self._infra_error}")
            return (
                0.0,
                f"infra: {self._infra_error}",
                {"error_category": self._infra_error, "infra_error": self._infra_error},
            )

        if self.dataset_env is None:
            return 0.0, "dataset env not set up", {}

        if hasattr(self.dataset_env, "attach_rollout"):
            try:
                self.dataset_env.attach_rollout(
                    agent=None,
                    task=self.instance.get("problem_statement") or "",
                    result=final_message or "",
                )
                self._log(f"attach_rollout: final_message {len(final_message or '')} chars for grading")
            except Exception as e:
                self._log(f"attach_rollout failed (grading continues): {e}")

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

        await asyncio.to_thread(self._dump_model_patch, extra)

        if extra.get("transport_error"):
            extra["error_category"] = REWARD_TESTBED_CORRUPTED
            self._log(f"reward-phase TransportError (returned): {test_output}")
            return reward, test_output, extra

        if self._reward_binarize:
            orig = reward
            reward = 1.0 if reward >= self._reward_binarize_threshold else 0.0
            extra = dict(extra or {})
            extra["reward_continuous"] = orig
            self._log(f"binarize: {orig:.4f} -> {reward} (thr={self._reward_binarize_threshold})")

        self._log(f"reward={reward}")
        self._log(f"test_output={test_output}")
        return reward, test_output, extra

    def _dump_model_patch(self, extra: dict) -> None:
        if self._dump_dir is None:
            return
        patch = extra.get("model_patch") or ""
        patch_error = extra.get("model_patch_error") or ""
        try:
            if patch_error:
                (self._dump_dir / "model_patch_error.txt").write_text(patch_error, encoding="utf-8")
            if patch:
                if not patch.endswith("\n"):
                    patch += "\n"
                (self._dump_dir / "model_patch.diff").write_text(patch, encoding="utf-8", newline="\n")
        except Exception as e:
            self._log(f"failed to dump model patch: {e}")


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
