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
"""Two-container pod support, as a subclass rather than a fork of the base backend.

The s3k tasks need a ``main`` container where the agent runs bash and a ``sidecar`` that
runs the task's MCP servers and its verifier, sharing an emptyDir volume for the workspace.
The pinned ``mimoagent`` builds a single-container pod, but it factors the manifest into
``KubernetesEnvironment._build_pod_body`` -- a pure function with one call site -- so the
extra containers can be appended without copying ``_start_pod``.

Selected from the mimoagent yaml by dotted path, because ``get_environment_class`` falls
back to importlib for any spec it does not recognise::

    environment:
      environment_class: recipes.general.general_agent.k8s_sidecar.SidecarKubernetesEnvironment
      shared_volumes: [workspace, system-vol]
      main_volume_mounts:
        - {name: workspace, mount_path: /work/workspace}
      sidecars:
        - name: sidecar
          command: [sleep, infinity]
          volume_mounts:
            - {name: workspace, mount_path: /work/workspace}
            - {name: system-vol, mount_path: /work/system}

``make_dataset_env`` keeps the extra yaml keys because ``filter_env_kwargs`` resolves the
accepted field set from this class's ``config_class`` default rather than from the base
config, so ``sidecars`` / ``shared_volumes`` / ``main_volume_mounts`` survive the filter.

Exec/copy routing is the other half of the job, and it is done here rather than in the
pinned package. The base ``execute`` / ``copy_to`` / ``copy_out`` open their exec streams
without a ``container=`` argument, which the API server rejects with 400 as soon as a pod
holds more than one container. Instead of copying ~500 lines of method bodies just to
thread one keyword through, the overrides below record the target container in a
thread-local and wrap ``self.v1_api`` so pod-exec calls pick it up. Nested helpers
(``copy_file_chunked`` / ``_write_chunk``, used for files above 32 MB) inherit the routing
for free, which the upstream implementation of this feature happens to miss.

The thread-local is load-bearing, not defensive: ``DefaultAgentConfig.tool_parallel_workers``
defaults to 8, so several tool calls share one environment object concurrently and
instance-level state would cross between them.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from mimoagent.environments.kubernetes import KubernetesEnvironment, KubernetesEnvironmentConfig

_ROUTE = threading.local()


class _ContainerRoutingCoreV1Api:
    """Injects ``container=`` into pod-exec calls and delegates everything else.

    Only ``connect_get_namespaced_pod_exec`` is intercepted; pod create/read must stay
    untouched. Verified against the kubernetes client rather than assumed: ``stream()``
    dispatches to ``_websocket_request``, which does ``api_client =
    api_method.__self__.api_client`` and then assigns ``api_client.request``. ``__self__``
    is this proxy, ``__getattr__`` hands back the *real* api_client object, and the
    assignment therefore lands on the real client -- so forwarding reads is enough and no
    ``__setattr__`` forwarding is needed.
    """

    def __init__(self, inner):
        self.__dict__["_inner"] = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)

    def connect_get_namespaced_pod_exec(self, *args, **kwargs):
        if "container" not in kwargs:
            target = getattr(_ROUTE, "container", None)
            if target:
                kwargs["container"] = target
        return self.__dict__["_inner"].connect_get_namespaced_pod_exec(*args, **kwargs)


@dataclass
class SidecarKubernetesEnvironmentConfig(KubernetesEnvironmentConfig):
    """Base config plus the three keys that describe a multi-container pod."""

    shared_volumes: list[str] = field(default_factory=list)
    main_volume_mounts: list[dict[str, Any]] = field(default_factory=list)
    sidecars: list[dict[str, Any]] = field(default_factory=list)


class SidecarKubernetesEnvironment(KubernetesEnvironment):
    """``KubernetesEnvironment`` that can also start sidecar containers."""

    def __init__(self, *, config_class: type = SidecarKubernetesEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._validate_multi_container_config()


    def _setup_kubernetes_client(self) -> None:
        super()._setup_kubernetes_client()
        self.v1_api = _ContainerRoutingCoreV1Api(self.v1_api)

    def _resolve_container(self, container: str | None) -> str | None:
        """Explicit target wins; otherwise address ``main`` once the pod is multi-container.

        Returns None for a single-container pod so the call stays byte-identical to the
        base behaviour -- the API server's own default then applies.
        """
        if container:
            return container
        return "main" if self.config.sidecars else None

    @contextmanager
    def _route(self, container: str | None):
        target = self._resolve_container(container)
        previous = getattr(_ROUTE, "container", None)
        _ROUTE.container = target
        try:
            yield
        finally:
            _ROUTE.container = previous

    def execute(self, command: str, cwd: str = "", timeout: int = None, container: str | None = None):
        with self._route(container):
            return super().execute(command, cwd, timeout)

    def copy_to(
        self,
        src_path: str,
        dest_path: str,
        *,
        timeout: int = 300,
        max_retries: int = 10,
        container: str | None = None,
        dereference: bool = False,
    ) -> None:
        if dereference:
            src_path = self._dereferenced(src_path)
        with self._route(container):
            return super().copy_to(src_path, dest_path, timeout=timeout, max_retries=max_retries)

    def copy_out(
        self,
        src_path: str,
        dest_path: str,
        *,
        timeout: int = 300,
        max_retries: int = 10,
        container: str | None = None,
    ) -> None:
        with self._route(container):
            return super().copy_out(src_path, dest_path, timeout=timeout, max_retries=max_retries)

    @staticmethod
    def _dereferenced(src_path: str) -> str:
        """Resolve ``src_path`` itself, and refuse a tree that hides symlinks deeper down.

        The base ``copy_to`` packs with tarfile's default ``dereference=False``, so a link
        arrives in the pod pointing at a path that does not exist there. Resolving the top
        level covers the layout this arm actually ships: the open_source_env bundle was
        built with its symlinks already resolved (measured: zero links across all 1435 task
        directories), and the environment class realpaths workspace/tools/system before
        uploading them.

        A hand-built task directory could still nest a link, and that failure is invisible
        -- the upload succeeds and the agent simply cannot read the file. Raise instead.
        """
        resolved = os.path.realpath(src_path)
        if os.path.isdir(resolved):
            for root, dirs, files in os.walk(resolved, followlinks=False):
                for name in dirs + files:
                    if os.path.islink(os.path.join(root, name)):
                        raise ValueError(
                            f"copy_to(dereference=True) cannot resolve nested symlinks: "
                            f"{os.path.join(root, name)!r} inside {src_path!r}. Materialise "
                            f"the link in the task directory before uploading."
                        )
        return resolved


    def _validate_multi_container_config(self) -> None:
        """Fail at construction rather than 20 minutes later as a pod event."""
        declared = set(self.config.shared_volumes)
        names: list[str] = []
        for sidecar in self.config.sidecars:
            name = sidecar.get("name")
            if not name or name == "main":
                raise ValueError(f"sidecar needs a name other than 'main': {sidecar!r}")
            names.append(name)
            for mount in sidecar.get("volume_mounts") or []:
                if mount.get("name") not in declared:
                    raise ValueError(
                        f"sidecar {name!r} mounts undeclared volume {mount.get('name')!r} "
                        f"(declared shared_volumes: {sorted(declared)})"
                    )
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate sidecar names: {names}")
        for mount in self.config.main_volume_mounts:
            if mount.get("name") not in declared:
                raise ValueError(
                    f"main mounts undeclared volume {mount.get('name')!r} (declared shared_volumes: {sorted(declared)})"
                )

    @staticmethod
    def _volume_mount_spec(mount: dict[str, Any]) -> dict[str, Any]:
        spec = {"name": mount["name"], "mountPath": mount["mount_path"]}
        if mount.get("read_only"):
            spec["readOnly"] = True
        return spec

    def _build_pod_body(self, env_vars: dict[str, str]) -> dict[str, Any]:
        body = super()._build_pod_body(env_vars)
        spec = body["spec"]

        if self.config.shared_volumes:
            spec.setdefault("volumes", []).extend({"name": name, "emptyDir": {}} for name in self.config.shared_volumes)
        if self.config.main_volume_mounts:
            spec["containers"][0].setdefault("volumeMounts", []).extend(
                self._volume_mount_spec(m) for m in self.config.main_volume_mounts
            )
        if not self.config.sidecars:
            return body

        if self.config.host_network:
            self.logger.warning(
                "sidecars configured: forcing hostNetwork=False "
                "(fixed sidecar ports would collide node-wide under hostNetwork)"
            )
        spec["hostNetwork"] = False

        for sidecar in self.config.sidecars:
            sidecar_env = {**env_vars, **(sidecar.get("env") or {})}
            container: dict[str, Any] = {
                "name": sidecar["name"],
                "image": sidecar.get("image") or self.config.image,
                "env": [{"name": k, "value": str(v)} for k, v in sidecar_env.items()],
                "imagePullPolicy": "IfNotPresent",
                "resources": {
                    "requests": {
                        "cpu": sidecar.get("cpu_request", "0.5"),
                        "memory": sidecar.get("memory_request", "1Gi"),
                    },
                    "limits": {
                        "cpu": sidecar.get("cpu_limit", "2"),
                        "memory": sidecar.get("memory_limit", "4Gi"),
                    },
                },
            }
            if sidecar.get("command"):
                container["command"] = sidecar["command"]
            if sidecar.get("args"):
                container["args"] = sidecar["args"]
            if sidecar.get("volume_mounts"):
                container["volumeMounts"] = [self._volume_mount_spec(m) for m in sidecar["volume_mounts"]]
            if sidecar.get("privileged"):
                container["securityContext"] = {"privileged": True}
            spec["containers"].append(container)

        return body

    def _container_ready(self, pod) -> bool:
        """Every container must be ready, not just ``main``.

        The base implementation returns True as soon as ``main`` is up, which would let
        setup run against MCP servers whose container has not started. Readiness here is
        still process-level: a container being ready says nothing about its ports being
        bound, so the task's own ``wait_ports`` handshake remains necessary.
        """
        if not self.config.sidecars:
            return super()._container_ready(pod)
        statuses = pod.status.container_statuses
        if not statuses or len(statuses) < 1 + len(self.config.sidecars):
            return False
        return all(cs.ready is True for cs in statuses)
