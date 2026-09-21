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
"""The s3k dataset environment and the two-container pod backend it needs.

``dataset_type: "general_agent"`` rows route to :class:`environment.GeneralAgentEnvironment`,
which assumes a pod that already has a sidecar -- it consumes the topology rather than
requesting it. The topology itself comes from the yaml ``environment:`` block, served by
:class:`k8s_sidecar.SidecarKubernetesEnvironment`.
"""

DATASET_TYPE = "general_agent"


def register_general_agent_env() -> None:
    """Make ``dataset_type: general_agent`` routable. Idempotent and additive.

    ``DATASET_REGISTRY`` is a plain module-level dict and ``detect_dataset_type``
    honours a declared type that is already in it, so this needs no change to the
    pinned mimoagent. ``setdefault`` rather than assignment: a caller who registered
    their own general-agent environment keeps it.
    """
    from mimoagent.environments.datasets import DATASET_REGISTRY

    from .environment import GeneralAgentEnvironment

    DATASET_REGISTRY.setdefault(DATASET_TYPE, GeneralAgentEnvironment)


__all__ = [
    "DATASET_TYPE",
    "register_general_agent_env",
]
