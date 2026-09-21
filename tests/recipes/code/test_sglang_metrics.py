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
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
ASYNC_SERVER = REPO_ROOT / "verl/workers/rollout/sglang_rollout/async_sglang_server.py"
EXPORTER = REPO_ROOT / "verl/workers/rollout/sglang_rollout/metrics_exporter.py"
LAUNCHER = REPO_ROOT / "recipes/code/run_train.sh"


def test_sglang_exporter_is_started_and_ray_receives_metric_env():
    server = ASYNC_SERVER.read_text()
    launcher = LAUNCHER.read_text()

    assert EXPORTER.exists()
    assert "start_metrics_server" in server
    assert "Started metrics exporter" in server
    assert "runtime_env.env_vars.ENABLE_METRIC" in launcher
    assert "runtime_env.env_vars.METRIC_PORT" in launcher
