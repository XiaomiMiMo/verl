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
"""CPU-only tests for the ARVO arm's launcher, config, and FQN wiring."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ARVO_SH = REPO_ROOT / "scripts" / "arvo" / "arvo.sh"
RUN_ARVO_SH = REPO_ROOT / "recipes" / "arvo" / "run_arvo.sh"
AGENT_LOOP_YAML = REPO_ROOT / "recipes" / "arvo" / "config" / "arvo_agent_loop.yaml"
HARNESS_YAML = REPO_ROOT / "config" / "agent" / "arvo" / "arvo.yaml"


# ---------------------------------------------------------------------------
# FQN / import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_agent_loop_target_importable(self):
        """The _target_ in arvo_agent_loop.yaml must be importable."""
        entries = yaml.safe_load(AGENT_LOOP_YAML.read_text())
        for entry in entries:
            target = entry["_target_"]
            module_path, cls_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            assert cls is not None, f"{target} resolved to None"

    def test_env_actor_importable(self):
        from recipes.arvo.env_actor import DatasetEnvActor, _UserRestrictedEnv

        assert DatasetEnvActor is not None
        assert _UserRestrictedEnv is not None

    def test_arvo_dataset_env_registered(self):
        from mimoagent.environments.datasets import DATASET_REGISTRY

        assert "arvo" in DATASET_REGISTRY


# ---------------------------------------------------------------------------
# Config structure tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_agent_loop_yaml_names_mimo_swe_agent(self):
        entries = yaml.safe_load(AGENT_LOOP_YAML.read_text())
        names = {e["name"] for e in entries}
        assert "mimo_swe_agent" in names

    def test_harness_profile_has_exec_user(self):
        cfg = yaml.safe_load(HARNESS_YAML.read_text())
        env = cfg.get("environment", {})
        assert env.get("exec_user") == "agent"

    def test_harness_profile_has_use_dataset_env(self):
        cfg = yaml.safe_load(HARNESS_YAML.read_text())
        assert cfg.get("use_dataset_env") is True

    def test_harness_profile_tools_are_lowercase(self):
        cfg = yaml.safe_load(HARNESS_YAML.read_text())
        tools = cfg.get("agent", {}).get("tools", [])
        names = [t["tool"] for t in tools]
        for name in names:
            assert name == name.lower(), f"tool {name!r} is not lowercase"

    def test_hydra_config_resolves(self):
        """arvo.yaml must resolve with dummy model/data paths."""
        env = {
            **os.environ,
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'third_party' / 'uni_agent'}:"
            f"{REPO_ROOT / 'third_party' / 'mimoagent-osr' / 'src'}",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.trainer.main_ppo",
                "--config-name=arvo",
                f"--config-path={REPO_ROOT / 'recipes' / 'arvo' / 'config'}",
                "hydra.searchpath=[pkg://verl.trainer.config]",
                "actor_rollout_ref.model.path=/dummy/model",
                "data.train_files=[/dummy/t.parquet]",
                "data.val_files=[/dummy/v.parquet]",
                "--cfg",
                "job",
                "--resolve",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert result.returncode == 0, f"Hydra resolve failed:\n{result.stderr[-1000:]}"
        assert len(result.stdout.strip().splitlines()) > 100


# ---------------------------------------------------------------------------
# Launcher preflight tests
# ---------------------------------------------------------------------------


class TestLauncherPreflight:
    """Run the launcher scripts in a subprocess and check preflight behavior."""

    @staticmethod
    def _run_arvo_sh(extra_env: dict, timeout: int = 30) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'third_party' / 'uni_agent'}:"
            f"{REPO_ROOT / 'third_party' / 'mimoagent-osr' / 'src'}",
            "PREFLIGHT_ONLY": "1",
            **extra_env,
        }
        return subprocess.run(
            ["bash", str(ARVO_SH)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def test_missing_model_path_fails(self):
        result = self._run_arvo_sh(
            {
                "TRAIN_DATA": "/dummy/t.parquet",
                "VAL_DATA": "/dummy/v.parquet",
            }
        )
        assert result.returncode != 0
        assert "MODEL_PATH" in result.stderr

    def test_missing_train_data_fails(self):
        result = self._run_arvo_sh(
            {
                "MODEL_PATH": "/dummy/model",
                "VAL_DATA": "/dummy/v.parquet",
            }
        )
        assert result.returncode != 0
        assert "TRAIN_DATA" in result.stderr

    def test_batch_mini_batch_mismatch_fails(self):
        result = self._run_arvo_sh(
            {
                "MODEL_PATH": "/dummy/model",
                "TRAIN_DATA": "/dummy/t.parquet",
                "VAL_DATA": "/dummy/v.parquet",
                "TRAIN_BATCH_SIZE": "32",
                "PPO_MINI_BATCH_SIZE": "64",
            }
        )
        assert result.returncode != 0
        assert "TRAIN_BATCH_SIZE" in result.stderr

    def test_infra_strategy_conflict_fails(self):
        result = self._run_arvo_sh(
            {
                "MODEL_PATH": "/dummy/model",
                "TRAIN_DATA": "/dummy/t.parquet",
                "VAL_DATA": "/dummy/v.parquet",
                "INVALID_REWARD_FOR_INFRA": "true",
                "DROP_INFRA_FROM_GROUP": "1",
            }
        )
        assert result.returncode != 0
        assert "conflicts" in result.stderr.lower() or "INVALID_REWARD_FOR_INFRA" in result.stderr


# ---------------------------------------------------------------------------
# ARVO dataset env unit tests
# ---------------------------------------------------------------------------


class TestArvoDatasetEnv:
    def test_parse_description_extracts_all_fields(self):
        from mimoagent.environments.datasets.arvo import _parse_description

        desc = "AddressSanitizer: heap-buffer-overflow in function `extract_name` in file `dnsmasq/src/rfc1035.c`"
        func, file_, san, err = _parse_description(desc)
        assert func == "extract_name"
        assert file_ == "dnsmasq/src/rfc1035.c"
        assert san == "AddressSanitizer"
        assert err == "heap-buffer-overflow"

    def test_parse_description_fails_on_empty(self):
        from mimoagent.environments.datasets.arvo import _parse_description

        with pytest.raises(ValueError, match="cannot parse function"):
            _parse_description("")

    def test_parse_description_no_file_still_works(self):
        from mimoagent.environments.datasets.arvo import _parse_description

        desc = "AddressSanitizer: heap-buffer-overflow in function `foo`"
        func, file_, san, err = _parse_description(desc)
        assert func == "foo"
        assert file_ == ""
        assert san == "AddressSanitizer"
        assert err == "heap-buffer-overflow"
