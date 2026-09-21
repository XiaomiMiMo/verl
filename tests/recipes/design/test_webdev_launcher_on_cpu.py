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
"""The two launcher layers, and the invariants a per-key config table cannot see.

``notes/scripts/verify_recipe_configs.py`` compares effective values key by key, but it
reads the launcher as text, so it cannot evaluate a branch. Everything here is either
branch-sensitive or a relationship between two files -- the kind of claim that is true when
written and quietly false after one of the files is edited alone.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
CONTROL = REPO / "scripts/design/webdev.sh"
RUNNER = REPO / "recipes/design/run_webdev.sh"
TRAIN_PROFILE = REPO / "config/agent/design/webdev.yaml"
EVAL_PROFILE = REPO / "config/agent/design/webdev-eval.yaml"
TRAIN_REGISTRY = REPO / "recipes/design/config/webdev_agent_loop.yaml"
EVAL_REGISTRY = REPO / "recipes/design/config/webdev_eval_agent_loop.yaml"


def _flat(node, prefix=""):
    out = {}
    if OmegaConf.is_dict(node):
        for k, v in node.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif OmegaConf.is_list(node):
        for i, v in enumerate(node):
            out.update(_flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = node
    return out


# ---------------------------------------------------------------------------
# evaluation must measure the model, not a differently-configured model
# ---------------------------------------------------------------------------


def test_the_eval_profile_differs_from_training_only_in_the_grader():
    """Prompt, tools, step limit and environment must be carried over unchanged.

    Otherwise an evaluation number is not comparable with the training checkpoint it came
    from: it would be measuring the model under a different prompt. YAML has no include, so
    the two files really are duplicates -- which is exactly why this assertion exists rather
    than a comment saying they agree.
    """
    train = {k: v for k, v in _flat(OmegaConf.load(TRAIN_PROFILE)).items() if not k.startswith("traj_grader")}
    ev = {k: v for k, v in _flat(OmegaConf.load(EVAL_PROFILE)).items() if not k.startswith("traj_grader")}
    assert train == ev, f"differs outside traj_grader: {sorted(set(train) ^ set(ev))}"
    assert train, "sanity: the comparison must not be over an empty dict"

    assert OmegaConf.load(TRAIN_PROFILE).traj_grader.correctness_mode == "design_group_v1"
    assert OmegaConf.load(EVAL_PROFILE).traj_grader.correctness_mode == "webdev_eval_v1"


def test_the_two_registries_differ_only_in_which_profile_they_name():
    """Evaluation must not change a harness parameter either -- not the per-turn budget, not
    a timeout, not the image cap."""
    train = OmegaConf.load(TRAIN_REGISTRY)
    ev = OmegaConf.load(EVAL_REGISTRY)
    assert len(train) == len(ev) == 1
    differing = [k for k in set(train[0]) | set(ev[0]) if train[0].get(k) != ev[0].get(k)]
    assert differing == ["config_path"], f"unexpected differences: {differing}"
    assert (REPO / train[0].config_path) == TRAIN_PROFILE
    assert (REPO / ev[0].config_path) == EVAL_PROFILE


def test_both_registries_point_at_a_profile_that_exists_and_a_class_that_imports():
    import importlib

    for registry in (TRAIN_REGISTRY, EVAL_REGISTRY):
        entry = OmegaConf.load(registry)[0]
        assert (REPO / entry.config_path).is_file(), entry.config_path
        module, _, cls = entry._target_.rpartition(".")
        assert hasattr(importlib.import_module(module), cls), entry._target_


# ---------------------------------------------------------------------------
# branch-sensitive: what each mode actually emits
# ---------------------------------------------------------------------------


def test_val_only_is_emitted_in_the_eval_branch_only():
    """A training run that set val_only would train on nothing and report a validation
    number; an evaluation run that omitted it would train instead of evaluate.

    Asserted on placement rather than on presence, because the config checker greps this
    file as text and would otherwise read the eval value for a training run.
    """
    src = RUNNER.read_text()
    assert src.count("trainer.val_only=True") == 1
    branch = src.index('if [ "${WEBDEV_MODE}" = "eval" ]; then')
    closing = src.index("\nfi\n", branch)
    assert branch < src.index("trainer.val_only=True") < closing

    # Sampling has to be named in the same branch: validation reads val_kwargs, which
    # defaults to greedy, so the rollout temperature above never reaches it.
    for key in ("do_sample", "temperature", "top_p", "top_k", "n=1"):
        idx = src.index(f"val_kwargs.{key}" if key != "n=1" else "val_kwargs.n=1")
        assert branch < idx < closing, f"val_kwargs.{key} is outside the eval branch"


def test_the_control_script_exports_one_line_per_mode_dependent_default():
    """The config checker reads these defaults line by line, so a mode-dependent value put
    inside an if/else would have both branches applied with the last one winning -- which is
    how a previous check reported the wrong scheduler strategy."""
    src = CONTROL.read_text()
    for var in ("TEST_FREQ", "VAL_BEFORE_TRAIN"):
        lines = [ln for ln in src.splitlines() if ln.startswith(f"export {var}=")]
        assert len(lines) == 1, f"{var} must be exported exactly once, found {len(lines)}"
        assert "WEBDEV_MODE" in lines[0], f"{var} must branch inside its own expansion"


@pytest.mark.parametrize("mode,test_freq,val_before", [("train", "-1", "False"), ("eval", "1", "True")])
def test_the_mode_dependent_defaults_resolve_as_intended(mode, test_freq, val_before):
    """Runs the export lines for real, rather than trusting the expansion by eye."""
    src = CONTROL.read_text()
    exports = [ln for ln in src.splitlines() if ln.startswith(("export TEST_FREQ=", "export VAL_BEFORE_TRAIN="))]
    script = f"WEBDEV_MODE={mode}\n" + "\n".join(exports) + '\nprintf "%s %s" "$TEST_FREQ" "$VAL_BEFORE_TRAIN"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.split() == [test_freq, val_before]


# ---------------------------------------------------------------------------
# the checks that exist to fail early
# ---------------------------------------------------------------------------


def _run_runner(tmp_path, **env):
    """Invoke the runner with a plausible-but-fake environment; return the completed proc."""
    base = {
        "MODEL_PATH": str(tmp_path / "ckpt"),
        "TRAIN_DATA": str(tmp_path / "train.parquet"),
        "VAL_DATA": str(tmp_path / "val.parquet"),
        "WEBDEV_MODE": "train",
        "TRAIN_BATCH_SIZE": "32",
        "PPO_MINI_BATCH_SIZE": "32",
        "PATH": os.environ["PATH"],
    }
    base.update(env)
    return subprocess.run(["bash", str(RUNNER)], capture_output=True, text=True, env=base)


def test_training_refuses_to_start_without_a_shared_dump_directory(tmp_path):
    """The per-rollout grader writes the judged screenshot there and the driver reads it back
    by path. Unset, every group is skipped and every reward stays 0.0 -- with no error."""
    proc = _run_runner(tmp_path)
    assert proc.returncode == 2
    assert "WEBDEV_DEBUG_DIR is unset" in proc.stderr
    assert "shared" in proc.stderr


def test_prompt_mean_refuses_a_batch_it_cannot_weight(tmp_path):
    """The mismatch does raise inside the actor, but only after the first rollout batch --
    which on this arm is tens of minutes of pod work already spent."""
    proc = _run_runner(tmp_path, TRAIN_BATCH_SIZE="32", PPO_MINI_BATCH_SIZE="16")
    assert proc.returncode == 1
    assert "prompt-mean requires" in proc.stderr


def test_a_missing_agent_submodule_is_named_rather_than_imported_blindly(tmp_path):
    proc = _run_runner(tmp_path, MIMOAGENT_SRC=str(tmp_path / "nope"))
    assert proc.returncode == 2
    assert "git submodule update --init" in proc.stderr


def test_the_control_script_reports_every_missing_variable_at_once(tmp_path):
    """Checking one at a time means a launch that fails four times before it starts."""
    proc = subprocess.run(
        ["bash", str(CONTROL)],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "MODEL_PATH": str(tmp_path / "ckpt")},
    )
    assert proc.returncode == 1
    for var in ("TRAIN_DATA", "VAL_DATA", "KUBECONFIG", "POD_PROXY", "DESIGN_GRADER_URL", "LLM_JUDGE_API_KEY"):
        assert var in proc.stderr, f"{var} missing from the single report"


def test_eval_mode_asks_for_the_judge_not_the_service(tmp_path):
    proc = subprocess.run(
        ["bash", str(CONTROL)],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "WEBDEV_MODE": "eval"},
    )
    assert proc.returncode == 1
    for var in ("WEBDEV_EVAL_JUDGE_BASE_URL", "WEBDEV_EVAL_JUDGE_API_KEY", "WEBDEV_EVAL_JUDGE_MODEL"):
        assert var in proc.stderr
    assert "DESIGN_GRADER_URL" not in proc.stderr, "evaluation needs no grading service"


def test_the_runtime_env_carries_what_the_pod_actors_and_hooks_read():
    """A bare export reaches the driver and nothing else: the trainer and the environment
    actors are Ray actors and inherit the raylet's environment, not this shell's."""
    src = RUNNER.read_text()
    for var in (
        "PYTHONPATH",
        "KUBECONFIG",
        "POD_PROXY",
        "WEBDEV_GRADE_MODE",
        "WEBDEV_GRADE_HTTP",
        "WEBDEV_DEBUG_DIR",
        "DROP_INFRA_FROM_GROUP",
        "TRAJECTORY_TIMEOUT",
        "ENV_SETUP_TIMEOUT",
        "REWARD_TIMEOUT",
        "ENV_NUM_CPUS",
        "FAIL_ON_ENV_SETUP_ERROR",
        "INVALID_REWARD_FOR_INFRA",
    ):
        assert f"runtime_env.env_vars.{var}=" in src, f"{var} never reaches the workers"

    # Ray's env_vars only accepts strings, and a bare 1 is parsed by Hydra as an int. The
    # numeric and boolean ones therefore have to be quoted at the Hydra level.
    for var in (
        "WEBDEV_GRADE_HTTP",
        "DROP_INFRA_FROM_GROUP",
        "TRAJECTORY_TIMEOUT",
        "ENV_SETUP_TIMEOUT",
        "REWARD_TIMEOUT",
        "ENV_NUM_CPUS",
        "FAIL_ON_ENV_SETUP_ERROR",
        "INVALID_REWARD_FOR_INFRA",
    ):
        line = next(ln for ln in src.splitlines() if f"runtime_env.env_vars.{var}=" in ln)
        assert '\\"' in line, f"{var} is forwarded unquoted; Ray will reject a non-string"


# --- site placement: blank in the profile, supplied by the environment -----------------
#
# The failure these guard against is silent and expensive: a pod that selects a tainted
# node pool without the matching toleration never leaves Pending, every rollout sits in
# `creating environment` until env_setup_timeout, and the step then trains on empty
# trajectories -- surfacing two layers away as a bare torch check failure inside Megatron.

PLACEMENT_VARS = ("WEBDEV_NODE_SELECTOR", "WEBDEV_TOLERATIONS", "WEBDEV_DNAT_PROXY_IP")


def test_both_profiles_ship_placement_blank():
    """A site's node pool, taint and proxy IP must not be committed to either profile."""
    for profile in (TRAIN_PROFILE, EVAL_PROFILE):
        env = OmegaConf.load(profile).environment
        assert env.node_selector == {}, f"{profile.name} commits a node_selector"
        assert env.tolerations == [], f"{profile.name} commits tolerations"
        assert env.dnat_proxy_ip == "", f"{profile.name} commits a dnat_proxy_ip"


def test_placement_reaches_the_pod_actor_and_survives_hydra():
    """Read inside a Ray actor, so the launcher has to forward them, and the two JSON ones
    have to come through Hydra as opaque strings rather than parsed structure."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    src = RUNNER.read_text()
    for var in PLACEMENT_VARS:
        assert f"runtime_env.env_vars.{var}=" in src, f"{var} never reaches the env actor"

    parser = OverridesParser.create()
    for value in ('{"pool":"x"}', '[{"key":"pool","operator":"Exists"}]', "10.0.0.1", ""):
        override = f"+ray_kwargs.ray_init.runtime_env.env_vars.WEBDEV_TOLERATIONS='{value}'"
        parsed = parser.parse_overrides([override])[0].value()
        assert parsed == value, f"Hydra mangled {value!r} into {parsed!r}"
        assert isinstance(parsed, str), f"Ray's env_vars rejects non-strings, got {type(parsed)}"


@pytest.mark.parametrize("var,key", [
    ("WEBDEV_NODE_SELECTOR", "node_selector"),
    ("WEBDEV_TOLERATIONS", "tolerations"),
    ("WEBDEV_DNAT_PROXY_IP", "dnat_proxy_ip"),
])
def test_unset_or_blank_placement_leaves_the_profile_alone(monkeypatch, var, key):
    from recipes.design.env_actor import _apply_placement_overrides

    for v in PLACEMENT_VARS:
        monkeypatch.delenv(v, raising=False)
    sentinel = {"node_selector": {"kept": "yes"}, "tolerations": [{"kept": "yes"}], "dnat_proxy_ip": "kept"}

    kwargs = dict(sentinel)
    _apply_placement_overrides(kwargs)
    assert kwargs == sentinel, "an unset variable overwrote the profile"

    monkeypatch.setenv(var, "   ")
    kwargs = dict(sentinel)
    _apply_placement_overrides(kwargs)
    assert kwargs[key] == sentinel[key], "a blank variable overwrote the profile"


def test_set_placement_overlays_the_profile(monkeypatch):
    import json

    from recipes.design.env_actor import _apply_placement_overrides

    tolerations = [{"key": "pool", "operator": "Equal", "value": "x", "effect": "NoSchedule"}]
    monkeypatch.setenv("WEBDEV_NODE_SELECTOR", json.dumps({"pool": "x"}))
    monkeypatch.setenv("WEBDEV_TOLERATIONS", json.dumps(tolerations))
    monkeypatch.setenv("WEBDEV_DNAT_PROXY_IP", "10.0.0.1")

    kwargs = {"node_selector": {}, "tolerations": [], "dnat_proxy_ip": ""}
    _apply_placement_overrides(kwargs)
    assert kwargs == {"node_selector": {"pool": "x"}, "tolerations": tolerations, "dnat_proxy_ip": "10.0.0.1"}


def test_unparseable_placement_refuses_to_start(monkeypatch):
    """Falling back to the blank default here would select the pool and drop the
    toleration -- the exact combination that hangs every pod in Pending."""
    from recipes.design.env_actor import _apply_placement_overrides

    monkeypatch.setenv("WEBDEV_TOLERATIONS", "[{not json")
    with pytest.raises(ValueError, match="WEBDEV_TOLERATIONS"):
        _apply_placement_overrides({"tolerations": []})
