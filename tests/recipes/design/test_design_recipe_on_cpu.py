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
"""CPU checks for the design recipe.

The reflective identifiers and the tool catalogue are the two things here that
fail late rather than at import: a wrong `pkg://` path or agent name surfaces
mid-rollout, and a reworded tool description surfaces as a slightly different
policy weeks later. Both are asserted here.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[3]
RECIPE_ROOT = REPO_ROOT / "recipes" / "design"

pytest.importorskip("mimoagent", reason="needs third_party/mimoagent-osr/src on PYTHONPATH")


# --- music: the scorer is the reward ----------------------------------------


def test_music_config_names_the_scorer_by_package_path():
    """`pkg://` resolves as a real import, so relative imports inside the
    scorer package work. A bare file path would load the module without a
    package context and the package's own `from .pipeline import ...` would
    fail -- which is why the original needed a sys.path shim and this does not.
    """
    config = yaml.safe_load((RECIPE_ROOT / "config" / "music.yaml").read_text())
    reward_fn = config["reward"]["custom_reward_function"]

    assert reward_fn["path"] == "pkg://recipes.design.music.scorer"
    assert reward_fn["name"] == "compute_score"

    from verl.utils.import_utils import load_extern_object

    loaded = load_extern_object(module_path=reward_fn["path"], object_name=reward_fn["name"])
    assert callable(loaded)


def test_music_scorer_signature_matches_the_reward_manager_call():
    """The manager calls by keyword. The default `data_source` dispatch once
    called a widened signature positionally, so `data_source` bound to
    `solution_str` and every row scored 0.0 against the empty string.
    """
    import inspect

    from recipes.design.music import scorer

    params = list(inspect.signature(scorer.compute_score).parameters)
    assert params[:4] == ["data_source", "solution_str", "ground_truth", "extra_info"]


def test_music_missing_toolchain_raises_instead_of_scoring_zero(monkeypatch):
    """A missing abc2midi applies to every sample, not one score. Returning 0.0
    for it leaves GRPO with no variance and the run merely looks unsuccessful.
    """
    from recipes.design.music import scorer

    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(scorer.Abc2MidiMissing):
        scorer.compute_score("music", "```abc\nX:1\nK:C\nCDEF|\n```")


def test_music_rows_carry_a_dataset_index():
    """Without `extra_info.index` every row reads 0, so the rollout-trace cap
    can never bind and `rollout_n` counts across the whole flattened batch
    instead of resetting per prompt.
    """
    from recipes.design.music.build_parquet import to_verl_row

    row = to_verl_row({"text": "write a reel", "src_id": "x", "bpm": 120}, 7)
    assert row["extra_info"]["index"] == 7
    assert row["prompt"] == [{"role": "user", "content": "write a reel"}]


# --- web-dev: the tool catalogue is prompt text -----------------------------


def _webdev_tool_schema():
    from recipes.design.webdev.tools import WebdevToolRegistry

    registry = WebdevToolRegistry.from_config(
        [
            {"tool": "Bash", "config": {"timeout": 60, "max_timeout": 600}},
            {"tool": "Read"},
            {"tool": "Write"},
            {"tool": "Edit"},
            {"tool": "Grep"},
            {"tool": "Glob"},
        ]
    )
    return {
        definition["function"]["name"]: definition["function"]["description"]
        for definition in registry.get_function_definitions()
    }


def test_webdev_agent_registers_without_touching_mimoagent():
    """`_AGENT_LOADERS` is a plain dict, so the arm adds itself rather than
    being vendored in. Idempotent, because the environment actor registers
    lazily on every rollout.
    """
    from mimoagent.agents.factory import get_agent_class

    from recipes.design.webdev import AGENT_TYPE, WebdevAgent, register_webdev_agent

    register_webdev_agent()
    register_webdev_agent()
    assert get_agent_class(AGENT_TYPE) is WebdevAgent


def test_webdev_agent_keeps_the_default_loop():
    """Not `CCAgent`. It appends a `<context_usage>` footer to tool results past
    80% of its window and adds stray-`<tool_call>` detection to `step()`,
    neither of which the run being reproduced had.
    """
    from mimoagent.agents.default import DefaultAgent

    from recipes.design.webdev import WebdevAgent

    assert issubclass(WebdevAgent, DefaultAgent)
    assert "step" not in WebdevAgent.__dict__
    assert "query" not in WebdevAgent.__dict__
    assert "_build_tool_registry" in WebdevAgent.__dict__


def test_webdev_catalogue_is_exactly_six_tools():
    assert set(_webdev_tool_schema()) == {"Bash", "Read", "Write", "Edit", "Grep", "Glob"}


@pytest.mark.parametrize("tool_name", ["Agent", "Compact", "agent", "compact", "actor", "task"])
def test_webdev_rejects_conversation_tools(tool_name):
    """These need `agent`/`model` in the tool context, which the environment
    actor cannot supply because the agent runs in a different process. They also
    fork or rewrite the conversation, which the single append-only token buffer
    cannot represent.
    """
    from mimoagent.tools import ToolException

    from recipes.design.webdev.tools import CONVERSATION_TOOLS, WebdevToolRegistry

    assert tool_name in CONVERSATION_TOOLS
    with pytest.raises(ToolException):
        WebdevToolRegistry.from_config([{"tool": tool_name}])


def test_webdev_tool_descriptions_are_the_ported_ones():
    """Descriptions are prompt text the policy reads every turn, so they are
    ported verbatim rather than taken from `mimoagent.tools.cc`. These three
    sentences are exactly where the packaged catalogue has since diverged --
    314 bytes the reproduced run's policy never saw.
    """
    schema = _webdev_tool_schema()

    # Present in the ported Read, dropped from the packaged one when image
    # support became opt-in. For an arm that builds pages, "you can see the
    # rendered result" is a capability hint, not decoration.
    assert "returned as an actual image you can SEE" in schema["Read"]

    # Added to the packaged Bash and Edit after this run.
    concurrency_note = "runs concurrently"
    assert concurrency_note not in schema["Bash"]
    assert concurrency_note not in schema["Edit"]


def test_webdev_grep_has_no_silent_fallback_to_plain_grep():
    """rg and grep differ in output, ignore-file and regex semantics, so a
    fallback would train the model against a moving contract.
    """
    source = (RECIPE_ROOT / "webdev" / "tools" / "grep.py").read_text()
    assert "no silent fallback to plain" in source
    assert "WEBDEV_RG_PATH" in source


# --- CUDA forward compatibility: both arms, driver shell AND workers -------------------


@pytest.mark.parametrize("launcher", ["run_music.sh", "run_webdev.sh"])
def test_both_arms_source_cuda_compat_before_launching(launcher):
    """On a host whose driver predates the image's CUDA, torch imports fine and
    `torch.cuda.is_available()` is quietly False; the run dies much later at NCCL init
    with a message that names neither the driver nor the loader path. docker/cuda_compat.sh
    is the conditional fix and it has to be sourced before Ray starts."""
    src = (RECIPE_ROOT / launcher).read_text()
    assert "cuda_compat.sh" in src, f"{launcher} never sources the compat shim"

    # Forwarded too: sourcing fixes THIS shell, and the workers are the processes that
    # have to see a GPU. Exporting without forwarding is the trap PATH already documents.
    assert "runtime_env.env_vars.LD_LIBRARY_PATH=" in src, (
        f"{launcher} sources the shim but never forwards LD_LIBRARY_PATH to the workers"
    )

    # Before `ray start`, not after: a raylet already running would not inherit it.
    assert src.index("cuda_compat.sh") < src.index("MAIN_CMD="), (
        f"{launcher} sources the shim after assembling the command"
    )


def test_cuda_compat_is_conditional_and_safe_to_source_anywhere():
    """Unconditional prepending would trade today's broken hosts for tomorrow's: forward
    compat runs a NEW userspace libcuda against an OLD kernel driver, so once the host
    driver is ahead, loading it is the wrong direction. Must also no-op with no GPU."""
    src = (REPO_ROOT / "docker" / "cuda_compat.sh").read_text()
    assert "nvidia-smi" in src, "no driver probe, so the prepend cannot be conditional"
    assert "-lt" in src, "no version comparison between driver and compat lib"
