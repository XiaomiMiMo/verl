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
"""Contract tests for the evaluation mode. No pod, no judge call, no network.

Each of the three things pinned here fails silently if it is wrong -- nothing raises, the
score just quietly becomes a different number:

1. **The scoring formula.** ``score = mean(visual, query_fulfillment, premium_assets)`` with
   ``visual = mean`` of five dimensions. Getting a weight wrong throws no exception; it
   invalidates every published figure.
2. **A judge that omits a dimension is a parse failure, not a zero on that dimension.**
   Defaulting to 0 invents a low score out of nothing.
3. **A render or judge failure is ``reward=None`` (masked), never 0.0.** Averaging infra
   failures as zeros misreads the mean: one measured run had 57.9% judge failures and
   reported 0.351 for an arm whose real mean was 0.677, which reversed the conclusion.

A fourth is pinned here too: the rendered prompt's hash. The band descriptions ARE the
scoring ruler, so an accidental edit has to fail loudly rather than shift the distribution.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest

from recipes.design.webdev import eval_mode as em
from recipes.design.webdev import eval_rubric as rb

# The prompt handed to the judge by the run these numbers come from. Not a checksum of our
# source -- the sha256 of the rendered string, so reformatting the module cannot drift it and
# editing a single band character cannot pass.
REFERENCE_PROMPT_SHA256 = "a4d3be63029e8fb28b469bf3d188816a7fa238b415749d7ff4aad1ca360b2997"


def _verdict(**over) -> str:
    d = {k: 0.8 for k in rb.ALL_KEYS}
    d["reason"] = "ok"
    d.update(over)
    return json.dumps(d)


class _Env:
    """Pod stand-in: ``execute`` returns the stdout line format ``shot.py`` agrees on."""

    def __init__(self, output="", reason="ok"):
        self.output, self.reason, self.cmds = output, reason, []

    def execute(self, cmd, cwd="/", timeout=90):
        self.cmds.append(cmd)
        return {"reason": self.reason, "output": self.output}


def _shot_output(jpeg_b64: str, console="[]", render_env=None) -> str:
    lines = [f"SHOT_B64:{jpeg_b64}", f"CONSOLE_ERRORS:{console}"]
    if render_env is not None:
        lines.append("RENDER_ENV:" + json.dumps(render_env))
    return "\n".join(lines) + "\n"


def _jpeg_b64(w=64, h=64) -> str:
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------- the ruler ----


def test_the_rendered_prompt_is_byte_for_byte_the_reference_runs():
    """The bands are not documentation of a policy; they are the policy.

    If this fails, either restore the wording or -- if the change is intended -- copy the
    file, change RUBRIC_ID, and update this hash, so that later analysis can still tell which
    ruler measured which batch.
    """
    prompt = rb.build_prompt()
    assert hashlib.sha256(prompt.encode()).hexdigest() == REFERENCE_PROMPT_SHA256
    assert rb.RUBRIC_ID == "rva1:mean(visual,query,asset)"


def test_prompt_has_no_leftover_placeholders_and_names_every_dimension():
    p = rb.build_prompt()
    assert "{query}" in p, "the caller formats this one in"
    for tok in ("{nq}", "{na}", "{richness}", "{dims}"):
        assert tok not in p, f"unsubstituted placeholder: {tok}"
    for k in rb.ALL_KEYS:
        assert k in p, f"the prompt never mentions dimension {k}"
    rendered = p.format(query="X")
    assert rendered.count("{") == rendered.count("}")


# ------------------------------------------------------------- the formula ----


def test_score_is_the_equal_weight_mean_of_three():
    v = em.parse_verdict(
        _verdict(
            layout_integrity=1.0,
            typography_hierarchy=1.0,
            color_harmony=1.0,
            whitespace=1.0,
            content_richness=0.0,
            query_fulfillment=0.5,
            premium_assets=0.2,
        )
    )
    assert v["visual"] == pytest.approx(0.8)  # (1+1+1+1+0)/5
    assert v["score"] == pytest.approx((0.8 + 0.5 + 0.2) / 3)


def test_visual_is_the_mean_of_five_dims_not_four():
    """``content_richness`` must count: it is the dimension that stops a one-screen shell
    from scoring well on the other four."""
    lo = em.parse_verdict(_verdict(content_richness=0.0))["visual"]
    hi = em.parse_verdict(_verdict(content_richness=1.0))["visual"]
    assert hi > lo
    assert hi - lo == pytest.approx(1.0 / len(rb.VISUAL_KEYS))


def test_asset_weighs_as_much_as_visual():
    """Equal weighting, checked behaviourally: moving premium_assets by 0.3 must move the
    total exactly as much as moving all of visual by 0.3.

    This is the point of the change from the earlier 0.6 / 0.1 / 0.2 -- premium_assets is the
    dimension that separates the arms most, and a 0.6-weighted visual was drowning it out.
    """
    base = em.parse_verdict(_verdict())["score"]
    by_asset = em.parse_verdict(_verdict(premium_assets=0.5))["score"]
    by_visual = em.parse_verdict(_verdict(**{k: 0.5 for k in rb.VISUAL_KEYS}))["score"]
    assert base - by_asset == pytest.approx(base - by_visual)


@pytest.mark.parametrize("bad", [-1.0, 2.0])
def test_out_of_range_is_clamped(bad):
    v = em.parse_verdict(_verdict(premium_assets=bad))
    assert 0.0 <= v["dims"]["premium_assets"] <= 1.0


# ------------------------------------------- a parse failure is not a score ----


@pytest.mark.parametrize("missing", rb.ALL_KEYS)
def test_a_missing_dimension_fails_instead_of_scoring_zero(missing):
    d = json.loads(_verdict())
    del d[missing]
    v = em.parse_verdict(json.dumps(d))
    assert "_failed" in v and missing in v["_failed"]
    assert "score" not in v, "a verdict short one dimension must not yield a score at all"


@pytest.mark.parametrize("raw", ["", "not json at all", "{broken", '{"layout_integrity": "abc"}'])
def test_unparseable_judge_output_fails(raw):
    assert "_failed" in em.parse_verdict(raw)


# ------------------------------------ infra failures are masked, never zero ----


def test_pod_exec_failure_is_masked():
    out = em.grade_eval(_Env(reason="timeout"), "/workspace/dist", "q", {})
    assert out["grader_reward"] is None
    assert out["grading"]["verdict"] == "drop"


def test_empty_screenshot_is_masked():
    out = em.grade_eval(_Env(_shot_output("")), "/workspace/dist", "q", {})
    assert out["grader_reward"] is None
    assert "no image" in out["grading"]["drop_reason"]


def test_dead_proxy_chain_is_masked_before_empty_shot():
    """With every proxy route dead the pod may produce no image at all. That ordering
    matters: checked second, it would read as a broken page and score 0."""
    env = _Env(_shot_output("", render_env={"proxy_failed": True, "fatal": ["cdn timeout"]}))
    out = em.grade_eval(env, "/workspace/dist", "q", {})
    assert out["grader_reward"] is None
    assert "render env failed" in out["grading"]["drop_reason"]


def test_judge_failure_is_masked_not_zero(monkeypatch):
    monkeypatch.setattr(em, "_judge", lambda *a, **k: ({"_failed": "boom"}, 4))
    out = em.grade_eval(_Env(_shot_output(_jpeg_b64())), "/workspace/dist", "q", {})
    assert out["grader_reward"] is None
    assert "judge" in out["grading"]["drop_reason"]


def test_a_misconfigured_judge_is_masked_rather_than_scored(monkeypatch):
    """No judge model and no endpoint is a launcher error that would otherwise affect every
    row. Dropping it keeps a whole evaluation from reporting zeros as if they were scores."""
    for var in (
        "WEBDEV_EVAL_JUDGE_MODEL",
        "WEBDEV_EVAL_JUDGE_BASE_URL",
        "LLM_JUDGE_BASE_URL",
        "GRADER_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    out = em.grade_eval(_Env(_shot_output(_jpeg_b64())), "/workspace/dist", "q", {})
    assert out["grader_reward"] is None
    assert "judge" in out["grading"]["drop_reason"]


def test_happy_path_returns_the_score_and_the_shot(monkeypatch):
    monkeypatch.setattr(em, "_judge", lambda *a, **k: (em.parse_verdict(_verdict()), 1))
    b64 = _jpeg_b64()
    out = em.grade_eval(_Env(_shot_output(b64)), "/workspace/dist", "build a shop", {})
    assert out["grader_reward"] == pytest.approx(0.8)
    assert out["grading"]["rubric"] == rb.RUBRIC_ID
    assert out["shot_jpg_b64"] == b64
    assert out["grading"]["mode"] == em.MODE


# ------------------------------------------------------- oversized shots ----


def test_oversized_shot_is_downscaled_keeping_aspect():
    """The judge refuses very large images permanently -- 23 MP accepted, 46.5 MP rejected --
    so shots are scaled before being sent.

    Proportionally rather than to a fixed width: a page that overflows horizontally has a
    width that is itself the defect ``layout_integrity`` exists to catch.
    """
    Image = pytest.importorskip("PIL.Image")
    Image.MAX_IMAGE_PIXELS = None
    big = _jpeg_b64(6000, 6000)  # 36 MP, over the 20 MP default
    out, shrunk = em._shrink(big)
    assert shrunk
    im = Image.open(io.BytesIO(base64.b64decode(out)))
    assert im.width * im.height <= em.MAX_MEGAPIXELS * 1e6
    assert im.width == im.height, "aspect ratio preserved"
    small = _jpeg_b64(64, 64)
    assert em._shrink(small) == (small, False), "under the limit, not re-encoded"


# ------------------------------------------------------------- dispatch ----


def test_grade_dispatches_the_eval_mode(monkeypatch):
    from recipes.design.webdev import grade as grade_mod

    seen = {}

    def _fake(env, deliver_dir, query, cfg):
        seen["cfg"] = cfg
        return {"grader_reward": 1.0}

    monkeypatch.setattr(em, "grade_eval", _fake)
    out = grade_mod.grade(_Env(), "/workspace/dist", "q", {"correctness_mode": "webdev_eval_v1"})
    assert out["grader_reward"] == 1.0
    assert seen["cfg"]["correctness_mode"] == "webdev_eval_v1"


def test_eval_mode_is_not_a_training_mode():
    """The two paths must not blur: the training grader serves only the group mode, and the
    evaluation grader only the evaluation mode."""
    from recipes.design.webdev import design_mode as dm
    from recipes.design.webdev import grade as grade_mod

    assert em.MODE in grade_mod.EVAL_MODES
    assert em.MODE not in grade_mod.DESIGN_MODES
    assert em.MODE not in dm.MODES
    out = dm.grade_design(_Env(), "/workspace/dist", "q", {"correctness_mode": em.MODE})
    assert out["grader_reward"] is None, "refused, rather than graded with the wrong ruler"


def test_the_shot_script_is_carried_verbatim():
    """Every non-obvious line in the in-pod render script encodes a diagnosed failure, and
    all of those failures arrive as a plausible screenshot that scores badly."""
    from recipes.design.webdev import shot

    assert hashlib.sha256(shot._SHOT_SCRIPT.encode()).hexdigest().startswith("a36f4aa3f1c02d08")
    # The loopback bypass must ride the launch arg, not playwright's dict key, which this
    # chromium build silently ignores -- that cost 100% blank shots once. (The script does
    # mention the word in a comment saying exactly this, so match the dict-key form.)
    assert "--proxy-bypass-list=127.0.0.1;localhost" in shot._SHOT_SCRIPT
    assert '"bypass":' not in shot._SHOT_SCRIPT


def test_http_rendering_is_what_the_launcher_turns_on(monkeypatch):
    """file:// blanks every runtime-rendered page, so the launcher serves dist over
    loopback http. Scores across that boundary are not comparable."""
    from recipes.design.webdev import shot

    monkeypatch.setenv("WEBDEV_GRADE_HTTP", "1")
    cmd = shot._build_shot_cmd("file:///workspace/dist/index.html", proxy="")
    assert "--serve-root" in cmd and "/workspace/dist" in cmd

    monkeypatch.setenv("WEBDEV_GRADE_HTTP", "0")
    assert "--serve-root" not in shot._build_shot_cmd("file:///workspace/dist/index.html")
