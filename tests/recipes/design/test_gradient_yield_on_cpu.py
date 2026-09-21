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
"""Gradient-yield metrics, and the guarantee that they cost other recipes nothing.

Two things are checked here. The first is the arithmetic: which rows produced no gradient,
and how that splits by infra failure. The second is a boundary condition on the three places
this arm touches verl's trainer -- each must sit behind an environment gate, so a run of any
other recipe executes none of it. That is a stated requirement, not a preference, which is
why it gets an assertion rather than a comment.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from recipes.design.webdev.gradient_yield import gradient_yield_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_BASE = REPO_ROOT / "verl/trainer/ppo/v1/trainer_base.py"


def _rows(adv_rows: list[list[float]], scores: list[float] | None = None):
    adv = torch.tensor(adv_rows, dtype=torch.float32)
    mask = torch.ones_like(adv)
    tls = None
    if scores is not None:
        tls = torch.zeros_like(adv)
        tls[:, -1] = torch.tensor(scores, dtype=torch.float32)
    return adv, mask, tls


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------


def test_a_row_is_dead_only_when_its_advantage_is_zero_everywhere():
    """ "Small" is not "dead": a tiny advantage still trains."""
    adv, mask, _ = _rows([[0.0, 0.0], [0.0, 1e-6], [0.5, -0.5], [0.0, 0.0]])
    m = gradient_yield_metrics(advantages=adv, response_mask=mask)
    assert m["training/adv/dead_rows"] == 2.0
    assert m["training/adv/dead_rows_ratio"] == pytest.approx(0.5)


def test_masked_tokens_cannot_keep_a_row_alive():
    """An advantage on a token outside the response mask is not trained on."""
    adv = torch.tensor([[0.0, 0.9], [0.0, 0.9]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    m = gradient_yield_metrics(advantages=adv, response_mask=mask)
    assert m["training/adv/dead_rows"] == 1.0, "row 0's only non-zero advantage is masked out"


def test_padding_rows_are_not_counted_as_dead():
    """Batch padding has no advantage by construction and would swamp the ratio."""
    adv, mask, _ = _rows([[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
    m = gradient_yield_metrics(advantages=adv, response_mask=mask, keep=np.array([True, True, False]))
    assert m["training/adv/dead_rows"] == 1.0
    assert m["training/adv/dead_rows_ratio"] == pytest.approx(0.5), "denominator excludes padding"


def test_dead_tokens_ratio_is_the_share_of_the_token_mean_denominator():
    """The reason dead rows matter: they dilute everyone else's gradient, in proportion to
    their token count rather than their row count."""
    adv = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    m = gradient_yield_metrics(advantages=adv, response_mask=mask)
    assert m["training/adv/dead_rows_ratio"] == pytest.approx(0.5)
    # One live token against four dead ones: half the rows, four fifths of the compute.
    assert m["training/adv/dead_tokens_ratio"] == pytest.approx(0.8)


def test_the_infra_split_is_what_makes_the_count_actionable():
    """An infra row at zero is intended; a non-infra row at zero means a group lost its
    spread, passed the sampling filter and trained on nothing."""
    adv, mask, _ = _rows([[0.0, 0.0], [0.0, 0.0], [0.4, -0.4], [0.0, 0.0]])
    m = gradient_yield_metrics(advantages=adv, response_mask=mask, is_infra=np.array([1.0, 0.0, 0.0, 1.0]))
    assert m["training/adv/dead_rows"] == 3.0
    assert m["training/adv/dead_rows_non_infra"] == 1.0, "only row 1 is a real problem"
    assert m["training/valid_rate"] == pytest.approx(0.5)


def test_no_infra_score_mean_undoes_the_deflation_infra_rows_cause():
    """Infra rows score 0.0 and sit inside score/mean, so a bad pod rate reads as a worse
    policy. Reproduces the measured arithmetic: 0.6387 * (512-166)/512 == 0.4317."""
    n, n_infra = 512, 166
    scores = [0.0] * n_infra + [0.6387] * (n - n_infra)
    adv, mask, tls = _rows([[0.1, -0.1]] * n, scores=scores)
    infra = np.array([1.0] * n_infra + [0.0] * (n - n_infra))

    naive = float(tls.sum(dim=-1).mean())
    m = gradient_yield_metrics(advantages=adv, response_mask=mask, token_level_scores=tls, is_infra=infra)
    assert naive == pytest.approx(0.4317, abs=1e-4), "what score/mean would report"
    assert m["training/no_infra/score_mean"] == pytest.approx(0.6387, abs=1e-4), "the policy"


def test_no_flags_yields_only_the_counts_that_do_not_need_them():
    adv, mask, _ = _rows([[0.0, 0.0], [0.3, 0.3]])
    m = gradient_yield_metrics(advantages=adv, response_mask=mask)
    assert "training/adv/dead_rows" in m
    assert "training/adv/dead_rows_non_infra" not in m
    assert "training/valid_rate" not in m


@pytest.mark.parametrize(
    "kwargs",
    [
        {"advantages": torch.zeros(0, 4), "response_mask": torch.zeros(0, 4)},
        {
            "advantages": torch.zeros(2, 4),
            "response_mask": torch.zeros(2, 4),
            "keep": np.array([False, False]),
        },
        {
            "advantages": torch.zeros(2, 4),
            "response_mask": torch.ones(2, 4),
            "is_infra": np.array([0.0, 0.0, 0.0]),  # wrong length
        },
    ],
)
def test_degenerate_inputs_return_fewer_keys_rather_than_raising(kwargs):
    """A metric helper that can kill a training step is worse than a missing metric."""
    m = gradient_yield_metrics(**kwargs)
    assert isinstance(m, dict)
    assert "training/adv/dead_rows_non_infra" not in m


# ---------------------------------------------------------------------------
# the boundary condition: other recipes must execute none of this
# ---------------------------------------------------------------------------


def test_every_touchpoint_in_the_trainer_is_behind_an_environment_gate():
    """Three places in verl's trainer serve this arm. Each must be gated.

    Without a gate the cost is not merely wasted work: a recipe that stores its
    ``extra_fields`` differently would take the fail-open branch and log a warning on every
    single step, which is a worse failure than not having the feature at all.

    Matching on source text is crude, but the alternative -- constructing the trainer -- needs
    a transfer-queue backend that is not installed, and this is precisely the kind of
    requirement that gets quietly dropped in a later refactor.
    """
    src = TRAINER_BASE.read_text()

    # The group rewrite and the gradient-yield metrics: gated on the launcher's mode var.
    calls = [
        "self._rewrite_webdev_group_rewards(",
        "from recipes.design.webdev.gradient_yield import gradient_yield_metrics",
    ]
    for call in calls:
        idx = src.index(call)
        preceding = src[:idx]
        gate = preceding.rindex('os.environ.get("WEBDEV_GRADE_MODE")')
        between = preceding[gate:]
        assert between.count("\n") <= 12, f"{call!r} is no longer directly under its WEBDEV_GRADE_MODE gate"

    # The infra exclusion changes advantages, so it carries its own explicit opt-in.
    assert 'os.environ.get("DROP_INFRA_FROM_GROUP", "0") == "1"' in src

    # And no other recipe-specific import may be unconditional at module scope.
    module_scope_imports = re.findall(r"^from recipes\..*$", src, re.M)
    assert module_scope_imports == [], (
        f"recipe imports must stay inside the gated blocks, found: {module_scope_imports}"
    )


def test_the_helper_needs_neither_verl_nor_mimoagent():
    """It is called from verl's trainer, so an import cycle here would be a runtime failure
    on the first step rather than at collection."""
    src = (REPO_ROOT / "recipes/design/webdev/gradient_yield.py").read_text()
    assert "import verl" not in src and "from verl" not in src
    assert "mimoagent" not in src
