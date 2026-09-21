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
"""Prompt-group normalization across unequal episodes and distributed splits."""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_prompt_loss_weights
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss

PROMPTS = ["a", "b", "a", "b", "b", "padding"]
MASK = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.bool,
)


def reference_prompt_mean(loss_mat, mask, prompt_ids):
    means = []
    for uid in sorted(set(prompt_ids)):
        indices = [i for i, name in enumerate(prompt_ids) if name == uid]
        group_mask = mask[indices]
        if group_mask.any():
            means.append(loss_mat[indices][group_mask].mean())
    return torch.stack(means).mean()


def test_unequal_groups_and_action_masks_match_document_formula():
    loss = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    weights = compute_prompt_loss_weights(MASK, PROMPTS)
    actual = agg_loss(loss, MASK, "prompt-mean", prompt_loss_weights=weights)
    torch.testing.assert_close(actual, reference_prompt_mean(loss, MASK, PROMPTS))
    torch.testing.assert_close((weights * MASK.sum(-1)).sum(), torch.tensor(1.0, dtype=torch.float64))
    assert weights[0] == weights[2]
    assert weights[1] == weights[3] == weights[4]
    assert weights[5] == 0
    # Excluded tool/context/padding positions must not affect the objective.
    altered = loss.masked_fill(~MASK, 1e9)
    torch.testing.assert_close(agg_loss(altered, MASK, "prompt-mean", prompt_loss_weights=weights), actual)


def test_prompt_mean_is_not_sequence_mean_or_batch_token_mean():
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    loss = torch.tensor([[1.0, 0, 0], [3.0, 3, 3], [8.0, 0, 0]])
    weights = compute_prompt_loss_weights(mask, ["a", "a", "b"])
    actual = agg_loss(loss, mask, "prompt-mean", prompt_loss_weights=weights)
    assert actual.item() == pytest.approx((2.5 + 8) / 2)
    assert actual.item() != pytest.approx(agg_loss(loss, mask, "seq-mean-token-mean").item())
    assert actual.item() != pytest.approx(agg_loss(loss, mask, "token-mean").item())


@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("micro_size", [1, 2])
def test_loss_and_gradients_are_invariant_to_dp_microbatches_and_reordering(dp_size, micro_size):
    values = torch.arange(36, dtype=torch.float64).reshape(6, 6).requires_grad_()
    weights = compute_prompt_loss_weights(MASK, PROMPTS)
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    rank_losses = []
    for shard in order.chunk(dp_size):
        rank_loss = 0
        for indices in shard.split(micro_size):
            rank_loss = rank_loss + agg_loss(
                values[indices],
                MASK[indices],
                "prompt-mean",
                dp_size=dp_size,
                prompt_loss_weights=weights[indices],
            )
        rank_losses.append(rank_loss)
    reduced = torch.stack(rank_losses).mean()
    reference = reference_prompt_mean(values, MASK, PROMPTS)
    torch.testing.assert_close(reduced, reference)
    torch.testing.assert_close(torch.autograd.grad(reduced, values)[0], torch.autograd.grad(reference, values)[0])


def test_missing_weights_and_empty_batch_fail_closed():
    with pytest.raises(ValueError, match="global prompt_loss_weights"):
        agg_loss(torch.ones(6, 6), MASK, "prompt-mean")
    with pytest.raises(ValueError, match="at least one prompt"):
        compute_prompt_loss_weights(torch.zeros(2, 3), ["a", "padding"])
    with pytest.raises(ValueError, match="Prompt IDs"):
        compute_prompt_loss_weights(MASK, ["a"])


@pytest.mark.parametrize("dp_size", [1, 2, 3])
def test_ppo_pipeline_uses_same_prompt_weights_for_policy_entropy_and_kl(dp_size):
    log_probs = torch.linspace(-1.0, 0.5, 36, dtype=torch.float64).reshape(6, 6).requires_grad_()
    old_log_probs = torch.full_like(log_probs, -0.6)
    advantages = torch.tensor([0.5, -0.2, -0.5, 0.8, -0.2, 0.0], dtype=torch.float64)[:, None].expand(-1, 6)
    weights = compute_prompt_loss_weights(MASK, PROMPTS)
    cfg = ActorConfig(
        strategy="megatron",
        rollout_n=16,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="prompt-mean",
        entropy_coeff=0.03,
        use_kl_loss=True,
        kl_loss_coef=0.02,
        kl_loss_type="kl",
    )
    rank_losses = []
    for indices in torch.tensor([4, 1, 5, 0, 3, 2]).chunk(dp_size):
        rank_loss = 0
        for index in indices:
            i = int(index)
            lp = log_probs[i : i + 1]
            ent = 1 - 0.2 * lp
            data = TensorDict(
                {
                    "prompts": torch.zeros(1, 1, dtype=torch.long),
                    "responses": torch.zeros(1, 6, dtype=torch.long),
                    "attention_mask": torch.ones(1, 7, dtype=torch.long),
                    "response_mask": MASK[i : i + 1],
                    "old_log_probs": old_log_probs[i : i + 1],
                    "advantages": advantages[i : i + 1],
                    "ref_log_prob": torch.full_like(lp, -0.7),
                    "prompt_loss_weights": weights[i : i + 1],
                },
                batch_size=[1],
            )
            tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=int(MASK.sum()), global_batch_size=6)
            output = {
                "log_probs": torch.cat([lp.flatten(), lp.new_zeros(1)]),
                "entropy": torch.cat([ent.flatten(), ent.new_zeros(1)]),
            }
            part, _ = ppo_loss(cfg, output, data)
            rank_loss = rank_loss + part
        rank_losses.append(rank_loss)
    actual = torch.stack(rank_losses).mean()
    ratio = (log_probs - old_log_probs).exp()
    clipped = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    policy = torch.where(advantages < 0, torch.minimum(clipped, -advantages * 3), clipped)
    combined = policy - 0.03 * (1 - 0.2 * log_probs) + 0.02 * (log_probs + 0.7)
    expected = reference_prompt_mean(combined, MASK, PROMPTS)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, log_probs)[0], torch.autograd.grad(expected, log_probs)[0])
