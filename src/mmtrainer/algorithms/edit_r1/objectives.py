"""Pure reward shaping and low-variance filtering for Edit-R1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..common import require_float_finite
from .config import EditR1Config


def expected_logit_score(
    score_logits: Tensor,
    *,
    score_values: Sequence[float] = (0, 1, 2, 3, 4, 5),
) -> Tensor:
    """Convert score-token logits to the normalized non-CoT Edit-R1 reward."""

    require_float_finite("score_logits", score_logits)
    if score_logits.ndim < 1 or score_logits.shape[-1] != len(score_values):
        raise ValueError("score_logits final dimension must match score_values")
    values = torch.as_tensor(tuple(score_values), device=score_logits.device, dtype=torch.float32)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("score_values must be a finite one-dimensional sequence")
    low, high = values.min(), values.max()
    if not bool((high > low).item()):
        raise ValueError("score_values must span a non-zero range")
    probabilities = torch.softmax(score_logits.float(), dim=-1)
    expectation = (probabilities * values).sum(dim=-1)
    normalized = (expectation - low) / (high - low)
    if not torch.isfinite(normalized).all():
        raise ValueError("expected Edit-R1 score is non-finite")
    return normalized


@dataclass(frozen=True)
class EditR1Optimality:
    """Optimality probabilities plus the paper's group-filtering diagnostics."""

    probabilities: Tensor
    centered_rewards: Tensor
    normalizer: Tensor
    advantages: Tensor
    sample_weight: Tensor
    group_mean: Tensor
    group_std: Tensor
    discarded_groups: Tensor

    @property
    def kept_group_fraction(self) -> Tensor:
        return (~self.discarded_groups).float().mean()


def edit_r1_optimality(
    raw_rewards: Tensor,
    config: EditR1Config | None = None,
) -> EditR1Optimality:
    """Map normalized MLLM rewards to DiffusionNFT weights and a group mask.

    The paper discards a group only when its mean is high *and* its standard
    deviation is low. This prevents tiny MLLM logit differences among already
    successful edits from being amplified by reward normalization.
    """

    config = config or EditR1Config()
    require_float_finite("raw_rewards", raw_rewards)
    if raw_rewards.ndim != 2:
        raise ValueError("raw_rewards must have shape [batch, group]")
    if raw_rewards.shape[1] != config.group_size:
        raise ValueError(
            f"raw_rewards group size must be {config.group_size}, got {raw_rewards.shape[1]}"
        )
    work = raw_rewards.float()
    if bool(((work < 0) | (work > 1)).any()):
        raise ValueError("Edit-R1 MLLM rewards must be normalized to [0, 1]")
    group_mean = work.mean(dim=1, keepdim=True)
    group_std = work.std(dim=1, correction=0, keepdim=True)
    centered = work - group_mean
    if config.advantage_normalizer == "global_std":
        normalizer = work.std(correction=0).reshape(1, 1)
    else:
        normalizer = group_std
    normalizer = torch.broadcast_to(normalizer + config.normalizer_epsilon, work.shape)
    advantages = centered / normalizer
    clipped = advantages.clamp(-config.advantage_clip, config.advantage_clip)
    probabilities = 0.5 + clipped / (2.0 * config.advantage_clip)
    discarded = (
        (group_mean > config.high_mean_threshold) & (group_std < config.low_std_threshold)
        if config.filter_low_variance_groups
        else torch.zeros_like(group_mean, dtype=torch.bool)
    )
    sample_weight = (~discarded).expand_as(work).to(torch.float32)
    probabilities = torch.where(
        sample_weight.bool(), probabilities, torch.full_like(probabilities, 0.5)
    )
    diagnostics = (
        probabilities,
        centered,
        normalizer,
        advantages,
        sample_weight,
        group_mean,
        group_std,
    )
    if not all(torch.isfinite(value).all() for value in diagnostics):
        raise ValueError("Edit-R1 optimality diagnostics must be finite")
    return EditR1Optimality(
        probabilities=probabilities,
        centered_rewards=centered,
        normalizer=normalizer,
        advantages=advantages,
        sample_weight=sample_weight,
        group_mean=group_mean,
        group_std=group_std,
        discarded_groups=discarded,
    )


__all__ = ["EditR1Optimality", "edit_r1_optimality", "expected_logit_score"]
