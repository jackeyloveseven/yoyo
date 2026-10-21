"""Group-relative advantage calculations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdvantageResult:
    """Normalized group advantages and diagnostics."""

    advantages: Tensor
    group_mean: Tensor
    group_std: Tensor
    zero_variance_mask: Tensor


def group_relative_advantages(
    rewards: Tensor,
    *,
    std_floor: float = 1e-8,
    clip_value: float | None = None,
) -> AdvantageResult:
    """Normalize rewards relative to each prompt's rollout group.

    ``rewards`` must have shape ``[batch, group]``. Groups whose standard
    deviation is at or below ``std_floor`` receive zero advantages and are
    marked in ``zero_variance_mask``. ``clip_value`` clips normalized values
    symmetrically when supplied.
    """

    if not isinstance(rewards, Tensor) or not rewards.is_floating_point():
        raise TypeError("rewards must be a floating-point torch.Tensor")
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [batch, group], got {tuple(rewards.shape)}")
    if std_floor <= 0:
        raise ValueError("std_floor must be positive")
    if clip_value is not None and clip_value <= 0:
        raise ValueError("clip_value must be positive when supplied")

    compute_rewards = (
        rewards.float() if rewards.dtype in (torch.float16, torch.bfloat16) else rewards
    )
    group_mean = compute_rewards.mean(dim=1)
    group_std = compute_rewards.std(dim=1, unbiased=False)
    zero_variance_mask = group_std <= std_floor
    denominator = group_std.clamp_min(std_floor)
    advantages = (compute_rewards - group_mean.unsqueeze(1)) / denominator.unsqueeze(1)
    advantages = torch.where(
        zero_variance_mask.unsqueeze(1), torch.zeros_like(advantages), advantages
    )
    if clip_value is not None:
        advantages = advantages.clamp(min=-clip_value, max=clip_value)

    return AdvantageResult(
        advantages=advantages,
        group_mean=group_mean,
        group_std=group_std,
        zero_variance_mask=zero_variance_mask,
    )


compute_group_advantages = group_relative_advantages

__all__ = ["AdvantageResult", "compute_group_advantages", "group_relative_advantages"]
