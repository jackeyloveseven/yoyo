"""PPO and reference-KL objectives for Flow-GRPO updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PPOMetrics:
    """Scalar losses and per-step PPO diagnostics."""

    loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    surrogate: Tensor
    ratio: Tensor
    clipped_ratio: Tensor
    per_step_kl: Tensor
    sampled_log_ratio: Tensor | None
    clip_fraction: Tensor
    approx_kl: Tensor
    valid_fraction: Tensor
    clipping_enabled: bool


@dataclass(frozen=True)
class PPOInputs:
    old_logprobs: Tensor
    new_logprobs: Tensor
    advantages: Tensor
    reference_kl: Tensor
    ref_logprobs: Tensor | None = None
    deterministic_mask: Tensor | None = None
    valid_mask: Tensor | None = None


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float | None = None
    kl_beta: float = 0.01


def _broadcast_advantages(advantages: Tensor, target: Tensor) -> Tensor:
    if advantages.shape == target.shape:
        return advantages
    if (
        advantages.ndim == 2
        and target.ndim == 3
        and tuple(advantages.shape) == tuple(target.shape[:2])
    ):
        return advantages.unsqueeze(-1).expand_as(target)
    raise ValueError(
        "advantages must have shape [batch, group] or match per-step logprob shape; "
        f"got {tuple(advantages.shape)} and target {tuple(target.shape)}"
    )


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=value.dtype)
    denominator = weights.sum().clamp_min(1)
    return (value * weights).sum() / denominator


def _validate_ppo(inputs: PPOInputs, config: PPOConfig) -> None:
    shape = inputs.old_logprobs.shape
    if inputs.new_logprobs.shape != shape or inputs.reference_kl.shape != shape:
        raise ValueError("old_logprobs, new_logprobs, and reference_kl must match")
    if inputs.ref_logprobs is not None and inputs.ref_logprobs.shape != shape:
        raise ValueError("ref_logprobs must match old_logprobs shape")
    if inputs.old_logprobs.ndim != 3:
        raise ValueError("per-step logprobs must have shape [batch, group, step]")
    if not inputs.old_logprobs.is_floating_point():
        raise TypeError("logprobs must use a floating dtype")
    if config.clip_epsilon is not None and not 0 < config.clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1) when supplied")
    if config.kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")


def _resolve_masks(inputs: PPOInputs) -> Tensor:
    shape = inputs.old_logprobs.shape
    deterministic = inputs.deterministic_mask
    if deterministic is None:
        deterministic = torch.zeros_like(inputs.old_logprobs, dtype=torch.bool)
    elif deterministic.dtype != torch.bool or deterministic.shape != shape:
        raise ValueError("deterministic_mask must be bool and match logprob shape")
    valid = inputs.valid_mask
    if valid is None:
        valid = torch.ones_like(inputs.old_logprobs, dtype=torch.bool)
    elif valid.dtype != torch.bool or valid.shape != shape:
        raise ValueError("valid_mask must be bool and match logprob shape")
    return valid & ~deterministic


def _ppo_surrogate(
    ratio: Tensor,
    advantages: Tensor,
    clip_epsilon: float | None,
) -> tuple[Tensor, Tensor, bool]:
    unclipped = ratio * advantages
    if clip_epsilon is None:
        return unclipped, ratio, False
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantages
    return torch.where(clipped < unclipped, clipped, unclipped), clipped_ratio, True


def ppo_clipped_surrogate(
    inputs: PPOInputs,
    config: PPOConfig | None = None,
) -> PPOMetrics:
    """Compute a per-step PPO surrogate plus a supplied reference KL.

    ``reference_kl`` must already be the closed-form Gaussian mean KL, such as
    :func:`gaussian_mean_kl`. ``ref_logprobs`` is accepted only to expose a
    sampled log-ratio diagnostic; ``new_logprobs - ref_logprobs`` is not used
    as the main KL term.

    ``clip_epsilon=None`` disables clipping explicitly. This is useful because
    SenseNova does not publish its PPO clip epsilon; callers should provide an
    explicit value before enabling the clipped objective.
    """

    config = config or PPOConfig()
    _validate_ppo(inputs, config)
    valid_mask = _resolve_masks(inputs)
    old_logprobs = inputs.old_logprobs
    compute_dtype = (
        torch.float32
        if old_logprobs.dtype in (torch.float16, torch.bfloat16)
        else old_logprobs.dtype
    )
    old_compute = old_logprobs.to(dtype=compute_dtype)
    new_compute = inputs.new_logprobs.to(device=old_logprobs.device, dtype=compute_dtype)
    reference_kl_compute = inputs.reference_kl.to(device=old_logprobs.device, dtype=compute_dtype)
    expanded_advantages = _broadcast_advantages(inputs.advantages, old_logprobs).to(
        dtype=compute_dtype
    )
    log_ratio = new_compute - old_compute
    ratio = torch.exp(log_ratio.clamp(min=-60, max=60))
    surrogate, clipped_ratio, clipping_enabled = _ppo_surrogate(
        ratio, expanded_advantages, config.clip_epsilon
    )

    policy_loss = -_masked_mean(surrogate, valid_mask)
    kl_loss = config.kl_beta * _masked_mean(reference_kl_compute, valid_mask)
    loss = policy_loss + kl_loss
    clip_fraction = _masked_mean((ratio != clipped_ratio).to(dtype=old_logprobs.dtype), valid_mask)
    approx_kl = 0.5 * _masked_mean(log_ratio.square(), valid_mask)
    valid_fraction = valid_mask.to(dtype=compute_dtype).mean()
    sampled_log_ratio = (
        None
        if inputs.ref_logprobs is None
        else new_compute - inputs.ref_logprobs.to(device=old_logprobs.device, dtype=compute_dtype)
    )

    return PPOMetrics(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        surrogate=surrogate,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        per_step_kl=reference_kl_compute,
        sampled_log_ratio=sampled_log_ratio,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
        valid_fraction=valid_fraction,
        clipping_enabled=clipping_enabled,
    )
