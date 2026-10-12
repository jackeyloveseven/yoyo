"""Model-independent AWM and RAM objectives for rectified-flow post-training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import (
    EventDims,
    normalize_event_dims,
    require_float_finite,
    require_matching_tensors,
    working_dtype,
)


def _require_finite_scalar(name: str, value: float, *, minimum: float = 0.0) -> None:
    candidate = torch.tensor(value, dtype=torch.float64)
    if not torch.isfinite(candidate) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _leading_shape(value: Tensor, dims: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(size for index, size in enumerate(value.shape) if index not in dims)


def _prepare_leading_tensor(
    name: str,
    value: Tensor,
    *,
    reference: Tensor,
    dims: tuple[int, ...],
) -> Tensor:
    require_float_finite(name, value)
    expected = _leading_shape(reference, dims)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have leading shape {expected}, got {tuple(value.shape)}")
    if value.device != reference.device:
        raise ValueError(f"{name} and velocity tensors must be on the same device")
    return value.to(dtype=working_dtype(reference, value))


def _event_mse(
    prediction: Tensor,
    target: Tensor,
    *,
    dims: tuple[int, ...],
) -> Tensor:
    dtype = working_dtype(prediction, target)
    error = (prediction.to(dtype) - target.to(dtype)).square()
    return error.mean(dim=dims) if dims else error


def _apply_sample_weight(
    per_sample: Tensor,
    sample_weight: Tensor | None,
    *,
    reference: Tensor,
    dims: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    if sample_weight is None:
        weight = torch.ones_like(per_sample)
    else:
        weight = _prepare_leading_tensor(
            "sample_weight", sample_weight, reference=reference, dims=dims
        )
        if (weight < 0).any():
            raise ValueError("sample_weight must be non-negative")
    return per_sample * weight, weight


def normalized_advantages(
    centered_rewards: Tensor,
    normalizer: Tensor,
    zero_variance_mask: Tensor,
    *,
    clip_value: float | None = None,
) -> Tensor:
    """Recover group/global normalized advantages from an ``OptimalityResult``.

    The separate arguments keep this helper independent of the DiffusionNFT
    dataclass while preserving the exact normalizer selected by the rollout.
    """

    require_float_finite("centered_rewards", centered_rewards)
    require_float_finite("normalizer", normalizer)
    if centered_rewards.shape != normalizer.shape:
        raise ValueError("centered_rewards and normalizer must have matching shapes")
    if centered_rewards.device != normalizer.device:
        raise ValueError("centered_rewards and normalizer must be on the same device")
    if not isinstance(zero_variance_mask, Tensor) or zero_variance_mask.dtype != torch.bool:
        raise TypeError("zero_variance_mask must be a boolean Tensor")
    if zero_variance_mask.shape != centered_rewards.shape:
        raise ValueError("zero_variance_mask must match centered_rewards shape")
    if zero_variance_mask.device != centered_rewards.device:
        raise ValueError("zero_variance_mask and rewards must be on the same device")
    if (normalizer < 0).any():
        raise ValueError("normalizer must be non-negative")
    if bool(((normalizer == 0) & ~zero_variance_mask).any()):
        raise ValueError("zero normalizer entries must be marked zero-variance")
    if clip_value is not None:
        _require_finite_scalar("clip_value", clip_value)
        if clip_value == 0:
            raise ValueError("clip_value must be positive when supplied")

    dtype = working_dtype(centered_rewards, normalizer)
    centered = centered_rewards.to(dtype)
    scale = normalizer.to(dtype)
    advantages = torch.where(
        zero_variance_mask,
        torch.zeros_like(centered),
        centered / scale.clamp_min(torch.finfo(dtype).tiny),
    )
    if clip_value is not None:
        advantages = advantages.clamp(-clip_value, clip_value)
    return advantages


@dataclass(frozen=True)
class AWMMetrics:
    """Advantage Weighted Matching loss and diagnostics."""

    total_loss: Tensor
    policy_loss: Tensor
    reference_loss: Tensor
    old_anchor_loss: Tensor
    policy_mse: Tensor
    old_mse: Tensor
    policy_matching_error: Tensor
    old_matching_error: Tensor
    ratio: Tensor
    clipped_ratio: Tensor
    clip_fraction: Tensor
    approx_kl: Tensor
    advantage_mean: Tensor
    advantage_abs_mean: Tensor
    sample_weight_mean: Tensor


@dataclass(frozen=True)
class AWMInputs:
    policy_velocity: Tensor
    old_velocity: Tensor
    target_velocity: Tensor
    advantages: Tensor
    reference_velocity: Tensor | None = None
    sample_weight: Tensor | None = None


@dataclass(frozen=True)
class AWMConfig:
    event_dims: EventDims = None
    clip_epsilon: float | None = 1.0
    reference_beta: float = 0.0
    old_anchor_beta: float = 0.0
    matching_power: float = 1.0
    matching_epsilon: float = 1e-10
    log_ratio_clip: float = 60.0


def _validate_awm(inputs: AWMInputs, config: AWMConfig) -> None:
    require_matching_tensors(
        policy_velocity=inputs.policy_velocity,
        old_velocity=inputs.old_velocity,
        target_velocity=inputs.target_velocity,
    )
    for name, value in (
        ("reference_beta", config.reference_beta),
        ("old_anchor_beta", config.old_anchor_beta),
        ("matching_power", config.matching_power),
        ("matching_epsilon", config.matching_epsilon),
        ("log_ratio_clip", config.log_ratio_clip),
    ):
        _require_finite_scalar(name, value)
        if name.startswith("matching_") or name == "log_ratio_clip":
            if value == 0:
                raise ValueError(f"{name} must be positive")
    if config.clip_epsilon is not None:
        _require_finite_scalar("clip_epsilon", config.clip_epsilon)
        if config.clip_epsilon == 0 or config.clip_epsilon > 1:
            raise ValueError("clip_epsilon must be in (0, 1] when supplied")
    if config.reference_beta > 0 and inputs.reference_velocity is None:
        raise ValueError("reference_velocity is required for positive reference_beta")
    if inputs.reference_velocity is not None:
        require_matching_tensors(
            policy_velocity=inputs.policy_velocity,
            reference_velocity=inputs.reference_velocity,
        )


def _matching_errors(
    inputs: AWMInputs,
    config: AWMConfig,
    dims: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    policy_mse = _event_mse(inputs.policy_velocity, inputs.target_velocity.detach(), dims=dims)
    old_mse = _event_mse(inputs.old_velocity.detach(), inputs.target_velocity.detach(), dims=dims)
    if config.matching_power == 1.0:
        policy_raw, old_raw = policy_mse, old_mse
    else:
        baseline = config.matching_epsilon**config.matching_power
        policy_raw = (
            (policy_mse + config.matching_epsilon).pow(config.matching_power) - baseline
        ) / config.matching_power
        old_raw = (
            (old_mse + config.matching_epsilon).pow(config.matching_power) - baseline
        ) / config.matching_power
    policy_error, weight = _apply_sample_weight(
        policy_raw,
        inputs.sample_weight,
        reference=inputs.policy_velocity,
        dims=dims,
    )
    return policy_mse, old_mse, policy_error, old_raw * weight, weight


def _awm_surrogate(
    policy_error: Tensor,
    old_error: Tensor,
    advantage: Tensor,
    config: AWMConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    log_ratio = (old_error.detach() - policy_error).clamp(
        min=-config.log_ratio_clip, max=config.log_ratio_clip
    )
    ratio = log_ratio.exp()
    if config.clip_epsilon is None:
        return -(ratio * advantage).mean(), ratio, ratio, log_ratio
    clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    return (
        -torch.where(clipped < unclipped, clipped, unclipped).mean(),
        ratio,
        clipped_ratio,
        log_ratio,
    )


def advantage_weighted_matching_loss(
    inputs: AWMInputs,
    config: AWMConfig | None = None,
) -> AWMMetrics:
    """Compute the official PPO-style Advantage Weighted Matching surrogate.

    A flow-matching error acts as a clean endpoint log-likelihood surrogate:
    ``log_ratio = old_mse - policy_mse``. At a fresh on-policy update where
    policy and old match, its gradient is the direct advantage-weighted
    flow-matching gradient. Optional reference and lagged-policy anchors are
    plain velocity MSE terms, matching the official implementation's base/EMA
    regularizers.
    """

    config = config or AWMConfig()
    _validate_awm(inputs, config)
    dims = normalize_event_dims(inputs.policy_velocity.ndim, config.event_dims)
    advantage = _prepare_leading_tensor(
        "advantages", inputs.advantages, reference=inputs.policy_velocity, dims=dims
    )
    policy_mse, old_mse, policy_error, old_error, weight = _matching_errors(inputs, config, dims)
    policy_loss, ratio, clipped_ratio, log_ratio = _awm_surrogate(
        policy_error, old_error, advantage, config
    )
    if inputs.reference_velocity is None:
        reference_per_sample = torch.zeros_like(policy_mse)
    else:
        reference_per_sample = _event_mse(
            inputs.policy_velocity, inputs.reference_velocity.detach(), dims=dims
        )
    old_anchor_per_sample = _event_mse(
        inputs.policy_velocity, inputs.old_velocity.detach(), dims=dims
    )
    reference_loss = reference_per_sample.mean()
    old_anchor_loss = old_anchor_per_sample.mean()
    total_loss = (
        policy_loss
        + config.reference_beta * reference_loss
        + config.old_anchor_beta * old_anchor_loss
    )
    diagnostics = (
        policy_mse,
        old_mse,
        policy_error,
        old_error,
        ratio,
        clipped_ratio,
        total_loss,
        reference_per_sample,
        old_anchor_per_sample,
    )
    if not all(torch.isfinite(value).all() for value in diagnostics):
        raise ValueError("AWM loss and diagnostics must be finite")

    return AWMMetrics(
        total_loss=total_loss,
        policy_loss=policy_loss,
        reference_loss=reference_loss,
        old_anchor_loss=old_anchor_loss,
        policy_mse=policy_mse,
        old_mse=old_mse,
        policy_matching_error=policy_error,
        old_matching_error=old_error,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        clip_fraction=(ratio != clipped_ratio).to(ratio.dtype).mean(),
        approx_kl=0.5 * log_ratio.square().mean(),
        advantage_mean=advantage.mean(),
        advantage_abs_mean=advantage.abs().mean(),
        sample_weight_mean=weight.mean(),
    )


@dataclass(frozen=True)
class RAMMetrics:
    """Reinforce Adjoint Matching loss, target, and diagnostics."""

    total_loss: Tensor
    policy_loss: Tensor
    per_sample_loss: Tensor
    target_velocity: Tensor
    target_velocity_rms: Tensor
    policy_reference_rms: Tensor
    advantage_mean: Tensor
    advantage_abs_mean: Tensor
    sample_weight_mean: Tensor


def reinforce_adjoint_matching_loss(
    policy_velocity: Tensor,
    old_velocity: Tensor,
    reference_velocity: Tensor,
    flow_target_velocity: Tensor,
    advantages: Tensor,
    *,
    reward_multiplier: float = 1.0,
    event_dims: EventDims = None,
    sample_weight: Tensor | None = None,
) -> RAMMetrics:
    """Compute the reward-corrected RAM velocity regression objective.

    The stopped target is
    ``v_ref + reward_multiplier * advantage * (noise_minus_clean - v_old)``.
    Endpoint samples, rewards, the lagged policy, and the reference therefore
    receive no gradient; only the leading policy prediction is differentiated.
    """

    require_matching_tensors(
        policy_velocity=policy_velocity,
        old_velocity=old_velocity,
        reference_velocity=reference_velocity,
        flow_target_velocity=flow_target_velocity,
    )
    _require_finite_scalar("reward_multiplier", reward_multiplier)
    dims = normalize_event_dims(policy_velocity.ndim, event_dims)
    advantage = _prepare_leading_tensor(
        "advantages", advantages, reference=policy_velocity, dims=dims
    )
    expanded_advantage = advantage.reshape(*advantage.shape, *([1] * len(dims)))
    dtype = working_dtype(
        policy_velocity,
        old_velocity,
        reference_velocity,
        flow_target_velocity,
    )
    target = reference_velocity.detach().to(dtype) + (
        float(reward_multiplier)
        * expanded_advantage
        * (flow_target_velocity.detach().to(dtype) - old_velocity.detach().to(dtype))
    )
    target = target.detach()
    per_sample_raw = _event_mse(policy_velocity, target, dims=dims)
    per_sample, weight = _apply_sample_weight(
        per_sample_raw,
        sample_weight,
        reference=policy_velocity,
        dims=dims,
    )
    policy_loss = per_sample.mean()
    if not all(torch.isfinite(value).all() for value in (target, per_sample, policy_loss)):
        raise ValueError("RAM loss and diagnostics must be finite")

    target_rms = target.square().mean().sqrt()
    policy_reference_rms = (
        (policy_velocity.to(dtype) - reference_velocity.detach().to(dtype)).square().mean().sqrt()
    )
    return RAMMetrics(
        total_loss=policy_loss,
        policy_loss=policy_loss,
        per_sample_loss=per_sample,
        target_velocity=target,
        target_velocity_rms=target_rms,
        policy_reference_rms=policy_reference_rms,
        advantage_mean=advantage.mean(),
        advantage_abs_mean=advantage.abs().mean(),
        sample_weight_mean=weight.mean(),
    )
