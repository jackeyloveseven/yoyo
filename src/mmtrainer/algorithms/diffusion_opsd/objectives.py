"""Paper-faithful clean-output objectives for DiffusionOPSD.

Implements Eqs. 1, 5--13 of *On-Policy Self-Distillation in Diffusion
Models* (arXiv:2608.24646).  Rollout/model details intentionally live in
adapters so this module can be tested without a diffusion checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import rectified_flow_clean_output, require_float_finite

EventDims = int | Sequence[int] | None
RewardFn = Callable[[Tensor], Tensor]


def _positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _event_dims(value: Tensor, event_dims: EventDims) -> tuple[int, ...]:
    if event_dims is None:
        if value.ndim < 3:
            raise ValueError("default event dims require [batch, group, ...]")
        return tuple(range(2, value.ndim))
    if isinstance(event_dims, int):
        if isinstance(event_dims, bool) or not 0 <= event_dims <= value.ndim:
            raise ValueError("event_dims count is outside the tensor rank")
        return tuple(range(value.ndim - event_dims, value.ndim))
    result: list[int] = []
    for dim in event_dims:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("event_dims must contain integers")
        resolved = dim + value.ndim if dim < 0 else dim
        if not 0 <= resolved < value.ndim:
            raise ValueError(f"event dimension {dim} is outside the tensor rank")
        result.append(resolved)
    if len(set(result)) != len(result):
        raise ValueError("event_dims must not contain duplicates")
    result.sort()
    if result and tuple(result) != tuple(range(result[0], value.ndim)):
        raise ValueError("event_dims must be a contiguous trailing block")
    return tuple(result)


def _leading_shape(value: Tensor, dims: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(size for index, size in enumerate(value.shape) if index not in dims)


def _broadcast_leading(value: Tensor, reference: Tensor, dims: tuple[int, ...]) -> Tensor:
    expected = _leading_shape(reference, dims)
    if tuple(value.shape) != expected:
        raise ValueError(f"leading tensor shape must be {expected}, got {tuple(value.shape)}")
    return value.reshape(*expected, *([1] * len(dims)))


def clean_output_prediction(query_state: Tensor, query_sigma: Tensor, velocity: Tensor) -> Tensor:
    """Eq. 1: convert a rectified-flow velocity into its clean prediction."""

    return rectified_flow_clean_output(query_state, query_sigma, velocity)


@dataclass(frozen=True)
class DiffusionOPSDWeights:
    raw_rewards: Tensor
    group_mean: Tensor
    global_mean: Tensor
    global_std: Tensor
    normalization: Tensor
    centered_scaled: Tensor
    weights: Tensor


def group_normalized_endpoint_weights(
    raw_rewards: Tensor,
    *,
    advantage_scale: float = 5.0,
    epsilon: float = 1e-4,
    global_mean: Tensor | None = None,
    global_std: Tensor | None = None,
) -> DiffusionOPSDWeights:
    """Eq. 5: per-prompt centering with one global rollout-batch std."""

    require_float_finite("raw_rewards", raw_rewards)
    if raw_rewards.ndim != 2 or raw_rewards.shape[1] < 2:
        raise ValueError("raw_rewards must be [prompt_batch, group] with group >= 2")
    _positive_float("advantage_scale", advantage_scale)
    _positive_float("epsilon", epsilon)
    work = raw_rewards.float()
    observed_mean = work.mean()
    observed_std = work.std(correction=0)
    mean = observed_mean if global_mean is None else global_mean.float()
    std = observed_std if global_std is None else global_std.float()
    for name, value in (("global_mean", mean), ("global_std", std)):
        require_float_finite(name, value)
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
    if bool((std < 0).item()):
        raise ValueError("global_std must be non-negative")
    group_mean = work.mean(dim=1, keepdim=True)
    normalization = advantage_scale * (std + epsilon)
    centered_scaled = (work - group_mean) / normalization
    weights = 0.5 + 0.5 * centered_scaled.clamp(-1.0, 1.0)
    return DiffusionOPSDWeights(
        raw_rewards=work,
        group_mean=group_mean,
        global_mean=mean.reshape(()),
        global_std=std.reshape(()),
        normalization=normalization.reshape(()),
        centered_scaled=centered_scaled,
        weights=weights,
    )


@dataclass(frozen=True)
class DiffusionOPSDTargets:
    anchor: Tensor
    positive: Tensor
    negative: Tensor
    radius: Tensor
    first_gradient_norm: Tensor
    reward_gradient_calls: int


def _reward_gradient(
    reward_fn: RewardFn, clean: Tensor, leading_shape: tuple[int, ...]
) -> tuple[Tensor, Tensor]:
    leaf = clean.detach().requires_grad_(True)
    reward = reward_fn(leaf)
    require_float_finite("local reward", reward)
    if tuple(reward.shape) != leading_shape:
        raise ValueError(f"local reward shape must be {leading_shape}, got {tuple(reward.shape)}")
    (gradient,) = torch.autograd.grad(reward.sum(), leaf, create_graph=False)
    require_float_finite("local reward gradient", gradient)
    return gradient.detach(), reward.detach()


def construct_bounded_reward_targets(
    anchor: Tensor,
    reward_fn: RewardFn,
    *,
    target_radius: float = 0.10,
    target_steps: int = 2,
    target_step_multiplier: float = 1.0,
    gradient_epsilon: float = 1e-12,
    event_dims: EventDims = None,
) -> DiffusionOPSDTargets:
    """Eqs. 6--9: normalized reward ascent/descent inside a trust region."""

    require_float_finite("anchor", anchor)
    for name, value in (
        ("target_radius", target_radius),
        ("target_step_multiplier", target_step_multiplier),
        ("gradient_epsilon", gradient_epsilon),
    ):
        _positive_float(name, value)
    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    dims = _event_dims(anchor, event_dims)
    leading = _leading_shape(anchor, dims)
    base = anchor.detach().float()
    radius = target_radius * torch.linalg.vector_norm(base, dim=dims)
    step_length = target_step_multiplier * radius / target_steps
    step_b = _broadcast_leading(step_length, base, dims)

    first_gradient, _ = _reward_gradient(reward_fn, base, leading)
    first_norm = torch.linalg.vector_norm(first_gradient.float(), dim=dims)

    def walk(sign: float) -> Tensor:
        current = base
        for index in range(target_steps):
            gradient = (
                first_gradient if index == 0 else _reward_gradient(reward_fn, current, leading)[0]
            )
            norm = torch.linalg.vector_norm(gradient.float(), dim=dims)
            norm_b = _broadcast_leading(norm, base, dims)
            direction = gradient.float() / (norm_b + gradient_epsilon)
            current = current + sign * step_b * direction
            displacement = current - base
            displacement_norm = torch.linalg.vector_norm(displacement, dim=dims)
            scale = torch.clamp(radius / (displacement_norm + gradient_epsilon), max=1.0)
            current = (base + displacement * _broadcast_leading(scale, base, dims)).detach()
        return current

    positive = walk(1.0)
    negative = walk(-1.0)
    for name, value in (("positive target", positive), ("negative target", negative)):
        require_float_finite(name, value)
        distance = torch.linalg.vector_norm(value - base, dim=dims)
        if bool((distance > radius + 1e-5 * radius.clamp_min(1.0)).any()):
            raise RuntimeError(f"{name} escaped the configured trust region")
    return DiffusionOPSDTargets(
        anchor=base,
        positive=positive.detach(),
        negative=negative.detach(),
        radius=radius.detach(),
        first_gradient_norm=first_norm.detach(),
        reward_gradient_calls=2 * target_steps - 1,
    )


@dataclass(frozen=True)
class DiffusionOPSDLoss:
    total_loss: Tensor
    positive_loss: Tensor
    negative_loss: Tensor
    per_sample: Tensor
    policy_clean: Tensor
    positive_branch: Tensor
    negative_branch: Tensor
    positive_normalizer: Tensor
    negative_normalizer: Tensor


@dataclass(frozen=True)
class DiffusionOPSDFittingInputs:
    query_state: Tensor
    query_sigma: Tensor
    policy_velocity: Tensor
    behavior_clean: Tensor
    positive_target: Tensor
    negative_target: Tensor
    weights: Tensor


@dataclass(frozen=True)
class DiffusionOPSDFittingConfig:
    branch_beta: float = 1.0
    advantage_scale: float = 5.0
    residual_epsilon: float = 1e-5
    event_dims: EventDims = None


def diffusion_opsd_fitting_loss(
    inputs: DiffusionOPSDFittingInputs,
    config: DiffusionOPSDFittingConfig | None = None,
) -> DiffusionOPSDLoss:
    """Eqs. 11--13: detached positive/negative finite fitting."""

    config = config or DiffusionOPSDFittingConfig()
    for name, value in (
        ("query_state", inputs.query_state),
        ("policy_velocity", inputs.policy_velocity),
        ("behavior_clean", inputs.behavior_clean),
        ("positive_target", inputs.positive_target),
        ("negative_target", inputs.negative_target),
        ("weights", inputs.weights),
    ):
        require_float_finite(name, value)
    for name, value in (
        ("branch_beta", config.branch_beta),
        ("advantage_scale", config.advantage_scale),
        ("residual_epsilon", config.residual_epsilon),
    ):
        _positive_float(name, value)
    expected = inputs.query_state.shape
    for name, value in (
        ("policy_velocity", inputs.policy_velocity),
        ("behavior_clean", inputs.behavior_clean),
        ("positive_target", inputs.positive_target),
        ("negative_target", inputs.negative_target),
    ):
        if value.shape != expected or value.device != inputs.query_state.device:
            raise ValueError(f"{name} must match query_state shape/device")
    dims = _event_dims(inputs.query_state, config.event_dims)
    leading = _leading_shape(inputs.query_state, dims)
    if tuple(inputs.weights.shape) != leading:
        raise ValueError(f"weights must have leading shape {leading}")
    if bool(((inputs.weights < 0) | (inputs.weights > 1)).any()):
        raise ValueError("weights must lie in [0, 1]")

    policy_clean = clean_output_prediction(
        inputs.query_state, inputs.query_sigma, inputs.policy_velocity
    )
    anchor = inputs.behavior_clean.detach().to(policy_clean.dtype)
    positive = inputs.positive_target.detach().to(policy_clean.dtype)
    negative = inputs.negative_target.detach().to(policy_clean.dtype)
    positive_branch = config.branch_beta * policy_clean + (1.0 - config.branch_beta) * anchor
    negative_branch = (1.0 + config.branch_beta) * anchor - config.branch_beta * policy_clean
    positive_normalizer = (
        (positive_branch.detach().double() - positive.double())
        .abs()
        .mean(dim=dims)
        .clamp_min(config.residual_epsilon)
        .to(policy_clean.dtype)
    )
    negative_normalizer = (
        (negative_branch.detach().double() - negative.double())
        .abs()
        .mean(dim=dims)
        .clamp_min(config.residual_epsilon)
        .to(policy_clean.dtype)
    )
    positive_loss = (positive_branch - positive).square().mean(dim=dims) / positive_normalizer
    negative_loss = (negative_branch - negative).square().mean(dim=dims) / negative_normalizer
    weight = inputs.weights.detach().to(device=policy_clean.device, dtype=policy_clean.dtype)
    per_sample = weight * positive_loss + (1.0 - weight) * negative_loss
    total = config.advantage_scale * per_sample.mean()
    require_float_finite("DiffusionOPSD fitting loss", total)
    return DiffusionOPSDLoss(
        total_loss=total,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        per_sample=per_sample,
        policy_clean=policy_clean,
        positive_branch=positive_branch,
        negative_branch=negative_branch,
        positive_normalizer=positive_normalizer,
        negative_normalizer=negative_normalizer,
    )


__all__ = [
    "DiffusionOPSDLoss",
    "DiffusionOPSDFittingConfig",
    "DiffusionOPSDFittingInputs",
    "DiffusionOPSDTargets",
    "DiffusionOPSDWeights",
    "clean_output_prediction",
    "construct_bounded_reward_targets",
    "diffusion_opsd_fitting_loss",
    "group_normalized_endpoint_weights",
]
