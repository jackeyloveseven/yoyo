"""Model-independent DiffusionNFT forward-process objectives."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from mmtrainer.algorithms.common import EventDims, normalize_event_dims, require_float_finite

Normalizer = Literal["group_std", "global_std"] | float | Tensor


def _upcast(value: Tensor) -> Tensor:
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _broadcast_like(value: Tensor, reference: Tensor, name: str) -> Tensor:
    candidate = value
    if candidate.ndim < reference.ndim:
        candidate = candidate.reshape(*candidate.shape, *([1] * (reference.ndim - candidate.ndim)))
    try:
        return torch.broadcast_to(candidate, reference.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to shape {tuple(reference.shape)}") from exc


@dataclass(frozen=True)
class OptimalityResult:
    """Raw reward normalization used as DiffusionNFT optimality probability."""

    probabilities: Tensor
    centered_rewards: Tensor
    normalizer: Tensor
    zero_variance_mask: Tensor


def reward_to_optimality(
    raw_rewards: Tensor,
    *,
    normalizer: Normalizer = "group_std",
    adv_clip_max: float = 5.0,
) -> OptimalityResult:
    """Map normalized advantages to DiffusionNFT optimality probabilities."""

    require_float_finite("raw_rewards", raw_rewards)
    if raw_rewards.ndim != 2:
        raise ValueError("raw_rewards must have shape [batch, group]")
    if not torch.isfinite(torch.tensor(adv_clip_max, dtype=torch.float64)) or adv_clip_max <= 0:
        raise ValueError("adv_clip_max must be finite and positive")
    work = _upcast(raw_rewards)
    centered = work - work.mean(dim=1, keepdim=True)
    if isinstance(normalizer, str) and normalizer == "group_std":
        scale = work.std(dim=1, correction=0, keepdim=True)
    elif isinstance(normalizer, str) and normalizer == "global_std":
        scale = work.std(correction=0).reshape(1, 1)
    elif isinstance(normalizer, str):
        raise ValueError("normalizer must be 'group_std', 'global_std', a scalar, or a Tensor")
    elif isinstance(normalizer, Tensor):
        require_float_finite("normalizer", normalizer)
        scale = _upcast(normalizer).to(device=work.device)
    else:
        if not torch.isfinite(torch.tensor(normalizer, dtype=torch.float64)):
            raise ValueError("normalizer must be finite")
        scale = torch.tensor(float(normalizer), dtype=work.dtype, device=work.device)

    scale = _broadcast_like(scale, work, "normalizer")
    if (scale < 0).any():
        raise ValueError("normalizer must be non-negative")
    zero_variance = scale == 0
    if isinstance(normalizer, str):
        # Stabilize measured standard deviations, preserving explicit scales.
        scale = scale + 1e-4
    normalized = torch.where(
        zero_variance, torch.zeros_like(centered), centered / scale.clamp_min(1e-30)
    )
    probabilities = 0.5 + normalized.clamp(-adv_clip_max, adv_clip_max) / (2.0 * adv_clip_max)
    probabilities = torch.where(zero_variance, torch.full_like(probabilities, 0.5), probabilities)
    return OptimalityResult(
        probabilities=probabilities,
        centered_rewards=centered,
        normalizer=scale,
        zero_variance_mask=zero_variance,
    )


def implicit_positive_velocity(policy: Tensor, old: Tensor, *, beta: float) -> Tensor:
    """Return ``(1 - beta) * old + beta * policy``."""

    if old.device != policy.device:
        old = old.to(policy.device)
    _validate_implicit_inputs(policy, old, beta)
    return (1.0 - beta) * _upcast(old.detach()) + beta * _upcast(policy)


def implicit_negative_velocity(policy: Tensor, old: Tensor, *, beta: float) -> Tensor:
    """Return ``(1 + beta) * old - beta * policy``."""

    if old.device != policy.device:
        old = old.to(policy.device)
    _validate_implicit_inputs(policy, old, beta)
    return (1.0 + beta) * _upcast(old.detach()) - beta * _upcast(policy)


def _validate_implicit_inputs(policy: Tensor, old: Tensor, beta: float) -> None:
    require_float_finite("policy velocity", policy)
    require_float_finite("old velocity", old)
    if policy.shape != old.shape:
        raise ValueError("policy and old velocities must have the same shape")
    if policy.device != old.device:
        old = old.to(policy.device)
    if not torch.isfinite(torch.tensor(beta, dtype=torch.float64)) or beta <= 0:
        raise ValueError("beta must be finite and positive")


def rectified_flow_noising(clean: Tensor, noise: Tensor, noise_fraction: Tensor) -> Tensor:
    """Apply ``x_t = (1 - t) * x0 + t * noise``."""

    _validate_forward_tensors(clean, noise, noise_fraction)
    clean, noise = _upcast(clean), _upcast(noise)
    t = _broadcast_like(
        noise_fraction.to(device=clean.device, dtype=clean.dtype),
        clean,
        "noise_fraction",
    )
    if ((t < 0) | (t > 1)).any():
        raise ValueError("noise_fraction must be in [0, 1]")
    return (1.0 - t) * clean + t * noise


def rectified_flow_target_velocity(clean: Tensor, noise: Tensor) -> Tensor:
    """Return the rectified-flow target velocity ``noise - clean``."""

    require_float_finite("clean", clean)
    require_float_finite("noise", noise)
    if clean.shape != noise.shape or clean.device != noise.device:
        raise ValueError("clean and noise must have matching shape and device")
    return _upcast(noise) - _upcast(clean)


def reconstruct_clean(noisy: Tensor, noise_fraction: Tensor, velocity: Tensor) -> Tensor:
    """Reconstruct clean data as ``x_t - t * v``."""

    require_float_finite("noisy", noisy)
    require_float_finite("noise_fraction", noise_fraction)
    require_float_finite("velocity", velocity)
    if noisy.shape != velocity.shape or noisy.device != velocity.device:
        raise ValueError("noisy and velocity must have matching shape and device")
    noisy, velocity = _upcast(noisy), _upcast(velocity)
    t = _broadcast_like(
        noise_fraction.to(device=noisy.device, dtype=noisy.dtype),
        noisy,
        "noise_fraction",
    )
    if ((t < 0) | (t > 1)).any():
        raise ValueError("noise_fraction must be in [0, 1]")
    return noisy - t * velocity


def _validate_forward_tensors(clean: Tensor, noise: Tensor, noise_fraction: Tensor) -> None:
    require_float_finite("clean", clean)
    require_float_finite("noise", noise)
    require_float_finite("noise_fraction", noise_fraction)
    if clean.shape != noise.shape or clean.device != noise.device:
        raise ValueError("clean and noise must have matching shape and device")


@dataclass(frozen=True)
class RectifiedFlowBatch:
    """One standard rectified-flow forward-noising batch."""

    clean: Tensor
    noisy: Tensor
    noise_fraction: Tensor
    target: Tensor
    model_timestep: Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("clean", self.clean),
            ("noisy", self.noisy),
            ("target", self.target),
        ):
            require_float_finite(name, value)
        if self.noisy.shape != self.clean.shape or self.target.shape != self.clean.shape:
            raise ValueError("clean, noisy, and target must have matching shapes")
        if self.noisy.device != self.clean.device or self.target.device != self.clean.device:
            raise ValueError("clean, noisy, and target must be on the same device")

        require_float_finite("noise_fraction", self.noise_fraction)
        broadcast_time = _broadcast_like(self.noise_fraction, self.clean, "noise_fraction")
        if ((broadcast_time < 0) | (broadcast_time > 1)).any():
            raise ValueError("noise_fraction must be in [0, 1]")
        if self.model_timestep is not None:
            require_float_finite("model_timestep", self.model_timestep)
            _broadcast_like(self.model_timestep, self.clean, "model_timestep")


def make_rectified_flow_batch(
    clean: Tensor,
    noise_fraction: Tensor,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> RectifiedFlowBatch:
    """Construct noisy inputs and target velocities for rectified-flow training."""

    require_float_finite("clean", clean)
    clean = _upcast(clean)
    if noise is None:
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
    noisy = rectified_flow_noising(clean, noise, noise_fraction)
    target = rectified_flow_target_velocity(clean, noise)
    return RectifiedFlowBatch(
        clean=clean, noisy=noisy, noise_fraction=_upcast(noise_fraction), target=target
    )


def _event_mean(value: Tensor, dims: tuple[int, ...]) -> Tensor:
    return value.mean(dim=dims) if dims else value


def _adaptive_clean_loss(
    prediction: Tensor,
    noisy: Tensor,
    clean: Tensor,
    noise_fraction: Tensor,
    dims: tuple[int, ...],
    eps: float,
) -> Tensor:
    reconstructed = reconstruct_clean(noisy, noise_fraction, prediction)
    error = _upcast(reconstructed) - _upcast(clean)
    denominator = _event_mean(error.abs(), dims).detach().clamp_min(eps)
    if dims:
        denominator = denominator.reshape(*denominator.shape, *([1] * len(dims)))
    return _event_mean(error.square() / denominator, dims)


@dataclass(frozen=True)
class DiffusionNFTMetrics:
    """DiffusionNFT objective, implicit policies, and scalar diagnostics."""

    positive_loss: Tensor
    negative_loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    total_loss: Tensor
    positive_per_sample: Tensor
    negative_per_sample: Tensor
    policy_per_sample: Tensor
    kl_per_sample: Tensor
    implicit_positive: Tensor
    implicit_negative: Tensor
    optimality_mean: Tensor
    policy_old_mse: Tensor


@dataclass(frozen=True)
class DiffusionNFTInputs:
    policy_velocity: Tensor
    old_velocity: Tensor
    target_velocity: Tensor
    optimality: Tensor
    clean: Tensor | None = None
    noisy: Tensor | None = None
    noise_fraction: Tensor | None = None
    reference_velocity: Tensor | None = None


@dataclass(frozen=True)
class DiffusionNFTLossConfig:
    beta: float
    event_dims: EventDims = None
    adv_clip_max: float = 5.0
    adaptive_clean_weighting: bool = False
    adaptive_eps: float = 1e-5
    kl_beta: float = 0.0


def _validate_loss_config(config: DiffusionNFTLossConfig) -> None:
    if not math.isfinite(config.adv_clip_max) or config.adv_clip_max <= 0:
        raise ValueError("adv_clip_max must be finite and positive")
    if not math.isfinite(config.kl_beta) or config.kl_beta < 0:
        raise ValueError("kl_beta must be finite and non-negative")
    if not math.isfinite(config.adaptive_eps) or config.adaptive_eps <= 0:
        raise ValueError("adaptive_eps must be finite and positive")


def _loss_dimensions(
    inputs: DiffusionNFTInputs,
    config: DiffusionNFTLossConfig,
) -> tuple[tuple[int, ...], Tensor, Tensor]:
    policy = inputs.policy_velocity
    target = inputs.target_velocity.to(policy.device)
    if target.shape != policy.shape:
        raise ValueError("target velocity must match policy velocity shape")
    dims = normalize_event_dims(policy.ndim, config.event_dims)
    leading = tuple(size for index, size in enumerate(policy.shape) if index not in dims)
    if tuple(inputs.optimality.shape) != leading:
        raise ValueError(
            f"optimality must match leading shape {leading}, got {tuple(inputs.optimality.shape)}"
        )
    if inputs.optimality.device != policy.device:
        raise ValueError("optimality and velocities must be on the same device")
    optimality = _upcast(inputs.optimality)
    if bool(((optimality < 0) | (optimality > 1)).any()):
        raise ValueError("optimality must be in [0, 1]")
    return dims, target, optimality


def _implicit_sample_losses(
    inputs: DiffusionNFTInputs,
    config: DiffusionNFTLossConfig,
    positive: Tensor,
    negative: Tensor,
    target: Tensor,
    dims: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    if not config.adaptive_clean_weighting:
        return (
            _event_mean((_upcast(positive) - _upcast(target)).square(), dims),
            _event_mean((_upcast(negative) - _upcast(target)).square(), dims),
        )
    if inputs.clean is None or inputs.noisy is None or inputs.noise_fraction is None:
        raise ValueError("adaptive weighting requires clean, noisy, and noise_fraction")
    require_float_finite("clean", inputs.clean)
    clean = inputs.clean.to(inputs.policy_velocity.device)
    if clean.shape != inputs.policy_velocity.shape:
        raise ValueError("clean must match policy velocity shape")
    return (
        _adaptive_clean_loss(
            positive,
            inputs.noisy,
            clean,
            inputs.noise_fraction,
            dims,
            config.adaptive_eps,
        ),
        _adaptive_clean_loss(
            negative,
            inputs.noisy,
            clean,
            inputs.noise_fraction,
            dims,
            config.adaptive_eps,
        ),
    )


def _kl_per_sample(
    inputs: DiffusionNFTInputs,
    config: DiffusionNFTLossConfig,
    policy_per_sample: Tensor,
    dims: tuple[int, ...],
) -> Tensor:
    reference = inputs.reference_velocity
    if reference is None:
        if config.kl_beta > 0:
            raise ValueError("reference_velocity is required when kl_beta is positive")
        return torch.zeros_like(policy_per_sample)
    require_float_finite("reference_velocity", reference)
    if reference.shape != inputs.policy_velocity.shape:
        raise ValueError("reference velocity must match policy velocity shape")
    if reference.device != inputs.policy_velocity.device:
        raise ValueError("reference velocity must match policy velocity device")
    return _event_mean(
        (_upcast(inputs.policy_velocity) - _upcast(reference.detach())).square(),
        dims,
    )


def diffusion_nft_loss(
    inputs: DiffusionNFTInputs,
    config: DiffusionNFTLossConfig,
) -> DiffusionNFTMetrics:
    """Compute the strict DiffusionNFT objective and optional regularizers."""

    _validate_implicit_inputs(inputs.policy_velocity, inputs.old_velocity, config.beta)
    require_float_finite("target_velocity", inputs.target_velocity)
    require_float_finite("optimality", inputs.optimality)
    _validate_loss_config(config)
    dims, target, optimality = _loss_dimensions(inputs, config)
    positive = implicit_positive_velocity(
        inputs.policy_velocity, inputs.old_velocity, beta=config.beta
    )
    negative = implicit_negative_velocity(
        inputs.policy_velocity, inputs.old_velocity, beta=config.beta
    )
    positive_per_sample, negative_per_sample = _implicit_sample_losses(
        inputs, config, positive, negative, target, dims
    )
    policy_per_sample = (
        optimality * positive_per_sample + (1.0 - optimality) * negative_per_sample
    ) / config.beta
    kl_per_sample = _kl_per_sample(inputs, config, policy_per_sample, dims)
    positive_loss = positive_per_sample.mean()
    negative_loss = negative_per_sample.mean()
    policy_loss = policy_per_sample.mean() * config.adv_clip_max
    kl_loss = kl_per_sample.mean()
    total_loss = policy_loss + config.kl_beta * kl_loss
    if not all(
        torch.isfinite(value).all()
        for value in (
            positive_per_sample,
            negative_per_sample,
            policy_per_sample,
            kl_per_sample,
            total_loss,
        )
    ):
        raise ValueError("DiffusionNFT loss and diagnostics must be finite")
    policy_old_mse = _event_mean(
        (_upcast(inputs.policy_velocity) - _upcast(inputs.old_velocity.detach())).square(),
        dims,
    ).mean()
    return DiffusionNFTMetrics(
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        total_loss=total_loss,
        positive_per_sample=positive_per_sample,
        negative_per_sample=negative_per_sample,
        policy_per_sample=policy_per_sample,
        kl_per_sample=kl_per_sample,
        implicit_positive=positive,
        implicit_negative=negative,
        optimality_mean=optimality.mean(),
        policy_old_mse=policy_old_mse,
    )


@torch.no_grad()
def ema_update_parameters(
    old_parameters: Iterable[Tensor],
    policy_parameters: Iterable[Tensor],
    *,
    decay: float,
) -> None:
    """Update ``old <- decay * old + (1 - decay) * policy`` in place."""

    if not torch.isfinite(torch.tensor(decay, dtype=torch.float64)) or not 0 <= decay <= 1:
        raise ValueError("decay must be finite and in [0, 1]")
    old_list = tuple(old_parameters)
    policy_list = tuple(policy_parameters)
    if len(old_list) != len(policy_list):
        raise ValueError("old and policy parameter collections must have the same length")
    for index, (old, policy) in enumerate(zip(old_list, policy_list, strict=True)):
        if not isinstance(old, Tensor) or not isinstance(policy, Tensor):
            raise TypeError("EMA parameters must be Tensors")
        if old.shape != policy.shape:
            raise ValueError(f"EMA parameter {index} shapes do not match")
        if old.device != policy.device:
            raise ValueError(f"EMA parameter {index} devices do not match")
        if old.dtype != policy.dtype:
            raise ValueError(f"EMA parameter {index} dtypes do not match")
        if not old.is_floating_point() or not policy.is_floating_point():
            raise TypeError("EMA parameters must use floating-point dtypes")
        if not torch.isfinite(old).all() or not torch.isfinite(policy).all():
            raise ValueError(f"EMA parameter {index} must contain only finite values")
        old.mul_(decay).add_(policy.detach(), alpha=1.0 - decay)
