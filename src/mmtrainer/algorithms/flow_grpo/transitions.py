"""Rectified-flow SDE transition parameterization used by Flow-GRPO."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import TransitionDistribution


def _as_tensor(value: Tensor | float, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return reference.new_tensor(value)


def _prefix_parameter(name: str, value: Tensor, sample: Tensor) -> Tensor:
    if value.ndim == 0:
        return value
    if value.ndim > sample.ndim or tuple(value.shape) != tuple(sample.shape[: value.ndim]):
        raise ValueError(
            f"{name} must be scalar or match a prefix of sample; "
            f"sample={tuple(sample.shape)}, {name}={tuple(value.shape)}"
        )
    return value


def _to_event_shape(value: Tensor, sample: Tensor) -> Tensor:
    if value.ndim == 0:
        return value
    return value.reshape(*value.shape, *([1] * (sample.ndim - value.ndim)))


def _validate_state(sample: Tensor, velocity: Tensor) -> None:
    if not isinstance(sample, Tensor) or not sample.is_floating_point():
        raise TypeError("sample must be a floating-point torch.Tensor")
    if not isinstance(velocity, Tensor) or not velocity.is_floating_point():
        raise TypeError("velocity must be a floating-point torch.Tensor")
    if sample.shape != velocity.shape:
        raise ValueError("sample and velocity must have identical shapes")
    if sample.ndim < 1:
        raise ValueError("sample must have at least one state dimension")


def _prepare_transition_parameters(
    sample: Tensor,
    sigma: Tensor | float,
    sigma_next: Tensor | float,
    noise_level: Tensor | float,
    sigma_max: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    values = tuple(
        _prefix_parameter(name, _as_tensor(value, sample), sample)
        for name, value in (
            ("sigma", sigma),
            ("sigma_next", sigma_next),
            ("noise_level", noise_level),
            ("sigma_max", sigma_max),
        )
    )
    sigma_t, sigma_next_t, noise_t, sigma_max_t = torch.broadcast_tensors(*values)
    if sample.ndim >= 4 and all(value.ndim == 0 for value in values):
        prefix = sample.shape[:3]
        return tuple(value.expand(prefix) for value in values)
    return sigma_t, sigma_next_t, noise_t, sigma_max_t


def _validate_transition_parameters(
    sigma: Tensor,
    sigma_next: Tensor,
    noise_level: Tensor,
    sigma_max: Tensor,
) -> None:
    for name, value in (
        ("sigma", sigma),
        ("sigma_next", sigma_next),
        ("noise_level", noise_level),
        ("sigma_max", sigma_max),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if bool(((sigma <= 0) | (sigma > 1)).any()):
        raise ValueError("sigma must satisfy 0 < sigma <= 1")
    if bool(((sigma_next < 0) | (sigma_next > sigma)).any()):
        raise ValueError("sigma_next must satisfy 0 <= sigma_next <= sigma")
    if bool((noise_level < 0).any()):
        raise ValueError("noise_level must be non-negative")
    if bool(((sigma_max <= 0) | (sigma_max > 1)).any()):
        raise ValueError("sigma_max must satisfy 0 < sigma_max <= 1")


def sde_step_with_logprob(
    sample: Tensor,
    velocity: Tensor,
    sigma: Tensor | float,
    sigma_next: Tensor | float,
    noise_level: Tensor | float,
    sigma_max: Tensor | float,
) -> TransitionDistribution:
    """Build one rectified-flow SDE transition distribution.

    The parameterization follows the official Flow-GRPO step:

    ``dt = sigma_next - sigma``
    ``std_dev = sqrt(sigma / (1 - sigma_safe)) * noise_level``
    ``mean = sample * (1 + std_dev**2 / (2*sigma) * dt)``
    ``     + velocity * (1 + std_dev**2 * (1-sigma) / (2*sigma)) * dt``
    ``transition_std = std_dev * sqrt(-dt)``

    ``sigma_safe`` equals ``sigma`` except at ``sigma == 1``, where the
    denominator uses ``1 - sigma_max``. ``sigma_next`` must not be greater
    than ``sigma`` because the flow integrates toward lower noise.
    """

    _validate_state(sample, velocity)

    # The official Flow-GRPO implementation explicitly upcasts this step:
    # the mean and Gaussian scale can overflow in fp16/bf16 near sigma=1.
    compute_dtype = (
        torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    )
    sample_compute = sample.to(dtype=compute_dtype)
    velocity_compute = velocity.to(device=sample.device, dtype=compute_dtype)

    sigma_t, sigma_next_t, noise_level_t, sigma_max_t = _prepare_transition_parameters(
        sample_compute, sigma, sigma_next, noise_level, sigma_max
    )
    _validate_transition_parameters(sigma_t, sigma_next_t, noise_level_t, sigma_max_t)

    at_one = sigma_t == 1
    sigma_safe = torch.where(at_one, sigma_max_t, sigma_t)
    denominator = 1 - sigma_safe
    if (denominator <= 0).any():
        raise ValueError("1 - sigma_safe must be positive; use sigma_max < 1 when sigma == 1")
    dt = sigma_next_t - sigma_t
    if (dt > 0).any():
        raise ValueError("dt=sigma_next-sigma must be non-positive")

    std_dev = torch.sqrt(sigma_t / denominator) * noise_level_t
    std_dev_sq = std_dev.square()
    sigma_event = _to_event_shape(sigma_t, sample_compute)
    dt_event = _to_event_shape(dt, sample_compute)
    std_dev_sq_event = _to_event_shape(std_dev_sq, sample_compute)
    mean = (
        sample_compute * (1 + std_dev_sq_event / (2 * sigma_event) * dt_event)
        + velocity_compute
        * (1 + std_dev_sq_event * (1 - sigma_event) / (2 * sigma_event))
        * dt_event
    )
    transition_std = std_dev * torch.sqrt((-dt).clamp_min(0))
    deterministic_mask = transition_std == 0
    return TransitionDistribution(
        mean=mean,
        std=transition_std,
        deterministic_mask=deterministic_mask,
    )


__all__ = ["sde_step_with_logprob"]
