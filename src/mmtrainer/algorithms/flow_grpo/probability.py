"""Gaussian transition probabilities and closed-form reference KL."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _require_transition_shape(name: str, value: Tensor, reference: Tensor) -> None:
    prefix = tuple(reference.shape[:3])
    if value.ndim == 0:
        return
    if value.ndim == 3 and tuple(value.shape) == prefix:
        return
    if value.ndim == reference.ndim and tuple(value.shape) == tuple(reference.shape):
        return
    if value.ndim == reference.ndim and tuple(value.shape[:3]) == prefix:
        if all(dim == 1 for dim in value.shape[3:]):
            return
    raise ValueError(
        f"{name} must be scalar, [batch, group, step], or state-shaped; "
        f"got {tuple(value.shape)} for reference {tuple(reference.shape)}"
    )


def _expand_transition_parameter(value: Tensor, reference: Tensor) -> Tensor:
    if value.ndim == 0:
        return value
    if value.ndim == 3 and reference.ndim > 3 and tuple(value.shape) == tuple(reference.shape[:3]):
        return value.reshape(*value.shape, *([1] * (reference.ndim - 3)))
    return value


def _prepare_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    prefix = tuple(reference.shape[:3])
    if mask is None:
        return torch.zeros(prefix, dtype=torch.bool, device=reference.device)
    if mask.dtype != torch.bool or tuple(mask.shape) != prefix:
        raise ValueError(f"deterministic_mask must be bool with shape {prefix}")
    return mask


def _reduce_event(values: Tensor, reduction: str) -> Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    event_dims = tuple(range(3, values.ndim))
    if reduction == "sum":
        return values.sum(dim=event_dims)
    return values.mean(dim=event_dims)


def _safe_std(std: Tensor, deterministic_mask: Tensor, eps: float) -> Tensor:
    if eps <= 0:
        raise ValueError("eps must be positive")
    mask = deterministic_mask.reshape(*deterministic_mask.shape, *([1] * (std.ndim - 3)))
    if ((std <= 0) & ~mask).any():
        raise ValueError("std must be positive for non-deterministic transitions")
    return torch.where(mask, torch.ones_like(std), std.clamp_min(eps))


def isotropic_gaussian_log_prob(
    value: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    deterministic_mask: Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Return per-transition log-probability for an isotropic Gaussian.

    Inputs use ``[batch, group, step, ...state]``. ``mean`` is intentionally
    state-shaped by default, so callers cannot accidentally replace a
    per-dimension flow mean with one scalar mean. ``std`` may be scalar,
    per-transition ``[batch, group, step]``, or state-shaped.

    Deterministic transitions are treated as having zero contribution. This
    avoids evaluating a singular Dirac density and keeps rollout/update masks
    numerically stable.
    """

    if value.shape != mean.shape:
        raise ValueError(
            f"value and mean must have identical shapes, got {value.shape} and {mean.shape}"
        )
    if value.ndim < 4:
        raise ValueError("value must have shape [batch, group, step, ...state]")
    _require_transition_shape("std", std, value)
    mask = _prepare_mask(deterministic_mask, value)
    compute_dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
    value_compute = value.to(dtype=compute_dtype)
    mean_compute = mean.to(device=value.device, dtype=compute_dtype)
    expanded_std = _expand_transition_parameter(std, value).to(
        device=value.device, dtype=compute_dtype
    )
    expanded_std = torch.broadcast_to(expanded_std, value.shape)
    safe_std = _safe_std(expanded_std, mask, eps)
    log_prob = -0.5 * (
        ((value_compute - mean_compute) / safe_std).square()
        + 2.0 * safe_std.log()
        + math.log(2.0 * math.pi)
    )
    log_prob = torch.where(
        mask.reshape(*mask.shape, *([1] * (value.ndim - 3))), torch.zeros_like(log_prob), log_prob
    )
    return _reduce_event(log_prob, reduction)


def gaussian_mean_kl(
    policy_mean: Tensor,
    ref_mean: Tensor,
    std: Tensor,
    *,
    deterministic_mask: Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Return the same-variance Gaussian mean KL surrogate per transition.

    For equal isotropic variance, the official Flow-GRPO reference term is
    ``((policy_mean - ref_mean) ** 2) / (2 * std ** 2)`` averaged or summed
    over action dimensions. This is the main KL path; a sampled log-ratio is
    intentionally not substituted here.
    """

    if policy_mean.shape != ref_mean.shape:
        raise ValueError("policy_mean and ref_mean must have identical shapes")
    if policy_mean.ndim < 4:
        raise ValueError("policy_mean must have shape [batch, group, step, ...state]")
    _require_transition_shape("std", std, policy_mean)
    mask = _prepare_mask(deterministic_mask, policy_mean)
    compute_dtype = (
        torch.float32 if policy_mean.dtype in (torch.float16, torch.bfloat16) else policy_mean.dtype
    )
    policy_compute = policy_mean.to(dtype=compute_dtype)
    ref_compute = ref_mean.to(device=policy_mean.device, dtype=compute_dtype)
    expanded_std = _expand_transition_parameter(std, policy_mean).to(
        device=policy_mean.device, dtype=compute_dtype
    )
    expanded_std = torch.broadcast_to(expanded_std, policy_mean.shape)
    safe_std = _safe_std(expanded_std, mask, eps)
    kl = 0.5 * ((policy_compute - ref_compute) / safe_std).square()
    kl = torch.where(
        mask.reshape(*mask.shape, *([1] * (policy_mean.ndim - 3))), torch.zeros_like(kl), kl
    )
    return _reduce_event(kl, reduction)
