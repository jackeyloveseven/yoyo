"""Core tensor containers for model-independent Flow-GRPO code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _require_floating(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")


def _require_numeric(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not (
        value.is_floating_point()
        or value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise TypeError(f"{name} must use a numeric dtype")


def _require_prefix_shape(name: str, value: Tensor, prefix: tuple[int, ...]) -> None:
    if value.ndim == 0:
        return
    if value.ndim < len(prefix) or tuple(value.shape[: len(prefix)]) != prefix:
        raise ValueError(
            f"{name} must be scalar or start with trajectory prefix {prefix}; "
            f"got shape {tuple(value.shape)}"
        )
    if value.ndim > len(prefix) and any(dim != 1 for dim in value.shape[len(prefix) :]):
        raise ValueError(
            f"{name} event dimensions must be singleton when supplied; "
            f"got shape {tuple(value.shape)}"
        )


def _trajectory_prefix(
    states: Tensor,
    next_states: Tensor,
    timesteps: Tensor,
    transition_std: Tensor,
) -> tuple[int, ...]:
    if states.ndim < 4:
        raise ValueError("states must be [batch, group, step, ...state]")
    if next_states.shape != states.shape:
        raise ValueError("states and next_states must have identical shapes")
    prefix = tuple(states.shape[:3])
    if tuple(timesteps.shape) != prefix:
        raise ValueError(f"timesteps must have shape {prefix}")
    _require_prefix_shape("transition_std", transition_std, prefix)
    return prefix


def _resolved_mask(mask: Tensor | None, prefix: tuple[int, ...], device: Any) -> Tensor:
    if mask is None:
        return torch.zeros(prefix, dtype=torch.bool, device=device)
    if mask.dtype != torch.bool:
        raise TypeError("deterministic_mask must use torch.bool")
    if tuple(mask.shape) != prefix:
        raise ValueError(f"deterministic_mask must have shape {prefix}")
    return mask


def _validate_trajectory_values(
    timesteps: Tensor,
    transition_std: Tensor,
) -> None:
    if transition_std.numel() and not torch.isfinite(transition_std).all():
        raise ValueError("transition_std must contain only finite values")
    if transition_std.numel() and bool((transition_std < 0).any()):
        raise ValueError("transition_std must be non-negative")
    if timesteps.numel() and not torch.isfinite(timesteps).all():
        raise ValueError("timesteps must contain only finite values")


def _validate_trajectory_devices(
    states: Tensor,
    next_states: Tensor,
    timesteps: Tensor,
    transition_std: Tensor,
    deterministic_mask: Tensor,
) -> None:
    for name, value in (
        ("next_states", next_states),
        ("timesteps", timesteps),
        ("transition_std", transition_std),
        ("deterministic_mask", deterministic_mask),
    ):
        if value.device != states.device:
            raise ValueError(f"{name} must be on the same device as states")


@dataclass(frozen=True)
class TrajectoryBatch:
    """A batched group of flow trajectories.

    Tensors use ``[batch, group, step, ...state]``. ``transition_std`` and
    ``deterministic_mask`` are per transition and therefore use
    ``[batch, group, step]`` (singleton state dimensions are also accepted).
    """

    states: Tensor
    next_states: Tensor
    timesteps: Tensor
    transition_std: Tensor
    deterministic_mask: Tensor | None = None

    def __post_init__(self) -> None:
        _require_floating("states", self.states)
        _require_floating("next_states", self.next_states)
        _require_numeric("timesteps", self.timesteps)
        _require_floating("transition_std", self.transition_std)

        prefix = _trajectory_prefix(
            self.states, self.next_states, self.timesteps, self.transition_std
        )
        mask = _resolved_mask(self.deterministic_mask, prefix, self.states.device)
        object.__setattr__(self, "deterministic_mask", mask)
        _validate_trajectory_values(self.timesteps, self.transition_std)
        _validate_trajectory_devices(
            self.states,
            self.next_states,
            self.timesteps,
            self.transition_std,
            mask,
        )

    @property
    def batch_size(self) -> int:
        """Number of prompt instances in the batch."""

        return self.states.shape[0]

    @property
    def group_size(self) -> int:
        """Number of rollouts per prompt."""

        return self.states.shape[1]

    @property
    def rollout_steps(self) -> int:
        """Number of stored transitions per rollout."""

        return self.states.shape[2]


@dataclass(frozen=True)
class TransitionDistribution:
    """Per-step isotropic Gaussian transition parameters."""

    mean: Tensor
    std: Tensor
    deterministic_mask: Tensor | None = None

    def __iter__(self):
        """Allow ``mean, std = distribution`` while retaining mask metadata."""

        yield self.mean
        yield self.std

    def __post_init__(self) -> None:
        _require_floating("mean", self.mean)
        _require_floating("std", self.std)
        if self.mean.ndim < 4:
            raise ValueError("mean must have shape [batch, group, step, ...state]")
        prefix = tuple(self.mean.shape[:3])
        _require_prefix_shape("std", self.std, prefix)
        if self.std.numel() and not torch.isfinite(self.std).all():
            raise ValueError("std must contain only finite values")
        if self.std.numel() and (self.std < 0).any():
            raise ValueError("std must be non-negative")

        if self.deterministic_mask is None:
            object.__setattr__(
                self,
                "deterministic_mask",
                torch.zeros(prefix, dtype=torch.bool, device=self.mean.device),
            )
        elif self.deterministic_mask.dtype != torch.bool:
            raise TypeError("deterministic_mask must use torch.bool")
        elif tuple(self.deterministic_mask.shape) != prefix:
            raise ValueError("deterministic_mask must match the std prefix")
        if self.mean.device != self.std.device:
            raise ValueError("mean and std must be on the same device")
        if self.deterministic_mask.device != self.mean.device:
            raise ValueError("deterministic_mask must be on the same device as mean")
