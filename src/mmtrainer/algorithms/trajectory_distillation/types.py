"""Tensor containers for Trajectory Distribution Matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from mmtrainer.algorithms.common import require_float_finite


@dataclass(frozen=True)
class TDMStudentSample:
    """One student trajectory anchor and its predicted clean endpoint."""

    clean: Tensor
    context: Any = None

    def __post_init__(self) -> None:
        require_float_finite("student clean", self.clean)


@dataclass(frozen=True)
class TDMScoreBatch:
    """Noised student anchor queried by teacher and fake score models."""

    noisy: Tensor
    noise_time: Tensor
    model_time: Tensor
    snr_weight: Tensor
    importance_weight: Tensor
    context: Any = None

    def __post_init__(self) -> None:
        for name, value in (
            ("score noisy", self.noisy),
            ("noise_time", self.noise_time),
            ("model_time", self.model_time),
            ("snr_weight", self.snr_weight),
            ("importance_weight", self.importance_weight),
        ):
            require_float_finite(name, value)
        if self.noise_time.shape != self.model_time.shape:
            raise ValueError("noise_time and model_time must have matching shapes")
        if self.snr_weight.shape != self.noise_time.shape:
            raise ValueError("snr_weight must match noise_time shape")
        if self.importance_weight.shape != self.noise_time.shape:
            raise ValueError("importance_weight must match noise_time shape")
        if bool(((self.noise_time <= 0) | (self.noise_time >= 1)).any()):
            raise ValueError("noise_time must lie strictly inside (0, 1)")
        if bool((self.snr_weight < 0).any()) or bool((self.importance_weight < 0).any()):
            raise ValueError("TDM score weights must be non-negative")


@dataclass(frozen=True)
class TDMTeacherRollout:
    """One teacher trajectory rollout and its terminal clean prediction."""

    clean: Tensor
    timesteps: Tensor | None = None
    trajectory: tuple[Tensor, ...] | None = None
    context: Any = None

    def __post_init__(self) -> None:
        require_float_finite("teacher clean", self.clean)
        if self.timesteps is not None:
            require_float_finite("teacher timesteps", self.timesteps)
        if self.trajectory is not None:
            for item in self.trajectory:
                require_float_finite("teacher trajectory state", item)


__all__ = ["TDMScoreBatch", "TDMStudentSample", "TDMTeacherRollout"]
