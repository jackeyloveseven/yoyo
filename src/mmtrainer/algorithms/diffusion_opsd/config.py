"""Validated paper defaults for DiffusionOPSD (arXiv:2608.24646)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

BackboneProfile = Literal["kio", "sirius", "sensenova"]


@dataclass(frozen=True)
class DiffusionOPSDConfig:
    """Model-independent algorithm settings.

    The paper uses ``query_sigma=0.273`` for the native few-step Turbo model
    and ``0.278`` for SD3.5-M.  Our three adapters are flow models and select
    the nearest non-zero value from their own native rollout schedule.
    """

    query_sigma: float = 0.273
    target_radius: float = 0.10
    target_steps: int = 2
    target_step_multiplier: float = 1.0
    branch_beta: float = 1.0
    advantage_scale: float = 5.0
    reward_std_epsilon: float = 1e-4
    gradient_epsilon: float = 1e-12
    residual_epsilon: float = 1e-5
    target_microbatch: int = 6
    fitting_steps: int = 1
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        finite_positive = {
            "query_sigma": self.query_sigma,
            "target_radius": self.target_radius,
            "target_step_multiplier": self.target_step_multiplier,
            "branch_beta": self.branch_beta,
            "advantage_scale": self.advantage_scale,
            "reward_std_epsilon": self.reward_std_epsilon,
            "gradient_epsilon": self.gradient_epsilon,
            "residual_epsilon": self.residual_epsilon,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.query_sigma >= 1:
            raise ValueError("query_sigma must lie strictly inside (0, 1)")
        if self.target_steps < 1:
            raise ValueError("target_steps must be positive")
        if (
            isinstance(self.target_microbatch, bool)
            or not isinstance(self.target_microbatch, int)
            or self.target_microbatch < 1
        ):
            raise ValueError("target_microbatch must be a positive integer")
        if self.fitting_steps < 1:
            raise ValueError("fitting_steps must be positive")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")


def diffusion_opsd_defaults(
    profile: BackboneProfile,
    **overrides: object,
) -> DiffusionOPSDConfig:
    """Return a fresh paper-aligned configuration for one internal backbone."""

    if profile not in {"kio", "sirius", "sensenova"}:
        raise ValueError(f"unsupported DiffusionOPSD profile: {profile}")
    base = DiffusionOPSDConfig(query_sigma=0.273)
    return replace(base, **overrides)


__all__ = ["BackboneProfile", "DiffusionOPSDConfig", "diffusion_opsd_defaults"]
