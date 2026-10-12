"""Validated settings for the FlowMimic-OPSD conditioning-path objective."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

BackboneProfile = Literal["kio", "sirius", "sensenova"]


@dataclass(frozen=True)
class FlowMimicOPSDConfig:
    """Shared settings for one favourable/naive clean-output mimic update."""

    query_sigma: float = 0.273
    temperature: float = 1.0
    mimic_weight: float = 1.0
    base_loss_weight: float = 1.0
    require_base_loss: bool = True
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("query_sigma", self.query_sigma),
            ("temperature", self.temperature),
            ("mimic_weight", self.mimic_weight),
            ("base_loss_weight", self.base_loss_weight),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.query_sigma >= 1:
            raise ValueError("query_sigma must lie strictly inside (0, 1)")
        if not isinstance(self.require_base_loss, bool):
            raise TypeError("require_base_loss must be boolean")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")


def flowmimic_opsd_defaults(
    profile: BackboneProfile,
    **overrides: object,
) -> FlowMimicOPSDConfig:
    if profile not in {"kio", "sirius", "sensenova"}:
        raise ValueError(f"unsupported FlowMimic-OPSD profile: {profile}")
    return replace(FlowMimicOPSDConfig(), **overrides)


__all__ = [
    "BackboneProfile",
    "FlowMimicOPSDConfig",
    "flowmimic_opsd_defaults",
]
