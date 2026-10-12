"""Validated Edit-R1 defaults from arXiv:2510.16888v3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Literal

EditR1Profile = Literal["sirius", "krea2", "sensenova"]
AdvantageNormalizer = Literal["global_std", "group_std"]
EDIT_R1_PAPER_REVISION = "arXiv:2510.16888v3"
EDIT_R1_UPSTREAM_REVISION = "4217f251a5abd3ca9ef2f9b4eee851a14f744f84"


@dataclass(frozen=True)
class EditR1SamplingConfig:
    resolution: int = 512
    rollout_steps: int = 6
    evaluation_steps: int = 15
    sampler: Literal["dpm2"] = "dpm2"
    group_size: int = 12
    groups_per_update: int = 24
    timestep_fraction: float = 0.99
    rollout_cfg_scale: float = 1.0
    evaluation_cfg_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "resolution",
            "rollout_steps",
            "evaluation_steps",
            "group_size",
            "groups_per_update",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.resolution % 32:
            raise ValueError("resolution must be divisible by 32")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if self.sampler != "dpm2":
            raise ValueError("Edit-R1 sampler must be dpm2")
        for name in (
            "timestep_fraction",
            "rollout_cfg_scale",
            "evaluation_cfg_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.timestep_fraction > 1:
            raise ValueError("timestep_fraction must be at most one")
        if self.rollout_cfg_scale < 1 or self.evaluation_cfg_scale < 1:
            raise ValueError("CFG scales must be at least one")


@dataclass(frozen=True)
class EditR1OptimizationConfig:
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4
    beta: float = 1.0
    kl_beta: float = 1e-4
    advantage_clip: float = 5.0
    advantage_normalizer: AdvantageNormalizer = "global_std"
    normalizer_epsilon: float = 1e-4
    ema_decay: float = 0.9
    adaptive_clean_weighting: bool = True
    adaptive_epsilon: float = 1e-5
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "beta",
            "advantage_clip",
            "normalizer_epsilon",
            "adaptive_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("adam_beta1", "adam_beta2", "ema_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in ("weight_decay", "kl_beta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.advantage_normalizer not in {"global_std", "group_std"}:
            raise ValueError("unsupported Edit-R1 advantage normalizer")
        if not isinstance(self.adaptive_clean_weighting, bool):
            raise TypeError("adaptive_clean_weighting must be a bool")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")


@dataclass(frozen=True)
class EditR1GroupFilterConfig:
    enabled: bool = True
    high_mean_threshold: float = 0.9
    low_std_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Edit-R1 group-filter enabled flag must be a bool")
        for name in ("high_mean_threshold", "low_std_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Edit-R1 reward thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class EditR1Config:
    """Model-neutral configuration split by lifecycle responsibility."""

    sampling: EditR1SamplingConfig = field(default_factory=EditR1SamplingConfig)
    optimization: EditR1OptimizationConfig = field(default_factory=EditR1OptimizationConfig)
    group_filter: EditR1GroupFilterConfig = field(default_factory=EditR1GroupFilterConfig)

    @property
    def candidates_per_update(self) -> int:
        return self.group_size * self.groups_per_update

    @property
    def resolution(self) -> int:
        return self.sampling.resolution

    @property
    def rollout_steps(self) -> int:
        return self.sampling.rollout_steps

    @property
    def evaluation_steps(self) -> int:
        return self.sampling.evaluation_steps

    @property
    def sampler(self) -> str:
        return self.sampling.sampler

    @property
    def group_size(self) -> int:
        return self.sampling.group_size

    @property
    def groups_per_update(self) -> int:
        return self.sampling.groups_per_update

    @property
    def timestep_fraction(self) -> float:
        return self.sampling.timestep_fraction

    @property
    def rollout_cfg_scale(self) -> float:
        return self.sampling.rollout_cfg_scale

    @property
    def evaluation_cfg_scale(self) -> float:
        return self.sampling.evaluation_cfg_scale

    @property
    def learning_rate(self) -> float:
        return self.optimization.learning_rate

    @property
    def beta(self) -> float:
        return self.optimization.beta

    @property
    def kl_beta(self) -> float:
        return self.optimization.kl_beta

    @property
    def advantage_clip(self) -> float:
        return self.optimization.advantage_clip

    @property
    def advantage_normalizer(self) -> AdvantageNormalizer:
        return self.optimization.advantage_normalizer

    @property
    def normalizer_epsilon(self) -> float:
        return self.optimization.normalizer_epsilon

    @property
    def ema_decay(self) -> float:
        return self.optimization.ema_decay

    @property
    def adaptive_clean_weighting(self) -> bool:
        return self.optimization.adaptive_clean_weighting

    @property
    def adaptive_epsilon(self) -> float:
        return self.optimization.adaptive_epsilon

    @property
    def max_grad_norm(self) -> float | None:
        return self.optimization.max_grad_norm

    @property
    def filter_low_variance_groups(self) -> bool:
        return self.group_filter.enabled

    @property
    def high_mean_threshold(self) -> float:
        return self.group_filter.high_mean_threshold

    @property
    def low_std_threshold(self) -> float:
        return self.group_filter.low_std_threshold


def edit_r1_defaults(
    profile: EditR1Profile,
    **overrides: object,
) -> EditR1Config:
    """Return paper defaults while accepting convenient flat field overrides."""

    if profile not in {"sirius", "krea2", "sensenova"}:
        raise ValueError(f"unsupported Edit-R1 profile: {profile}")
    base = EditR1Config()
    sampling_names = {item.name for item in fields(EditR1SamplingConfig)}
    optimization_names = {item.name for item in fields(EditR1OptimizationConfig)}
    filter_aliases = {
        "filter_low_variance_groups": "enabled",
        "high_mean_threshold": "high_mean_threshold",
        "low_std_threshold": "low_std_threshold",
    }
    sampling_overrides = {
        name: value for name, value in overrides.items() if name in sampling_names
    }
    optimization_overrides = {
        name: value for name, value in overrides.items() if name in optimization_names
    }
    filter_overrides = {
        filter_aliases[name]: value for name, value in overrides.items() if name in filter_aliases
    }
    known = sampling_names | optimization_names | set(filter_aliases)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown Edit-R1 config overrides: {unknown}")
    return EditR1Config(
        sampling=replace(base.sampling, **sampling_overrides),
        optimization=replace(base.optimization, **optimization_overrides),
        group_filter=replace(base.group_filter, **filter_overrides),
    )


__all__ = [
    "AdvantageNormalizer",
    "EDIT_R1_PAPER_REVISION",
    "EDIT_R1_UPSTREAM_REVISION",
    "EditR1Config",
    "EditR1GroupFilterConfig",
    "EditR1OptimizationConfig",
    "EditR1Profile",
    "EditR1SamplingConfig",
    "edit_r1_defaults",
]
