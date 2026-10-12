"""Configuration defaults for the reusable Flow-GRPO primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlowGRPOConfig:
    """Public SenseNova Stage-5 defaults without model/trainer assumptions."""

    group_size: int = 16
    rollout_steps: int = 10
    noise_level: float = 0.7
    cfg_scale: float = 4.0
    kl_beta: float = 0.01
    learning_rate: float = 1e-5
    prompts_per_epoch: int = 48
    ocr_epochs: int = 600
    dynamic_resolution_warmup_epochs: int = 200
    general_rl_epochs: int = 1600
    clip_epsilon: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "group_size",
            "rollout_steps",
            "prompts_per_epoch",
            "ocr_epochs",
            "dynamic_resolution_warmup_epochs",
            "general_rl_epochs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 1:
            raise ValueError("cfg_scale must be finite and at least 1")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.clip_epsilon is not None and not (0 < self.clip_epsilon < 1):
            raise ValueError("clip_epsilon must be in (0, 1) when supplied")
