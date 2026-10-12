"""Explicit learning-rate schedules for multimodal flow and diffusion training."""

from __future__ import annotations

import math
from typing import Any, Literal

from torch.optim.lr_scheduler import LambdaLR

ScheduleKind = Literal["constant", "cosine", "wsd"]


def _validate_schedule_params(
    total_steps: int,
    warmup_steps: int,
    kind: ScheduleKind,
    wsd_decay_ratio: float,
) -> None:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup < total_steps")
    if kind not in {"constant", "cosine", "wsd"}:
        raise ValueError(f"unsupported schedule: {kind}")
    if not 0 < wsd_decay_ratio <= 1:
        raise ValueError("wsd_decay_ratio must be within (0, 1]")


def lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    kind: ScheduleKind = "cosine",
    wsd_decay_ratio: float = 0.2,
) -> float:
    """Return the exact warmup/cosine or warmup-stable-decay LR multiplier."""
    _validate_schedule_params(total_steps, warmup_steps, kind, wsd_decay_ratio)
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    if kind == "constant":
        return 1.0
    if step >= total_steps:
        return 0.0
    if kind == "wsd":
        decay_start = int(total_steps * (1.0 - wsd_decay_ratio))
        if step < decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(
    optimizer: Any,
    *,
    total_steps: int,
    warmup_steps: int,
    kind: ScheduleKind = "cosine",
    wsd_decay_ratio: float = 0.2,
) -> LambdaLR:
    """Build a PyTorch ``LambdaLR`` from the schedule multiplier."""
    _validate_schedule_params(total_steps, warmup_steps, kind, wsd_decay_ratio)
    return LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            kind=kind,
            wsd_decay_ratio=wsd_decay_ratio,
        ),
    )
