"""Sliding ODE/SDE windows for MixGRPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressiveMixWindow:
    """Move one contiguous SDE window from high to low noise.

    ``optimizable_steps`` may exclude a terminal ODE-only step.  Once the
    window reaches the final valid start it stays there instead of wrapping to
    high noise again.
    """

    optimizable_steps: int
    window_size: int
    shift_interval: int = 25
    stride: int = 1

    def __post_init__(self) -> None:
        if self.optimizable_steps <= 0:
            raise ValueError("optimizable_steps must be positive")
        if not 1 <= self.window_size <= self.optimizable_steps:
            raise ValueError("window_size must be within optimizable_steps")
        if self.shift_interval <= 0:
            raise ValueError("shift_interval must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    @property
    def final_start(self) -> int:
        return self.optimizable_steps - self.window_size

    def start(self, update_index: int) -> int:
        if update_index < 0:
            raise ValueError("update_index must be non-negative")
        shift_count = update_index // self.shift_interval
        return min(shift_count * self.stride, self.final_start)

    def indices(self, update_index: int) -> tuple[int, ...]:
        left = self.start(update_index)
        return tuple(range(left, left + self.window_size))


__all__ = ["ProgressiveMixWindow"]
