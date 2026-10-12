"""Pluggable reward composition and alternating reward-group scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


class Reward(Protocol):
    """Callable reward interface used by :class:`RewardComposer`."""

    def __call__(self, samples: Any, context: Any = None) -> Tensor: ...


@dataclass(frozen=True)
class WeightedReward:
    """A named reward callable and its linear-combination weight."""

    name: str
    reward: Reward
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward name must not be empty")
        if not torch.isfinite(torch.tensor(self.weight)):
            raise ValueError("reward weight must be finite")


@dataclass(frozen=True)
class RewardGroup:
    """A weighted sum of reward components."""

    name: str
    components: tuple[WeightedReward, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward group name must not be empty")
        if not self.components:
            raise ValueError("reward group must contain at least one component")


@dataclass(frozen=True)
class AlternatingRewardSchedule:
    """Description of a periodic reward-group schedule."""

    group_names: tuple[str, ...]
    period_epochs: int = 1
    start_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.group_names:
            raise ValueError("group_names must not be empty")
        if self.period_epochs <= 0:
            raise ValueError("period_epochs must be positive")
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")

    def group_for_epoch(self, epoch: int) -> str:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch < self.start_epoch:
            return self.group_names[0]
        index = (epoch - self.start_epoch) // self.period_epochs
        return self.group_names[index % len(self.group_names)]


@dataclass(frozen=True)
class RewardComposition:
    """Reward tensor plus component tensors for logging."""

    reward: Tensor
    group_name: str
    components: dict[str, Tensor]


class RewardComposer:
    """Compose weighted rewards and select groups by epoch."""

    def __init__(
        self,
        groups: Sequence[RewardGroup],
        schedule: AlternatingRewardSchedule | None = None,
    ) -> None:
        self._groups = {group.name: group for group in groups}
        if len(self._groups) != len(groups):
            raise ValueError("reward group names must be unique")
        if not self._groups:
            raise ValueError("at least one reward group is required")
        if schedule is not None and any(name not in self._groups for name in schedule.group_names):
            raise ValueError("reward schedule references an unknown group")
        self.schedule = schedule

    @property
    def groups(self) -> tuple[str, ...]:
        """Available reward-group names."""

        return tuple(self._groups)

    def group_for_epoch(self, epoch: int) -> RewardGroup:
        name = self.schedule.group_for_epoch(epoch) if self.schedule else self.groups[0]
        return self._groups[name]

    def compose(self, samples: Any, context: Any = None, *, epoch: int = 0) -> RewardComposition:
        group = self.group_for_epoch(epoch)
        values: dict[str, Tensor] = {}
        total: Tensor | None = None
        for component in group.components:
            value = component.reward(samples, context)
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"reward {component.name!r} must return a floating-point Tensor")
            values[component.name] = value
            total = component.weight * value if total is None else total + component.weight * value
        assert total is not None
        return RewardComposition(reward=total, group_name=group.name, components=values)

    def __call__(self, samples: Any, context: Any = None, *, epoch: int = 0) -> Tensor:
        """Return only the composed reward tensor."""

        return self.compose(samples, context, epoch=epoch).reward
