"""Strict behavior-policy and checkpoint EMA state for DiffusionOPSD."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

NamedParameters = Iterable[tuple[str, torch.nn.Parameter]]


def behavior_policy_retention(optimizer_update: int) -> float:
    """Paper App. C.1: ``min(1e-3 * u, 0.5)`` after update ``u``."""

    if isinstance(optimizer_update, bool) or optimizer_update < 1:
        raise ValueError("optimizer_update must be a positive integer")
    return min(1e-3 * optimizer_update, 0.5)


def checkpoint_policy_retention(optimizer_update: int) -> float:
    """Paper App. C.1 checkpoint averaging schedule."""

    if isinstance(optimizer_update, bool) or optimizer_update < 1:
        raise ValueError("optimizer_update must be a positive integer")
    return min((optimizer_update + 1.0) / (optimizer_update + 10.0), 0.9)


def _named(parameters: NamedParameters) -> dict[str, torch.nn.Parameter]:
    items = list(parameters)
    result = dict(items)
    if not result or any(not name for name in result):
        raise ValueError("EMA parameters must have unique non-empty names")
    if len(result) != len(items):
        raise ValueError("EMA parameter names must be unique")
    return result


class EMAParameterStore:
    """Independent named-parameter snapshot with fail-closed restore semantics."""

    def __init__(
        self, parameters: NamedParameters, *, device: torch.device | str | None = None
    ) -> None:
        current = _named(parameters)
        self._values = {
            name: parameter.detach().clone().to(device=device)
            for name, parameter in current.items()
        }
        self.updates = 0

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._values)

    @torch.no_grad()
    def update(self, parameters: NamedParameters, *, retention: float) -> None:
        if not math.isfinite(retention) or not 0 <= retention < 1:
            raise ValueError("EMA retention must be finite and in [0, 1)")
        current = _named(parameters)
        if tuple(current) != self.names:
            raise ValueError("EMA parameter names/order changed")
        for name, destination in self._values.items():
            source = current[name]
            if source.shape != destination.shape:
                raise ValueError(f"EMA parameter shape changed: {name}")
            destination.mul_(retention).add_(
                source.detach().to(device=destination.device, dtype=destination.dtype),
                alpha=1.0 - retention,
            )
        self.updates += 1

    @torch.no_grad()
    def copy_to(self, parameters: NamedParameters) -> None:
        current = _named(parameters)
        if tuple(current) != self.names:
            raise ValueError("EMA restore parameter names/order changed")
        for name, destination in current.items():
            source = self._values[name]
            if source.shape != destination.shape:
                raise ValueError(f"EMA restore parameter shape changed: {name}")
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "diffusion-opsd-ema-v1",
            "updates": self.updates,
            "values": {name: value.detach().cpu() for name, value in self._values.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format") != "diffusion-opsd-ema-v1":
            raise ValueError("unsupported DiffusionOPSD EMA format")
        values = state.get("values")
        if not isinstance(values, Mapping) or tuple(values) != self.names:
            raise ValueError("DiffusionOPSD EMA tensor inventory changed")
        updates = state.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("DiffusionOPSD EMA update count is invalid")
        for name, destination in self._values.items():
            source = values[name]
            if not isinstance(source, Tensor) or source.shape != destination.shape:
                raise ValueError(f"DiffusionOPSD EMA tensor is invalid: {name}")
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
        self.updates = updates


__all__ = [
    "EMAParameterStore",
    "behavior_policy_retention",
    "checkpoint_policy_retention",
]
