"""Shared tensor validation and event-dimension conventions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

EventDims = int | Sequence[int] | None


def require_float_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def working_dtype(*values: Tensor) -> torch.dtype:
    if any(value.dtype in (torch.float16, torch.bfloat16) for value in values):
        return torch.float32
    return values[0].dtype


def normalize_event_dims(ndim: int, event_dims: EventDims) -> tuple[int, ...]:
    if event_dims is None:
        return tuple(range(1, ndim)) if ndim > 1 else ()
    if isinstance(event_dims, int):
        if isinstance(event_dims, bool) or event_dims < 0 or event_dims > ndim:
            raise ValueError(f"event_dims count must be in [0, {ndim}]")
        return tuple(range(ndim - event_dims, ndim))

    normalized: list[int] = []
    for dim in event_dims:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("event_dims must contain integers")
        resolved = dim + ndim if dim < 0 else dim
        if resolved < 0 or resolved >= ndim:
            raise ValueError(f"event dimension {dim} is out of range for rank {ndim}")
        normalized.append(resolved)
    if len(set(normalized)) != len(normalized):
        raise ValueError("event_dims must not contain duplicates")
    normalized.sort()
    if normalized and tuple(normalized) != tuple(range(normalized[0], ndim)):
        raise ValueError("event_dims must be a contiguous trailing block")
    return tuple(normalized)


def broadcast_strict_fraction(name: str, value: Tensor, reference: Tensor) -> Tensor:
    require_float_finite(name, value)
    candidate = value
    while candidate.ndim < reference.ndim:
        candidate = candidate.unsqueeze(-1)
    try:
        candidate = torch.broadcast_to(candidate, reference.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to the reference tensor") from error
    if bool(((candidate <= 0) | (candidate >= 1)).any()):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return candidate.to(device=reference.device, dtype=reference.dtype)


def require_matching_tensors(**values: Tensor) -> None:
    first_name, first = next(iter(values.items()))
    require_float_finite(first_name, first)
    for name, value in tuple(values.items())[1:]:
        require_float_finite(name, value)
        if value.shape != first.shape or value.device != first.device:
            raise ValueError(f"{name} must match {first_name} shape and device")


def rectified_flow_clean_output(
    query_state: Tensor,
    query_sigma: Tensor,
    velocity: Tensor,
) -> Tensor:
    require_matching_tensors(query_state=query_state, velocity=velocity)
    sigma = broadcast_strict_fraction("query_sigma", query_sigma, query_state)
    return query_state - sigma * velocity
