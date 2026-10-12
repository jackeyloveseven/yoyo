"""Trajectory Distribution Matching losses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import require_float_finite, require_matching_tensors

EventDims = int | Sequence[int] | None


def _event_dims(ndim: int, value: EventDims) -> tuple[int, ...]:
    if value is None:
        return tuple(range(1, ndim))
    if isinstance(value, int):
        if isinstance(value, bool) or value < 0 or value > ndim:
            raise ValueError("event_dims count is invalid")
        return tuple(range(ndim - value, ndim))
    dims = sorted((dim + ndim if dim < 0 else dim) for dim in value)
    if len(set(dims)) != len(dims) or any(dim < 0 or dim >= ndim for dim in dims):
        raise ValueError("event_dims contains invalid or duplicate dimensions")
    if dims and tuple(dims) != tuple(range(dims[0], ndim)):
        raise ValueError("event_dims must be a contiguous trailing block")
    return tuple(dims)


def _compute_dtype(*values: Tensor) -> torch.dtype:
    return (
        torch.float32
        if any(value.dtype in (torch.float16, torch.bfloat16) for value in values)
        else values[0].dtype
    )


def _leading_shape(value: Tensor, dims: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(size for index, size in enumerate(value.shape) if index not in dims)


def _weight(name: str, value: Tensor, reference: Tensor, dims: tuple[int, ...]) -> Tensor:
    require_float_finite(name, value)
    expected = _leading_shape(reference, dims)
    if value.shape != expected:
        raise ValueError(f"{name} must have leading shape {expected}")
    if value.device != reference.device or bool((value < 0).any()):
        raise ValueError(f"{name} must be non-negative on the prediction device")
    return value.to(dtype=_compute_dtype(reference, value))


@dataclass(frozen=True)
class TDMFakeScoreMetrics:
    """Fake-score denoising regression and diagnostics."""

    total_loss: Tensor
    per_sample_loss: Tensor
    raw_mse: Tensor
    fake_student_rms: Tensor


def tdm_fake_score_loss(
    fake_clean: Tensor,
    student_clean: Tensor,
    snr_weight: Tensor,
    importance_weight: Tensor,
    *,
    event_dims: EventDims = None,
) -> TDMFakeScoreMetrics:
    """Fit the fake score to the current student trajectory distribution."""

    require_matching_tensors(fake_clean=fake_clean, student_clean=student_clean)
    dims = _event_dims(fake_clean.ndim, event_dims)
    dtype = _compute_dtype(fake_clean, student_clean)
    squared = (fake_clean.to(dtype) - student_clean.detach().to(dtype)).square()
    raw_mse = squared.mean(dim=dims) if dims else squared
    snr = _weight("snr_weight", snr_weight, fake_clean, dims)
    importance = _weight("importance_weight", importance_weight, fake_clean, dims)
    per_sample = raw_mse * snr * importance
    total = per_sample.mean()
    if not torch.isfinite(total):
        raise ValueError("TDM fake-score loss must be finite")
    return TDMFakeScoreMetrics(
        total_loss=total,
        per_sample_loss=per_sample,
        raw_mse=raw_mse,
        fake_student_rms=squared.mean().sqrt(),
    )


@dataclass(frozen=True)
class TDMGeneratorMetrics:
    """Student cooperative target loss and diagnostics."""

    total_loss: Tensor
    per_sample_loss: Tensor
    cooperative_target: Tensor
    normalizer: Tensor
    teacher_fake_delta_rms: Tensor
    student_teacher_rms: Tensor


def tdm_generator_loss(
    student_clean: Tensor,
    teacher_clean: Tensor,
    fake_clean: Tensor,
    *,
    teacher_unconditional_clean: Tensor | None = None,
    cfg_scale: float = 1.0,
    huber_c: float | None = 1e-3,
    normalizer_floor: float = 1e-6,
    event_dims: EventDims = None,
) -> TDMGeneratorMetrics:
    """Move the student along the stopped ``teacher - fake`` score direction."""

    require_matching_tensors(
        student_clean=student_clean,
        teacher_clean=teacher_clean,
        fake_clean=fake_clean,
    )
    if teacher_unconditional_clean is not None:
        require_matching_tensors(
            student_clean=student_clean,
            teacher_unconditional_clean=teacher_unconditional_clean,
        )
    if not torch.isfinite(torch.tensor(cfg_scale, dtype=torch.float64)) or cfg_scale < 1:
        raise ValueError("cfg_scale must be finite and at least 1")
    if cfg_scale > 1 and teacher_unconditional_clean is None:
        raise ValueError("CFG requires teacher_unconditional_clean")
    if huber_c is not None and (
        not torch.isfinite(torch.tensor(huber_c, dtype=torch.float64)) or huber_c <= 0
    ):
        raise ValueError("huber_c must be finite and positive when supplied")
    if (
        not torch.isfinite(torch.tensor(normalizer_floor, dtype=torch.float64))
        or normalizer_floor <= 0
    ):
        raise ValueError("normalizer_floor must be finite and positive")

    dims = _event_dims(student_clean.ndim, event_dims)
    dtype = _compute_dtype(student_clean, teacher_clean, fake_clean)
    student = student_clean.to(dtype)
    teacher = teacher_clean.detach().to(dtype)
    fake = fake_clean.detach().to(dtype)
    if teacher_unconditional_clean is None:
        teacher_cfg = teacher
    else:
        unconditional = teacher_unconditional_clean.detach().to(dtype)
        teacher_cfg = unconditional + cfg_scale * (teacher - unconditional)
    cooperative = (student.detach() + teacher_cfg - fake).detach()
    normalizer_error = (student.detach() - teacher_cfg).abs()
    normalizer = (normalizer_error.mean(dim=dims) if dims else normalizer_error).clamp_min(
        normalizer_floor
    )
    expanded_normalizer = normalizer.reshape(*normalizer.shape, *([1] * len(dims)))
    delta = student - cooperative
    if huber_c is None:
        pointwise = delta.square()
    else:
        pointwise = torch.sqrt(delta.square() + huber_c**2) - huber_c
    pointwise = pointwise / expanded_normalizer
    per_sample = pointwise.mean(dim=dims) if dims else pointwise
    total = per_sample.mean()
    if not torch.isfinite(total):
        raise ValueError("TDM generator loss must be finite")
    return TDMGeneratorMetrics(
        total_loss=total,
        per_sample_loss=per_sample,
        cooperative_target=cooperative,
        normalizer=normalizer,
        teacher_fake_delta_rms=(teacher_cfg - fake).square().mean().sqrt(),
        student_teacher_rms=(student.detach() - teacher_cfg).square().mean().sqrt(),
    )


__all__ = [
    "TDMFakeScoreMetrics",
    "TDMGeneratorMetrics",
    "tdm_fake_score_loss",
    "tdm_generator_loss",
]
