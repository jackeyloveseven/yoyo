"""Flow-matching DPO with FireRed's asymmetric positive anchoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from mmtrainer.algorithms.common import (
    EventDims,
    normalize_event_dims,
    require_float_finite,
    working_dtype,
)


def flow_matching_event_mse(
    prediction: Tensor,
    target: Tensor,
    *,
    event_dims: EventDims = None,
) -> Tensor:
    """Return flow-matching MSE reduced over event dimensions only.

    With ``event_dims=None``, tensors are interpreted as ``[batch, ...event]``.
    Use an integer event rank or explicit trailing dimensions when there are
    multiple leading dimensions, for example ``event_dims=3`` for
    ``[batch, group, channels, height, width]``.
    """

    require_float_finite("prediction", prediction)
    require_float_finite("target", target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    dims = normalize_event_dims(prediction.ndim, event_dims)
    work_dtype = working_dtype(prediction, target)
    squared_error = (prediction.to(work_dtype) - target.to(work_dtype)).square()
    return squared_error.mean(dim=dims) if dims else squared_error


@dataclass(frozen=True)
class AsymmetricDPOMetrics:
    """Asymmetric DPO objective and per-pair diagnostics."""

    loss: Tensor
    dpo_loss: Tensor
    psr_loss: Tensor
    logits: Tensor
    win_diff: Tensor
    lose_diff: Tensor
    implicit_accuracy: Tensor
    double_degradation_fraction: Tensor


@dataclass(frozen=True)
class AsymmetricDPOInputs:
    policy_chosen: Tensor
    policy_rejected: Tensor
    reference_chosen: Tensor
    reference_rejected: Tensor
    chosen_target: Tensor
    rejected_target: Tensor


@dataclass(frozen=True)
class AsymmetricDPOConfig:
    beta: float
    win_weight: float = 1.0
    psr_weight: float = 0.0
    event_dims: EventDims = None


def asymmetric_dpo_loss(
    inputs: AsymmetricDPOInputs,
    config: AsymmetricDPOConfig,
) -> AsymmetricDPOMetrics:
    """Compute FireRed-style asymmetric flow-matching DPO plus PSR.

    ``win_diff`` and ``lose_diff`` are policy-minus-reference event MSEs.
    Reference predictions and flow-matching targets are detached so only the
    policy predictions receive gradients.
    """

    for name, value in (
        ("beta", config.beta),
        ("win_weight", config.win_weight),
        ("psr_weight", config.psr_weight),
    ):
        if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
            raise ValueError(f"{name} must be finite")
    if config.beta <= 0:
        raise ValueError("beta must be positive")
    if config.win_weight <= 0:
        raise ValueError("win_weight must be positive")
    if config.psr_weight < 0:
        raise ValueError("psr_weight must be non-negative")

    tensors = {
        "policy_chosen": inputs.policy_chosen,
        "policy_rejected": inputs.policy_rejected,
        "reference_chosen": inputs.reference_chosen,
        "reference_rejected": inputs.reference_rejected,
        "chosen_target": inputs.chosen_target,
        "rejected_target": inputs.rejected_target,
    }
    for name, value in tensors.items():
        require_float_finite(name, value)
    expected_shape = inputs.policy_chosen.shape
    if any(value.shape != expected_shape for value in tensors.values()):
        shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        raise ValueError(f"all DPO tensors must have the same shape, got {shapes}")
    if any(value.device != inputs.policy_chosen.device for value in tensors.values()):
        raise ValueError("all DPO tensors must be on the same device")

    chosen_target = inputs.chosen_target.detach()
    rejected_target = inputs.rejected_target.detach()
    policy_chosen_loss = flow_matching_event_mse(
        inputs.policy_chosen,
        chosen_target,
        event_dims=config.event_dims,
    )
    policy_rejected_loss = flow_matching_event_mse(
        inputs.policy_rejected,
        rejected_target,
        event_dims=config.event_dims,
    )
    reference_chosen_loss = flow_matching_event_mse(
        inputs.reference_chosen.detach(),
        chosen_target,
        event_dims=config.event_dims,
    )
    reference_rejected_loss = flow_matching_event_mse(
        inputs.reference_rejected.detach(),
        rejected_target,
        event_dims=config.event_dims,
    )
    win_diff = policy_chosen_loss - reference_chosen_loss
    lose_diff = policy_rejected_loss - reference_rejected_loss
    logits = config.beta * (lose_diff - config.win_weight * win_diff)
    if not torch.isfinite(logits).all():
        raise ValueError("DPO logits must contain only finite values")

    dpo_loss = -F.logsigmoid(logits).mean()
    psr_loss = config.psr_weight * policy_chosen_loss.mean()
    loss = dpo_loss + psr_loss
    if not torch.isfinite(loss):
        raise ValueError("DPO loss must be finite")
    implicit_accuracy = (logits > 0).to(logits.dtype).mean()
    double_degradation_fraction = ((win_diff > 0) & (lose_diff > 0)).to(logits.dtype).mean()
    return AsymmetricDPOMetrics(
        loss=loss,
        dpo_loss=dpo_loss,
        psr_loss=psr_loss,
        logits=logits,
        win_diff=win_diff,
        lose_diff=lose_diff,
        implicit_accuracy=implicit_accuracy,
        double_degradation_fraction=double_degradation_fraction,
    )
