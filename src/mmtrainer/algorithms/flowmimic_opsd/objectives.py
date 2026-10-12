"""Clean-output distribution mimicry from FlowMimic Sec. 3.3, Eqs. 30--43."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import rectified_flow_clean_output, require_float_finite


def flow_clean_output(
    query_state: Tensor,
    query_sigma: Tensor,
    velocity: Tensor,
) -> Tensor:
    """Rectified-flow clean estimate: ``x0_hat = xt - sigma * velocity``."""

    return rectified_flow_clean_output(query_state, query_sigma, velocity)


@dataclass(frozen=True)
class FlowMimicOPSDLoss:
    total_loss: Tensor
    mimic_loss: Tensor
    base_loss: Tensor
    per_sample_kl: Tensor
    per_sample_reverse_kl: Tensor
    naive_clean: Tensor
    favorable_clean: Tensor
    naive_log_probabilities: Tensor
    favorable_log_probabilities: Tensor


@dataclass(frozen=True)
class FlowMimicOPSDInputs:
    query_state: Tensor
    query_sigma: Tensor
    naive_velocity: Tensor
    favorable_velocity: Tensor
    base_loss: Tensor | None


@dataclass(frozen=True)
class FlowMimicOPSDLossConfig:
    temperature: float = 1.0
    mimic_weight: float = 1.0
    base_loss_weight: float = 1.0
    require_base_loss: bool = True


def flowmimic_clean_distribution_loss(
    inputs: FlowMimicOPSDInputs,
    config: FlowMimicOPSDLossConfig | None = None,
) -> FlowMimicOPSDLoss:
    """Main FlowMimic KL with live gradients through both conditioning paths.

    The favourable distribution is the first argument of the paper's asymmetric
    KL. It is intentionally *not* detached: both paths share the same trainable
    model and the main FlowMimic objective performs mutual capability alignment.
    """

    config = config or FlowMimicOPSDLossConfig()
    for name, value in (
        ("temperature", config.temperature),
        ("mimic_weight", config.mimic_weight),
        ("base_loss_weight", config.base_loss_weight),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not torch.isfinite(torch.tensor(float(value), dtype=torch.float64))
        ):
            raise ValueError(f"{name} must be finite")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not isinstance(config.require_base_loss, bool):
        raise TypeError("require_base_loss must be boolean")
    for name, value in (
        ("query_state", inputs.query_state),
        ("naive_velocity", inputs.naive_velocity),
        ("favorable_velocity", inputs.favorable_velocity),
    ):
        require_float_finite(name, value)
    if inputs.naive_velocity.shape != inputs.query_state.shape:
        raise ValueError("naive_velocity must match query_state")
    if inputs.favorable_velocity.shape != inputs.query_state.shape:
        raise ValueError("favorable_velocity must match query_state")

    naive_clean = flow_clean_output(inputs.query_state, inputs.query_sigma, inputs.naive_velocity)
    favorable_clean = flow_clean_output(
        inputs.query_state,
        inputs.query_sigma,
        inputs.favorable_velocity,
    )
    if naive_clean.ndim < 3:
        raise ValueError("FlowMimic tensors must be [batch, group, ...]")
    leading = naive_clean.shape[:2]
    naive_flat = naive_clean.float().reshape(*leading, -1) / float(config.temperature)
    favorable_flat = favorable_clean.float().reshape(*leading, -1) / float(config.temperature)
    naive_log = torch.log_softmax(naive_flat, dim=-1)
    favorable_log = torch.log_softmax(favorable_flat, dim=-1)
    naive_probability = naive_log.exp()
    favorable_probability = favorable_log.exp()
    per_sample_kl = (favorable_probability * (favorable_log - naive_log)).sum(dim=-1)
    per_sample_reverse = (naive_probability * (naive_log - favorable_log)).sum(dim=-1)
    mimic_loss = per_sample_kl.mean()

    if inputs.base_loss is None:
        if config.require_base_loss:
            raise ValueError("FlowMimic main training requires the native flow-matching base loss")
        resolved_base = mimic_loss.new_zeros(())
    else:
        require_float_finite("base_loss", inputs.base_loss)
        if inputs.base_loss.numel() != 1:
            raise ValueError("base_loss must be scalar")
        resolved_base = inputs.base_loss.reshape(()).to(
            device=mimic_loss.device,
            dtype=mimic_loss.dtype,
        )
    total = float(config.base_loss_weight) * resolved_base + float(config.mimic_weight) * mimic_loss
    require_float_finite("FlowMimic total loss", total)
    return FlowMimicOPSDLoss(
        total_loss=total,
        mimic_loss=mimic_loss,
        base_loss=resolved_base,
        per_sample_kl=per_sample_kl,
        per_sample_reverse_kl=per_sample_reverse,
        naive_clean=naive_clean,
        favorable_clean=favorable_clean,
        naive_log_probabilities=naive_log,
        favorable_log_probabilities=favorable_log,
    )


__all__ = [
    "FlowMimicOPSDLoss",
    "FlowMimicOPSDInputs",
    "FlowMimicOPSDLossConfig",
    "flow_clean_output",
    "flowmimic_clean_distribution_loss",
]
