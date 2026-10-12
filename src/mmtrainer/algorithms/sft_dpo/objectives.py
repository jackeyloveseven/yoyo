"""Model-neutral flow-matching SFT, DPO, and OPD objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowDpoLoss:
    """All tensors needed for backward, logging, and global DPO metrics."""

    loss: torch.Tensor
    dpo: torch.Tensor
    reference_mse: torch.Tensor
    sft_anchor: torch.Tensor
    logits: torch.Tensor
    policy_chosen_error: torch.Tensor
    policy_rejected_error: torch.Tensor
    reference_chosen_error: torch.Tensor
    reference_rejected_error: torch.Tensor

    @property
    def chosen_delta(self) -> torch.Tensor:
        return self.policy_chosen_error - self.reference_chosen_error

    @property
    def rejected_delta(self) -> torch.Tensor:
        return self.policy_rejected_error - self.reference_rejected_error


@dataclass(frozen=True)
class FlowDistillationLoss:
    """Dense per-step supervision for an on-policy flow trajectory."""

    loss: torch.Tensor
    per_sample: torch.Tensor
    student_target: torch.Tensor
    teacher_target: torch.Tensor


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-sample MSE averaged over all non-batch dimensions in FP32."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    sample_dims = tuple(range(1, prediction.ndim))
    if not sample_dims:
        raise ValueError("predictions must include a batch dimension")
    return ((prediction.float() - target.float()) ** 2).mean(dim=sample_dims)


def flow_sft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute standard or per-sample weighted flow-matching velocity MSE loss."""
    errors = per_sample_mse(prediction, target)
    if sample_weights is None:
        return errors.mean()
    weights = sample_weights.to(device=errors.device, dtype=errors.dtype).reshape(-1)
    if weights.shape != errors.shape:
        raise ValueError(
            f"sample_weights shape {tuple(weights.shape)} != batch {tuple(errors.shape)}"
        )
    return (errors * weights).mean()


def _broadcast_time(
    time: torch.Tensor, reference: torch.Tensor, *, name: str
) -> torch.Tensor:
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError(
            f"{name} must have shape ({reference.shape[0]},), got {tuple(time.shape)}"
        )
    return time.to(device=reference.device, dtype=reference.dtype).view(
        reference.shape[0], *([1] * (reference.ndim - 1))
    )


def flow_opd_ode_loss(
    *,
    latents: torch.Tensor,
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    delta_t: torch.Tensor,
) -> FlowDistillationLoss:
    """DiffusionOPD one-step deterministic transition matching loss."""
    if (
        latents.shape != student_velocity.shape
        or latents.shape != teacher_velocity.shape
    ):
        raise ValueError("OPD latent/student/teacher shapes must match")
    expanded_delta = _broadcast_time(delta_t, latents, name="delta_t")
    student_mean = latents + expanded_delta * student_velocity
    teacher_mean = latents + expanded_delta * teacher_velocity.detach()
    per_sample = per_sample_mse(student_mean, teacher_mean)
    return FlowDistillationLoss(
        loss=0.5 * per_sample.mean(),
        per_sample=per_sample,
        student_target=student_mean,
        teacher_target=teacher_mean,
    )


def flow_dpo_loss(
    *,
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    chosen_target: torch.Tensor,
    rejected_target: torch.Tensor,
    beta: float,
    sft_anchor_weight: float = 0.0,
) -> FlowDpoLoss:
    """Compute flow-matching DPO loss with optional SFT anchor regularization."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    if sft_anchor_weight < 0:
        raise ValueError("sft_anchor_weight must be non-negative")
    p_chosen = per_sample_mse(policy_chosen, chosen_target)
    p_rejected = per_sample_mse(policy_rejected, rejected_target)
    r_chosen = per_sample_mse(reference_chosen, chosen_target)
    r_rejected = per_sample_mse(reference_rejected, rejected_target)
    logits = -0.5 * beta * ((p_chosen - r_chosen) - (p_rejected - r_rejected))
    dpo = -F.logsigmoid(logits).mean()
    ref_mse = 0.5 * (
        per_sample_mse(policy_chosen, reference_chosen)
        + per_sample_mse(policy_rejected, reference_rejected)
    ).mean()
    sft_anchor = p_chosen.mean()
    total = dpo + sft_anchor_weight * sft_anchor
    return FlowDpoLoss(
        loss=total,
        dpo=dpo,
        reference_mse=ref_mse,
        sft_anchor=sft_anchor,
        logits=logits,
        policy_chosen_error=p_chosen,
        policy_rejected_error=p_rejected,
        reference_chosen_error=r_chosen,
        reference_rejected_error=r_rejected,
    )
