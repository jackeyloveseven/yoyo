"""Distillation and regularization primitives: DMD, GT-Reg, NormGuard, and REPA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NormGuardResult:
    """Diagnostics for NormGuard relative velocity-norm hinge regularization."""

    penalty: torch.Tensor
    norm_ratio: torch.Tensor
    active_fraction: torch.Tensor


@dataclass(frozen=True)
class DMDLossResult:
    """Generator distribution-matching loss and gradient diagnostics."""

    loss: torch.Tensor
    gradient_norm: torch.Tensor


def norm_guard_loss(
    student_velocity: torch.Tensor,
    reference_velocity: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> NormGuardResult:
    """Compute NormGuard (arXiv:2606.27771) relative velocity-norm hinge penalty."""
    if student_velocity.shape != reference_velocity.shape:
        raise ValueError(
            f"velocity shape mismatch: "
            f"{tuple(student_velocity.shape)} != {tuple(reference_velocity.shape)}"
        )
    if student_velocity.ndim < 2:
        raise ValueError("velocities must include batch and feature dimensions")
    reduce_dims = tuple(range(1, student_velocity.ndim))
    sq_theta = student_velocity.double().pow(2).sum(dim=reduce_dims)
    sq_ref = reference_velocity.detach().double().pow(2).sum(dim=reduce_dims).clamp_min(eps)
    excess = (sq_theta - sq_ref) / sq_ref
    penalty = excess.clamp_min(0.0).mean().float()
    ratio = (sq_theta.detach() / sq_ref).mean().float()
    active_frac = (excess > 0).float().mean().detach()
    return NormGuardResult(
        penalty=penalty,
        norm_ratio=ratio,
        active_fraction=active_frac,
    )


def dmd_distribution_matching_loss(
    x0_generated: torch.Tensor,
    x0_real: torch.Tensor,
    x0_fake: torch.Tensor,
    *,
    global_norm: bool = False,
    eps: float = 1e-6,
) -> DMDLossResult:
    """Compute DMD2 distribution matching loss in clean-latent ``x0`` space."""
    if x0_generated.shape != x0_real.shape or x0_generated.shape != x0_fake.shape:
        raise ValueError("x0_generated, x0_real, and x0_fake must have identical shapes")
    grad = x0_fake.detach().float() - x0_real.detach().float()
    if global_norm:
        scale = grad.abs().mean().clamp_min(eps)
    else:
        reduce_dims = tuple(range(1, x0_generated.ndim))
        scale = (
            (x0_generated.detach().float() - x0_real.detach().float())
            .abs()
            .mean(dim=reduce_dims, keepdim=True)
            .clamp_min(eps)
        )
    normalized_grad = torch.nan_to_num(grad / scale)
    target = (x0_generated.detach().float() - normalized_grad).detach()
    loss = 0.5 * F.mse_loss(x0_generated.float(), target)
    return DMDLossResult(
        loss=loss,
        gradient_norm=normalized_grad.norm().detach(),
    )


def dmd_fake_score_loss(
    fake_velocity: torch.Tensor,
    x0_generated: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Compute the critic (fake score) flow-matching MSE loss on detached generated samples."""
    if fake_velocity.shape != x0_generated.shape or fake_velocity.shape != noise.shape:
        raise ValueError("fake_velocity, x0_generated, and noise shapes must match")
    target_velocity = noise.detach().float() - x0_generated.detach().float()
    return F.mse_loss(fake_velocity.float(), target_velocity)


def gt_regression_loss(
    x0_generated: torch.Tensor,
    x0_ground_truth: torch.Tensor,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    fake_velocity: torch.Tensor,
) -> torch.Tensor:
    """Compute MagicDistill (arXiv:2503.13319 Eq. 7) ground-truth regression loss."""
    if x0_generated.shape != x0_ground_truth.shape or x0_generated.shape != x_t.shape:
        raise ValueError("GT regression latent shapes must match")
    expanded_sigma = sigma.double().view(x0_generated.shape[0], *([1] * (x0_generated.ndim - 1)))
    with torch.no_grad():
        x0_target = x_t.double() - expanded_sigma * fake_velocity.detach().double()
        grad = torch.nan_to_num(x0_target - x0_ground_truth.detach().double()).float()
    return 0.5 * F.mse_loss(x0_generated.float(), (x0_generated.float() - grad).detach())


class RepaProjector(nn.Module):
    """Lightweight MLP projecting DiT hidden states to visual encoder feature space."""

    def __init__(
        self,
        dit_hidden_size: int,
        encoder_dim: int,
        *,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if dit_hidden_size <= 0 or encoder_dim <= 0 or hidden_dim <= 0:
            raise ValueError("RepaProjector dimensions must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(dit_hidden_size),
            nn.Linear(dit_hidden_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states.float())


def repa_cosine_alignment_loss(
    projected_hidden: torch.Tensor,
    encoder_tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute REPA negative cosine similarity loss between aligned token grids."""
    if projected_hidden.shape != encoder_tokens.shape:
        raise ValueError(
            f"REPA token shape mismatch: "
            f"{tuple(projected_hidden.shape)} != {tuple(encoder_tokens.shape)}"
        )
    proj_norm = F.normalize(projected_hidden.float(), dim=-1)
    enc_norm = F.normalize(encoder_tokens.detach().float(), dim=-1)
    cosine = (proj_norm * enc_norm).sum(dim=-1)
    return -cosine.mean()
