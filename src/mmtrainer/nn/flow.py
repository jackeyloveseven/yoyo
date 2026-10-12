"""Reusable flow-matching interpolation and timestep sampling components."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowSample:
    """A noised latent and its velocity target at one continuous time."""

    noisy: torch.Tensor
    target: torch.Tensor
    sigmas: torch.Tensor | None = None

    @property
    def noisy_latents(self) -> torch.Tensor:
        return self.noisy

    @property
    def target_velocity(self) -> torch.Tensor:
        return self.target


def flow_sample(
    clean: torch.Tensor,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    *,
    sigmas: torch.Tensor | None = None,
) -> FlowSample:
    """Construct ``x_t = (1 - t) * x_0 + t * noise`` and ``v = noise - x_0``."""
    resolved_time = time if time is not None else sigmas
    if resolved_time is None:
        resolved_time = torch.rand(clean.shape[0], device=clean.device, dtype=torch.float32)
    if noise is None:
        noise = torch.randn_like(clean)
    if clean.shape != noise.shape:
        raise ValueError(
            f"clean/noise shape mismatch: {tuple(clean.shape)} != {tuple(noise.shape)}"
        )
    if clean.ndim < 2:
        raise ValueError("flow latents must include batch and feature dimensions")
    if resolved_time.ndim != 1 or resolved_time.shape[0] != clean.shape[0]:
        raise ValueError(
            f"time must have shape (batch,), "
            f"got {tuple(resolved_time.shape)} for batch={clean.shape[0]}"
        )
    expanded_time = resolved_time.view(clean.shape[0], *([1] * (clean.ndim - 1))).to(
        device=clean.device, dtype=clean.dtype
    )
    return FlowSample(
        noisy=(1.0 - expanded_time) * clean + expanded_time * noise,
        target=noise - clean,
        sigmas=resolved_time,
    )


def paired_flow_samples(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    time: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    *,
    sigmas: torch.Tensor | None = None,
) -> tuple[FlowSample, FlowSample]:
    """Noisify a preference pair with the exact same noise and time."""
    if chosen.shape != rejected.shape:
        raise ValueError(
            f"chosen/rejected shape mismatch: {tuple(chosen.shape)} != {tuple(rejected.shape)}"
        )
    resolved_time = time if time is not None else sigmas
    if resolved_time is None:
        resolved_time = torch.rand(chosen.shape[0], device=chosen.device, dtype=torch.float32)
    if noise is None:
        noise = torch.randn_like(chosen)
    return (
        flow_sample(chosen, resolved_time, noise),
        flow_sample(rejected, resolved_time, noise),
    )


def euler_flow_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma_curr: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
) -> torch.Tensor:
    """First-order Euler ODE step for rectified flow trajectories."""
    dt = sigma_next - sigma_curr
    if isinstance(dt, torch.Tensor):
        dt = dt.view(latents.shape[0], *([1] * (latents.ndim - 1))).to(
            device=latents.device, dtype=latents.dtype
        )
    return latents + dt * velocity.to(dtype=latents.dtype)


def dpm2_flow_step(
    latents: torch.Tensor,
    v_curr: torch.Tensor,
    v_mid: torch.Tensor,
    sigma_curr: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
) -> torch.Tensor:
    """Second-order midpoint/DPM2 ODE step for flow-matching rollout."""
    blended_v = 0.5 * (v_curr.to(latents.dtype) + v_mid.to(latents.dtype))
    return euler_flow_step(latents, blended_v, sigma_curr, sigma_next)


def sample_timesteps_shift(
    batch_size: int,
    device: torch.device,
    shift: float = 2.2,
    sigmoid_scale: float = 1.0,
) -> torch.Tensor:
    """Sample shifted logit-normal continuous times in ``(0, 1)``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    if sigmoid_scale < 0:
        raise ValueError("sigmoid_scale must be non-negative")
    logits = torch.randn(batch_size, device=device) * sigmoid_scale
    times = torch.sigmoid(logits)
    return (times * shift) / (1.0 + (shift - 1.0) * times)


def shifted_flow_sigmas(
    t_norm: torch.Tensor,
    base_shift: float,
    max_shift: float,
    image_seq_len: int,
    max_image_seq_len: int = 4096,
    base_image_seq_len: int = 256,
) -> torch.Tensor:
    """Apply sequence-length-dependent time shift (FLUX / Klein style)."""
    slope = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    intercept = base_shift - slope * base_image_seq_len
    mu = math.exp(slope * image_seq_len + intercept)
    return mu / (mu + (1.0 / t_norm - 1.0))


def empirical_rollout_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute sequence- and step-dependent time shift ``mu`` for short rollouts."""
    if image_seq_len <= 0 or num_steps <= 0:
        raise ValueError("image_seq_len and num_steps must be positive")
    short_slope, short_intercept = 8.73809524e-05, 1.89833333
    long_slope, long_intercept = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(long_slope * image_seq_len + long_intercept)
    mu_short = short_slope * image_seq_len + short_intercept
    mu_long = long_slope * image_seq_len + long_intercept
    step_slope = (mu_long - mu_short) / 190.0
    step_intercept = mu_long - 200.0 * step_slope
    return float(step_slope * num_steps + step_intercept)


def resolution_shifted_timesteps(
    seq_len: int,
    steps: int,
    *,
    x1: float = 256.0,
    x2: float = 4096.0,
    y1: float = 0.5,
    y2: float = 1.15,
    sigma: float = 1.0,
    mu: float | None = None,
) -> list[float]:
    """Build a resolution-aware 1 -> 0 flow-matching timestep schedule."""
    if seq_len <= 0 or steps <= 0:
        raise ValueError("seq_len and steps must be positive")
    ts = torch.linspace(1.0, 0.0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    shifted[0] = 1.0
    shifted[-1] = 0.0
    return [float(item) for item in shifted.tolist()]
