"""Rectified-flow specialization of Trajectory Distribution Matching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from mmtrainer.algorithms.common import require_float_finite
from mmtrainer.nn.flow import euler_flow_step

from .types import TDMScoreBatch, TDMStudentSample, TDMTeacherRollout

TDMModelRole = Literal["student", "teacher", "teacher_unconditional", "fake"]
InitialNoiseFn = Callable[[Any, torch.Generator | None], Tensor]
VelocityFn = Callable[[Tensor, Tensor, Any, TDMModelRole], Tensor]
TimeGridFn = Callable[[Any, int], Tensor]
ModelTimeFn = Callable[[Tensor, Any], Tensor]
ImportanceWeightFn = Callable[[Tensor, Tensor, Tensor, Any], Tensor]


@dataclass(frozen=True)
class RectifiedFlowAnchorContext:
    anchor_index: int
    anchor_time: Tensor
    next_time: Tensor


@dataclass(frozen=True)
class RectifiedFlowTDMConfig:
    event_dims: int
    snr_cap: float = 5.0
    share_time_across_leading_dims: int = 0


def heun_midpoint_step(
    latents: Tensor,
    v_curr: Tensor,
    v_next: Tensor,
    sigma_curr: Tensor | float,
    sigma_next: Tensor | float,
) -> Tensor:
    """Second-order Heun midpoint step for rectified flow trajectories."""
    blended_v = 0.5 * (v_curr.to(latents.dtype) + v_next.to(latents.dtype))
    return euler_flow_step(latents, blended_v, sigma_curr, sigma_next)


def interpolate_descending_time_grid(
    noise_time: Tensor,
    noise_grid: Tensor,
    model_grid: Tensor,
) -> Tensor:
    """Linearly map continuous noise times onto a paired model-time grid."""

    for name, value in (
        ("noise_time", noise_time),
        ("noise_grid", noise_grid),
        ("model_grid", model_grid),
    ):
        require_float_finite(name, value)
    if noise_grid.ndim != 1 or model_grid.shape != noise_grid.shape:
        raise ValueError("paired time grids must be one-dimensional and equal length")
    if noise_grid.numel() < 2 or not bool((noise_grid[:-1] > noise_grid[1:]).all()):
        raise ValueError("noise_grid must be strictly descending")
    if bool((noise_time > noise_grid[0]).any()) or bool((noise_time < noise_grid[-1]).any()):
        raise ValueError("noise_time lies outside the paired grid")
    ascending = noise_grid.flip(0)
    upper_asc = torch.searchsorted(ascending, noise_time.contiguous(), right=True)
    upper_asc = upper_asc.clamp(1, ascending.numel() - 1)
    lower_asc = upper_asc - 1
    low_noise = ascending[lower_asc]
    high_noise = ascending[upper_asc]
    fraction = (noise_time - low_noise) / (high_noise - low_noise).clamp_min(1e-12)
    ascending_model = model_grid.flip(0)
    low_model = ascending_model[lower_asc]
    high_model = ascending_model[upper_asc]
    return low_model + fraction * (high_model - low_model)


class RectifiedFlowTDMAdapter:
    """TDM trajectory/noising logic parameterized by model-side callbacks."""

    def __init__(
        self,
        initial_noise_fn: InitialNoiseFn,
        velocity_fn: VelocityFn,
        time_grid_fn: TimeGridFn,
        config: RectifiedFlowTDMConfig,
        model_time_fn: ModelTimeFn | None = None,
        importance_weight_fn: ImportanceWeightFn | None = None,
    ) -> None:
        if config.event_dims <= 0:
            raise ValueError("event_dims must be positive")
        if (
            not torch.isfinite(torch.tensor(config.snr_cap, dtype=torch.float64))
            or config.snr_cap <= 0
        ):
            raise ValueError("snr_cap must be finite and positive")
        if config.share_time_across_leading_dims < 0:
            raise ValueError("share_time_across_leading_dims must be non-negative")
        self.initial_noise_fn = initial_noise_fn
        self.velocity_fn = velocity_fn
        self.time_grid_fn = time_grid_fn
        self.event_dims = int(config.event_dims)
        self.model_time_fn = model_time_fn or (lambda noise_time, conditioning: noise_time)
        self.importance_weight_fn = importance_weight_fn
        self.snr_cap = float(config.snr_cap)
        self.share_time_across_leading_dims = int(config.share_time_across_leading_dims)

    def _time_grid(self, conditioning: Any, student_steps: int) -> Tensor:
        grid = self.time_grid_fn(conditioning, student_steps)
        require_float_finite("TDM time grid", grid)
        if grid.ndim != 1 or grid.numel() != student_steps + 1:
            raise ValueError("TDM time grid must contain student_steps + 1 points")
        if not torch.isclose(grid[0], grid.new_tensor(1.0)) or not torch.isclose(
            grid[-1], grid.new_tensor(0.0)
        ):
            raise ValueError("TDM time grid must run from 1 to 0")
        if not bool((grid[:-1] > grid[1:]).all()):
            raise ValueError("TDM time grid must be strictly decreasing")
        return grid

    def _velocity(
        self,
        state: Tensor,
        noise_time: Tensor,
        conditioning: Any,
        role: TDMModelRole,
    ) -> Tensor:
        if role not in ("student", "teacher", "teacher_unconditional", "fake"):
            raise ValueError(f"unsupported model role: {role}")
        model_time = self.model_time_fn(noise_time, conditioning)
        velocity = self.velocity_fn(state, model_time, conditioning, role)
        require_float_finite(f"{role} velocity", velocity)
        if velocity.shape != state.shape or velocity.device != state.device:
            raise ValueError(f"{role} velocity must match state shape and device")
        return velocity

    @staticmethod
    def _broadcast_time(time: Tensor, reference: Tensor, event_dims: int) -> Tensor:
        return time.reshape(*time.shape, *([1] * event_dims)).to(
            device=reference.device, dtype=reference.dtype
        )

    def sample_student(
        self,
        conditioning: Any,
        *,
        student_steps: int,
        requires_grad: bool,
        generator: torch.Generator | None = None,
    ) -> TDMStudentSample:
        if student_steps <= 0:
            raise ValueError("student_steps must be positive")
        grid = self._time_grid(conditioning, student_steps)
        anchor_index = int(
            torch.randint(
                student_steps,
                (1,),
                device=grid.device,
                generator=generator,
            ).item()
        )
        state = self.initial_noise_fn(conditioning, generator)
        require_float_finite("initial student noise", state)
        if state.ndim <= self.event_dims:
            raise ValueError("student state lacks leading dimensions")
        with torch.set_grad_enabled(requires_grad):
            current = state
            for step in range(anchor_index):
                velocity = self._velocity(current, grid[step], conditioning, "student")
                current = current + (grid[step + 1] - grid[step]) * velocity
            anchor_time = grid[anchor_index]
            anchor_velocity = self._velocity(current, anchor_time, conditioning, "student")
            clean = current - anchor_time * anchor_velocity
        return TDMStudentSample(
            clean=clean,
            context=RectifiedFlowAnchorContext(
                anchor_index=anchor_index,
                anchor_time=anchor_time,
                next_time=grid[anchor_index + 1],
            ),
        )

    def prepare_score_batch(
        self,
        sample: TDMStudentSample,
        conditioning: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> TDMScoreBatch:
        if not isinstance(sample.context, RectifiedFlowAnchorContext):
            raise TypeError("rectified-flow TDM sample lacks anchor context")
        clean = sample.clean
        leading_shape = clean.shape[: clean.ndim - self.event_dims]
        if self.share_time_across_leading_dims >= len(leading_shape):
            raise ValueError("cannot share TDM time across every leading dimension")
        sampled_shape = (
            leading_shape[: -self.share_time_across_leading_dims]
            if self.share_time_across_leading_dims
            else leading_shape
        )
        anchor = sample.context.anchor_time.to(device=clean.device, dtype=torch.float32)
        following = sample.context.next_time.to(device=clean.device, dtype=torch.float32)
        interval = anchor - following
        if not bool(interval > 0):
            raise ValueError("TDM anchor interval must be positive")
        uniform = torch.rand(
            sampled_shape,
            device=clean.device,
            dtype=torch.float32,
            generator=generator,
        ).clamp(1e-4, 1.0 - 1e-4)
        if self.share_time_across_leading_dims:
            uniform = uniform.reshape(
                *sampled_shape,
                *([1] * self.share_time_across_leading_dims),
            ).expand(leading_shape)
        noise_time = following + uniform * interval
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        expanded = self._broadcast_time(noise_time, clean, self.event_dims)
        noisy = (1.0 - expanded) * clean + expanded * noise
        model_time = self.model_time_fn(noise_time, conditioning)
        require_float_finite("TDM model time", model_time)
        if model_time.shape != noise_time.shape:
            raise ValueError("model_time_fn must preserve the leading time shape")
        snr = ((1.0 - noise_time) / noise_time).square().clamp_max(self.snr_cap)
        importance = (
            torch.ones_like(noise_time)
            if self.importance_weight_fn is None
            else self.importance_weight_fn(clean, noise, noise_time, conditioning)
        )
        return TDMScoreBatch(
            noisy=noisy,
            noise_time=noise_time,
            model_time=model_time,
            snr_weight=snr,
            importance_weight=importance,
            context=sample.context,
        )

    def predict_clean(
        self,
        batch: TDMScoreBatch,
        conditioning: Any,
        *,
        model: Literal["teacher", "teacher_unconditional", "fake"],
    ) -> Tensor:
        velocity = self.velocity_fn(
            batch.noisy,
            batch.model_time,
            conditioning,
            model,
        )
        require_float_finite(f"{model} velocity", velocity)
        if velocity.shape != batch.noisy.shape or velocity.device != batch.noisy.device:
            raise ValueError(f"{model} velocity must match score-batch state")
        expanded = self._broadcast_time(batch.noise_time, batch.noisy, self.event_dims)
        return batch.noisy - expanded * velocity

    def rollout_teacher(
        self,
        conditioning: Any,
        *,
        teacher_steps: int,
        method: Literal["euler", "heun"] = "euler",
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> TDMTeacherRollout:
        if teacher_steps <= 0:
            raise ValueError("teacher_steps must be positive")
        if not torch.isfinite(torch.tensor(cfg_scale, dtype=torch.float64)) or cfg_scale < 1:
            raise ValueError("cfg_scale must be finite and at least 1")
        if method not in ("euler", "heun"):
            raise ValueError(f"unsupported stepping method: {method}")
        grid = self._time_grid(conditioning, teacher_steps)
        state = self.initial_noise_fn(conditioning, generator)
        require_float_finite("initial teacher noise", state)
        current = state
        trajectory = [current] if return_trajectory else None
        for step in range(teacher_steps):
            t_curr = grid[step]
            t_next = grid[step + 1]
            t_curr_val = float(t_curr.item())
            t_next_val = float(t_next.item())
            if cfg_scale > 1.0:
                v_cond = self._velocity(current, t_curr, conditioning, "teacher")
                v_uncond = self._velocity(current, t_curr, conditioning, "teacher_unconditional")
                v_curr = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_curr = self._velocity(current, t_curr, conditioning, "teacher")
            if method == "euler":
                current = euler_flow_step(current, v_curr, t_curr_val, t_next_val)
            else:
                euler_next = euler_flow_step(current, v_curr, t_curr_val, t_next_val)
                if cfg_scale > 1.0:
                    v_next_cond = self._velocity(euler_next, t_next, conditioning, "teacher")
                    v_next_uncond = self._velocity(
                        euler_next, t_next, conditioning, "teacher_unconditional"
                    )
                    v_next = v_next_uncond + cfg_scale * (v_next_cond - v_next_uncond)
                else:
                    v_next = self._velocity(euler_next, t_next, conditioning, "teacher")
                current = heun_midpoint_step(current, v_curr, v_next, t_curr_val, t_next_val)
            if return_trajectory and trajectory is not None:
                trajectory.append(current)
        return TDMTeacherRollout(
            clean=current,
            timesteps=grid,
            trajectory=tuple(trajectory) if trajectory is not None else None,
        )


__all__ = [
    "RectifiedFlowAnchorContext",
    "RectifiedFlowTDMAdapter",
    "RectifiedFlowTDMConfig",
    "TDMModelRole",
    "TDMTeacherRollout",
    "euler_flow_step",
    "heun_midpoint_step",
    "interpolate_descending_time_grid",
]
