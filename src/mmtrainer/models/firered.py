"""FireRed-Image-Edit aspect-ratio sizing, candidate interleaving, and adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mmtrainer.nn.flow import euler_flow_step, flow_sample
from mmtrainer.plugins.base import BaseModelAdapter

FIRERED_LORA_TARGET_MODULES: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


@dataclass(frozen=True)
class FireRedEditBatch:
    noisy_latents: torch.Tensor
    target_velocity: torch.Tensor
    sigmas: torch.Tensor
    reference_latents: torch.Tensor | None = None
    prompt_embeds: torch.Tensor | None = None


def calculate_aspect_dimensions(
    target_area: int,
    ratio: float,
    *,
    multiple_of: int = 32,
) -> tuple[int, int]:
    """Compute (width, height) preserving aspect ratio and rounded to multiple_of."""
    if target_area <= 0 or ratio <= 0:
        raise ValueError("target_area and ratio must be positive")
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    aligned_w = max(multiple_of, round(width / multiple_of) * multiple_of)
    aligned_h = max(multiple_of, round(height / multiple_of) * multiple_of)
    return int(aligned_w), int(aligned_h)


def interleave_candidates(streams: list[torch.Tensor]) -> torch.Tensor:
    """Interleave K candidate tensors of shape [B, ...] into [B * K, ...]."""
    if not streams:
        raise ValueError("streams must not be empty")
    stacked = torch.stack(streams, dim=1)
    bsz, num_candidates = stacked.shape[0], stacked.shape[1]
    return stacked.reshape(bsz * num_candidates, *stacked.shape[2:])


class FireRedAdapter(BaseModelAdapter):
    """Composable model adapter for FireRed-Image-Edit workflows."""

    model_id = "firered"
    default_lora_target_modules = FIRERED_LORA_TARGET_MODULES

    def prepare_sft_batch(self, **kwargs: Any) -> FireRedEditBatch:
        sample = flow_sample(
            kwargs["clean_latents"],
            sigmas=kwargs.get("sigmas"),
            noise=kwargs.get("noise"),
        )
        return FireRedEditBatch(
            noisy_latents=sample.noisy_latents,
            target_velocity=sample.target_velocity,
            sigmas=sample.sigmas,
            reference_latents=kwargs.get("reference_latents"),
            prompt_embeds=kwargs.get("prompt_embeds"),
        )

    def step_trajectory(
        self,
        latents: torch.Tensor,
        velocity: torch.Tensor,
        sigma_curr: torch.Tensor | float,
        sigma_next: torch.Tensor | float,
    ) -> torch.Tensor:
        return euler_flow_step(latents, velocity, sigma_curr, sigma_next)

    def predict_velocity(
        self, transformer: Any, batch: FireRedEditBatch, **kwargs: Any
    ) -> torch.Tensor:
        inputs = batch.noisy_latents
        if batch.reference_latents is not None:
            inputs = torch.cat([inputs, batch.reference_latents], dim=1)
        out = transformer(
            hidden_states=inputs,
            timestep=batch.sigmas,
            encoder_hidden_states=batch.prompt_embeds,
            **kwargs,
        )
        raw = out[0] if isinstance(out, tuple) else out
        return raw[:, : batch.noisy_latents.shape[1]].float()


__all__ = [
    "FIRERED_LORA_TARGET_MODULES",
    "FireRedAdapter",
    "FireRedEditBatch",
    "calculate_aspect_dimensions",
    "euler_flow_step",
    "interleave_candidates",
]
