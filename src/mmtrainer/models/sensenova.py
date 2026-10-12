"""SenseNova-SI and HiDream multi-stream conditioning and rollout adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mmtrainer.nn.flow import dpm2_flow_step, flow_sample, paired_flow_samples
from mmtrainer.plugins.base import BaseModelAdapter

SENSENOVA_LORA_TARGET_MODULES: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


@dataclass(frozen=True)
class MultiStreamLatentBatch:
    noisy_latents: torch.Tensor
    target_velocity: torch.Tensor
    sigmas: torch.Tensor
    pooled_projections: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None


class SenseNovaAdapter(BaseModelAdapter):
    """Composable model adapter for SenseNova-SI / Sirius-Diff workflows."""

    model_id = "sensenova"
    default_lora_target_modules = SENSENOVA_LORA_TARGET_MODULES

    def prepare_sft_batch(self, **kwargs: Any) -> MultiStreamLatentBatch:
        sample = flow_sample(
            kwargs["target_latents"],
            sigmas=kwargs.get("sigmas"),
            noise=kwargs.get("noise"),
        )
        return MultiStreamLatentBatch(
            noisy_latents=sample.noisy_latents,
            target_velocity=sample.target_velocity,
            sigmas=sample.sigmas,
            pooled_projections=kwargs.get("pooled_projections"),
            encoder_hidden_states=kwargs.get("encoder_hidden_states"),
        )

    def dpm2_step(
        self,
        latents: torch.Tensor,
        v_curr: torch.Tensor,
        v_mid: torch.Tensor,
        sigma_curr: torch.Tensor | float,
        sigma_next: torch.Tensor | float,
    ) -> torch.Tensor:
        return dpm2_flow_step(latents, v_curr, v_mid, sigma_curr, sigma_next)

    def predict_velocity(
        self, transformer: Any, batch: MultiStreamLatentBatch, **kwargs: Any
    ) -> torch.Tensor:
        res = transformer(
            hidden_states=batch.noisy_latents,
            timestep=batch.sigmas,
            encoder_hidden_states=batch.encoder_hidden_states,
            pooled_projections=batch.pooled_projections,
            **kwargs,
        )
        return res[0].float() if isinstance(res, tuple) else res.float()


class HiDreamAdapter(BaseModelAdapter):
    """Composable model adapter for HiDream T2I and preference alignment."""

    model_id = "hidream"
    default_lora_target_modules = SENSENOVA_LORA_TARGET_MODULES

    def prepare_dpo_batch(
        self, **kwargs: Any
    ) -> tuple[MultiStreamLatentBatch, MultiStreamLatentBatch]:
        win_sample, lose_sample = paired_flow_samples(
            kwargs["win_latents"],
            kwargs["lose_latents"],
            sigmas=kwargs.get("sigmas"),
        )
        enc = kwargs.get("encoder_hidden_states")
        pooled = kwargs.get("pooled_projections")
        return (
            MultiStreamLatentBatch(
                noisy_latents=win_sample.noisy_latents,
                target_velocity=win_sample.target_velocity,
                sigmas=win_sample.sigmas,
                pooled_projections=pooled,
                encoder_hidden_states=enc,
            ),
            MultiStreamLatentBatch(
                noisy_latents=lose_sample.noisy_latents,
                target_velocity=lose_sample.target_velocity,
                sigmas=lose_sample.sigmas,
                pooled_projections=pooled,
                encoder_hidden_states=enc,
            ),
        )
