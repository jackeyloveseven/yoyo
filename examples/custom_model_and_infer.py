"""Example 6: Onboarding a Custom Model Architecture + Multi-Step Flow Inference.

Demonstrates the scaffold philosophy:
1. Subclass `BaseModelAdapter` and `CatalogModelPlugin` for any new model architecture.
2. Register the new model into `PluginRegistry`.
3. Train it with `flow_sft_loss` and run rollout inference with `resolution_shifted_timesteps`,
   `euler_flow_step`, and `dpm2_flow_step`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mmtrainer.algorithms.sft_dpo import flow_sft_loss
from mmtrainer.config import ComponentSpec
from mmtrainer.nn import (
    dpm2_flow_step,
    euler_flow_step,
    flow_sample,
    resolution_shifted_timesteps,
)
from mmtrainer.plugins.base import BaseModelAdapter, CatalogModelPlugin
from mmtrainer.registry import PluginKind, RegistryEntry, build_builtin_registries


class NextGenFlowBackbone(nn.Module):
    """Minimal example of a new diffusion/flow architecture."""

    def __init__(self, channels: int = 8, cond_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels + cond_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, channels),
        )

    def forward(
        self, latents: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        t_feat = timestep.view(-1, 1).to(dtype=latents.dtype)
        return self.net(torch.cat([latents, cond, t_feat], dim=-1))


class NextGenFlowAdapter(BaseModelAdapter):
    """Custom model adapter implementing the mmtrainer DiffusionModelAdapter contract."""

    model_id = "nextgen_flow"
    default_lora_target_modules = ("net.0", "net.2")

    def prepare_sft_batch(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        sample = flow_sample(kwargs["clean_latents"], sigmas=kwargs.get("sigmas"))
        return {
            "noisy_latents": sample.noisy_latents,
            "target_velocity": sample.target_velocity,
            "sigmas": sample.sigmas,
            "cond": kwargs["cond"],
        }

    def predict_velocity(
        self, transformer: Any, batch: dict[str, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        del kwargs
        return transformer(batch["noisy_latents"], batch["sigmas"], batch["cond"]).float()


class NextGenFlowPlugin(CatalogModelPlugin):
    plugin_id = "nextgen_flow"
    capabilities = frozenset({"diffusion_alignment", "image_edit", "image_training", "t2i"})
    adapter_import_path = f"{__name__}:NextGenFlowAdapter"


def run_custom_model_and_infer_demo(steps: int = 4) -> dict[str, Any]:
    torch.manual_seed(99)
    registries = build_builtin_registries()
    registries.models.register(
        RegistryEntry.create(
            plugin_id="nextgen_flow",
            kind=PluginKind.MODEL,
            import_path=f"{__name__}:NextGenFlowPlugin",
            capabilities=NextGenFlowPlugin.capabilities,
        )
    )
    plugin_cls = registries.models.resolve("nextgen_flow")
    plugin = plugin_cls(ComponentSpec(plugin_id="nextgen_flow"))
    adapter = plugin.build_adapter()
    backbone = NextGenFlowBackbone(channels=8, cond_dim=16)

    clean = torch.randn(2, 8)
    cond = torch.randn(2, 16)
    sft_batch = adapter.prepare_sft_batch(clean_latents=clean, cond=cond)
    pred_v = adapter.predict_velocity(backbone, sft_batch)
    sft_loss = flow_sft_loss(pred_v, sft_batch["target_velocity"])

    schedule = resolution_shifted_timesteps(seq_len=1024, steps=steps)
    latents = torch.randn(2, 8)
    with torch.no_grad():
        for idx in range(steps):
            t_curr = torch.full((2,), schedule[idx])
            t_next = torch.full((2,), schedule[idx + 1])
            v_curr = backbone(latents, t_curr, cond)
            if idx % 2 == 0:
                latents = euler_flow_step(latents, v_curr, t_curr, t_next)
            else:
                t_mid = 0.5 * (t_curr + t_next)
                latents_mid = euler_flow_step(latents, v_curr, t_curr, t_mid)
                v_mid = backbone(latents_mid, t_mid, cond)
                latents = dpm2_flow_step(latents, v_curr, v_mid, t_curr, t_next)

    return {
        "registered_plugin_id": plugin.plugin_id,
        "adapter_model_id": adapter.model_id,
        "sft_loss": float(sft_loss.detach().item()),
        "rollout_steps": steps,
        "output_shape": tuple(latents.shape),
    }


if __name__ == "__main__":
    print(run_custom_model_and_infer_demo())
