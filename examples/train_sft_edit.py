"""Example 1: Composable Flow-Matching SFT for Image Editing / T2I.

Combines:
- Model Adapter & Backbone: `Krea2Adapter`, `SingleStreamDiT`, `SingleMMDiTConfig`
- Objective: `flow_sft_loss`
- Schedule: `build_lr_scheduler`
"""

from __future__ import annotations

from typing import Any

import torch

from mmtrainer.algorithms.sft_dpo import flow_sft_loss
from mmtrainer.models.krea2 import Krea2Adapter, SingleMMDiTConfig, SingleStreamDiT
from mmtrainer.nn import build_lr_scheduler, sample_timesteps_shift


def run_sft_demo(steps: int = 3, batch_size: int = 2) -> dict[str, Any]:
    torch.manual_seed(42)
    config = SingleMMDiTConfig(
        in_channels=16,
        hidden_size=64,
        num_heads=4,
        depth=2,
        context_in_dim=32,
    )
    model = SingleStreamDiT(config)
    adapter = Krea2Adapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(
        optimizer,
        kind="cosine",
        warmup_steps=1,
        total_steps=steps,
    )

    losses: list[float] = []
    for _ in range(steps):
        target_latents = torch.randn(batch_size, 4, 8, 8)
        cond_latents = torch.randn(batch_size, 4, 8, 8)
        txt_tokens = torch.randn(batch_size, 6, 32)
        sigmas = sample_timesteps_shift(batch_size, target_latents.device, shift=2.0)

        sft_batch = adapter.prepare_sft_batch(
            target_latents=target_latents,
            cond_latents=cond_latents,
            txt_tokens=txt_tokens,
            sigmas=sigmas,
            patch_size=2,
        )
        pred_velocity = adapter.predict_velocity(model, sft_batch.forward)
        loss = flow_sft_loss(pred_velocity, sft_batch.target_velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().item()))

    return {
        "model_id": adapter.model_id,
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "last_lr": float(scheduler.get_last_lr()[0]),
    }


if __name__ == "__main__":
    print(run_sft_demo())
