"""Example 2: Composable Coupled-Noise Flow-DPO Preference Optimization.

Combines:
- Model Adapter & Backbone: `Krea2Adapter`, `SingleStreamDiT`
- Preference Objective: `flow_dpo_loss`
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from mmtrainer.algorithms.sft_dpo import flow_dpo_loss
from mmtrainer.models.krea2 import Krea2Adapter, SingleMMDiTConfig, SingleStreamDiT
from mmtrainer.models.qwen_image import split_pair_prediction


def run_dpo_demo(steps: int = 3, batch_size: int = 2) -> dict[str, Any]:
    torch.manual_seed(7)
    config = SingleMMDiTConfig(
        in_channels=16,
        hidden_size=64,
        num_heads=4,
        depth=2,
        context_in_dim=32,
    )
    policy_model = SingleStreamDiT(config)
    ref_model = copy.deepcopy(policy_model).eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    adapter = Krea2Adapter()
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)

    last_metrics: dict[str, float] = {}
    for _ in range(steps):
        win_latents = torch.randn(batch_size, 4, 8, 8)
        lose_latents = torch.randn(batch_size, 4, 8, 8)
        cond_latents = torch.randn(batch_size, 4, 8, 8)
        txt_tokens = torch.randn(batch_size, 6, 32)

        dpo_batch = adapter.prepare_dpo_batch(
            win_latents=win_latents,
            lose_latents=lose_latents,
            cond_latents=cond_latents,
            txt_tokens=txt_tokens,
            patch_size=2,
        )
        policy_pred = adapter.predict_velocity(policy_model, dpo_batch.forward)
        policy_win, policy_lose = split_pair_prediction(policy_pred)

        with torch.no_grad():
            ref_pred = adapter.predict_velocity(ref_model, dpo_batch.forward)
            ref_win, ref_lose = split_pair_prediction(ref_pred)

        result = flow_dpo_loss(
            policy_chosen=policy_win,
            policy_rejected=policy_lose,
            reference_chosen=ref_win,
            reference_rejected=ref_lose,
            chosen_target=dpo_batch.win_target_velocity,
            rejected_target=dpo_batch.lose_target_velocity,
            beta=50.0,
            sft_anchor_weight=0.1,
        )
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        last_metrics = {
            "loss": float(result.loss.detach().item()),
            "dpo_loss": float(result.dpo.detach().item()),
            "implicit_accuracy": float((result.logits > 0).float().mean().item()),
        }

    return {"model_id": adapter.model_id, "steps": steps, **last_metrics}


if __name__ == "__main__":
    print(run_dpo_demo())
