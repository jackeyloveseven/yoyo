"""Example 4: Composable Online RL with DiffusionNFT + VLM & Identity Rewards.

Combines:
- DiffusionNFT Objective: `DiffusionNFTInputs`, `DiffusionNFTLossConfig`,
  `diffusion_nft_loss`, `make_rectified_flow_batch`, `reward_to_optimality`,
  `ema_update_parameters`
- Rewards: `parse_rational_rewards_scores`, `cosine_identity_loss`
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from mmtrainer.algorithms.diffusion_alignment import (
    DiffusionNFTInputs,
    DiffusionNFTLossConfig,
    diffusion_nft_loss,
    ema_update_parameters,
    make_rectified_flow_batch,
    reward_to_optimality,
)
from mmtrainer.rewards.identity import cosine_identity_loss
from mmtrainer.rewards.protocols import parse_rational_rewards_scores


def run_diffusionnft_demo(batch_size: int = 4) -> dict[str, Any]:
    torch.manual_seed(19)
    policy_net = nn.Linear(8, 8)
    old_net = copy.deepcopy(policy_net).eval()
    ref_net = copy.deepcopy(policy_net).eval()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-3)

    clean_latents = torch.randn(batch_size, 8)
    noise = torch.randn_like(clean_latents)
    timesteps = torch.linspace(0.2, 0.8, batch_size)
    rf_batch = make_rectified_flow_batch(
        clean_latents, timesteps, noise=noise
    )

    mock_vlm_outputs = [
        '<think>good</think><answer>{"Instruction_Following": 5, "Consistency": 4, "Visual_Quality": 5}</answer>',
        '<think>ok</think><answer>{"Instruction_Following": 3, "Consistency": 4, "Visual_Quality": 3}</answer>',
        '<think>poor</think><answer>{"Instruction_Following": 2, "Consistency": 2, "Visual_Quality": 2}</answer>',
        '<think>great</think><answer>{"Instruction_Following": 5, "Consistency": 5, "Visual_Quality": 4}</answer>',
    ]
    vlm_scores = torch.tensor(
        [parse_rational_rewards_scores(txt)["overall"] for txt in mock_vlm_outputs],
        dtype=torch.float32,
    ).view(2, 2)
    optimality = reward_to_optimality(
        vlm_scores, normalizer="group_std"
    ).probabilities.reshape(batch_size)

    pred_v = policy_net(rf_batch.noisy)
    with torch.no_grad():
        old_v = old_net(rf_batch.noisy)
        ref_v = ref_net(rf_batch.noisy)

    nft_metrics = diffusion_nft_loss(
        DiffusionNFTInputs(
            policy_velocity=pred_v,
            old_velocity=old_v,
            target_velocity=rf_batch.target,
            optimality=optimality,
            clean=rf_batch.clean,
            noisy=rf_batch.noisy,
            noise_fraction=rf_batch.noise_fraction,
            reference_velocity=ref_v,
        ),
        DiffusionNFTLossConfig(beta=0.5, kl_beta=0.01),
    )
    id_loss = cosine_identity_loss(pred_v, clean_latents)
    total_loss = nft_metrics.total_loss + 0.05 * id_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    ema_update_parameters(old_net.parameters(), policy_net.parameters(), decay=0.95)

    return {
        "total_loss": float(total_loss.detach().item()),
        "policy_loss": float(nft_metrics.policy_loss.detach().item()),
        "kl_loss": float(nft_metrics.kl_loss.detach().item()),
        "identity_loss": float(id_loss.detach().item()),
        "mean_optimality": float(optimality.mean().item()),
    }


if __name__ == "__main__":
    print(run_diffusionnft_demo())
