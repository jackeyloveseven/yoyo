"""Example 5: Composable Trajectory RL with Flow-GRPO & DiffusionOPSD.

Combines:
- Flow-GRPO: `sde_step_with_logprob`, `isotropic_gaussian_log_prob`, `gaussian_mean_kl`,
  `group_relative_advantages`, `PPOInputs`, `PPOConfig`, `ppo_clipped_surrogate`
- DiffusionOPSD: `DiffusionOPSDFittingInputs`, `DiffusionOPSDFittingConfig`,
  `diffusion_opsd_fitting_loss`, `group_normalized_endpoint_weights`
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mmtrainer.algorithms.diffusion_opsd import (
    DiffusionOPSDFittingConfig,
    DiffusionOPSDFittingInputs,
    clean_output_prediction,
    diffusion_opsd_fitting_loss,
    group_normalized_endpoint_weights,
)
from mmtrainer.algorithms.flow_grpo import (
    PPOConfig,
    PPOInputs,
    gaussian_mean_kl,
    group_relative_advantages,
    isotropic_gaussian_log_prob,
    ppo_clipped_surrogate,
    sde_step_with_logprob,
)


def run_flow_grpo_and_opsd_demo(batch_size: int = 2, group_size: int = 2) -> dict[str, Any]:
    torch.manual_seed(29)
    policy_net = nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-3)

    latents = torch.randn(batch_size, group_size, 1, 8)
    sigma_curr = torch.full((batch_size, group_size, 1), 0.6)
    sigma_next = torch.full((batch_size, group_size, 1), 0.4)

    with torch.no_grad():
        old_v = policy_net(latents)
        old_dist = sde_step_with_logprob(
            sample=latents,
            velocity=old_v,
            sigma=sigma_curr,
            sigma_next=sigma_next,
            noise_level=0.7,
            sigma_max=0.99,
        )
        next_sample = old_dist.mean + 0.1 * torch.randn_like(old_dist.mean)
        old_lp = isotropic_gaussian_log_prob(next_sample, old_dist.mean, old_dist.std)

    rewards = torch.tensor([[0.9, 0.2], [0.8, 0.4]])
    adv = group_relative_advantages(rewards).advantages

    curr_v = policy_net(latents)
    curr_dist = sde_step_with_logprob(
        sample=latents,
        velocity=curr_v,
        sigma=sigma_curr,
        sigma_next=sigma_next,
        noise_level=0.7,
        sigma_max=0.99,
    )
    new_lp = isotropic_gaussian_log_prob(next_sample, curr_dist.mean, curr_dist.std)
    ref_kl = gaussian_mean_kl(curr_dist.mean, old_dist.mean.detach(), curr_dist.std)

    ppo_metrics = ppo_clipped_surrogate(
        PPOInputs(
            old_logprobs=old_lp.detach(),
            new_logprobs=new_lp,
            advantages=adv,
            reference_kl=ref_kl,
        ),
        PPOConfig(clip_epsilon=0.2, kl_beta=0.01),
    )

    opsd_weights = group_normalized_endpoint_weights(rewards).weights
    latents_step0 = latents.squeeze(2)
    sigma_step0 = sigma_curr.squeeze(2)
    curr_v_step0 = curr_v.squeeze(2)
    old_v_step0 = old_v.squeeze(2).detach()
    behav_clean = clean_output_prediction(latents_step0, sigma_step0, old_v_step0)
    opsd_res = diffusion_opsd_fitting_loss(
        DiffusionOPSDFittingInputs(
            query_state=latents_step0,
            query_sigma=sigma_step0,
            policy_velocity=curr_v_step0,
            behavior_clean=behav_clean,
            positive_target=behav_clean + 0.05,
            negative_target=behav_clean - 0.05,
            weights=opsd_weights,
        ),
        DiffusionOPSDFittingConfig(branch_beta=1.0),
    )

    total_loss = ppo_metrics.loss + 0.1 * opsd_res.total_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": float(total_loss.detach().item()),
        "ppo_policy_loss": float(ppo_metrics.policy_loss.detach().item()),
        "ppo_kl_loss": float(ppo_metrics.kl_loss.detach().item()),
        "opsd_loss": float(opsd_res.total_loss.detach().item()),
    }


if __name__ == "__main__":
    print(run_flow_grpo_and_opsd_demo())
