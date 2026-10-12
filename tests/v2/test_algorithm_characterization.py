from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _losses() -> dict[str, float]:
    from mmtrainer.algorithms.diffusion_alignment import (
        AsymmetricDPOConfig,
        AsymmetricDPOInputs,
        DiffusionNFTInputs,
        DiffusionNFTLossConfig,
        asymmetric_dpo_loss,
        diffusion_nft_loss,
    )
    from mmtrainer.algorithms.diffusion_opsd import (
        DiffusionOPSDFittingConfig,
        DiffusionOPSDFittingInputs,
        diffusion_opsd_fitting_loss,
    )
    from mmtrainer.algorithms.flow_grpo import (
        PPOConfig,
        PPOInputs,
        ppo_clipped_surrogate,
    )
    from mmtrainer.algorithms.flowmimic_opsd import (
        FlowMimicOPSDInputs,
        FlowMimicOPSDLossConfig,
        flowmimic_clean_distribution_loss,
    )
    from mmtrainer.algorithms.reward_matching import (
        AWMConfig,
        AWMInputs,
        advantage_weighted_matching_loss,
    )

    torch.manual_seed(7)
    dpo_tensors = [torch.randn(2, 3, 4) for _ in range(6)]
    dpo = asymmetric_dpo_loss(
        AsymmetricDPOInputs(*dpo_tensors),
        AsymmetricDPOConfig(beta=0.7, win_weight=1.2, psr_weight=0.1, event_dims=1),
    )
    policy, old, target = (torch.randn(2, 3, 4) for _ in range(3))
    optimality = torch.rand(2, 3)
    reference = torch.randn(2, 3, 4)
    nft = diffusion_nft_loss(
        DiffusionNFTInputs(policy, old, target, optimality, reference_velocity=reference),
        DiffusionNFTLossConfig(beta=0.8, event_dims=1, adv_clip_max=4.0, kl_beta=0.02),
    )
    old_logprobs = torch.randn(2, 3, 4)
    new_logprobs = torch.randn(2, 3, 4)
    advantages = torch.randn(2, 3)
    reference_kl = torch.rand(2, 3, 4)
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[..., 0] = True
    ppo = ppo_clipped_surrogate(
        PPOInputs(
            old_logprobs,
            new_logprobs,
            advantages,
            reference_kl,
            deterministic_mask=mask,
        ),
        PPOConfig(clip_epsilon=0.2, kl_beta=0.03),
    )
    query = torch.randn(2, 3, 4)
    sigma = torch.full((2, 3), 0.4)
    velocity, behavior, positive, negative = (torch.randn(2, 3, 4) for _ in range(4))
    weights = torch.rand(2, 3)
    opsd = diffusion_opsd_fitting_loss(
        DiffusionOPSDFittingInputs(query, sigma, velocity, behavior, positive, negative, weights),
        DiffusionOPSDFittingConfig(branch_beta=0.8, advantage_scale=3.0, event_dims=1),
    )
    naive, favorable = (torch.randn(2, 3, 4) for _ in range(2))
    mimic = flowmimic_clean_distribution_loss(
        FlowMimicOPSDInputs(query, sigma, naive, favorable, torch.tensor(0.25)),
        FlowMimicOPSDLossConfig(temperature=0.8, mimic_weight=1.2, base_loss_weight=0.7),
    )
    matching_advantage = torch.randn(2, 3)
    sample_weight = torch.rand(2, 3)
    awm = advantage_weighted_matching_loss(
        AWMInputs(
            policy,
            old,
            target,
            matching_advantage,
            reference_velocity=reference,
            sample_weight=sample_weight,
        ),
        AWMConfig(
            event_dims=1,
            clip_epsilon=0.2,
            reference_beta=0.03,
            old_anchor_beta=0.04,
            matching_power=0.8,
        ),
    )
    return {
        "asymmetric_dpo": float(dpo.loss),
        "diffusion_nft": float(nft.total_loss),
        "flow_grpo_ppo": float(ppo.loss),
        "diffusion_opsd": float(opsd.total_loss),
        "flowmimic_opsd": float(mimic.total_loss),
        "advantage_weighted_matching": float(awm.total_loss),
    }


def test_v2_primitives_match_final_v1_characterization() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[2]
        / "receipts"
        / "v2"
        / "v1_v2_algorithm_characterization.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed = _losses()
    for name, value in observed.items():
        expected = receipt["algorithms"][name]
        assert expected["max_abs_diff"] == 0.0
        assert value == pytest.approx(expected["golden_loss"], abs=1e-7, rel=1e-6)
