from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.diffusion_alignment import (  # noqa: E402
    DiffusionNFTInputs,
    DiffusionNFTLossConfig,
    diffusion_nft_loss,
    make_rectified_flow_batch,
    reconstruct_clean,
    reward_to_optimality,
)


@pytest.mark.parametrize("normalizer", ["group_std", "global_std"])
def test_nearly_constant_rewards_remain_neutral(normalizer: str) -> None:
    rewards = torch.tensor([[0.5, 0.500001, 0.500002, 0.500003]])
    result = reward_to_optimality(rewards, normalizer=normalizer)

    assert (result.probabilities - 0.5).abs().max() < 0.002
    assert not result.zero_variance_mask.any()
    assert (result.normalizer >= 1e-4).all()


def test_constant_rewards_and_explicit_normalizers_keep_their_contracts() -> None:
    constant = reward_to_optimality(torch.full((2, 4), 0.5))
    assert torch.equal(constant.probabilities, torch.full((2, 4), 0.5))
    assert constant.zero_variance_mask.all()
    for normalizer in (2.0, torch.tensor([[2.0]])):
        explicit = reward_to_optimality(
            torch.tensor([[0.0, 2.0]]), normalizer=normalizer, adv_clip_max=1.0
        )
        torch.testing.assert_close(explicit.probabilities, torch.tensor([[0.25, 0.75]]))
        torch.testing.assert_close(explicit.normalizer, torch.full((1, 2), 2.0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("supplied_noise", [False, True])
def test_half_precision_forward_batch_reconstructs_clean(dtype, supplied_noise: bool) -> None:
    clean = torch.randn(1, 4, 3, 8, 8, generator=torch.Generator().manual_seed(123)).to(dtype)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(456)).to(dtype)
    batch = make_rectified_flow_batch(
        clean,
        torch.tensor([[0.025, 0.05, 0.5, 0.975]]),
        noise=noise if supplied_noise else None,
        generator=torch.Generator().manual_seed(456),
    )

    assert batch.clean.dtype == batch.noisy.dtype == batch.target.dtype == torch.float32
    reconstructed = reconstruct_clean(batch.noisy, batch.noise_fraction, batch.target)
    torch.testing.assert_close(reconstructed, clean.float(), atol=1e-6, rtol=1e-6)
    if not supplied_noise:
        expected_noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(456))
        torch.testing.assert_close(batch.target + batch.clean, expected_noise, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_neutral_on_policy_loss_has_no_update(dtype) -> None:
    rng = torch.Generator().manual_seed(87)
    policy = torch.randn(2, 3, 5, 5, generator=rng).to(dtype).requires_grad_(True)
    clean = torch.randn(policy.shape, generator=rng).to(dtype)
    noisy = torch.randn(policy.shape, generator=rng).to(dtype)
    result = diffusion_nft_loss(
        DiffusionNFTInputs(
            policy,
            policy.detach().clone(),
            torch.zeros_like(policy),
            torch.full((2,), 0.5),
            clean=clean,
            noisy=noisy,
            noise_fraction=torch.tensor([0.2, 0.8]),
        ),
        DiffusionNFTLossConfig(beta=0.1, event_dims=3, adaptive_clean_weighting=True),
    )
    gradient = torch.autograd.grad(result.total_loss, policy)[0]

    assert result.implicit_positive.dtype == result.implicit_negative.dtype == torch.float32
    assert gradient.float().norm() < 1e-6
