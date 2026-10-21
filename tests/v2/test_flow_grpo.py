"""Numerical invariant and boundary tests for Flow-GRPO mathematical primitives."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.flow_grpo.advantages import (  # noqa: E402
    AdvantageResult,
    compute_group_advantages,
    group_relative_advantages,
)
from mmtrainer.algorithms.flow_grpo.objectives import (  # noqa: E402
    PPOConfig,
    PPOInputs,
    PPOMetrics,
    ppo_clipped_surrogate,
)
from mmtrainer.algorithms.flow_grpo.probability import (  # noqa: E402
    gaussian_mean_kl,
    isotropic_gaussian_log_prob,
)

# ---------------------------------------------------------------------------
# Advantages: compute_group_advantages / group_relative_advantages
# ---------------------------------------------------------------------------


def test_group_relative_advantages_alias() -> None:
    """compute_group_advantages is an alias for group_relative_advantages."""
    assert compute_group_advantages is group_relative_advantages


def test_advantages_group_normalization() -> None:
    """Each prompt's rollout group has zero mean and unit variance."""
    generator = torch.Generator().manual_seed(42)
    rewards = torch.randn(4, 16, generator=generator) * 3.5 + 2.0
    result: AdvantageResult = compute_group_advantages(rewards)

    assert result.advantages.shape == (4, 16)
    assert not result.zero_variance_mask.any()

    # Per-group mean should be 0.0 and sample std (unbiased=False) should be 1.0
    for group_idx in range(4):
        group_adv = result.advantages[group_idx]
        assert group_adv.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert group_adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)

    torch.testing.assert_close(result.group_mean, rewards.mean(dim=1))
    torch.testing.assert_close(result.group_std, rewards.std(dim=1, unbiased=False))


def test_advantages_zero_variance_groups_handling() -> None:
    """Groups with constant rewards produce 0 advantages without NaN."""
    rewards = torch.tensor(
        [
            [5.0, 5.0, 5.0, 5.0],  # zero variance
            [1.0, 2.0, 3.0, 4.0],  # normal variance
            [-2.5, -2.5, -2.5, -2.5],  # zero variance
        ]
    )
    result = compute_group_advantages(rewards)

    assert result.zero_variance_mask[0].item() is True
    assert result.zero_variance_mask[1].item() is False
    assert result.zero_variance_mask[2].item() is True

    # Zero-variance groups must be exactly zero without NaNs
    torch.testing.assert_close(result.advantages[0], torch.zeros(4))
    torch.testing.assert_close(result.advantages[2], torch.zeros(4))
    assert not torch.isnan(result.advantages).any()

    # Normal variance group is normalized
    assert result.advantages[1].mean().item() == pytest.approx(0.0, abs=1e-5)
    assert result.advantages[1].std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_advantages_epsilon_stability() -> None:
    """std_floor prevents division by near-zero variance."""
    # std of this group is ~1e-7, smaller than std_floor=1e-4
    rewards = torch.tensor([[1.0, 1.0 + 1e-7, 1.0 - 1e-7, 1.0]])
    result = compute_group_advantages(rewards, std_floor=1e-4)

    assert result.zero_variance_mask.all()
    torch.testing.assert_close(result.advantages, torch.zeros_like(rewards))
    assert not torch.isnan(result.advantages).any()


def test_advantages_clipping_bounds() -> None:
    """Advantage clipping clamps values symmetrically to [-clip_value, clip_value]."""
    # Create extreme outlier to ensure normalized advantages exceed 1.5
    rewards = torch.tensor([[0.0, 0.0, 0.0, 100.0]])
    unclipped = compute_group_advantages(rewards, clip_value=None)
    assert unclipped.advantages.max().item() > 1.5

    clipped = compute_group_advantages(rewards, clip_value=1.5)
    assert clipped.advantages.max().item() == pytest.approx(1.5, abs=1e-6)
    assert clipped.advantages.min().item() >= -1.5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_advantages_half_precision_upcasting(dtype: torch.dtype) -> None:
    """Half-precision rewards compute stats in float32 without precision collapse."""
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    result = compute_group_advantages(rewards)

    assert result.group_mean.dtype == torch.float32
    assert result.group_std.dtype == torch.float32
    assert result.advantages.dtype == torch.float32
    assert result.advantages.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert result.advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)


def test_advantages_validation_errors() -> None:
    """Invalid input shapes, types, and configurations raise appropriate errors."""
    with pytest.raises(TypeError, match="rewards must be a floating-point"):
        compute_group_advantages(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    with pytest.raises(ValueError, match="rewards must have shape"):
        compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="std_floor must be positive"):
        compute_group_advantages(torch.randn(2, 4), std_floor=0.0)

    with pytest.raises(ValueError, match="clip_value must be positive"):
        compute_group_advantages(torch.randn(2, 4), clip_value=-1.0)


# ---------------------------------------------------------------------------
# Objectives: ppo_clipped_surrogate
# ---------------------------------------------------------------------------


def _ppo_inputs(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    reference_kl: torch.Tensor | None = None,
    **kwargs,
) -> PPOInputs:
    if reference_kl is None:
        reference_kl = torch.zeros_like(old_logprobs)
    return PPOInputs(
        old_logprobs=old_logprobs,
        new_logprobs=new_logprobs,
        advantages=advantages,
        reference_kl=reference_kl,
        **kwargs,
    )


def test_ppo_identity_ratio_loss_equals_negative_advantage_mean() -> None:
    """When new_logprobs == old_logprobs, ratio is 1.0 and policy_loss is -advantages.mean()."""
    logprobs = torch.randn(2, 4, 3)
    advantages = torch.randn(2, 4)
    inputs = _ppo_inputs(logprobs, logprobs.clone(), advantages)
    metrics: PPOMetrics = ppo_clipped_surrogate(inputs, PPOConfig(clip_epsilon=0.2, kl_beta=0.0))

    torch.testing.assert_close(metrics.ratio, torch.ones_like(logprobs))
    torch.testing.assert_close(metrics.clipped_ratio, torch.ones_like(logprobs))
    assert metrics.clip_fraction.item() == pytest.approx(0.0, abs=1e-6)
    expected_policy_loss = -advantages.mean().item()
    assert metrics.policy_loss.item() == pytest.approx(expected_policy_loss, abs=1e-5)
    assert metrics.loss.item() == pytest.approx(expected_policy_loss, abs=1e-5)


def test_ppo_clipping_activation() -> None:
    """Ratios exceeding (1 - eps, 1 + eps) are clamped in surrogate calculation."""
    clip_epsilon = 0.2
    # Upper bound is 1.2, lower bound is 0.8
    # Log diff of log(1.5) gives ratio 1.5 > 1.2
    old_lp = torch.zeros(1, 2, 1)
    new_lp = torch.tensor([[[math.log(1.5)], [math.log(0.5)]]])
    # Positive advantage for upper clip, negative advantage for lower clip
    advantages = torch.tensor([[2.0, -2.0]])

    inputs = _ppo_inputs(old_lp, new_lp, advantages)
    metrics = ppo_clipped_surrogate(inputs, PPOConfig(clip_epsilon=clip_epsilon, kl_beta=0.0))

    assert metrics.clipping_enabled is True
    # Group 0: ratio = 1.5, clipped_ratio = 1.2, adv = 2.0. Unclipped = 3.0, clipped = 2.4.
    # torch.where(clipped < unclipped, clipped, unclipped) picks 2.4
    assert metrics.ratio[0, 0, 0].item() == pytest.approx(1.5, abs=1e-5)
    assert metrics.clipped_ratio[0, 0, 0].item() == pytest.approx(1.2, abs=1e-5)
    assert metrics.surrogate[0, 0, 0].item() == pytest.approx(2.4, abs=1e-5)

    # Group 1: ratio = 0.5, clipped_ratio = 0.8, adv = -2.0. Unclipped = -1.0, clipped = -1.6.
    # torch.where(clipped < unclipped, clipped, unclipped) picks -1.6
    assert metrics.ratio[0, 1, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert metrics.clipped_ratio[0, 1, 0].item() == pytest.approx(0.8, abs=1e-5)
    assert metrics.surrogate[0, 1, 0].item() == pytest.approx(-1.6, abs=1e-5)

    assert metrics.clip_fraction.item() == pytest.approx(1.0, abs=1e-6)


def test_ppo_clipping_disabled_when_none() -> None:
    """When clip_epsilon is None, clipping is explicitly disabled."""
    old_lp = torch.zeros(1, 1, 1)
    new_lp = torch.tensor([[[math.log(2.0)]]])
    advantages = torch.tensor([[3.0]])
    inputs = _ppo_inputs(old_lp, new_lp, advantages)

    metrics = ppo_clipped_surrogate(inputs, PPOConfig(clip_epsilon=None, kl_beta=0.0))
    assert metrics.clipping_enabled is False
    torch.testing.assert_close(metrics.ratio, metrics.clipped_ratio)
    assert metrics.surrogate.item() == pytest.approx(6.0, abs=1e-5)


def test_ppo_deterministic_transition_mask() -> None:
    """Deterministic transitions are zeroed out and excluded from surrogate/loss."""
    old_lp = torch.zeros(1, 1, 4)
    new_lp = torch.zeros(1, 1, 4)
    advantages = torch.tensor([[1.0]])
    ref_kl = torch.tensor([[[0.1, 0.2, 0.3, 100.0]]])  # Step 3 has extreme KL

    deterministic_mask = torch.tensor([[[False, False, False, True]]])
    inputs = _ppo_inputs(
        old_lp,
        new_lp,
        advantages,
        reference_kl=ref_kl,
        deterministic_mask=deterministic_mask,
    )
    metrics = ppo_clipped_surrogate(inputs, PPOConfig(kl_beta=0.5))

    # Masked transition at index 3 should be ignored in masked_mean
    # Valid transitions: steps 0, 1, 2 with reference_kl 0.1, 0.2, 0.3 (mean = 0.2)
    assert metrics.valid_fraction.item() == pytest.approx(0.75, abs=1e-6)
    expected_kl_loss = 0.5 * 0.2
    assert metrics.kl_loss.item() == pytest.approx(expected_kl_loss, abs=1e-5)


def test_ppo_reference_kl_penalty_scaling() -> None:
    """Reference KL penalty scales linearly with kl_beta."""
    old_lp = torch.randn(2, 3, 4)
    advantages = torch.randn(2, 3)
    ref_kl = torch.rand(2, 3, 4)

    inputs = _ppo_inputs(old_lp, old_lp.clone(), advantages, reference_kl=ref_kl)
    metrics_0 = ppo_clipped_surrogate(inputs, PPOConfig(kl_beta=0.0))
    metrics_1 = ppo_clipped_surrogate(inputs, PPOConfig(kl_beta=0.02))
    metrics_2 = ppo_clipped_surrogate(inputs, PPOConfig(kl_beta=0.04))

    assert metrics_0.kl_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics_0.loss.item() == pytest.approx(metrics_0.policy_loss.item(), abs=1e-6)

    # kl_beta=0.04 should have twice the kl_loss of kl_beta=0.02
    assert metrics_2.kl_loss.item() == pytest.approx(2.0 * metrics_1.kl_loss.item(), rel=1e-5)
    torch.testing.assert_close(metrics_1.policy_loss, metrics_2.policy_loss)


def test_ppo_non_finite_rejection() -> None:
    """NaN or Inf in inputs triggers ValueError."""
    shape = (2, 2, 2)
    valid_old = torch.zeros(shape)
    valid_new = torch.zeros(shape)
    valid_adv = torch.zeros(2, 2)
    valid_kl = torch.zeros(shape)

    for field, bad_tensor in [
        ("old_logprobs", torch.full(shape, float("nan"))),
        ("new_logprobs", torch.full(shape, float("inf"))),
        ("advantages", torch.full((2, 2), float("nan"))),
        ("reference_kl", torch.full(shape, float("inf"))),
    ]:
        kwargs = {
            "old_logprobs": valid_old,
            "new_logprobs": valid_new,
            "advantages": valid_adv,
            "reference_kl": valid_kl,
        }
        kwargs[field] = bad_tensor
        with pytest.raises(ValueError, match="must contain only finite values"):
            ppo_clipped_surrogate(PPOInputs(**kwargs))


def test_ppo_validation_contracts() -> None:
    """Mismatched shapes, invalid dtypes, and invalid configs raise clear errors."""
    valid_lp = torch.zeros(2, 3, 4)
    valid_adv = torch.zeros(2, 3)
    valid_kl = torch.zeros(2, 3, 4)

    # Shape mismatch
    with pytest.raises(ValueError, match="must match"):
        ppo_clipped_surrogate(
            _ppo_inputs(valid_lp, torch.zeros(2, 3, 5), valid_adv, reference_kl=valid_kl)
        )

    # ndim != 3
    with pytest.raises(ValueError, match="must have shape"):
        ppo_clipped_surrogate(
            _ppo_inputs(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                valid_adv,
                reference_kl=torch.zeros(2, 3),
            )
        )

    # clip_epsilon outside (0, 1)
    with pytest.raises(ValueError, match="clip_epsilon must be in"):
        ppo_clipped_surrogate(
            _ppo_inputs(valid_lp, valid_lp, valid_adv, valid_kl),
            PPOConfig(clip_epsilon=1.5),
        )

    # kl_beta negative
    with pytest.raises(ValueError, match="kl_beta must be non-negative"):
        ppo_clipped_surrogate(
            _ppo_inputs(valid_lp, valid_lp, valid_adv, valid_kl),
            PPOConfig(kl_beta=-0.1),
        )


# ---------------------------------------------------------------------------
# Probability: isotropic_gaussian_log_prob & gaussian_mean_kl
# ---------------------------------------------------------------------------


def test_isotropic_gaussian_log_prob_analytic_value() -> None:
    """Log probability matches closed-form isotropic Gaussian density."""
    # Shape: [batch=1, group=1, step=1, state=1]
    value = torch.tensor([[[[0.0]]]])
    mean = torch.tensor([[[[0.0]]]])
    std = torch.tensor([[[[1.0]]]])

    # log p(0; 0, 1) = -0.5 * ln(2*pi)
    expected = -0.5 * math.log(2.0 * math.pi)
    lp = isotropic_gaussian_log_prob(value, mean, std)
    assert lp.item() == pytest.approx(expected, abs=1e-6)

    # Non-zero offset: x=2, mu=0, sigma=2
    # log p(2; 0, 2) = -0.5 * ((2/2)^2 + 2*ln(2) + ln(2*pi))
    lp_offset = isotropic_gaussian_log_prob(
        torch.tensor([[[[2.0]]]]),
        mean,
        torch.tensor([[[[2.0]]]]),
    )
    expected_offset = -0.5 * (1.0 + 2.0 * math.log(2.0) + math.log(2.0 * math.pi))
    assert lp_offset.item() == pytest.approx(expected_offset, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_isotropic_gaussian_log_prob_half_precision_upcasting(dtype: torch.dtype) -> None:
    """Half precision inputs are upcast to float32 for log-probability computation."""
    value = torch.tensor([[[[1.0, -1.0]]]], dtype=dtype)
    mean = torch.tensor([[[[0.0, 0.0]]]], dtype=dtype)
    std = torch.tensor([[[[1.0, 1.0]]]], dtype=dtype)

    lp = isotropic_gaussian_log_prob(value, mean, std)
    assert lp.dtype == torch.float32
    expected = -0.5 * (1.0 + math.log(2.0 * math.pi))
    assert lp.item() == pytest.approx(expected, abs=1e-4)


def test_isotropic_gaussian_log_prob_deterministic_mask() -> None:
    """Deterministic transitions receive zero contribution."""
    value = torch.randn(1, 1, 3, 4)
    mean = torch.randn(1, 1, 3, 4)
    std = torch.ones(1, 1, 3, 1)
    mask = torch.tensor([[[True, False, False]]])

    lp = isotropic_gaussian_log_prob(value, mean, std, deterministic_mask=mask)
    assert lp.shape == (1, 1, 3)
    assert lp[0, 0, 0].item() == 0.0
    assert lp[0, 0, 1].item() != 0.0


def test_gaussian_mean_kl_analytic_and_non_negativity() -> None:
    """Closed-form Gaussian mean KL: 0.5 * ||mu_1 - mu_2||^2 / sigma^2 >= 0."""
    generator = torch.Generator().manual_seed(99)
    policy_mean = torch.randn(2, 3, 4, 8, generator=generator)
    ref_mean = torch.randn(2, 3, 4, 8, generator=generator)
    std = torch.full((2, 3, 4, 1), 1.5)

    kl = gaussian_mean_kl(policy_mean, ref_mean, std)
    # KL must be strictly non-negative
    assert (kl >= 0).all()

    # Identical distributions have exactly 0 KL
    same_kl = gaussian_mean_kl(policy_mean, policy_mean, std)
    torch.testing.assert_close(same_kl, torch.zeros_like(same_kl))

    # Analytic single-element check: mu_1=1.0, mu_2=3.0, std=2.0 -> 0.5 * 4 / 4 = 0.5
    analytic_kl = gaussian_mean_kl(
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[3.0]]]]),
        torch.tensor([[[[2.0]]]]),
    )
    assert analytic_kl.item() == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gaussian_mean_kl_half_precision_upcasting(dtype: torch.dtype) -> None:
    """Half precision KL upcasts to float32."""
    p_mean = torch.tensor([[[[2.0]]]], dtype=dtype)
    r_mean = torch.tensor([[[[0.0]]]], dtype=dtype)
    std = torch.tensor([[[[2.0]]]], dtype=dtype)

    kl = gaussian_mean_kl(p_mean, r_mean, std)
    assert kl.dtype == torch.float32
    assert kl.item() == pytest.approx(0.5, abs=1e-5)
