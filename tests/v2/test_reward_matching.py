"""Unit tests for Advantage Weighted Matching (AWM) and Reinforce Adjoint Matching (RAM)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.reward_matching import (  # noqa: E402
    AWMConfig,
    AWMInputs,
    AWMMetrics,
    RAMMetrics,
    advantage_weighted_matching_loss,
    normalized_advantages,
    reinforce_adjoint_matching_loss,
)


def _batch_velocities(
    shape: tuple[int, ...] = (2, 3, 4, 4),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(99)
    policy = torch.randn(shape, generator=gen)
    old = torch.randn(shape, generator=gen)
    target = torch.randn(shape, generator=gen)
    return policy, old, target


# ==============================================================================
# 1. normalized_advantages Tests
# ==============================================================================


def test_normalized_advantages_zero_variance_mask_neutrality() -> None:
    """Zero-variance mask forces advantages to 0 even if centered rewards are non-zero."""
    centered = torch.tensor([5.0, -3.0, 2.0, -10.0])
    normalizer = torch.tensor([1.5, 0.8, 1.0, 2.0])
    # Elements 1 and 3 are masked out as zero-variance
    mask = torch.tensor([False, True, False, True])

    adv = normalized_advantages(centered, normalizer, mask)

    # Masked positions are strictly 0.0
    assert adv[1].item() == pytest.approx(0.0, abs=1e-7)
    assert adv[3].item() == pytest.approx(0.0, abs=1e-7)

    # Unmasked positions match centered / normalizer
    assert adv[0].item() == pytest.approx(5.0 / 1.5, rel=1e-6)
    assert adv[2].item() == pytest.approx(2.0 / 1.0, rel=1e-6)


def test_normalized_advantages_normalizer_clamping_and_non_negative() -> None:
    """Normalizer must be non-negative, and zero normalizer must be marked in zero-variance mask."""
    centered = torch.tensor([1.0, 2.0])
    # Negative normalizer rejected
    with pytest.raises(ValueError, match="normalizer must be non-negative"):
        normalized_advantages(centered, torch.tensor([-1.0, 1.0]), torch.tensor([False, False]))

    # Zero normalizer not marked in zero_variance_mask rejected
    with pytest.raises(ValueError, match="zero normalizer entries must be marked zero-variance"):
        normalized_advantages(centered, torch.tensor([0.0, 1.0]), torch.tensor([False, False]))

    # Zero normalizer marked in mask is accepted and yields 0 advantage
    adv = normalized_advantages(centered, torch.tensor([0.0, 1.0]), torch.tensor([True, False]))
    assert adv[0].item() == pytest.approx(0.0, abs=1e-7)
    assert adv[1].item() == pytest.approx(2.0, rel=1e-6)

    # Near-zero positive normalizer is clamped to finfo.tiny without NaN/Inf
    tiny_normalizer = torch.tensor([1e-45, 1.0])
    adv_tiny = normalized_advantages(centered, tiny_normalizer, torch.tensor([False, False]))
    assert torch.isfinite(adv_tiny).all()


def test_normalized_advantages_clip_value_enforcement() -> None:
    """Advantages are clamped strictly within [-clip_value, clip_value]."""
    centered = torch.tensor([10.0, -15.0, 1.0])
    normalizer = torch.tensor([1.0, 1.0, 1.0])
    mask = torch.tensor([False, False, False])

    clip_val = 2.5
    adv = normalized_advantages(centered, normalizer, mask, clip_value=clip_val)

    assert adv[0].item() == pytest.approx(2.5, abs=1e-7)
    assert adv[1].item() == pytest.approx(-2.5, abs=1e-7)
    assert adv[2].item() == pytest.approx(1.0, abs=1e-7)
    assert bool((adv >= -clip_val).all()) and bool((adv <= clip_val).all())

    # Invalid clip_value
    with pytest.raises(ValueError, match="clip_value must be positive when supplied"):
        normalized_advantages(centered, normalizer, mask, clip_value=0.0)
    with pytest.raises(ValueError, match="clip_value must be finite and >= 0.0"):
        normalized_advantages(centered, normalizer, mask, clip_value=-1.0)


def test_normalized_advantages_validation() -> None:
    """Shape, type, device, and finite validation for normalized_advantages."""
    centered = torch.tensor([1.0, 2.0])
    normalizer = torch.tensor([1.0, 2.0])
    mask = torch.tensor([False, False])

    # Shape mismatch
    with pytest.raises(ValueError, match="must have matching shapes"):
        normalized_advantages(centered, normalizer[:1], mask[:1])
    with pytest.raises(ValueError, match="must match centered_rewards shape"):
        normalized_advantages(centered, normalizer, mask[:1])

    # Type error if mask not bool
    with pytest.raises(TypeError, match="must be a boolean Tensor"):
        normalized_advantages(centered, normalizer, torch.tensor([0, 1]))  # type: ignore[arg-type]

    # Non-finite values
    with pytest.raises(ValueError, match="must contain only finite values"):
        normalized_advantages(torch.tensor([float("nan"), 1.0]), normalizer, mask)
    with pytest.raises(ValueError, match="must contain only finite values"):
        normalized_advantages(centered, torch.tensor([float("inf"), 1.0]), mask)


# ==============================================================================
# 2. advantage_weighted_matching_loss (AWM) Tests
# ==============================================================================


def test_awm_closed_form_exact_zero_loss() -> None:
    """When policy matches old and target with zero advantages, total loss is strictly 0."""
    state = torch.randn(2, 3, 4, 4)
    advantages = torch.zeros(2)

    inputs = AWMInputs(
        policy_velocity=state,
        old_velocity=state,
        target_velocity=state,
        advantages=advantages,
    )
    config = AWMConfig(reference_beta=0.0, old_anchor_beta=0.0)

    metrics = advantage_weighted_matching_loss(inputs, config)
    assert isinstance(metrics, AWMMetrics)
    assert metrics.total_loss.ndim == 0
    assert metrics.total_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.policy_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.policy_mse.abs().max().item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.old_mse.abs().max().item() == pytest.approx(0.0, abs=1e-7)
    assert (
        metrics.ratio.item()
        if metrics.ratio.ndim == 0
        else metrics.ratio.mean().item() == pytest.approx(1.0, abs=1e-6)
    )
    assert metrics.clipped_ratio.mean().item() == pytest.approx(1.0, abs=1e-6)
    assert metrics.clip_fraction.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.approx_kl.item() == pytest.approx(0.0, abs=1e-7)


def test_awm_on_policy_advantage_loss() -> None:
    """When policy matches old velocity, ratio is 1.0 and policy_loss == -mean(advantages)."""
    policy, _, target = _batch_velocities((2, 3, 4, 4))
    old = policy.clone()
    advantages = torch.tensor([1.5, -0.5])

    inputs = AWMInputs(
        policy_velocity=policy,
        old_velocity=old,
        target_velocity=target,
        advantages=advantages,
    )
    config = AWMConfig(clip_epsilon=0.2, reference_beta=0.0, old_anchor_beta=0.0)

    metrics = advantage_weighted_matching_loss(inputs, config)
    # Ratio = exp(old_mse - policy_mse) = exp(0) = 1.0, clipped_ratio = 1.0
    # policy_loss = -mean(1.0 * advantages) = -mean([1.5, -0.5]) = -0.5
    assert metrics.policy_loss.item() == pytest.approx(-0.5, rel=1e-6)
    assert metrics.total_loss.item() == pytest.approx(-0.5, rel=1e-6)
    assert metrics.clip_fraction.item() == pytest.approx(0.0, abs=1e-7)


def test_awm_ratio_clipping() -> None:
    """Ratio is clamped to [1 - clip_epsilon, 1 + clip_epsilon] with PPO min-surrogate."""
    # Construct exact policy and old velocities such that old_mse - policy_mse is large
    shape = (2, 2)
    # Target at 0.0
    target = torch.zeros(shape)
    # Policy at 0.0 -> policy_mse = 0.0
    policy = torch.zeros(shape)
    # Old at 2.0 -> old_mse = 4.0
    old = torch.full(shape, 2.0)

    # With matching_power=1.0:
    # policy_error = 0.0, old_error = 4.0
    # log_ratio = old_error - policy_error = 4.0
    # ratio = exp(4.0) = ~54.598
    clip_eps = 0.2
    # clipped_ratio = clamp(ratio, 0.8, 1.2) = 1.2

    # Case 1: Positive advantage (e.g. +2.0)
    # unclipped = ratio * adv = 54.598 * 2.0 = 109.196
    # clipped = clipped_ratio * adv = 1.2 * 2.0 = 2.4
    # min surrogate selects clipped: loss = -2.4
    adv_pos = torch.tensor([2.0, 2.0])
    inputs_pos = AWMInputs(
        policy_velocity=policy, old_velocity=old, target_velocity=target, advantages=adv_pos
    )
    config = AWMConfig(event_dims=1, clip_epsilon=clip_eps)

    metrics_pos = advantage_weighted_matching_loss(inputs_pos, config)
    assert metrics_pos.clipped_ratio[0].item() == pytest.approx(1.0 + clip_eps, rel=1e-5)
    assert metrics_pos.policy_loss.item() == pytest.approx(-(1.0 + clip_eps) * 2.0, rel=1e-5)
    assert metrics_pos.clip_fraction.item() == pytest.approx(1.0, abs=1e-6)

    # Case 2: Negative advantage (e.g. -2.0) with policy_mse > old_mse
    # Policy at 2.0 -> policy_mse = 4.0
    # Old at 0.0 -> old_mse = 0.0
    # log_ratio = 0.0 - 4.0 = -4.0 -> ratio = exp(-4.0) = ~0.0183
    # clipped_ratio = 0.8
    # unclipped = 0.0183 * (-2) = -0.0366
    # clipped = 0.8 * (-2) = -1.6
    # torch.where(clipped < unclipped, clipped, unclipped): -1.6 < -0.0366 is True, so selects -1.6
    # loss = -(-1.6) = +1.6
    adv_neg = torch.tensor([-2.0, -2.0])
    inputs_neg = AWMInputs(
        policy_velocity=old, old_velocity=policy, target_velocity=target, advantages=adv_neg
    )
    metrics_neg = advantage_weighted_matching_loss(inputs_neg, config)
    assert metrics_neg.clipped_ratio[0].item() == pytest.approx(1.0 - clip_eps, rel=1e-5)
    assert metrics_neg.policy_loss.item() == pytest.approx(-((1.0 - clip_eps) * (-2.0)), rel=1e-5)
    assert metrics_neg.clip_fraction.item() == pytest.approx(1.0, abs=1e-6)


def test_awm_weighting_policy_vs_anchor_vs_reference() -> None:
    """total_loss = policy_loss + reference_beta * ref_loss + old_anchor_beta * old_loss."""
    policy, old, target = _batch_velocities((2, 3, 4, 4))
    reference = torch.randn_like(policy)
    advantages = torch.tensor([0.5, -0.5])

    inputs = AWMInputs(
        policy_velocity=policy,
        old_velocity=old,
        target_velocity=target,
        advantages=advantages,
        reference_velocity=reference,
    )
    ref_beta = 0.4
    old_beta = 0.3
    config = AWMConfig(reference_beta=ref_beta, old_anchor_beta=old_beta)

    metrics = advantage_weighted_matching_loss(inputs, config)

    # Independently compute reference_loss and old_anchor_loss
    expected_ref = (policy - reference).square().mean(dim=(1, 2, 3)).mean()
    expected_old = (policy - old).square().mean(dim=(1, 2, 3)).mean()

    assert metrics.reference_loss.item() == pytest.approx(expected_ref.item(), rel=1e-5)
    assert metrics.old_anchor_loss.item() == pytest.approx(expected_old.item(), rel=1e-5)

    expected_total = (
        metrics.policy_loss + ref_beta * metrics.reference_loss + old_beta * metrics.old_anchor_loss
    )
    assert metrics.total_loss.item() == pytest.approx(expected_total.item(), rel=1e-5)

    # Missing reference_velocity when reference_beta > 0 raises ValueError
    inputs_no_ref = AWMInputs(
        policy_velocity=policy, old_velocity=old, target_velocity=target, advantages=advantages
    )
    with pytest.raises(
        ValueError, match="reference_velocity is required for positive reference_beta"
    ):
        advantage_weighted_matching_loss(inputs_no_ref, config)


def test_awm_non_finite_rejection() -> None:
    """NaN or Inf in any velocity or advantage tensor raises ValueError."""
    policy, old, target = _batch_velocities((2, 3, 4, 4))
    advantages = torch.tensor([1.0, 1.0])

    nan_policy = policy.clone()
    nan_policy[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite values"):
        advantage_weighted_matching_loss(AWMInputs(nan_policy, old, target, advantages))

    inf_adv = advantages.clone()
    inf_adv[1] = float("inf")
    with pytest.raises(ValueError, match="must contain only finite values"):
        advantage_weighted_matching_loss(AWMInputs(policy, old, target, inf_adv))


def test_awm_gradient_flow() -> None:
    """Only policy receives gradients; old, target, and reference are detached."""
    policy, old, target = _batch_velocities((2, 3, 4, 4))
    reference = torch.randn_like(policy)
    policy.requires_grad_(True)
    old.requires_grad_(True)
    target.requires_grad_(True)
    reference.requires_grad_(True)

    inputs = AWMInputs(policy, old, target, torch.tensor([1.0, -1.0]), reference_velocity=reference)
    config = AWMConfig(reference_beta=0.1, old_anchor_beta=0.1)
    metrics = advantage_weighted_matching_loss(inputs, config)
    metrics.total_loss.backward()

    assert policy.grad is not None
    assert policy.grad.abs().sum().item() > 0
    assert old.grad is None
    assert target.grad is None
    assert reference.grad is None


# ==============================================================================
# 3. reinforce_adjoint_matching_loss (RAM) Tests
# ==============================================================================


def test_ram_score_difference_matching_closed_form() -> None:
    """When policy matches the adjoint target, RAM loss is strictly 0."""
    policy, old, flow_target = _batch_velocities((2, 3, 4, 4))
    reference = policy.clone()
    advantages = torch.tensor([0.0, 0.0])

    # When advantage is 0, target = reference = policy -> loss is strictly 0.0
    metrics = reinforce_adjoint_matching_loss(
        policy_velocity=policy,
        old_velocity=old,
        reference_velocity=reference,
        flow_target_velocity=flow_target,
        advantages=advantages,
        reward_multiplier=1.0,
    )
    assert isinstance(metrics, RAMMetrics)
    assert metrics.total_loss.ndim == 0
    assert metrics.total_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.policy_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.policy_reference_rms.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(metrics.target_velocity, reference)


def test_ram_advantage_directional_movement() -> None:
    """Positive advantage moves policy toward flow_target, negative moves away."""
    # Consider 1D vectors
    reference = torch.tensor([[0.0, 0.0]])
    old = torch.tensor([[1.0, 1.0]])
    flow_target = torch.tensor([[3.0, 3.0]])
    # direction = flow_target - old = [2.0, 2.0]
    # target = reference + reward_multiplier * advantage * direction

    # Positive advantage (+1.0)
    adv_pos = torch.tensor([1.0])
    policy_pos = reference.clone().requires_grad_(True)
    metrics_pos = reinforce_adjoint_matching_loss(
        policy_velocity=policy_pos,
        old_velocity=old,
        reference_velocity=reference,
        flow_target_velocity=flow_target,
        advantages=adv_pos,
        reward_multiplier=1.0,
        event_dims=1,
    )
    # target = [0, 0] + 1.0 * 1.0 * [2.0, 2.0] = [2.0, 2.0]
    assert torch.allclose(metrics_pos.target_velocity, torch.tensor([[2.0, 2.0]]))
    metrics_pos.total_loss.backward()

    # Gradient of 0.5 * (policy - target)^2 with mean over 2 elements:
    # d/d(policy) = (policy - target) = [0-2, 0-2] = [-2.0, -2.0]
    # Update step: policy_next = policy - lr * grad = [0, 0] - lr * [-2, -2]
    # moves toward target!
    assert policy_pos.grad is not None
    assert (policy_pos.grad < 0).all()

    # Negative advantage (-1.0)
    adv_neg = torch.tensor([-1.0])
    policy_neg = reference.clone().requires_grad_(True)
    metrics_neg = reinforce_adjoint_matching_loss(
        policy_velocity=policy_neg,
        old_velocity=old,
        reference_velocity=reference,
        flow_target_velocity=flow_target,
        advantages=adv_neg,
        reward_multiplier=1.0,
        event_dims=1,
    )
    # target = [0, 0] + 1.0 * (-1.0) * [2.0, 2.0] = [-2.0, -2.0]
    assert torch.allclose(metrics_neg.target_velocity, torch.tensor([[-2.0, -2.0]]))
    metrics_neg.total_loss.backward()
    # Gradient is positive -> update moves away from target!
    assert policy_neg.grad is not None
    assert (policy_neg.grad > 0).all()


def test_ram_advantage_clipping_bounds_target_deviation() -> None:
    """Normalized advantage clipping bounds the maximal displacement of the RAM target."""
    policy, old, flow_target = _batch_velocities((2, 3, 4, 4))
    reference = torch.randn_like(policy)

    # Massive raw rewards
    raw_centered = torch.tensor([1000.0, -2000.0])
    normalizer = torch.tensor([1.0, 1.0])
    mask = torch.tensor([False, False])

    clip_val = 1.5
    clipped_adv = normalized_advantages(raw_centered, normalizer, mask, clip_value=clip_val)
    assert clipped_adv[0].item() == pytest.approx(1.5)
    assert clipped_adv[1].item() == pytest.approx(-1.5)

    metrics = reinforce_adjoint_matching_loss(
        policy_velocity=policy,
        old_velocity=old,
        reference_velocity=reference,
        flow_target_velocity=flow_target,
        advantages=clipped_adv,
        reward_multiplier=2.0,
    )

    # Displacements are bounded by multiplier * clip_val * |flow_target - old|
    displacement = (metrics.target_velocity - reference).abs()
    max_expected = 2.0 * clip_val * (flow_target - old).abs()
    assert (displacement <= max_expected + 1e-5).all()


def test_ram_sample_weight_effect() -> None:
    """sample_weight scales per-sample loss linearly and rejects negative weights."""
    policy, old, flow_target = _batch_velocities((2, 3, 4, 4))
    reference = torch.randn_like(policy)
    adv = torch.tensor([1.0, 1.0])

    unweighted = reinforce_adjoint_matching_loss(policy, old, reference, flow_target, adv)

    weight = torch.tensor([2.0, 0.5])
    weighted = reinforce_adjoint_matching_loss(
        policy, old, reference, flow_target, adv, sample_weight=weight
    )

    expected_per_sample = unweighted.per_sample_loss * weight
    assert torch.allclose(weighted.per_sample_loss, expected_per_sample)

    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        reinforce_adjoint_matching_loss(
            policy, old, reference, flow_target, adv, sample_weight=torch.tensor([-1.0, 1.0])
        )


def test_ram_non_finite_rejection() -> None:
    """Non-finite velocities or rewards raise ValueError in RAM."""
    policy, old, flow_target = _batch_velocities((2, 3, 4, 4))
    reference = torch.randn_like(policy)
    adv = torch.tensor([1.0, 1.0])

    nan_flow = flow_target.clone()
    nan_flow[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite values"):
        reinforce_adjoint_matching_loss(policy, old, reference, nan_flow, adv)

    with pytest.raises(ValueError, match="reward_multiplier must be finite"):
        reinforce_adjoint_matching_loss(
            policy, old, reference, flow_target, adv, reward_multiplier=float("inf")
        )


def test_ram_gradient_flow_and_target_detach() -> None:
    """RAM target is detached; only policy_velocity receives gradients."""
    policy, old, flow_target = _batch_velocities((2, 3, 4, 4))
    reference = torch.randn_like(policy)
    policy.requires_grad_(True)
    old.requires_grad_(True)
    flow_target.requires_grad_(True)
    reference.requires_grad_(True)

    metrics = reinforce_adjoint_matching_loss(
        policy_velocity=policy,
        old_velocity=old,
        reference_velocity=reference,
        flow_target_velocity=flow_target,
        advantages=torch.tensor([1.0, -1.0]),
    )
    assert not metrics.target_velocity.requires_grad
    metrics.total_loss.backward()

    assert policy.grad is not None
    assert policy.grad.abs().sum().item() > 0
    assert old.grad is None
    assert flow_target.grad is None
    assert reference.grad is None
