"""Numerical invariant and boundary tests for DiffusionOPSD primitives."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.diffusion_opsd.ema import (  # noqa: E402
    EMAParameterStore,
    ModelEMA,
    behavior_policy_retention,
    checkpoint_policy_retention,
)
from mmtrainer.algorithms.diffusion_opsd.objectives import (  # noqa: E402
    DiffusionOPSDFittingConfig,
    DiffusionOPSDFittingInputs,
    DiffusionOPSDLoss,
    clean_output_prediction,
    construct_bounded_reward_targets,
    diffusion_opsd_fitting_loss,
    group_normalized_endpoint_weights,
)
from mmtrainer.algorithms.diffusion_opsd.runtime import (  # noqa: E402
    distributed_reward_moments,
    require_all_ranks,
)

# ---------------------------------------------------------------------------
# Objectives: diffusion_opsd_fitting_loss
# ---------------------------------------------------------------------------


def _fitting_inputs(
    *,
    shape: tuple[int, ...] = (2, 4, 3, 4, 4),
    weights: torch.Tensor | None = None,
    seed: int = 42,
) -> DiffusionOPSDFittingInputs:
    generator = torch.Generator().manual_seed(seed)
    query_state = torch.randn(shape, generator=generator)
    query_sigma = torch.full(shape[:2], 0.3)
    policy_velocity = torch.randn(shape, generator=generator)
    behavior_clean = torch.randn(shape, generator=generator)
    positive_target = torch.randn(shape, generator=generator)
    negative_target = torch.randn(shape, generator=generator)
    if weights is None:
        weights = torch.rand(shape[:2], generator=generator)
    return DiffusionOPSDFittingInputs(
        query_state=query_state,
        query_sigma=query_sigma,
        policy_velocity=policy_velocity,
        behavior_clean=behavior_clean,
        positive_target=positive_target,
        negative_target=negative_target,
        weights=weights,
    )


def test_clean_output_prediction_analytic() -> None:
    """Eq. 1: clean prediction is x0_hat = x_t - sigma * velocity."""
    state = torch.tensor([1.0, 2.0, 3.0])
    sigma = torch.tensor(0.5)
    velocity = torch.tensor([2.0, 0.0, -2.0])
    # 1.0 - 0.5*2 = 0.0; 2.0 - 0.5*0 = 2.0; 3.0 - 0.5*(-2) = 4.0
    expected = torch.tensor([0.0, 2.0, 4.0])
    pred = clean_output_prediction(state, sigma, velocity)
    torch.testing.assert_close(pred, expected)


def test_diffusion_opsd_branch_loss_weighting() -> None:
    """per_sample loss is exact convex combination of positive and negative branch losses."""
    # When weights = 1.0, per_sample must equal positive_loss
    inputs_pos = _fitting_inputs(weights=torch.ones(2, 4))
    loss_pos: DiffusionOPSDLoss = diffusion_opsd_fitting_loss(inputs_pos)
    torch.testing.assert_close(loss_pos.per_sample, loss_pos.positive_loss)

    # When weights = 0.0, per_sample must equal negative_loss
    inputs_neg = _fitting_inputs(weights=torch.zeros(2, 4))
    loss_neg: DiffusionOPSDLoss = diffusion_opsd_fitting_loss(inputs_neg)
    torch.testing.assert_close(loss_neg.per_sample, loss_neg.negative_loss)

    # When weights = 0.25, per_sample must equal 0.25 * pos + 0.75 * neg
    inputs_quarter = _fitting_inputs(weights=torch.full((2, 4), 0.25))
    loss_quarter = diffusion_opsd_fitting_loss(inputs_quarter)
    expected_sample = 0.25 * loss_quarter.positive_loss + 0.75 * loss_quarter.negative_loss
    torch.testing.assert_close(loss_quarter.per_sample, expected_sample)


def test_diffusion_opsd_advantage_scaling() -> None:
    """total_loss scales linearly with advantage_scale."""
    inputs = _fitting_inputs()
    loss_1 = diffusion_opsd_fitting_loss(inputs, DiffusionOPSDFittingConfig(advantage_scale=2.0))
    loss_2 = diffusion_opsd_fitting_loss(inputs, DiffusionOPSDFittingConfig(advantage_scale=6.0))

    assert loss_2.total_loss.item() == pytest.approx(3.0 * loss_1.total_loss.item(), rel=1e-5)
    torch.testing.assert_close(loss_1.per_sample, loss_2.per_sample)


def test_diffusion_opsd_non_finite_rejection() -> None:
    """NaN or Inf in any fitting input raises ValueError."""
    shape = (2, 2, 4)
    base_inputs = _fitting_inputs(shape=shape, weights=torch.full((2, 2), 0.5))

    for field in (
        "query_state",
        "policy_velocity",
        "behavior_clean",
        "positive_target",
        "negative_target",
    ):
        bad_val = getattr(base_inputs, field).clone()
        bad_val[0, 0, 0] = float("nan")
        kwargs = {
            "query_state": base_inputs.query_state,
            "query_sigma": base_inputs.query_sigma,
            "policy_velocity": base_inputs.policy_velocity,
            "behavior_clean": base_inputs.behavior_clean,
            "positive_target": base_inputs.positive_target,
            "negative_target": base_inputs.negative_target,
            "weights": base_inputs.weights,
        }
        kwargs[field] = bad_val
        with pytest.raises(ValueError, match="must contain only finite values"):
            diffusion_opsd_fitting_loss(DiffusionOPSDFittingInputs(**kwargs))

    # Weights with Inf
    bad_weights = base_inputs.weights.clone()
    bad_weights[0, 0] = float("inf")
    with pytest.raises(ValueError, match="must contain only finite values"):
        diffusion_opsd_fitting_loss(
            DiffusionOPSDFittingInputs(
                query_state=base_inputs.query_state,
                query_sigma=base_inputs.query_sigma,
                policy_velocity=base_inputs.policy_velocity,
                behavior_clean=base_inputs.behavior_clean,
                positive_target=base_inputs.positive_target,
                negative_target=base_inputs.negative_target,
                weights=bad_weights,
            )
        )


def test_diffusion_opsd_weights_range_validation() -> None:
    """Weights outside [0, 1] raise ValueError."""
    shape = (2, 2, 4)
    base_inputs = _fitting_inputs(shape=shape)

    for out_of_bounds in (-0.1, 1.2):
        bad_weights = torch.full((2, 2), out_of_bounds)
        with pytest.raises(ValueError, match=r"weights must lie in \[0, 1\]"):
            diffusion_opsd_fitting_loss(
                DiffusionOPSDFittingInputs(
                    query_state=base_inputs.query_state,
                    query_sigma=base_inputs.query_sigma,
                    policy_velocity=base_inputs.policy_velocity,
                    behavior_clean=base_inputs.behavior_clean,
                    positive_target=base_inputs.positive_target,
                    negative_target=base_inputs.negative_target,
                    weights=bad_weights,
                )
            )


def test_group_normalized_endpoint_weights_contract() -> None:
    """Eq. 5: weights are bounded in [0, 1] and centered around group mean."""
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    weights_res = group_normalized_endpoint_weights(rewards, advantage_scale=5.0)

    # Weights must strictly lie within [0, 1]
    assert (weights_res.weights >= 0.0).all()
    assert (weights_res.weights <= 1.0).all()
    # Symmetrically centered: reward above group mean has weight > 0.5, below has < 0.5
    assert (weights_res.weights[:, 0] < 0.5).all()
    assert (weights_res.weights[:, 3] > 0.5).all()

    # Rejection of non-finite rewards
    bad_rewards = rewards.clone()
    bad_rewards[0, 0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite values"):
        group_normalized_endpoint_weights(bad_rewards)


def test_construct_bounded_reward_targets_trust_region() -> None:
    """Ascent/descent targets stay strictly inside configured target_radius trust region."""
    anchor = torch.randn(2, 3, 8)

    def simple_reward(clean: torch.Tensor) -> torch.Tensor:
        return clean.sum(dim=-1)

    target_radius = 0.15
    targets = construct_bounded_reward_targets(
        anchor,
        simple_reward,
        target_radius=target_radius,
        target_steps=3,
        event_dims=1,
    )
    dims = (2,)
    base = anchor.float()
    radius = target_radius * torch.linalg.vector_norm(base, dim=dims)

    pos_dist = torch.linalg.vector_norm(targets.positive - base, dim=dims)
    neg_dist = torch.linalg.vector_norm(targets.negative - base, dim=dims)

    # Must stay within trust region
    assert (pos_dist <= radius + 1e-5).all()
    assert (neg_dist <= radius + 1e-5).all()


# ---------------------------------------------------------------------------
# EMA: ModelEMA / EMAParameterStore
# ---------------------------------------------------------------------------


def test_model_ema_alias() -> None:
    """ModelEMA is an alias for EMAParameterStore."""
    assert ModelEMA is EMAParameterStore


def test_ema_retention_schedules() -> None:
    """Paper App. C.1 retention schedules for behavior and checkpoint policies."""
    # Behavior policy: min(1e-3 * u, 0.5)
    assert behavior_policy_retention(100) == pytest.approx(0.1, abs=1e-6)
    assert behavior_policy_retention(500) == pytest.approx(0.5, abs=1e-6)
    assert behavior_policy_retention(1000) == pytest.approx(0.5, abs=1e-6)

    with pytest.raises(ValueError, match="positive integer"):
        behavior_policy_retention(0)

    # Checkpoint policy: min((u + 1) / (u + 10), 0.9)
    assert checkpoint_policy_retention(1) == pytest.approx(2.0 / 11.0, abs=1e-6)
    assert checkpoint_policy_retention(80) == pytest.approx(0.9, abs=1e-6)
    assert checkpoint_policy_retention(200) == pytest.approx(0.9, abs=1e-6)

    with pytest.raises(ValueError, match="positive integer"):
        checkpoint_policy_retention(-1)


def test_ema_shadow_parameter_decay_rate() -> None:
    """Shadow parameters update as convex combination: retention * dest + (1 - retention) * src."""
    param_a = torch.nn.Parameter(torch.tensor([10.0, 20.0]))
    param_b = torch.nn.Parameter(torch.tensor([100.0]))
    params = [("a", param_a), ("b", param_b)]

    store = ModelEMA(params)
    assert store.updates == 0

    # New model values
    param_a.data.copy_(torch.tensor([20.0, 30.0]))
    param_b.data.copy_(torch.tensor([200.0]))

    # retention = 0.8
    # new_a = 0.8 * [10, 20] + 0.2 * [20, 30] = [8 + 4, 16 + 6] = [12.0, 22.0]
    # new_b = 0.8 * 100 + 0.2 * 200 = 80 + 40 = 120.0
    store.update(params, retention=0.8)
    assert store.updates == 1

    shadow = store.state_dict()["values"]
    torch.testing.assert_close(shadow["a"], torch.tensor([12.0, 22.0]))
    torch.testing.assert_close(shadow["b"], torch.tensor([120.0]))


def test_ema_parameter_copy_and_device_placement() -> None:
    """copy_to correctly restores parameters to model and respects device."""
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    params = [("weight", param)]

    store = ModelEMA(params, device="cpu")
    # Mutate model param
    param.data.fill_(999.0)

    # copy_to should restore shadow values to model param
    store.copy_to(params)
    torch.testing.assert_close(param.data, torch.tensor([1.0, 2.0, 3.0]))


def test_ema_state_dict_roundtrip() -> None:
    """state_dict and load_state_dict preserve updates and tensor contents."""
    param = torch.nn.Parameter(torch.tensor([5.0]))
    params = [("w", param)]

    store1 = ModelEMA(params)
    param.data.copy_(torch.tensor([15.0]))
    store1.update(params, retention=0.5)

    saved = store1.state_dict()
    assert saved["updates"] == 1

    param_fresh = torch.nn.Parameter(torch.tensor([0.0]))
    store2 = ModelEMA([("w", param_fresh)])
    store2.load_state_dict(saved)

    assert store2.updates == 1
    torch.testing.assert_close(store2.state_dict()["values"]["w"], torch.tensor([10.0]))


def test_ema_invalid_inputs_raise() -> None:
    """Invalid parameter names, retention values, or shape mismatches raise."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    # Duplicate names
    with pytest.raises(ValueError, match="unique"):
        ModelEMA([("x", param), ("x", param)])

    # Retention out of [0, 1)
    store = ModelEMA([("x", param)])
    with pytest.raises(ValueError, match="retention must be finite and in"):
        store.update([("x", param)], retention=1.0)
    with pytest.raises(ValueError, match="retention must be finite and in"):
        store.update([("x", param)], retention=-0.1)


# ---------------------------------------------------------------------------
# Runtime: distributed_reward_moments & require_all_ranks
# ---------------------------------------------------------------------------


def test_distributed_reward_moments_single_process() -> None:
    """Reward moments match sample population mean and population std."""
    # N=5: [1, 2, 3, 4, 5] -> sum=15, sum_sq=55, mean=3.0, var=55/5 - 9 = 2.0, std=sqrt(2)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, std = distributed_reward_moments(rewards)

    assert mean.item() == pytest.approx(3.0, abs=1e-6)
    assert std.item() == pytest.approx(math.sqrt(2.0), abs=1e-6)

    # 2D tensor matches flattened computation
    rewards_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean_2d, std_2d = distributed_reward_moments(rewards_2d)
    expected_mean = rewards_2d.float().mean().item()
    expected_std = rewards_2d.float().std(correction=0).item()
    assert mean_2d.item() == pytest.approx(expected_mean, abs=1e-6)
    assert std_2d.item() == pytest.approx(expected_std, abs=1e-6)


def test_distributed_reward_moments_rejections() -> None:
    """Non-floating, empty, or non-finite rewards raise errors."""
    with pytest.raises(TypeError, match="must be a floating-point"):
        distributed_reward_moments(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError, match="non-empty and finite"):
        distributed_reward_moments(torch.tensor([]))

    with pytest.raises(ValueError, match="non-empty and finite"):
        distributed_reward_moments(torch.tensor([1.0, float("nan"), 3.0]))


def test_require_all_ranks_single_process() -> None:
    """In single-process mode, condition=True passes and condition=False raises."""
    mock_accelerator = SimpleNamespace(num_processes=1, device="cpu")

    # condition=True passes
    require_all_ranks(mock_accelerator, True, message="should not raise")

    # condition=False raises RuntimeError by default
    with pytest.raises(RuntimeError, match="condition failed"):
        require_all_ranks(mock_accelerator, False, message="condition failed")

    # Custom exception type
    with pytest.raises(ValueError, match="custom error"):
        require_all_ranks(
            mock_accelerator,
            False,
            message="custom error",
            error_type=ValueError,
        )

    # Non-boolean condition raises TypeError
    with pytest.raises(TypeError, match="distributed gate condition must be boolean"):
        require_all_ranks(mock_accelerator, 1, message="not a bool")  # type: ignore

    # Empty message raises ValueError
    with pytest.raises(ValueError, match="distributed gate message must be non-empty"):
        require_all_ranks(mock_accelerator, True, message="")
