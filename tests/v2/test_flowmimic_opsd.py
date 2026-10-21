"""Numerical invariant and boundary tests for FlowMimic-OPSD primitives."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.flowmimic_opsd.config import (  # noqa: E402
    FlowMimicOPSDConfig,
    flowmimic_opsd_defaults,
)
from mmtrainer.algorithms.flowmimic_opsd.objectives import (  # noqa: E402
    FlowMimicOPSDInputs,
    FlowMimicOPSDLoss,
    FlowMimicOPSDLossConfig,
    flow_clean_output,
    flowmimic_clean_distribution_loss,
)


def _mimic_inputs(
    *,
    shape: tuple[int, ...] = (2, 3, 4, 4),
    seed: int = 123,
    base_loss: torch.Tensor | None = None,
    favorable_same_as_naive: bool = False,
) -> FlowMimicOPSDInputs:
    generator = torch.Generator().manual_seed(seed)
    query_state = torch.randn(shape, generator=generator)
    query_sigma = torch.full(shape[:2], 0.25)
    naive_velocity = torch.randn(shape, generator=generator)
    if favorable_same_as_naive:
        favorable_velocity = naive_velocity.clone()
    else:
        favorable_velocity = torch.randn(shape, generator=generator)
    if base_loss is None:
        base_loss = torch.tensor(0.42)
    return FlowMimicOPSDInputs(
        query_state=query_state,
        query_sigma=query_sigma,
        naive_velocity=naive_velocity,
        favorable_velocity=favorable_velocity,
        base_loss=base_loss,
    )


def test_flow_clean_output_analytic() -> None:
    """Rectified-flow clean estimate: x0_hat = xt - sigma * velocity."""
    state = torch.tensor([5.0, -2.0, 0.0])
    sigma = torch.tensor(0.5)
    velocity = torch.tensor([2.0, -4.0, 1.0])
    # 5.0 - 0.5*2 = 4.0; -2.0 - 0.5*(-4) = 0.0; 0.0 - 0.5*1 = -0.5
    expected = torch.tensor([4.0, 0.0, -0.5])
    reconstructed = flow_clean_output(state, sigma, velocity)
    torch.testing.assert_close(reconstructed, expected)


def test_flowmimic_zero_kl_when_distributions_identical() -> None:
    """When naive and favorable velocities are identical, KL divergence is zero."""
    inputs = _mimic_inputs(favorable_same_as_naive=True, base_loss=torch.tensor(0.5))
    loss: FlowMimicOPSDLoss = flowmimic_clean_distribution_loss(
        inputs, FlowMimicOPSDLossConfig(base_loss_weight=1.0, mimic_weight=1.0)
    )

    # Clean outputs and log probabilities must match
    torch.testing.assert_close(loss.naive_clean, loss.favorable_clean)
    torch.testing.assert_close(loss.naive_log_probabilities, loss.favorable_log_probabilities)

    # KL divergences must be zero
    assert loss.mimic_loss.item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(loss.per_sample_kl, torch.zeros_like(loss.per_sample_kl))
    torch.testing.assert_close(
        loss.per_sample_reverse_kl, torch.zeros_like(loss.per_sample_reverse_kl)
    )

    # Total loss equals base loss
    assert loss.total_loss.item() == pytest.approx(0.5, abs=1e-6)


def test_flowmimic_positive_kl_when_distributions_differ() -> None:
    """When velocities differ, asymmetric KL is strictly positive."""
    inputs = _mimic_inputs(favorable_same_as_naive=False)
    loss = flowmimic_clean_distribution_loss(inputs)

    assert loss.mimic_loss.item() > 0.0
    assert (loss.per_sample_kl > 0.0).all()
    assert (loss.per_sample_reverse_kl > 0.0).all()


def test_flowmimic_live_gradients_through_both_conditioning_paths() -> None:
    """Both naive and favorable paths receive live gradients (mutual capability alignment)."""
    shape = (2, 2, 4)
    query_state = torch.randn(shape)
    query_sigma = torch.full((2, 2), 0.25)
    naive_vel = torch.randn(shape, requires_grad=True)
    fav_vel = torch.randn(shape, requires_grad=True)

    inputs = FlowMimicOPSDInputs(
        query_state=query_state,
        query_sigma=query_sigma,
        naive_velocity=naive_vel,
        favorable_velocity=fav_vel,
        base_loss=torch.tensor(0.1),
    )
    result = flowmimic_clean_distribution_loss(inputs)
    result.total_loss.backward()

    # Both paths must have non-zero gradients
    assert naive_vel.grad is not None
    assert fav_vel.grad is not None
    assert naive_vel.grad.abs().sum().item() > 0.0
    assert fav_vel.grad.abs().sum().item() > 0.0


def test_flowmimic_temperature_scaling() -> None:
    """Higher temperature flattens distributions and reduces KL divergence."""
    inputs = _mimic_inputs(seed=456)
    loss_cool = flowmimic_clean_distribution_loss(inputs, FlowMimicOPSDLossConfig(temperature=0.5))
    loss_warm = flowmimic_clean_distribution_loss(inputs, FlowMimicOPSDLossConfig(temperature=2.0))
    loss_hot = flowmimic_clean_distribution_loss(inputs, FlowMimicOPSDLossConfig(temperature=10.0))

    # Higher temperature smooths the softmax logits, reducing KL
    assert loss_hot.mimic_loss.item() < loss_warm.mimic_loss.item() < loss_cool.mimic_loss.item()


def test_flowmimic_base_and_mimic_weight_combination() -> None:
    """total_loss is exact linear combination: base_loss_weight * base + mimic_weight * mimic."""
    base_val = 1.5
    inputs = _mimic_inputs(base_loss=torch.tensor(base_val))

    for b_weight, m_weight in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (0.1, 3.0)]:
        cfg = FlowMimicOPSDLossConfig(base_loss_weight=b_weight, mimic_weight=m_weight)
        loss = flowmimic_clean_distribution_loss(inputs, cfg)

        expected_total = b_weight * base_val + m_weight * loss.mimic_loss.item()
        assert loss.total_loss.item() == pytest.approx(expected_total, abs=1e-5)


def test_flowmimic_require_base_loss_toggle() -> None:
    """When base_loss is None, require_base_loss controls whether error is raised."""
    shape = (2, 2, 4)
    inputs_none = FlowMimicOPSDInputs(
        query_state=torch.randn(shape),
        query_sigma=torch.full((2, 2), 0.2),
        naive_velocity=torch.randn(shape),
        favorable_velocity=torch.randn(shape),
        base_loss=None,
    )

    # Default require_base_loss=True raises ValueError
    with pytest.raises(ValueError, match="FlowMimic main training requires the native"):
        flowmimic_clean_distribution_loss(
            inputs_none, FlowMimicOPSDLossConfig(require_base_loss=True)
        )

    # require_base_loss=False defaults base_loss to 0.0
    loss = flowmimic_clean_distribution_loss(
        inputs_none, FlowMimicOPSDLossConfig(require_base_loss=False, mimic_weight=1.5)
    )
    assert loss.base_loss.item() == 0.0
    assert loss.total_loss.item() == pytest.approx(1.5 * loss.mimic_loss.item(), abs=1e-6)


def test_flowmimic_non_finite_rejection() -> None:
    """NaN or Inf in inputs triggers ValueError."""
    shape = (2, 2, 4)
    base_in = _mimic_inputs(shape=shape)

    for field in ("query_state", "naive_velocity", "favorable_velocity"):
        bad_tensor = getattr(base_in, field).clone()
        bad_tensor[0, 0, 0] = float("nan")
        kwargs = {
            "query_state": base_in.query_state,
            "query_sigma": base_in.query_sigma,
            "naive_velocity": base_in.naive_velocity,
            "favorable_velocity": base_in.favorable_velocity,
            "base_loss": base_in.base_loss,
        }
        kwargs[field] = bad_tensor
        with pytest.raises(ValueError, match="must contain only finite values"):
            flowmimic_clean_distribution_loss(FlowMimicOPSDInputs(**kwargs))

    # Base loss with Inf
    with pytest.raises(ValueError, match="must contain only finite values"):
        flowmimic_clean_distribution_loss(
            FlowMimicOPSDInputs(
                query_state=base_in.query_state,
                query_sigma=base_in.query_sigma,
                naive_velocity=base_in.naive_velocity,
                favorable_velocity=base_in.favorable_velocity,
                base_loss=torch.tensor(float("inf")),
            )
        )


def test_flowmimic_config_validation() -> None:
    """Invalid temperature, weights, or config types raise errors."""
    inputs = _mimic_inputs()

    for invalid_temp in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            flowmimic_clean_distribution_loss(
                inputs, FlowMimicOPSDLossConfig(temperature=invalid_temp)
            )

    for invalid_weight in (0.0, -0.5, float("inf")):
        with pytest.raises(ValueError):
            flowmimic_clean_distribution_loss(
                inputs, FlowMimicOPSDLossConfig(mimic_weight=invalid_weight)
            )
        with pytest.raises(ValueError):
            flowmimic_clean_distribution_loss(
                inputs, FlowMimicOPSDLossConfig(base_loss_weight=invalid_weight)
            )


def test_flowmimic_profile_defaults() -> None:
    """FlowMimicOPSDConfig and flowmimic_opsd_defaults instantiate with valid profiles."""
    for profile in ("kio", "sirius", "sensenova"):
        cfg = flowmimic_opsd_defaults(profile)
        assert isinstance(cfg, FlowMimicOPSDConfig)
        assert cfg.query_sigma == 0.273
        assert cfg.temperature == 1.0

    with pytest.raises(ValueError, match="unsupported FlowMimic-OPSD profile"):
        flowmimic_opsd_defaults("invalid_profile")  # type: ignore
