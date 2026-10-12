"""LCM 一致性蒸馏的数值不变量、detach 边界与失败模式单测。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.latent_consistency import (  # noqa: E402
    LCMConsistencyConfig,
    LCMConsistencyInputs,
    lcm_consistency_loss,
)
from mmtrainer.nn.flow import euler_flow_step  # noqa: E402


def _batch(shape: tuple[int, ...] = (3, 4, 2, 2)):
    """构造一批固定随机数,并返回(noisy, velocity, sigma_curr, sigma_prev)。"""
    generator = torch.Generator().manual_seed(11)
    noisy = torch.randn(shape, generator=generator)
    velocity = torch.randn(shape, generator=generator)
    sigma_curr = torch.tensor([0.8, 0.5, 0.25])
    sigma_prev = torch.tensor([0.6, 0.3, 0.05])
    return noisy, velocity, sigma_curr, sigma_prev


def _inputs(noisy, online, target, sigma_curr, sigma_prev) -> LCMConsistencyInputs:
    return LCMConsistencyInputs(noisy, online, target, sigma_curr, sigma_prev)


def test_identical_velocities_close_the_consistency_pair() -> None:
    """online 与 target 速度完全相同时,一致对严格闭合:残差与损失都为 0。"""
    torch.manual_seed(0)
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    result = lcm_consistency_loss(_inputs(noisy, velocity, velocity, sigma_curr, sigma_prev))

    # 同一速度实测:两步干净预测的差 = Δt ·(v_prev - v_curr) = 0,且逐样本损失形状为 (batch,)。
    assert result.per_sample_loss.shape == (3,)
    assert result.total_loss.ndim == 0
    assert result.consistency_rms.item() == pytest.approx(0.0, abs=1e-5)
    assert result.total_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert result.raw_huber.abs().max().item() == pytest.approx(0.0, abs=1e-7)


def test_exact_consistency_pair_yields_zero_loss() -> None:
    """按规格构造的一致对(同一模型在 x_t 与 x_{t'} 处查询)给出零损失。

    零速度场下 ``x0 = x_t = x_{t'}`` 恒定,LCM 的固定点在此被精确命中。
    """

    torch.manual_seed(1)
    noisy, velocity, sigma_curr, sigma_prev = _batch()

    def zero_velocity(state: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # 常量速度场 v ≡ 0:x0 在整条轨迹上恒定,一致性目标与在线预测严格相等。
        del sigma
        return torch.zeros_like(state)

    state_prev = euler_flow_step(
        noisy, zero_velocity(noisy, sigma_curr), sigma_curr, sigma_prev
    ).detach()
    online = zero_velocity(noisy, sigma_curr)
    target = zero_velocity(state_prev, sigma_prev)
    result = lcm_consistency_loss(_inputs(noisy, online, target, sigma_curr, sigma_prev))

    assert torch.equal(state_prev, noisy)
    assert result.consistency_rms.item() == pytest.approx(0.0, abs=1e-7)
    assert result.raw_huber.abs().max().item() == pytest.approx(0.0, abs=1e-7)
    assert result.total_loss.item() == pytest.approx(0.0, abs=1e-7)


def test_loss_is_monotonic_in_target_perturbation() -> None:
    """目标速度偏离越大,一致性与损失越大(伪 Huber 单调)。"""
    torch.manual_seed(2)
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    scales = (0.0, 0.02, 0.2)
    rms = []
    losses = []
    for scale in scales:
        target = velocity + scale * torch.ones_like(velocity)
        result = lcm_consistency_loss(_inputs(noisy, velocity, target, sigma_curr, sigma_prev))
        rms.append(result.consistency_rms.item())
        losses.append(result.total_loss.item())

    assert rms[0] < rms[1] < rms[2]
    assert losses[0] < losses[1] < losses[2]


def test_snr_weight_scales_per_sample_loss_only() -> None:
    """snr_weight 只加权 per_sample_loss/total_loss,不改动 raw_huber 与 RMS。"""
    torch.manual_seed(3)
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    target = velocity + 0.1 * torch.ones_like(velocity)
    weight = torch.tensor([0.5, 1.0, 2.0])
    plain = lcm_consistency_loss(_inputs(noisy, velocity, target, sigma_curr, sigma_prev))
    weighted = lcm_consistency_loss(
        _inputs(noisy, velocity, target, sigma_curr, sigma_prev),
        LCMConsistencyConfig(snr_weight=weight),
    )

    torch.testing.assert_close(weighted.raw_huber, plain.raw_huber)
    torch.testing.assert_close(weighted.consistency_rms, plain.consistency_rms)
    torch.testing.assert_close(weighted.per_sample_loss, plain.per_sample_loss * weight)
    assert not torch.allclose(weighted.total_loss, plain.total_loss)


def test_gradient_flows_only_through_online_velocity() -> None:
    """loss 对 online_velocity 有非零梯度,对 target_velocity 完全无梯度(detach 生效)。"""
    torch.manual_seed(4)
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    online = velocity.clone().requires_grad_(True)
    target = (velocity + 0.05).clone().requires_grad_(True)
    result = lcm_consistency_loss(_inputs(noisy, online, target, sigma_curr, sigma_prev))
    result.total_loss.backward()

    assert online.grad is not None and online.grad.abs().sum() > 0
    assert target.grad is None


def test_gradient_matches_analytical_stopped_bootstrap_reference() -> None:
    """梯度与"自举状态 detach、只有 x0_online 带梯度"的解析参考逐元素一致。"""
    torch.manual_seed(5)
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    online = velocity.clone().requires_grad_(True)
    target = velocity + 0.05
    result = lcm_consistency_loss(_inputs(noisy, online, target, sigma_curr, sigma_prev))
    gradient = torch.autograd.grad(result.total_loss, online)[0]

    # 参考实现:x_{t'} 与 x0_target 一律视为常数(自举项不产生梯度路径)。
    shape = (-1, *([1] * (noisy.ndim - 1)))
    clean_target = (
        euler_flow_step(noisy, velocity, sigma_curr, sigma_prev) - sigma_prev.view(shape) * target
    ).detach()
    clean_online = noisy - sigma_curr.view(shape) * online
    residual = clean_online - clean_target
    reference_loss = (torch.sqrt(residual.square() + 1e-3**2) - 1e-3).mean()
    reference_gradient = torch.autograd.grad(reference_loss, online)[0]

    torch.testing.assert_close(result.total_loss, reference_loss)
    torch.testing.assert_close(gradient, reference_gradient)


def test_sigma_ordering_violations_raise() -> None:
    """sigma_prev >= sigma_curr(方向偏离 x0)必须报错。"""
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    with pytest.raises(ValueError, match="strictly less than sigma_curr"):
        lcm_consistency_loss(_inputs(noisy, velocity, velocity, sigma_prev, sigma_curr))
    with pytest.raises(ValueError, match="strictly less than sigma_curr"):
        lcm_consistency_loss(
            _inputs(noisy, velocity, velocity, torch.full((3,), 0.4), torch.full((3,), 0.4))
        )


@pytest.mark.parametrize(
    ("sigma_curr", "sigma_prev"),
    [
        (torch.tensor([1.5, 0.5, 0.25]), torch.tensor([0.6, 0.3, 0.05])),
        (torch.tensor([0.8, 0.5, 0.25]), torch.tensor([0.6, 0.3, -0.05])),
    ],
)
def test_sigma_out_of_range_raises(sigma_curr, sigma_prev) -> None:
    """sigma 越出 [0, 1] 必须报错。"""
    noisy, velocity, _, _ = _batch()
    with pytest.raises(ValueError, match="inside \\[0, 1\\]"):
        lcm_consistency_loss(_inputs(noisy, velocity, velocity, sigma_curr, sigma_prev))


def test_shape_and_device_mismatches_give_readable_errors() -> None:
    """形状/维度不匹配给出可读错误,而不是底层广播报错。"""
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    with pytest.raises(ValueError, match="target_velocity must match noisy"):
        lcm_consistency_loss(_inputs(noisy, velocity, velocity[:2], sigma_curr, sigma_prev))
    with pytest.raises(ValueError, match="sigma_curr must have shape \\(3,\\)"):
        lcm_consistency_loss(_inputs(noisy, velocity, velocity, sigma_curr[:2], sigma_prev))
    with pytest.raises(ValueError, match="snr_weight must have leading shape"):
        lcm_consistency_loss(
            _inputs(noisy, velocity, velocity, sigma_curr, sigma_prev),
            LCMConsistencyConfig(snr_weight=torch.ones(3, 4, 2, 2)),
        )
    with pytest.raises(ValueError, match="snr_weight"):
        lcm_consistency_loss(
            _inputs(noisy, velocity, velocity, sigma_curr, sigma_prev),
            LCMConsistencyConfig(snr_weight=torch.tensor([0.5, -1.0, 1.0])),
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_promotes_to_fp32_and_stays_finite(dtype) -> None:
    """FP16/BF16 输入上溢 FP32 再算,损失有限且与 FP32 参考接近。"""
    torch.manual_seed(6)
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    half = _inputs(
        noisy.to(dtype),
        velocity.to(dtype),
        (velocity + 0.1).to(dtype),
        sigma_curr.to(dtype),
        sigma_prev.to(dtype),
    )
    result = lcm_consistency_loss(half)
    assert torch.isfinite(result.total_loss)
    assert result.total_loss.dtype == torch.float32

    reference = lcm_consistency_loss(
        _inputs(noisy, velocity, velocity + 0.1, sigma_curr, sigma_prev)
    )
    assert result.total_loss.item() == pytest.approx(reference.total_loss.item(), rel=1e-2)


@pytest.mark.parametrize(
    "field",
    ["noisy", "online_velocity", "target_velocity", "sigma_curr", "sigma_prev"],
)
def test_non_finite_inputs_raise_before_returning(field: str) -> None:
    """任一输入含 nan/inf 都在返回前 raise。"""
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    payload = {
        "noisy": noisy,
        "online_velocity": velocity,
        "target_velocity": velocity,
        "sigma_curr": sigma_curr,
        "sigma_prev": sigma_prev,
    }
    payload[field] = payload[field].clone()
    payload[field].view(-1)[0] = float("nan")
    inputs = LCMConsistencyInputs(**payload)
    with pytest.raises(ValueError, match="finite"):
        lcm_consistency_loss(inputs)


def test_invalid_huber_c_is_rejected() -> None:
    """huber_c 非法值(负数/非有限)必须报错。"""
    noisy, velocity, sigma_curr, sigma_prev = _batch()
    inputs = _inputs(noisy, velocity, velocity, sigma_curr, sigma_prev)
    with pytest.raises(ValueError, match="huber_c must be finite and non-negative"):
        lcm_consistency_loss(inputs, LCMConsistencyConfig(huber_c=-1e-3))
    with pytest.raises(ValueError, match="huber_c must be finite and non-negative"):
        lcm_consistency_loss(inputs, LCMConsistencyConfig(huber_c=float("nan")))
    with pytest.raises(TypeError, match="huber_c must be a real number"):
        lcm_consistency_loss(inputs, LCMConsistencyConfig(huber_c=True))


def test_registered_objective_resolves_to_the_loss_function() -> None:
    """注册表条目指向真实可调用的零件函数(与 catalog 声明一致)。"""
    from mmtrainer.algorithms.catalog import LatentConsistencyPlugin
    from mmtrainer.registry import build_builtin_registries

    entry = build_builtin_registries().objectives.entry("latent_consistency")
    assert entry.metadata["required_model_capabilities"] == frozenset({"image_training"})
    assert entry.metadata["reward_required"] is False
    assert LatentConsistencyPlugin.resolve_objective_fn() is lcm_consistency_loss
