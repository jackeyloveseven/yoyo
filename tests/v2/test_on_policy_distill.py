"""On-Policy Distillation 教师轨迹对齐蒸馏的数值不变量、detach 边界与失败模式单测。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.on_policy_distill import (  # noqa: E402
    OnPolicyDistillConfig,
    OnPolicyDistillInputs,
    on_policy_distill_loss,
)


def _batch(shape: tuple[int, ...] = (3, 4, 2, 2)):
    """构造一批固定随机数,返回(x_t, student_velocity, teacher_x0, sigma)。"""
    generator = torch.Generator().manual_seed(7)
    x_t = torch.randn(shape, generator=generator)
    student_velocity = torch.randn(shape, generator=generator)
    teacher_x0 = torch.randn(shape, generator=generator)
    sigma = torch.tensor([0.8, 0.5, 0.25])
    return x_t, student_velocity, teacher_x0, sigma


def _inputs(x_t, velocity, teacher, sigma, weight=None) -> OnPolicyDistillInputs:
    return OnPolicyDistillInputs(
        student_velocity=velocity,
        teacher_x0=teacher,
        x_t=x_t,
        sigma=sigma,
        sample_weight=weight,
    )


def test_loss_zero_when_student_matches_teacher() -> None:
    """学生一步重建的 x0 与教师完全相同时,损失为 0。"""
    torch.manual_seed(0)
    x_t, _, teacher_x0, sigma = _batch()
    # 反解出能精确重建 teacher_x0 的速度:v = (x_t - x0_teacher) / sigma。
    event_shape = (-1, *([1] * (x_t.ndim - 1)))
    velocity = (x_t - teacher_x0) / sigma.view(event_shape)
    result = on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, sigma))

    assert result.per_sample_loss.shape == (3,)
    assert result.total_loss.ndim == 0
    assert result.total_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert result.mse_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert result.x0_rms.item() == pytest.approx(0.0, abs=1e-5)


def test_loss_positive_when_student_differs() -> None:
    """学生与教师不一致时损失严格为正,且等于逐样本 MSE 均值。"""
    torch.manual_seed(1)
    x_t, velocity, teacher_x0, sigma = _batch()
    result = on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, sigma))

    event_shape = (-1, *([1] * (x_t.ndim - 1)))
    clean_student = x_t - sigma.view(event_shape) * velocity
    expected = (clean_student - teacher_x0).square().mean(dim=(1, 2, 3)).mean()
    assert result.total_loss.item() > 0
    assert result.total_loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_gradient_flows_to_student_not_teacher() -> None:
    """梯度只回传到 student_velocity,teacher_x0 即使带梯度也被截断。"""
    torch.manual_seed(2)
    x_t, velocity, teacher_x0, sigma = _batch()
    velocity = velocity.clone().requires_grad_(True)
    teacher = teacher_x0.clone().requires_grad_(True)  # 调用方漏 detach 的情况
    result = on_policy_distill_loss(_inputs(x_t, velocity, teacher, sigma))
    result.total_loss.backward()

    assert velocity.grad is not None
    assert velocity.grad.abs().sum().item() > 0
    assert teacher.grad is None


def test_config_validation() -> None:
    """非法配置在构造期就报错。"""
    with pytest.raises(ValueError, match="at least 1"):
        OnPolicyDistillConfig(num_distill_steps=0)
    with pytest.raises(ValueError, match="non-negative"):
        OnPolicyDistillConfig(min_distill_idx=-1)
    with pytest.raises(ValueError, match="positive"):
        OnPolicyDistillConfig(loss_weight=-1.0)


def test_shape_mismatch_raises() -> None:
    """形状或设备不匹配给出可读错误。"""
    torch.manual_seed(3)
    x_t, velocity, teacher_x0, sigma = _batch()
    with pytest.raises(ValueError, match="teacher_x0 must match x_t"):
        on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0[:, :, :1, :], sigma))
    with pytest.raises(ValueError, match=r"sigma must have shape \(3,\)"):
        on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, sigma[:2]))
    with pytest.raises(ValueError, match=r"sample_weight must have leading shape"):
        on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, sigma, torch.ones(2)))


def test_sample_weight_scales_loss() -> None:
    """逐样本权重线性放大对应样本的损失贡献。"""
    torch.manual_seed(4)
    x_t, velocity, teacher_x0, sigma = _batch()
    plain = on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, sigma))
    doubled = on_policy_distill_loss(
        _inputs(x_t, velocity, teacher_x0, sigma, weight=2.0 * torch.ones(3))
    )
    assert doubled.total_loss.item() == pytest.approx(2.0 * plain.total_loss.item(), rel=1e-5)


def test_loss_weight_scales_total() -> None:
    """config.loss_weight 只缩放 total_loss,不改 per_sample 的未加权 MSE 诊断。"""
    torch.manual_seed(5)
    x_t, velocity, teacher_x0, sigma = _batch()
    base = on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, sigma))
    scaled = on_policy_distill_loss(
        _inputs(x_t, velocity, teacher_x0, sigma), OnPolicyDistillConfig(loss_weight=3.0)
    )
    assert scaled.total_loss.item() == pytest.approx(3.0 * base.total_loss.item(), rel=1e-5)
    assert scaled.mse_loss.item() == pytest.approx(base.mse_loss.item(), rel=1e-6)


def test_non_finite_inputs_raise() -> None:
    """nan/inf 输入在优化前就报错,不会带进优化器。"""
    torch.manual_seed(6)
    x_t, velocity, teacher_x0, sigma = _batch()
    bad_teacher = teacher_x0.clone()
    bad_teacher[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="teacher_x0"):
        on_policy_distill_loss(_inputs(x_t, velocity, bad_teacher, sigma))
    bad_sigma = sigma.clone()
    bad_sigma[1] = float("inf")
    with pytest.raises(ValueError, match="sigma"):
        on_policy_distill_loss(_inputs(x_t, velocity, teacher_x0, bad_sigma))
