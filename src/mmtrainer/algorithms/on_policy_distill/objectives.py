"""On-Policy Distillation(在线策略蒸馏)少步蒸馏的教师轨迹对齐损失。

与 LCM 的相邻步自举一致性不同,本方法的监督来自**教师多步轨迹的终点**:学生
从 ``x_t`` 一步预测干净样本,教师从同一个 ``x_t`` 走若干 Euler 步到干净样本,
两者做逐样本 MSE。教师轨迹由调用方以 no_grad 构造并 detach 后传入,因此梯度
只沿学生一条路径回传。

流程(严格按仓库 rectified-flow 约定,见 :mod:`mmtrainer.nn.flow`):

1. ``x0_student = x_t - sigma * v_θ(x_t, t)``(保留梯度,rectified_flow_clean_output);
2. ``x0_teacher`` 由调用方多步 Euler 构造并 ``detach`` 传入(零件再 detach 一次);
3. ``L = loss_weight * w(t) * mean((x0_student - x0_teacher)^2)``。

参考 DiffSynth-Studio 的 OPD 实现(数学对照,不抄其框架结构):
``examples/qwen_image_21/model_training/special/opd/train_opd_lora.py``。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import (
    normalize_event_dims,
    rectified_flow_clean_output,
    require_matching_tensors,
    working_dtype,
)
from mmtrainer.algorithms.latent_consistency.types import _batch_sigma

from .types import (
    OnPolicyDistillConfig,
    OnPolicyDistillInputs,
    _optional_sample_weight,
)


@dataclass(frozen=True)
class OnPolicyDistillMetrics:
    """在线策略蒸馏损失及诊断量,除 ``per_sample_loss`` 外全部为标量张量。"""

    total_loss: Tensor
    per_sample_loss: Tensor
    mse_loss: Tensor
    x0_rms: Tensor


def on_policy_distill_loss(
    inputs: OnPolicyDistillInputs,
    config: OnPolicyDistillConfig | None = None,
) -> OnPolicyDistillMetrics:
    """计算一次在线策略蒸馏的教师轨迹对齐损失。

    契约要点:

    - ``student_velocity`` 必须带梯度;``teacher_x0`` 一律 ``detach``,梯度只沿
      学生项回传。
    - ``sigma`` 为 ``(batch,)`` 且落在 ``[0, 1]``;FP16/BF16 输入统一上溢到 FP32。
    - 非有限输入或非有限结果都在返回前 ``raise ValueError``。
    """

    config = config or OnPolicyDistillConfig()
    require_matching_tensors(x_t=inputs.x_t, student_velocity=inputs.student_velocity)
    require_matching_tensors(x_t=inputs.x_t, teacher_x0=inputs.teacher_x0)
    sigma = _batch_sigma("sigma", inputs.sigma, inputs.x_t)

    dims = normalize_event_dims(inputs.x_t.ndim, config.event_dims)
    dtype = working_dtype(inputs.x_t, inputs.student_velocity, inputs.teacher_x0)
    x_t = inputs.x_t.to(dtype)
    student_velocity = inputs.student_velocity.to(dtype)
    clean_student = rectified_flow_clean_output(x_t, sigma.to(dtype), student_velocity)
    clean_teacher = inputs.teacher_x0.detach().to(dtype)

    delta = clean_student - clean_teacher
    pointwise = delta.square()
    per_sample_mse = pointwise.mean(dim=dims) if dims else pointwise
    weight = _optional_sample_weight("sample_weight", inputs.sample_weight, inputs.x_t, dims)
    per_sample = per_sample_mse * weight
    total = float(config.loss_weight) * per_sample.mean()
    if not torch.isfinite(total):
        raise ValueError("on-policy distillation loss must be finite")

    mse = per_sample_mse.mean().detach()
    x0_rms = delta.detach().square().mean().sqrt() if delta.numel() else total.new_zeros(())
    return OnPolicyDistillMetrics(
        total_loss=total,
        per_sample_loss=per_sample,
        mse_loss=mse,
        x0_rms=x0_rms,
    )


__all__ = [
    "OnPolicyDistillConfig",
    "OnPolicyDistillInputs",
    "OnPolicyDistillMetrics",
    "on_policy_distill_loss",
]
