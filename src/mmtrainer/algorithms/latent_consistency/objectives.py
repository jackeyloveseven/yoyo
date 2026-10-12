"""LCM(Latent Consistency Model)少步蒸馏的一致性自举损失。

学生 ``f_θ`` 把任意时刻 ``t`` 的加噪样本一步映射回干净样本 ``x0``。监督来自相邻
ODE 步之间的一致性:用在线速度走一个 Euler 步到更早时刻 ``t' = t - Δt``,再要求
学生在两处给出同一个干净样本。整条目标里没有教师多步 rollout,也没有独立
fake-score 网络——区别于 TDM/DMD 系。

流程(严格按仓库 rectified-flow 约定,见 :mod:`mmtrainer.nn.flow`):

1. ``x0_online = x_t - t * v_θ(x_t, t)``(保留梯度);
2. ``x_{t'} = euler_flow_step(x_t, v_θ(x_t, t).detach(), t, t')``(自举状态,不建二阶);
3. ``x0_target = (x_{t'} - t' * v_θ⁻(x_{t'}, t')).detach()``;
4. ``L = w(t) * (sqrt(Δx0² + c²) - c)``,``Δx0 = x0_online - x0_target``,``c`` 默认
   与仓库 TDM 的伪 Huber 阈值一致(``1e-3``)。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmtrainer.algorithms.common import (
    normalize_event_dims,
    require_matching_tensors,
    working_dtype,
)
from mmtrainer.nn.flow import euler_flow_step

from .types import LCMConsistencyConfig, LCMConsistencyInputs, _batch_sigma, _sample_weight


@dataclass(frozen=True)
class LCMConsistencyMetrics:
    """一致性损失及诊断量,全部为标量张量(per_sample_loss 除外)。"""

    total_loss: Tensor
    per_sample_loss: Tensor
    raw_huber: Tensor
    consistency_rms: Tensor


def _pseudo_huber(delta: Tensor, huber_c: float) -> Tensor:
    """伪 Huber ``sqrt(delta^2 + c^2) - c``;``c == 0`` 时退化为 ``|delta|``。"""
    return torch.sqrt(delta.square() + huber_c**2) - huber_c


def lcm_consistency_loss(
    inputs: LCMConsistencyInputs,
    config: LCMConsistencyConfig | None = None,
) -> LCMConsistencyMetrics:
    """计算一次 LCM 一致性自举损失。

    契约要点:

    - ``online_velocity`` 必须带梯度;``target_velocity`` 与自举状态 ``x_{t'}`` 一律
      ``detach``,因此梯度只会经由在线项回传(一阶一致性,无二阶路径)。
    - ``sigma_prev`` 必须严格小于 ``sigma_curr``(Euler 步方向朝干净样本),两者都
      落在 ``[0, 1]``;FP16/BF16 输入统一上溢到 FP32 后再算。
    - 非有限输入或非有限结果都在返回前 ``raise ValueError``。
    """

    config = config or LCMConsistencyConfig()
    if isinstance(config.huber_c, bool) or not isinstance(config.huber_c, (int, float)):
        raise TypeError("huber_c must be a real number")
    huber_c = float(config.huber_c)
    if not torch.isfinite(torch.tensor(huber_c, dtype=torch.float64)) or huber_c < 0:
        raise ValueError("huber_c must be finite and non-negative")

    require_matching_tensors(noisy=inputs.noisy, online_velocity=inputs.online_velocity)
    require_matching_tensors(noisy=inputs.noisy, target_velocity=inputs.target_velocity)
    sigma_curr = _batch_sigma("sigma_curr", inputs.sigma_curr, inputs.noisy)
    sigma_prev = _batch_sigma("sigma_prev", inputs.sigma_prev, inputs.noisy)
    if bool((sigma_prev >= sigma_curr).any()):
        raise ValueError(
            "sigma_prev must be strictly less than sigma_curr "
            "(one Euler step toward the clean sample)"
        )

    dims = normalize_event_dims(inputs.noisy.ndim, config.event_dims)
    dtype = working_dtype(inputs.noisy, inputs.online_velocity, inputs.target_velocity)
    event_shape = (-1, *([1] * (inputs.noisy.ndim - 1)))
    time_curr = sigma_curr.detach().to(device=inputs.noisy.device, dtype=dtype).view(event_shape)
    time_prev = sigma_prev.detach().to(device=inputs.noisy.device, dtype=dtype).view(event_shape)
    noisy = inputs.noisy.to(dtype)
    online_velocity = inputs.online_velocity.to(dtype)
    with torch.no_grad():
        # 自举状态只用在线速度的 detach 值:梯度不经过 Euler 步,只有一阶路径。
        bootstrap_velocity = inputs.online_velocity.detach().to(dtype)
        previous_state = euler_flow_step(noisy, bootstrap_velocity, time_curr, time_prev)
        velocity_at_previous = inputs.target_velocity.to(dtype)
        clean_target = (previous_state - time_prev * velocity_at_previous).detach()
    clean_online = noisy - time_curr * online_velocity
    delta = clean_online - clean_target
    pointwise = _pseudo_huber(delta, huber_c)
    raw_huber = pointwise.mean(dim=dims) if dims else pointwise
    if config.snr_weight is None:
        per_sample = raw_huber
    else:
        weight = _sample_weight("snr_weight", config.snr_weight, inputs.noisy, dims)
        per_sample = raw_huber * weight
    total = per_sample.mean()
    if not torch.isfinite(total):
        raise ValueError("LCM consistency loss must be finite")

    zero = total.new_zeros(())
    if delta.numel():
        consistency_rms = delta.detach().square().mean().sqrt()
    else:
        consistency_rms = zero
    return LCMConsistencyMetrics(
        total_loss=total,
        per_sample_loss=per_sample,
        raw_huber=raw_huber,
        consistency_rms=consistency_rms,
    )


__all__ = [
    "LCMConsistencyConfig",
    "LCMConsistencyInputs",
    "LCMConsistencyMetrics",
    "lcm_consistency_loss",
]
