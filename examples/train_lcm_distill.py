"""Example 7: Latent Consistency Model (LCM) few-step distillation.

组合:
- 一致性蒸馏目标: `lcm_consistency_loss`, `LCMConsistencyInputs`, `LCMConsistencyConfig`
- 流匹配原语: `flow_sample`, `euler_flow_step`
- 学生在线网络 + EMA 目标网络,演示"学生一步映射 vs 相邻步自举一致性"。
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from mmtrainer.algorithms.latent_consistency import (
    LCMConsistencyConfig,
    LCMConsistencyInputs,
    lcm_consistency_loss,
)
from mmtrainer.nn import euler_flow_step, flow_sample


def _ema_update(online: nn.Module, target: nn.Module, decay: float) -> None:
    """用衰减 ``decay`` 把在线参数指数滑动平均进目标网络(调用方职责,非零件)。"""
    with torch.no_grad():
        for t_param, o_param in zip(target.parameters(), online.parameters(), strict=True):
            t_param.mul_(decay).add_(o_param.detach(), alpha=1.0 - decay)


def run_lcm_demo(batch_size: int = 4) -> dict[str, Any]:
    torch.manual_seed(21)
    # 学生在线网络与其 EMA 目标网络(EMA 是 LCM 的目标网络,不回传梯度)。
    student = nn.Linear(16, 16)
    target_net = copy.deepcopy(student).eval()
    for param in target_net.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    # 造一批流匹配加噪样本 x_t = (1-t) x0 + t noise,t ∈ (0,1)。
    clean = torch.randn(batch_size, 8, 16)
    sigma_curr = torch.rand(batch_size) * 0.5 + 0.4  # t ∈ [0.4, 0.9]
    delta_t = torch.full((batch_size,), 0.1)
    sigma_prev = sigma_curr - delta_t  # t' = t - Δt,朝干净样本方向退一步
    sample = flow_sample(clean, time=sigma_curr)

    # 在线学生:当前样本的一步干净预测(带梯度)。
    online_velocity = student(sample.noisy_latents)

    # 目标网络:先用在线速度的 detach 值把 x_t 自举到 x_{t'},再在 x_{t'} 上查 EMA。
    with torch.no_grad():
        prev_state = euler_flow_step(
            sample.noisy_latents,
            online_velocity.detach(),
            sigma_curr,
            sigma_prev,
        )
        target_velocity = target_net(prev_state)

    metrics = lcm_consistency_loss(
        LCMConsistencyInputs(
            noisy=sample.noisy_latents,
            online_velocity=online_velocity,
            target_velocity=target_velocity,
            sigma_curr=sample.sigmas,
            sigma_prev=sigma_prev,
        ),
        LCMConsistencyConfig(huber_c=1e-3),
    )

    optimizer.zero_grad()
    metrics.total_loss.backward()
    optimizer.step()
    _ema_update(student, target_net, decay=0.999)

    # 学生一步映射的干净样本 vs 自举一致性目标的差距(consistency_rms 应随训练下降)。
    clean_online = sample.noisy_latents - sample.sigmas.view(-1, 1, 1) * online_velocity
    return {
        "total_loss": float(metrics.total_loss.detach().item()),
        "raw_huber_mean": float(metrics.raw_huber.mean().detach().item()),
        "consistency_rms": float(metrics.consistency_rms.detach().item()),
        "clean_online_shape": tuple(clean_online.shape),
        "sigma_curr_range": (float(sigma_curr.min()), float(sigma_curr.max())),
    }


if __name__ == "__main__":
    print(run_lcm_demo())
