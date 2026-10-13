"""Example 8: On-Policy Distillation(在线策略蒸馏)少步蒸馏。

组合:
- 教师轨迹对齐目标: `on_policy_distill_loss`, `OnPolicyDistillInputs`, `OnPolicyDistillConfig`
- 流匹配原语: `flow_sample`, `euler_flow_step`
- 学生一步预测 vs 教师多步 Euler 终点,演示"轨迹对齐"监督。

与 LCM(示例 7)的区别:LCM 用相邻步自举一致性、无需教师;本方法用教师多步
轨迹终点做监督,信号更稳但每个样本要多跑几步教师前向。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mmtrainer.algorithms.on_policy_distill import (
    OnPolicyDistillConfig,
    OnPolicyDistillInputs,
    on_policy_distill_loss,
)
from mmtrainer.nn import euler_flow_step, flow_sample


def run_on_policy_distill_demo(
    batch_size: int = 4,
    num_distill_steps: int = 4,
) -> dict[str, Any]:
    torch.manual_seed(31)
    # 学生在线网络与教师网络(教师不回传梯度;生产中教师通常是更强的冻结模型)。
    student = nn.Linear(16, 16)
    teacher = nn.Linear(16, 16)
    for param in teacher.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    # 造一批流匹配加噪样本 x_t = (1-t) x0 + t noise,t ∈ (0,1)。
    clean = torch.randn(batch_size, 8, 16)
    sigma = torch.rand(batch_size) * 0.6 + 0.2  # t ∈ [0.2, 0.8]
    sample = flow_sample(clean, time=sigma)

    # 学生:当前样本的一步干净预测(带梯度)。
    student_velocity = student(sample.noisy_latents)

    # 教师:从同一个 x_t 出发走 num_distill_steps 个 Euler 步到 t=0(全程 no_grad)。
    with torch.no_grad():
        state = sample.noisy_latents
        edges = torch.linspace(0.0, 1.0, num_distill_steps + 1)
        for index in range(num_distill_steps):
            t_curr = sigma * (1.0 - edges[index])
            t_next = sigma * (1.0 - edges[index + 1])
            state = euler_flow_step(state, teacher(state), t_curr, t_next)
        teacher_x0 = state  # 终点即教师的干净样本估计

    metrics = on_policy_distill_loss(
        OnPolicyDistillInputs(
            student_velocity=student_velocity,
            teacher_x0=teacher_x0,
            x_t=sample.noisy_latents,
            sigma=sample.sigmas,
        ),
        OnPolicyDistillConfig(num_distill_steps=num_distill_steps),
    )

    optimizer.zero_grad()
    metrics.total_loss.backward()
    optimizer.step()

    # 学生一步重建 vs 教师多步终点的差距(x0_rms 应随训练下降)。
    clean_student = sample.noisy_latents - sample.sigmas.view(-1, 1, 1) * student_velocity
    return {
        "total_loss": float(metrics.total_loss.detach().item()),
        "mse_loss": float(metrics.mse_loss.detach().item()),
        "x0_rms": float(metrics.x0_rms.detach().item()),
        "clean_student_shape": tuple(clean_student.shape),
        "num_distill_steps": num_distill_steps,
    }


if __name__ == "__main__":
    print(run_on_policy_distill_demo())
