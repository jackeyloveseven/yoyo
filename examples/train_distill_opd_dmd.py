"""Example 3: Composable Distillation (OPD + DMD + NormGuard + REPA + LoRA-MoE).

Combines:
- LoRA-MoE Router: `LoRAExpertSpec`, `NativeLoRAMoEConfig`, `NativeLoRARouter`
- Distillation Objectives: `flow_opd_ode_loss`, `dmd_distribution_matching_loss`,
  `norm_guard_loss`, `RepaProjector`, `repa_cosine_alignment_loss`
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mmtrainer.algorithms.sft_dpo import (
    RepaProjector,
    dmd_distribution_matching_loss,
    flow_opd_ode_loss,
    norm_guard_loss,
    repa_cosine_alignment_loss,
)
from mmtrainer.nn import LoRAExpertSpec, NativeLoRAMoEConfig, NativeLoRARouter


def run_distill_demo(batch_size: int = 2) -> dict[str, Any]:
    torch.manual_seed(11)
    student_proj = nn.Linear(16, 16)
    repa_head = RepaProjector(dit_hidden_size=16, encoder_dim=24, hidden_dim=32)
    moe_config = NativeLoRAMoEConfig(
        enabled=True,
        experts=(
            LoRAExpertSpec(name="lighting", path="adapters/lighting.safetensors"),
            LoRAExpertSpec(name="detail", path="adapters/detail.safetensors"),
        ),
        context_dim=16,
        hidden_dim=32,
        num_blocks=8,
        block_group_size=4,
        top_k=1,
    )
    moe_router = NativeLoRARouter(moe_config)

    optimizer = torch.optim.AdamW(
        list(student_proj.parameters())
        + list(repa_head.parameters())
        + list(moe_router.parameters()),
        lr=1e-3,
    )

    hidden = torch.randn(batch_size, 8, 16)
    sigmas = torch.tensor([0.3, 0.7])
    routing_gates = moe_router(hidden, sigmas)
    expert_scale = routing_gates[:, 0, 0].view(batch_size, 1, 1)
    student_velocity = student_proj(hidden) * expert_scale

    latents = torch.randn(batch_size, 8, 16)
    teacher_velocity = torch.randn(batch_size, 8, 16)
    fake_x0 = torch.randn(batch_size, 8, 16)
    real_x0 = torch.randn(batch_size, 8, 16)
    vision_target_feats = torch.randn(batch_size, 8, 24)
    delta_t = torch.full((batch_size,), -0.25)

    opd_res = flow_opd_ode_loss(
        latents=latents,
        student_velocity=student_velocity,
        teacher_velocity=teacher_velocity,
        delta_t=delta_t,
    )
    dmd_res = dmd_distribution_matching_loss(
        x0_generated=opd_res.student_target,
        x0_real=real_x0,
        x0_fake=fake_x0,
    )
    ng_res = norm_guard_loss(student_velocity, teacher_velocity)
    repa_loss = repa_cosine_alignment_loss(
        repa_head(hidden),
        vision_target_feats,
    )
    aux_loss = moe_router.auxiliary_loss()

    total_loss = (
        opd_res.loss
        + 0.2 * dmd_res.loss
        + 0.1 * ng_res.penalty
        + 0.1 * repa_loss
        + aux_loss
    )
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": float(total_loss.detach().item()),
        "opd_loss": float(opd_res.loss.detach().item()),
        "dmd_loss": float(dmd_res.loss.detach().item()),
        "norm_guard_ratio": float(ng_res.norm_ratio.detach().item()),
        "repa_loss": float(repa_loss.detach().item()),
        "routing_shape": tuple(routing_gates.shape),
    }


if __name__ == "__main__":
    print(run_distill_demo())
