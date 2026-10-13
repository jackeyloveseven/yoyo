"""End-to-end unit tests for composable mmtrainer components and example scripts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from check_structure import audit_training_code
from examples.custom_model_and_infer import run_custom_model_and_infer_demo
from examples.train_distill_opd_dmd import run_distill_demo
from examples.train_dpo_preference import run_dpo_demo
from examples.train_lcm_distill import run_lcm_demo
from examples.train_on_policy_distill import run_on_policy_distill_demo
from examples.train_rl_diffusionnft import run_diffusionnft_demo
from examples.train_rl_flow_grpo import run_flow_grpo_and_opsd_demo
from examples.train_sft_edit import run_sft_demo
from mmtrainer.algorithms.sft_dpo import (
    RepaProjector,
    dmd_distribution_matching_loss,
    dmd_fake_score_loss,
    flow_dpo_loss,
    flow_opd_ode_loss,
    flow_sft_loss,
    gt_regression_loss,
    norm_guard_loss,
    repa_cosine_alignment_loss,
)
from mmtrainer.config import ComponentSpec
from mmtrainer.datasets.manifest import (
    JsonlEditDataset,
    JsonlEditPairDataset,
    JsonlT2IDataset,
)
from mmtrainer.models.firered import (
    FireRedAdapter,
    calculate_aspect_dimensions,
    interleave_candidates,
)
from mmtrainer.models.krea2 import (
    Krea2Adapter,
    SingleMMDiTConfig,
    SingleStreamDiT,
    diffusers_to_native_key,
    pad_text_features,
)
from mmtrainer.models.sensenova import HiDreamAdapter, SenseNovaAdapter
from mmtrainer.nn import (
    LoRAExpertSpec,
    NativeLoRAMoEConfig,
    NativeLoRARouter,
    block_group_for_module_name,
    build_accelerate_config,
    build_lr_scheduler,
    dpm2_flow_step,
    empirical_rollout_mu,
    euler_flow_step,
    flow_sample,
    paired_flow_samples,
    resolution_shifted_timesteps,
    shifted_flow_sigmas,
)
from mmtrainer.registry import build_builtin_registries
from mmtrainer.rewards.identity import (
    ARCFACE_REFERENCE_LANDMARKS_112,
    align_face,
    cosine_identity_loss,
)
from mmtrainer.rewards.protocols import (
    CallableRewardScorer,
    build_edit_messages,
    parse_rational_rewards_scores,
)

TRAINING_CODE_ROOT = Path(__file__).resolve().parents[2]


def test_structure_gate_passes() -> None:
    assert audit_training_code(TRAINING_CODE_ROOT) == []


def test_nn_flow_schedules_fsdp2_and_lora_moe() -> None:
    clean = torch.randn(2, 4, 8, 8)
    sample = flow_sample(clean)
    assert sample.noisy_latents.shape == clean.shape
    assert sample.target_velocity.shape == clean.shape
    assert sample.sigmas is not None and sample.sigmas.shape == (2,)

    win_s, lose_s = paired_flow_samples(clean, torch.randn_like(clean))
    assert torch.allclose(win_s.sigmas, lose_s.sigmas)

    stepped_euler = euler_flow_step(clean, sample.target_velocity, 0.8, 0.6)
    stepped_dpm2 = dpm2_flow_step(clean, sample.target_velocity, sample.target_velocity, 0.8, 0.6)
    assert torch.allclose(stepped_euler, stepped_dpm2)

    assert len(resolution_shifted_timesteps(1024, 8)) == 9
    assert empirical_rollout_mu(1024, 10) > 0
    assert shifted_flow_sigmas(torch.tensor([0.5]), 0.5, 1.15, 1024).shape == (1,)

    accel_cfg = build_accelerate_config(model_family="klein", world_size=8)
    assert accel_cfg["num_processes"] == 8

    param = torch.nn.Parameter(torch.randn(4, 4))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = build_lr_scheduler(opt, total_steps=4, warmup_steps=1, kind="wsd")
    opt.step()
    sched.step()
    assert sched.get_last_lr()[0] > 0

    moe_cfg = NativeLoRAMoEConfig(
        enabled=True,
        experts=(LoRAExpertSpec(name="exp1", path="exp1.safetensors"),),
        context_dim=8,
        hidden_dim=16,
        num_blocks=8,
        block_group_size=4,
    )
    router = NativeLoRARouter(moe_cfg)
    gates = router(torch.randn(2, 4, 8), torch.tensor([0.2, 0.8]))
    assert gates.shape == (2, 3, 2)
    assert router.auxiliary_loss().ndim == 0
    assert block_group_for_module_name("single_blocks.5.qkv", num_blocks=8, block_group_size=4) == 1


def test_sft_dpo_and_distillation_objectives() -> None:
    pred = torch.randn(2, 6, 8, requires_grad=True)
    target = torch.randn(2, 6, 8)
    assert flow_sft_loss(pred, target).ndim == 0

    dpo_out = flow_dpo_loss(
        policy_chosen=pred,
        policy_rejected=pred + 0.1,
        reference_chosen=target,
        reference_rejected=target + 0.2,
        chosen_target=target,
        rejected_target=target,
        beta=10.0,
    )
    assert dpo_out.loss.ndim == 0

    opd_out = flow_opd_ode_loss(
        latents=target,
        student_velocity=pred,
        teacher_velocity=target,
        delta_t=torch.tensor([-0.2, -0.2]),
    )
    assert opd_out.loss.ndim == 0

    ng_out = norm_guard_loss(pred * 1.5, target)
    assert ng_out.penalty.ndim == 0

    dmd_out = dmd_distribution_matching_loss(pred, target, target + 0.1)
    assert dmd_out.loss.ndim == 0
    assert dmd_fake_score_loss(pred, target, torch.randn_like(target)).ndim == 0
    assert (
        gt_regression_loss(pred, target, target + 0.1, torch.tensor([0.5, 0.5]), target).ndim == 0
    )

    proj = RepaProjector(dit_hidden_size=8, encoder_dim=12, hidden_dim=16)
    assert repa_cosine_alignment_loss(proj(pred), torch.randn(2, 6, 12)).ndim == 0


def test_model_adapters_and_registry_wiring() -> None:
    registries = build_builtin_registries()
    for entry in registries.models.entries():
        cls = registries.models.resolve(entry.plugin_id)
        plugin = cls(ComponentSpec(plugin_id=entry.plugin_id))
        adapter = plugin.build_adapter()
        assert adapter.model_id == entry.plugin_id

    for entry in registries.objectives.entries():
        cls = registries.objectives.resolve(entry.plugin_id)
        assert callable(cls.resolve_objective_fn())

    for entry in registries.rewards.entries():
        cls = registries.rewards.resolve(entry.plugin_id)
        plugin = cls(ComponentSpec(plugin_id=entry.plugin_id))
        scorer = plugin.build_scorer(default_score=0.75)
        assert scorer(["sample"]) == [0.75]

    w, h = calculate_aspect_dimensions(1024 * 1024, 16 / 9, multiple_of=32)
    assert w % 32 == 0 and h % 32 == 0
    interleaved = interleave_candidates([torch.zeros(2, 4), torch.ones(2, 4)])
    assert interleaved.shape == (4, 4)

    padded, mask = pad_text_features([torch.ones(3, 4), torch.ones(5, 4)], torch.zeros(4), pad_to=6)
    assert padded.shape == (2, 6, 4) and mask.shape == (2, 6)
    assert (
        diffusers_to_native_key("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight")
        == "single_blocks.0.linear1.weight"
    )

    krea_dit = SingleStreamDiT(
        SingleMMDiTConfig(in_channels=16, hidden_size=32, num_heads=4, depth=1, context_in_dim=12)
    )
    krea_adapter = Krea2Adapter()
    krea_batch = krea_adapter.prepare_sft_batch(
        target_latents=torch.randn(2, 4, 4, 4),
        cond_latents=torch.randn(2, 4, 4, 4),
        txt_tokens=torch.randn(2, 3, 12),
        patch_size=2,
    )
    vel = krea_adapter.predict_velocity(krea_dit, krea_batch.forward)
    assert vel.shape == krea_batch.target_velocity.shape

    fr_adapter = FireRedAdapter()
    fr_batch = fr_adapter.prepare_sft_batch(clean_latents=torch.randn(2, 4, 8))
    assert fr_batch.noisy_latents.shape == (2, 4, 8)

    sn_adapter = SenseNovaAdapter()
    sn_batch = sn_adapter.prepare_sft_batch(target_latents=torch.randn(2, 4, 8))
    assert sn_batch.noisy_latents.shape == (2, 4, 8)

    hd_adapter = HiDreamAdapter()
    hd_win, hd_lose = hd_adapter.prepare_dpo_batch(
        win_latents=torch.randn(2, 4, 8),
        lose_latents=torch.randn(2, 4, 8),
    )
    assert torch.allclose(hd_win.sigmas, hd_lose.sigmas)


def test_datasets_and_rewards(tmp_path: Path) -> None:
    images = (("in.png", (255, 0, 0)), ("out.png", (0, 255, 0)), ("lose.png", (0, 0, 255)))
    for name, color in images:
        Image.new("RGB", (64, 64), color=color).save(tmp_path / name)

    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "prompt": "make it green",
                "input_image": "in.png",
                "output_image": "out.png",
                "win_image": "out.png",
                "lose_image": "lose.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    t2i_ds = JsonlT2IDataset(jsonl, resolution=32)
    assert t2i_ds[0]["target_tensor"].shape == (3, 32, 32)

    edit_ds = JsonlEditDataset(jsonl, resolution=32)
    assert edit_ds[0]["control_tensor"].shape == (3, 32, 32)

    pair_ds = JsonlEditPairDataset(jsonl, resolution=32)
    assert pair_ds[0]["win_tensor"].shape == (3, 32, 32)
    assert pair_ds[0]["lose_tensor"].shape == (3, 32, 32)

    landmarks = torch.tensor(ARCFACE_REFERENCE_LANDMARKS_112, dtype=torch.float32)
    face_img = torch.randn(1, 3, 112, 112, requires_grad=True)
    aligned = align_face(face_img, landmarks, out_size=112)
    assert aligned.shape == (1, 3, 112, 112)
    loss = cosine_identity_loss(aligned.flatten(1), torch.randn(1, 3 * 112 * 112))
    loss.backward()
    assert face_img.grad is not None

    msgs = build_edit_messages(
        "add glasses",
        source_image_url="data:image/png;base64,AAA",
        edited_image_url="data:image/png;base64,BBB",
    )
    assert len(msgs) == 2
    scores = parse_rational_rewards_scores(
        "<think>good</think><answer>"
        '{"Instruction_Following": 5, "Consistency": 3, "Visual_Quality": 5}</answer>'
    )
    assert 0.0 <= scores["overall"] <= 1.0
    scorer = CallableRewardScorer(fn=lambda items: [1.0 for _ in items])
    assert scorer(["a", "b"]) == [1.0, 1.0]


def test_all_composable_examples_run() -> None:
    assert run_sft_demo(steps=2)["steps"] == 2
    assert run_dpo_demo(steps=2)["steps"] == 2
    assert "total_loss" in run_distill_demo()
    assert "total_loss" in run_diffusionnft_demo()
    assert "total_loss" in run_flow_grpo_and_opsd_demo()
    assert run_custom_model_and_infer_demo(steps=2)["rollout_steps"] == 2
    assert "total_loss" in run_lcm_demo()
    assert "total_loss" in run_on_policy_distill_demo()
