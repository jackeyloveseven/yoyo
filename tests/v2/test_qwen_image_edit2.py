"""Qwen-Image-Edit 2.x 多图适配器的形状、防重影与失败模式单测。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.models.qwen_image_edit2 import (  # noqa: E402
    QwenImageEdit2Adapter,
    prepare_qwen_edit2_dpo_batch,
    prepare_qwen_edit2_sft_batch,
)


class _FakeVAE:
    """与 encode_vae_image 契约一致的最小 VAE:encode 返回 latent_dist.sample。"""

    def __init__(self, z_dim: int = 16) -> None:
        self.config = SimpleNamespace(
            latents_mean=[0.0] * z_dim, latents_std=[1.0] * z_dim, z_dim=z_dim
        )

    def encode(self, pixel_values: torch.Tensor) -> Any:
        # encode_vae_image 先 unsqueeze(2),故入参为 5D (B, C, 1, H, W)。
        b, _, _, h, w = pixel_values.shape
        z_dim = self.config.z_dim
        latents = (
            pixel_values[:, :1, :, ::2, ::2].expand(b, z_dim, 1, h // 2, w // 2).contiguous() * 0.5
        )
        assert latents.dim() == 5 and latents.shape[2] == 1 and latents.shape[1] == z_dim

        class _Dist:
            @staticmethod
            def sample() -> torch.Tensor:
                return latents

        return SimpleNamespace(latent_dist=_Dist())


def _image(
    batch: int, height: int = 8, width: int = 8, generator: torch.Generator | None = None
) -> torch.Tensor:
    gen = generator or torch.Generator().manual_seed(3)
    return torch.randn(batch, 3, height, width, generator=gen)


def _pipeline_factory() -> Any:
    """最小 diffusers 契约 mock:_pack_latents 与 encode_prompt。"""

    def _pack(latents: torch.Tensor, batch_size: int, channels: int, h: int, w: int):
        tokens = (
            latents.permute(0, 2, 3, 4, 1)
            .reshape(batch_size, h // 2, 2, w // 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, (h // 2) * (w // 2), channels * 4)
        )
        return tokens

    def _encode_prompt(image: list, prompt: list[str], device: Any, **kw: Any):
        batch_size = len(prompt)
        del image, device
        embeds = torch.ones(batch_size, 7, 32)
        mask = torch.ones(batch_size, 7, dtype=torch.long)
        return embeds, mask

    return SimpleNamespace(_pack_latents=_pack, encode_prompt=_encode_prompt)


def test_sft_multi_control_shapes_and_streams() -> None:
    """2 张控制图:hidden_states 拼三流;img_shapes 每流独立、不得共用 tuple。"""
    pipeline, vae = _pipeline_factory(), _FakeVAE()
    batch = 2
    control = torch.stack([_image(batch), _image(batch)], dim=1)
    target = _image(batch)
    sft = prepare_qwen_edit2_sft_batch(
        pipeline,
        vae,
        prompts=["edit a", "edit b"],
        control_pil_images=[object(), object()],
        control_tensors=control,
        target_tensors=target,
        weight_dtype=torch.float32,
    )
    forward = sft.forward
    # 8x8 像素 → mock VAE 降 2 → 4x4 latent → 2x2 patch → 4 token/流,三流共 12。
    per_stream = (4 // 2) * (4 // 2)
    assert forward.hidden_states.shape[1] == per_stream * 3
    assert forward.hidden_states.shape[2] == 16 * 4  # 每通道 z_dim=16 × patch 4
    assert forward.target_token_count == per_stream
    # 每样本 img_shapes 是 3 个独立形状对象(防 invoke-ai#9155 的 MSRoPE 重影)。
    assert len(forward.img_shapes) == batch
    assert [len(shapes) for shapes in forward.img_shapes] == [3, 3]


def test_sft_single_control_still_works() -> None:
    """1 张控制图(退化为原版行为)也应成功,控制图上限内即合法。"""
    pipeline, vae = _pipeline_factory(), _FakeVAE()
    control = torch.stack([_image(1)], dim=1)
    sft = prepare_qwen_edit2_sft_batch(
        pipeline,
        vae,
        prompts=["edit"],
        control_pil_images=[object()],
        control_tensors=control,
        target_tensors=_image(1),
        weight_dtype=torch.float32,
    )
    assert sft.target_velocity.shape[0] == 1


def test_control_tensor_validation() -> None:
    """4 张控制图(超 2509+ 上限)与 4D 张量都在构造期报错。"""
    pipeline, vae = _pipeline_factory(), _FakeVAE()
    with pytest.raises(ValueError, match="1-3 control images"):
        prepare_qwen_edit2_sft_batch(
            pipeline,
            vae,
            prompts=["x"],
            control_pil_images=[object()],
            control_tensors=torch.stack([_image(1)] * 4, dim=1),
            target_tensors=_image(1),
            weight_dtype=torch.float32,
        )
    with pytest.raises(ValueError, match=r"\(batch, num_control"):
        prepare_qwen_edit2_sft_batch(
            pipeline,
            vae,
            prompts=["x"],
            control_pil_images=[object()],
            control_tensors=_image(1),
            target_tensors=_image(1),
            weight_dtype=torch.float32,
        )


def test_dpo_batch_couples_pair_and_streams() -> None:
    """DPO:赢/输共享同一批控制流与成对噪声,沿 batch 维拼接一次前向。"""
    pipeline, vae = _pipeline_factory(), _FakeVAE()
    batch = 2
    control = torch.stack([_image(batch)], dim=1)
    dpo = prepare_qwen_edit2_dpo_batch(
        pipeline,
        vae,
        prompts=["pref a", "pref b"],
        control_pil_images=[object()],
        control_tensors=control,
        win_tensors=_image(batch),
        lose_tensors=_image(batch),
        weight_dtype=torch.float32,
    )
    forward = dpo.forward
    # batch 维翻倍(赢+输),timestep/mask 同步翻倍。
    assert forward.hidden_states.shape[0] == 2 * batch
    assert forward.timestep.shape[0] == 2 * batch
    assert forward.encoder_hidden_states_mask.shape[0] == 2 * batch
    assert len(forward.img_shapes) == 2 * batch
    # 赢/输速度目标同形;paired_flow_samples 保证两支共享同一组 sigma。
    assert dpo.win_target_velocity.shape == dpo.lose_target_velocity.shape
    assert dpo.sigmas.shape[0] == batch


def test_adapter_surface() -> None:
    """适配器表面契约:model_id、LoRA 靶点继承原版模块名。"""
    adapter = QwenImageEdit2Adapter()
    assert adapter.model_id == "qwen_image_edit_2x"
    assert len(adapter.default_lora_target_modules) == 12
    assert "to_q" in adapter.default_lora_target_modules
