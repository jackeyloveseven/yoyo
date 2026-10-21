"""Unit tests for Qwen-Image model adapter conditioning, packing, and velocity prediction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from mmtrainer.models.qwen_image import (
    QWEN_LORA_TARGET_MODULES,
    QwenForwardBatch,
    QwenImageAdapter,
    encode_qwen_prompt,
    encode_vae_image,
    pack_qwen_latents,
    predict_qwen,
    prepare_qwen_dpo_batch,
    prepare_qwen_sft_batch,
    split_pair_prediction,
)


class MockQwenVAE:
    """Mock VAE adhering to Qwen-Image 3D latent encoding contract."""

    def __init__(self, z_dim: int = 16) -> None:
        self.config = SimpleNamespace(
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
            z_dim=z_dim,
        )

    def encode(self, pixel_values: torch.Tensor) -> Any:
        batch_size, _, _, height, width = pixel_values.shape
        latent_tensor = torch.ones(
            batch_size,
            self.config.z_dim,
            1,
            max(1, height // 2),
            max(1, width // 2),
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )

        class _LatentDist:
            @staticmethod
            def sample() -> torch.Tensor:
                return latent_tensor

        return SimpleNamespace(latent_dist=_LatentDist())


class MockQwenPipeline:
    """Mock Pipeline adhering to Qwen 3D latent packing and prompt encoding contract."""

    def __init__(self, text_dim: int = 32) -> None:
        self.text_dim = text_dim

    def _pack_latents(
        self,
        latents: torch.Tensor,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        tokens = (
            latents.permute(0, 2, 3, 4, 1)
            .reshape(batch_size, height // 2, 2, width // 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        )
        return tokens

    def encode_prompt(
        self,
        image: list[Any],
        prompt: list[str],
        device: torch.device,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image, kwargs
        batch_size = len(prompt)
        seq_len = 8
        embeds = torch.ones(batch_size, seq_len, self.text_dim, device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        # Set partial mask to test txt_seq_lens
        mask[:, -2:] = 0
        return embeds, mask


class MockQwenTransformer(nn.Module):
    """Mock Qwen transformer predicting velocities and propagating gradients."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Any = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        img_shapes: list[list[tuple[int, int, int]]] | None = None,
        txt_seq_lens: list[int] | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        return (self.linear(hidden_states),)


def test_encode_vae_image_shapes_and_standardization() -> None:
    vae = MockQwenVAE(z_dim=16)
    images = torch.randn(2, 3, 8, 8)

    latents_fp32 = encode_vae_image(vae, images, torch.float32)
    assert latents_fp32.shape == (2, 16, 1, 4, 4)
    assert latents_fp32.dtype == torch.float32

    latents_bf16 = encode_vae_image(vae, images, torch.bfloat16)
    assert latents_bf16.shape == (2, 16, 1, 4, 4)
    assert latents_bf16.dtype == torch.bfloat16

    # Verify standardization with non-zero mean and non-unit std
    vae.config.latents_mean = [0.25] * 16
    vae.config.latents_std = [0.5] * 16
    scaled = encode_vae_image(vae, images, torch.float32)
    expected_val = (1.0 - 0.25) / 0.5  # 1.5
    assert torch.allclose(scaled, torch.full_like(scaled, expected_val))


def test_encode_qwen_prompt_shapes() -> None:
    pipeline = MockQwenPipeline(text_dim=32)
    prompts = ["add a hat", "remove background"]
    embeds, mask = encode_qwen_prompt(pipeline, [object(), object()], prompts, torch.device("cpu"))
    assert embeds.shape == (2, 8, 32)
    assert mask.shape == (2, 8)
    assert mask.sum().item() == 12  # 6 per sample with 2 masked out


def test_pack_qwen_latents_patch_flattening() -> None:
    pipeline = MockQwenPipeline()
    latents = torch.randn(2, 16, 1, 4, 4)
    packed = pack_qwen_latents(pipeline, latents)
    # 4x4 spatial -> 2x2 patches of 2x2 -> 4 tokens, channels * 4 = 64
    assert packed.shape == (2, 4, 64)


def test_prepare_qwen_sft_batch_two_stream_construction() -> None:
    pipeline = MockQwenPipeline(text_dim=32)
    vae = MockQwenVAE(z_dim=16)
    prompts = ["edit cat", "edit dog"]
    control_tensors = torch.randn(2, 3, 8, 8)
    target_tensors = torch.randn(2, 3, 8, 8)

    sft = prepare_qwen_sft_batch(
        pipeline=pipeline,
        vae=vae,
        prompts=prompts,
        control_pil_images=[object(), object()],
        control_tensors=control_tensors,
        target_tensors=target_tensors,
        weight_dtype=torch.float32,
    )
    forward = sft.forward
    # 4 noisy target tokens + 4 control tokens = 8 tokens total
    assert forward.hidden_states.shape == (2, 8, 64)
    assert forward.target_token_count == 4
    assert forward.encoder_hidden_states.shape == (2, 8, 32)
    assert forward.encoder_hidden_states_mask.shape == (2, 8)
    assert forward.timestep.shape == (2,)

    # img_shapes should contain 2 streams per sample
    assert len(forward.img_shapes) == 2
    assert forward.img_shapes[0] == [(1, 2, 2), (1, 2, 2)]
    # txt_seq_lens should reflect non-masked prompt length (6)
    assert forward.txt_seq_lens == [6, 6]

    assert sft.target_velocity.shape == (2, 4, 64)
    assert sft.target_velocity.dtype == torch.float32
    assert sft.sigmas.shape == (2,)


def test_prepare_qwen_dpo_batch_pairs_and_splitting() -> None:
    pipeline = MockQwenPipeline(text_dim=32)
    vae = MockQwenVAE(z_dim=16)
    prompts = ["preference task a", "preference task b"]
    control_tensors = torch.randn(2, 3, 8, 8)
    win_tensors = torch.randn(2, 3, 8, 8)
    lose_tensors = torch.randn(2, 3, 8, 8)

    dpo = prepare_qwen_dpo_batch(
        pipeline=pipeline,
        vae=vae,
        prompts=prompts,
        control_pil_images=[object(), object()],
        control_tensors=control_tensors,
        win_tensors=win_tensors,
        lose_tensors=lose_tensors,
        weight_dtype=torch.float32,
    )
    forward = dpo.forward
    # Batch dimension doubled: 2 win + 2 lose = 4
    assert forward.hidden_states.shape == (4, 8, 64)
    assert forward.target_token_count == 4
    assert forward.encoder_hidden_states.shape == (4, 8, 32)
    assert forward.encoder_hidden_states_mask.shape == (4, 8)
    assert forward.timestep.shape == (4,)
    assert len(forward.img_shapes) == 4
    assert forward.txt_seq_lens == [6, 6, 6, 6]

    # Target velocities retain single batch size (2)
    assert dpo.win_target_velocity.shape == (2, 4, 64)
    assert dpo.lose_target_velocity.shape == (2, 4, 64)
    assert dpo.sigmas.shape == (2,)


def test_predict_qwen_slicing_and_gradient_flow() -> None:
    transformer = MockQwenTransformer(hidden_dim=64)
    hidden_states = torch.randn(2, 8, 64, requires_grad=True)
    forward_batch = QwenForwardBatch(
        hidden_states=hidden_states,
        encoder_hidden_states=torch.randn(2, 8, 32),
        encoder_hidden_states_mask=torch.ones(2, 8, dtype=torch.long),
        timestep=torch.tensor([0.3, 0.7]),
        img_shapes=[[(1, 2, 2), (1, 2, 2)], [(1, 2, 2), (1, 2, 2)]],
        txt_seq_lens=[8, 8],
        target_token_count=4,
    )

    pred = predict_qwen(transformer, forward_batch)
    # Only target tokens (first 4 tokens) returned, control tokens sliced out
    assert pred.shape == (2, 4, 64)
    assert pred.dtype == torch.float32

    # Check pair splitting
    win_pred, lose_pred = split_pair_prediction(pred)
    assert win_pred.shape == (1, 4, 64)
    assert lose_pred.shape == (1, 4, 64)

    # Gradient check
    loss = pred.sum()
    loss.backward()
    assert transformer.linear.weight.grad is not None
    assert hidden_states.grad is not None
    # Control tokens (index 4 to 8) were sliced out so should have zero gradient
    assert torch.all(hidden_states.grad[:, 4:, :] == 0.0)
    assert not torch.all(hidden_states.grad[:, :4, :] == 0.0)


def test_qwen_image_adapter_integration() -> None:
    adapter = QwenImageAdapter()
    assert adapter.model_id == "qwen_image"
    assert adapter.default_lora_target_modules == QWEN_LORA_TARGET_MODULES

    pipeline = MockQwenPipeline(text_dim=32)
    vae = MockQwenVAE(z_dim=16)
    sft = adapter.prepare_sft_batch(
        pipeline=pipeline,
        vae=vae,
        prompts=["sample"],
        control_pil_images=[object()],
        control_tensors=torch.randn(1, 3, 8, 8),
        target_tensors=torch.randn(1, 3, 8, 8),
        weight_dtype=torch.float32,
    )
    assert sft.forward.hidden_states.shape == (1, 8, 64)

    dpo = adapter.prepare_dpo_batch(
        pipeline=pipeline,
        vae=vae,
        prompts=["sample"],
        control_pil_images=[object()],
        control_tensors=torch.randn(1, 3, 8, 8),
        win_tensors=torch.randn(1, 3, 8, 8),
        lose_tensors=torch.randn(1, 3, 8, 8),
        weight_dtype=torch.float32,
    )
    assert dpo.forward.hidden_states.shape == (2, 8, 64)

    transformer = MockQwenTransformer(hidden_dim=64)
    vel = adapter.predict_velocity(transformer, sft.forward)
    assert vel.shape == sft.target_velocity.shape
