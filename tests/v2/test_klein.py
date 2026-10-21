"""Unit tests for FLUX.2-klein model adapter conditioning and velocity prediction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from mmtrainer.models.klein import (
    KLEIN_LORA_TARGET_MODULES,
    KleinAdapter,
    KleinForwardBatch,
    encode_klein_latents,
    encode_klein_reference_tokens,
    predict_klein,
    prepare_klein_dpo_batch,
    prepare_klein_sft_batch,
)


class MockKleinVAE:
    """Mock VAE adhering to FLUX.2-klein VAE contract."""

    def __init__(self, in_channels: int = 16, patch_channels: int = 16) -> None:
        self.in_channels = in_channels
        self.config = SimpleNamespace(batch_norm_eps=1e-5)
        self.bn = SimpleNamespace(
            running_mean=torch.zeros(patch_channels),
            running_var=torch.ones(patch_channels),
        )

    def encode(self, images: torch.Tensor) -> Any:
        batch_size, _, height, width = images.shape
        latent_h = max(1, height // 2)
        latent_w = max(1, width // 2)
        latent_mode = torch.ones(
            batch_size,
            self.in_channels,
            latent_h,
            latent_w,
            dtype=images.dtype,
            device=images.device,
        )

        class _LatentDist:
            @staticmethod
            def mode() -> torch.Tensor:
                return latent_mode

        return SimpleNamespace(latent_dist=_LatentDist())


class MockKleinPipeline:
    """Mock Pipeline adhering to Klein patchify, pack, and ID preparation contract."""

    def __init__(self, hidden_dim: int = 16) -> None:
        self.hidden_dim = hidden_dim

    def _patchify_latents(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

    def _prepare_latent_ids(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = latents.shape
        return torch.zeros(batch_size, height * width, 3, device=latents.device)

    def _prepare_image_ids(self, per_sample_refs: list[list[torch.Tensor]]) -> torch.Tensor:
        batch_size = len(per_sample_refs)
        total_tokens = 0
        if per_sample_refs and per_sample_refs[0]:
            total_tokens = sum(ref.shape[1] * ref.shape[2] for ref in per_sample_refs[0])
        return torch.zeros(batch_size, total_tokens, 3)

    def encode_prompt(
        self,
        prompt: list[str],
        device: torch.device,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prompt) * num_images_per_prompt
        seq_len = 6
        prompt_embeds = torch.ones(batch_size, seq_len, self.hidden_dim, device=device)
        text_ids = torch.zeros(batch_size, seq_len, 3, device=device)
        return prompt_embeds, text_ids


class MockKleinDiT(nn.Module):
    """Simple linear DiT for testing velocity prediction and backward gradients."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: Any = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        return (self.proj(hidden_states),)


def test_encode_klein_latents_shapes_and_dtypes() -> None:
    vae = MockKleinVAE(in_channels=16, patch_channels=16)
    pipeline = MockKleinPipeline(hidden_dim=16)
    images = torch.randn(2, 3, 8, 8)

    latents_fp32 = encode_klein_latents(pipeline, vae, images, torch.float32)
    assert latents_fp32.shape == (2, 16, 4, 4)
    assert latents_fp32.dtype == torch.float32

    latents_bf16 = encode_klein_latents(pipeline, vae, images, torch.bfloat16)
    assert latents_bf16.shape == (2, 16, 4, 4)
    assert latents_bf16.dtype == torch.bfloat16

    # Verify batch norm standardization arithmetic
    vae.bn.running_mean = torch.full((16,), 0.5)
    vae.bn.running_var = torch.full((16,), 4.0 - 1e-5)
    standardized = encode_klein_latents(pipeline, vae, images, torch.float32)
    expected_val = (1.0 - 0.5) / 2.0  # (latent_mode - mean) / sqrt(var + eps)
    assert torch.allclose(standardized, torch.full_like(standardized, expected_val), atol=1e-4)


def test_encode_klein_reference_tokens_optionality() -> None:
    vae = MockKleinVAE(in_channels=16, patch_channels=16)
    pipeline = MockKleinPipeline(hidden_dim=16)

    assert encode_klein_reference_tokens(pipeline, vae, None, torch.float32) == (None, None)
    assert encode_klein_reference_tokens(pipeline, vae, [], torch.float32) == (None, None)

    ref_tensor = torch.randn(2, 3, 8, 8)
    packed_single, ids_single = encode_klein_reference_tokens(
        pipeline, vae, ref_tensor, torch.float32
    )
    assert packed_single is not None and ids_single is not None
    assert packed_single.shape == (2, 16, 16)
    assert ids_single.shape == (2, 16, 3)

    ref_streams = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)]
    packed_multi, ids_multi = encode_klein_reference_tokens(
        pipeline, vae, ref_streams, torch.float32
    )
    assert packed_multi is not None and ids_multi is not None
    assert packed_multi.shape == (2, 32, 16)
    assert ids_multi.shape == (2, 32, 3)


def test_prepare_klein_sft_batch_without_and_with_references() -> None:
    vae = MockKleinVAE(in_channels=16, patch_channels=16)
    pipeline = MockKleinPipeline(hidden_dim=16)
    prompts = ["a photo of a cat", "a photo of a dog"]
    target_images = torch.randn(2, 3, 8, 8)

    # 1. Without reference images
    sft = prepare_klein_sft_batch(
        pipeline,
        vae,
        prompts=prompts,
        target_images=target_images,
        weight_dtype=torch.float32,
    )
    assert sft.forward.hidden_states.shape == (2, 16, 16)
    assert sft.forward.target_token_count == 16
    assert sft.forward.encoder_hidden_states.shape == (2, 6, 16)
    assert sft.forward.timestep.shape == (2,)
    assert sft.forward.img_ids.shape == (2, 16, 3)
    assert sft.forward.txt_ids.shape == (2, 6, 3)
    assert sft.target_velocity.shape == (2, 16, 16)
    assert sft.target_velocity.dtype == torch.float32
    assert sft.sigmas.shape == (2,)

    # 2. With reference images
    ref_images = torch.randn(2, 3, 8, 8)
    sft_ref = prepare_klein_sft_batch(
        pipeline,
        vae,
        prompts=prompts,
        target_images=target_images,
        weight_dtype=torch.float32,
        reference_images=ref_images,
    )
    assert sft_ref.forward.hidden_states.shape == (2, 32, 16)
    assert sft_ref.forward.target_token_count == 16
    assert sft_ref.forward.img_ids.shape == (2, 32, 3)
    assert sft_ref.target_velocity.shape == (2, 16, 16)


def test_prepare_klein_dpo_batch_pairs_and_sigmas() -> None:
    vae = MockKleinVAE(in_channels=16, patch_channels=16)
    pipeline = MockKleinPipeline(hidden_dim=16)
    prompts = ["portrait editing", "landscape editing"]
    win_images = torch.randn(2, 3, 8, 8)
    lose_images = torch.randn(2, 3, 8, 8)
    ref_images = torch.randn(2, 3, 8, 8)

    dpo = prepare_klein_dpo_batch(
        pipeline,
        vae,
        prompts=prompts,
        win_images=win_images,
        lose_images=lose_images,
        weight_dtype=torch.float32,
        reference_images=ref_images,
    )
    forward = dpo.forward
    # Batch dimension doubled: 2 win + 2 lose = 4
    assert forward.hidden_states.shape == (4, 32, 16)
    assert forward.target_token_count == 16
    assert forward.encoder_hidden_states.shape == (4, 6, 16)
    assert forward.timestep.shape == (4,)
    assert forward.img_ids.shape == (4, 32, 3)
    assert forward.txt_ids.shape == (4, 6, 3)

    # Velocities retain per-stream single batch size
    assert dpo.win_target_velocity.shape == (2, 16, 16)
    assert dpo.lose_target_velocity.shape == (2, 16, 16)
    assert dpo.sigmas.shape == (2,)


def test_predict_klein_slicing_and_gradient_flow() -> None:
    transformer = MockKleinDiT(hidden_dim=16)
    hidden_states = torch.randn(2, 32, 16, requires_grad=True)
    forward_batch = KleinForwardBatch(
        hidden_states=hidden_states,
        encoder_hidden_states=torch.randn(2, 6, 16),
        timestep=torch.tensor([0.4, 0.6]),
        img_ids=torch.zeros(2, 32, 3),
        txt_ids=torch.zeros(2, 6, 3),
        target_token_count=16,
    )

    pred = predict_klein(transformer, forward_batch)
    # Output should slice only target tokens, ignoring the 16 reference tokens
    assert pred.shape == (2, 16, 16)
    assert pred.dtype == torch.float32

    # Verify gradient propagation to transformer parameters and hidden inputs
    loss = pred.sum()
    loss.backward()
    assert transformer.proj.weight.grad is not None
    assert hidden_states.grad is not None
    # Gradients for sliced out tokens should be zero
    assert torch.all(hidden_states.grad[:, 16:, :] == 0.0)
    assert not torch.all(hidden_states.grad[:, :16, :] == 0.0)


def test_klein_adapter_integration() -> None:
    adapter = KleinAdapter()
    assert adapter.model_id == "klein"
    assert adapter.default_lora_target_modules == KLEIN_LORA_TARGET_MODULES

    vae = MockKleinVAE(in_channels=16, patch_channels=16)
    pipeline = MockKleinPipeline(hidden_dim=16)
    sft = adapter.prepare_sft_batch(
        pipeline=pipeline,
        vae=vae,
        prompts=["test prompt"],
        target_images=torch.randn(1, 3, 8, 8),
        weight_dtype=torch.float32,
    )
    assert sft.forward.hidden_states.shape == (1, 16, 16)

    dpo = adapter.prepare_dpo_batch(
        pipeline=pipeline,
        vae=vae,
        prompts=["test prompt"],
        win_images=torch.randn(1, 3, 8, 8),
        lose_images=torch.randn(1, 3, 8, 8),
        weight_dtype=torch.float32,
    )
    assert dpo.forward.hidden_states.shape == (2, 16, 16)

    transformer = MockKleinDiT(hidden_dim=16)
    vel = adapter.predict_velocity(transformer, sft.forward)
    assert vel.shape == sft.target_velocity.shape
