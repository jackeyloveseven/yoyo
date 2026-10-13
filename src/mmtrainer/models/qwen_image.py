"""Qwen-Image conditioning, 3D latent packing, and velocity prediction adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mmtrainer.nn.flow import flow_sample, paired_flow_samples
from mmtrainer.plugins.base import BaseModelAdapter

QWEN_LORA_TARGET_MODULES: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
)


@dataclass(frozen=True)
class QwenForwardBatch:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_hidden_states_mask: torch.Tensor
    timestep: torch.Tensor
    img_shapes: list[list[tuple[int, int, int]]]
    txt_seq_lens: list[int]
    target_token_count: int


@dataclass(frozen=True)
class QwenSftBatch:
    forward: QwenForwardBatch
    target_velocity: torch.Tensor
    sigmas: torch.Tensor


@dataclass(frozen=True)
class QwenDpoBatch:
    forward: QwenForwardBatch
    win_target_velocity: torch.Tensor
    lose_target_velocity: torch.Tensor
    sigmas: torch.Tensor


def encode_vae_image(vae: Any, images: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    pixel_values = images.unsqueeze(2).to(dtype=weight_dtype)
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents.device, dtype=latents.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std, device=latents.device, dtype=latents.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    return (latents - latents_mean) / latents_std


def encode_qwen_prompt(
    pipeline: Any,
    control_pil_images: list[Any],
    prompts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """编码文本与控制图条件,返回 (prompt_embeds, prompt_mask)。"""
    prompt_embeds, prompt_mask = pipeline.encode_prompt(
        image=control_pil_images,
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=1024,
    )
    return prompt_embeds, prompt_mask


def pack_qwen_latents(pipeline: Any, latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _, height, width = latents.shape
    return pipeline._pack_latents(latents, batch_size, channels, height, width)


def _qwen_shape_metadata(
    latents: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    *,
    num_streams: int,
) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
    batch_size, _, _, height, width = latents.shape
    patch_shape = (1, height // 2, width // 2)
    img_shapes = [[patch_shape] * num_streams for _ in range(batch_size)]
    txt_seq_lens = [int(val) for val in prompt_embeds_mask.sum(dim=1).tolist()]
    return img_shapes, txt_seq_lens


def prepare_qwen_sft_batch(
    pipeline: Any,
    vae: Any,
    prompts: list[str],
    control_pil_images: list[Any],
    control_tensors: torch.Tensor,
    target_tensors: torch.Tensor,
    weight_dtype: torch.dtype,
    *,
    sigmas: torch.Tensor | None = None,
) -> QwenSftBatch:
    with torch.no_grad():
        prompt_embeds, prompt_mask = encode_qwen_prompt(
            pipeline, control_pil_images, prompts, target_tensors.device
        )
        target_latents = encode_vae_image(vae, target_tensors, weight_dtype)
        control_latents = encode_vae_image(vae, control_tensors, weight_dtype)

    sample = flow_sample(target_latents, sigmas=sigmas)
    packed_noisy = pack_qwen_latents(pipeline, sample.noisy_latents)
    packed_control = pack_qwen_latents(pipeline, control_latents)
    packed_target = pack_qwen_latents(pipeline, sample.target_velocity)
    hidden_states = torch.cat([packed_noisy, packed_control], dim=1)
    img_shapes, txt_seq_lens = _qwen_shape_metadata(target_latents, prompt_mask, num_streams=2)
    forward = QwenForwardBatch(
        hidden_states=hidden_states.to(dtype=weight_dtype),
        encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
        encoder_hidden_states_mask=prompt_mask,
        timestep=sample.sigmas.to(dtype=weight_dtype),
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        target_token_count=int(packed_noisy.shape[1]),
    )
    return QwenSftBatch(
        forward=forward,
        target_velocity=packed_target.float(),
        sigmas=sample.sigmas,
    )


def prepare_qwen_dpo_batch(
    pipeline: Any,
    vae: Any,
    prompts: list[str],
    control_pil_images: list[Any],
    control_tensors: torch.Tensor,
    win_tensors: torch.Tensor,
    lose_tensors: torch.Tensor,
    weight_dtype: torch.dtype,
) -> QwenDpoBatch:
    with torch.no_grad():
        prompt_embeds, prompt_mask = encode_qwen_prompt(
            pipeline, control_pil_images, prompts, win_tensors.device
        )
        win_latents = encode_vae_image(vae, win_tensors, weight_dtype)
        lose_latents = encode_vae_image(vae, lose_tensors, weight_dtype)
        control_latents = encode_vae_image(vae, control_tensors, weight_dtype)

    win_sample, lose_sample = paired_flow_samples(win_latents, lose_latents)
    packed_win = pack_qwen_latents(pipeline, win_sample.noisy_latents)
    packed_lose = pack_qwen_latents(pipeline, lose_sample.noisy_latents)
    packed_ctrl = pack_qwen_latents(pipeline, control_latents)
    hidden_states = torch.cat(
        [
            torch.cat([packed_win, packed_ctrl], dim=1),
            torch.cat([packed_lose, packed_ctrl], dim=1),
        ],
        dim=0,
    )
    doubled_mask = prompt_mask.repeat(2, 1)
    img_shapes, txt_seq_lens = _qwen_shape_metadata(
        win_latents.repeat(2, 1, 1, 1, 1), doubled_mask, num_streams=2
    )
    forward = QwenForwardBatch(
        hidden_states=hidden_states.to(dtype=weight_dtype),
        encoder_hidden_states=prompt_embeds.repeat(2, 1, 1).to(dtype=weight_dtype),
        encoder_hidden_states_mask=doubled_mask,
        timestep=win_sample.sigmas.repeat(2).to(dtype=weight_dtype),
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        target_token_count=int(packed_win.shape[1]),
    )
    return QwenDpoBatch(
        forward=forward,
        win_target_velocity=pack_qwen_latents(pipeline, win_sample.target_velocity).float(),
        lose_target_velocity=pack_qwen_latents(pipeline, lose_sample.target_velocity).float(),
        sigmas=win_sample.sigmas,
    )


def predict_qwen(transformer: Any, batch: QwenForwardBatch) -> torch.Tensor:
    prediction = transformer(
        hidden_states=batch.hidden_states,
        timestep=batch.timestep,
        guidance=None,
        encoder_hidden_states_mask=batch.encoder_hidden_states_mask,
        encoder_hidden_states=batch.encoder_hidden_states,
        img_shapes=batch.img_shapes,
        txt_seq_lens=batch.txt_seq_lens,
        return_dict=False,
    )[0]
    return prediction[:, : batch.target_token_count, :].float()


def split_pair_prediction(
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    win_pred, lose_pred = prediction.chunk(2, dim=0)
    return win_pred, lose_pred


class QwenImageAdapter(BaseModelAdapter):
    """Composable model adapter for Qwen-Image-Edit SFT, DPO, and alignment."""

    model_id = "qwen_image"
    default_lora_target_modules = QWEN_LORA_TARGET_MODULES

    def prepare_sft_batch(self, **kwargs: Any) -> QwenSftBatch:
        return prepare_qwen_sft_batch(**kwargs)

    def prepare_dpo_batch(self, **kwargs: Any) -> QwenDpoBatch:
        return prepare_qwen_dpo_batch(**kwargs)

    def predict_velocity(
        self, transformer: Any, batch: QwenForwardBatch, **kwargs: Any
    ) -> torch.Tensor:
        del kwargs
        return predict_qwen(transformer, batch)
