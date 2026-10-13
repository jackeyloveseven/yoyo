"""Qwen-Image-Edit 2.x(2509/2511)多图编辑适配器。

复用原版 `qwen_image` 的 VAE 编码、latent 打包与前向协议——2.x 与原版共享
Qwen2.5-VL 文本编码器与 Wan2.1-VAE,MMDiT 前向签名不变,对应 diffusers 的
`QwenImageEditPlusPipeline`。差异只在**控制图数量**:2509+ 经图像拼接训练支持
1-3 张参考图。

关键教训(invoke-ai#9155):`img_shapes` 里 noisy 段与每个 control 段必须**各自**
记录 `(1, H//2, W//2)`,跨流共用同一 tuple 会让 MSRoPE 给两段相同空间位置,
跨注意力无法区分,产生重影。本零件按流独立推导形状。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mmtrainer.models.qwen_image import (
    QwenForwardBatch,
    encode_qwen_prompt,
    encode_vae_image,
    pack_qwen_latents,
    predict_qwen,
)
from mmtrainer.nn.flow import flow_sample, paired_flow_samples
from mmtrainer.plugins.base import BaseModelAdapter

QWEN_EDIT2_LORA_TARGET_MODULES: tuple[str, ...] = (
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

MAX_CONTROL_IMAGES = 3


@dataclass(frozen=True)
class QwenEdit2SftBatch:
    forward: QwenForwardBatch
    target_velocity: torch.Tensor
    sigmas: torch.Tensor


@dataclass(frozen=True)
class QwenEdit2DpoBatch:
    forward: QwenForwardBatch
    win_target_velocity: torch.Tensor
    lose_target_velocity: torch.Tensor
    sigmas: torch.Tensor


def _validate_control_tensors(control_tensors: torch.Tensor) -> None:
    """控制图必须是 ``(batch, num_control, C, H, W)`` 且 1-3 张(2509+ 的上限)。"""
    if control_tensors.ndim != 5:
        raise ValueError(
            "control_tensors must be (batch, num_control, C, H, W), "
            f"got {tuple(control_tensors.shape)}"
        )
    num_control = control_tensors.shape[1]
    if not 1 <= num_control <= MAX_CONTROL_IMAGES:
        raise ValueError(
            f"Qwen-Image-Edit 2.x supports 1-{MAX_CONTROL_IMAGES} control images, got {num_control}"
        )


def _pack_latent_streams(
    pipeline: Any,
    *latents_list: torch.Tensor,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int]]]:
    """逐流打包 5D latent,返回 (packed 列表, 各流独立的 patch 形状)。

    每流形状从自己的 latent 推导,不复用相邻流的 tuple(见模块 docstring 的
    MSRoPE 重影教训)。
    """
    packed = [pack_qwen_latents(pipeline, latents) for latents in latents_list]
    shapes = [(1, latents.shape[-2] // 2, latents.shape[-1] // 2) for latents in latents_list]
    return packed, shapes


def _encode_controls(
    vae: Any,
    control_tensors: torch.Tensor,
    weight_dtype: torch.dtype,
) -> list[torch.Tensor]:
    """把 ``(batch, num_control, C, H, W)`` 逐张 VAE 编码为 5D latent 列表。"""
    return [
        encode_vae_image(vae, control_tensors[:, stream], weight_dtype)
        for stream in range(control_tensors.shape[1])
    ]


def _forward_batch(
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    packed_streams: list[torch.Tensor],
    stream_shapes: list[tuple[int, int, int]],
    sigmas: torch.Tensor,
    weight_dtype: torch.dtype,
) -> QwenForwardBatch:
    """沿 token 维拼各流,组装前向批次(noisy 流在前,control 流在后)。"""
    hidden_states = torch.cat(packed_streams, dim=1)
    batch_size = hidden_states.shape[0]
    forward = QwenForwardBatch(
        hidden_states=hidden_states.to(dtype=weight_dtype),
        encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
        encoder_hidden_states_mask=prompt_mask,
        timestep=sigmas.to(dtype=weight_dtype),
        img_shapes=[list(stream_shapes) for _ in range(batch_size)],
        txt_seq_lens=[int(val) for val in prompt_mask.sum(dim=1).tolist()],
        target_token_count=int(packed_streams[0].shape[1]),
    )
    return forward


def prepare_qwen_edit2_sft_batch(
    pipeline: Any,
    vae: Any,
    prompts: list[str],
    # Plus 管线契约:image 是嵌套列表——每个 prompt 一组 1-3 张控制图。
    control_pil_images: list[list[Any]],
    control_tensors: torch.Tensor,
    target_tensors: torch.Tensor,
    weight_dtype: torch.dtype,
    *,
    sigmas: torch.Tensor | None = None,
) -> QwenEdit2SftBatch:
    """构造多图编辑 SFT 批次:1-3 张控制图 + 1 张目标图,一次前向预测速度。"""
    _validate_control_tensors(control_tensors)
    with torch.no_grad():
        prompt_embeds, prompt_mask = encode_qwen_prompt(
            pipeline, control_pil_images, prompts, target_tensors.device
        )
        target_latents = encode_vae_image(vae, target_tensors, weight_dtype)
        control_streams = _encode_controls(vae, control_tensors, weight_dtype)

    sample = flow_sample(target_latents, sigmas=sigmas)
    packed_streams, stream_shapes = _pack_latent_streams(
        pipeline, sample.noisy_latents, *control_streams
    )
    forward = _forward_batch(
        prompt_embeds,
        prompt_mask,
        packed_streams,
        stream_shapes,
        sample.sigmas,
        weight_dtype,
    )
    packed_target = pack_qwen_latents(pipeline, sample.target_velocity)
    return QwenEdit2SftBatch(
        forward=forward,
        target_velocity=packed_target.float(),
        sigmas=sample.sigmas,
    )


def prepare_qwen_edit2_dpo_batch(
    pipeline: Any,
    vae: Any,
    prompts: list[str],
    # Plus 管线契约:image 是嵌套列表——每个 prompt 一组 1-3 张控制图。
    control_pil_images: list[list[Any]],
    control_tensors: torch.Tensor,
    win_tensors: torch.Tensor,
    lose_tensors: torch.Tensor,
    weight_dtype: torch.dtype,
) -> QwenEdit2DpoBatch:
    """构造多图编辑 DPO 批次:赢/输共享成对噪声与控制流,沿 batch 维拼接一次前向。"""
    _validate_control_tensors(control_tensors)
    with torch.no_grad():
        prompt_embeds, prompt_mask = encode_qwen_prompt(
            pipeline, control_pil_images, prompts, win_tensors.device
        )
        win_latents = encode_vae_image(vae, win_tensors, weight_dtype)
        lose_latents = encode_vae_image(vae, lose_tensors, weight_dtype)
        control_streams = _encode_controls(vae, control_tensors, weight_dtype)

    win_sample, lose_sample = paired_flow_samples(win_latents, lose_latents)
    win_packed, win_shapes = _pack_latent_streams(
        pipeline, win_sample.noisy_latents, *control_streams
    )
    lose_packed, lose_shapes = _pack_latent_streams(
        pipeline, lose_sample.noisy_latents, *control_streams
    )
    # 两支各自的流形状随各自分支记录(控制流相同,但 noisy 分辨率可能不同)。
    batch_win = win_packed[0].shape[0]
    batch_lose = lose_packed[0].shape[0]
    txt_seq_lens = [int(val) for val in prompt_mask.sum(dim=1).tolist()]
    forward = QwenForwardBatch(
        hidden_states=torch.cat(
            [torch.cat(win_packed, dim=1), torch.cat(lose_packed, dim=1)], dim=0
        ).to(dtype=weight_dtype),
        encoder_hidden_states=prompt_embeds.repeat(2, 1, 1).to(dtype=weight_dtype),
        encoder_hidden_states_mask=prompt_mask.repeat(2, 1),
        timestep=win_sample.sigmas.repeat(2).to(dtype=weight_dtype),
        img_shapes=[list(win_shapes) for _ in range(batch_win)]
        + [list(lose_shapes) for _ in range(batch_lose)],
        txt_seq_lens=txt_seq_lens * 2,
        target_token_count=int(win_packed[0].shape[1]),
    )
    return QwenEdit2DpoBatch(
        forward=forward,
        win_target_velocity=pack_qwen_latents(pipeline, win_sample.target_velocity).float(),
        lose_target_velocity=pack_qwen_latents(pipeline, lose_sample.target_velocity).float(),
        sigmas=win_sample.sigmas,
    )


class QwenImageEdit2Adapter(BaseModelAdapter):
    """Qwen-Image-Edit 2.x(2509/2511)多图编辑适配器,复用原版前向协议。"""

    model_id = "qwen_image_edit_2x"
    default_lora_target_modules = QWEN_EDIT2_LORA_TARGET_MODULES

    def prepare_sft_batch(self, **kwargs: Any) -> QwenEdit2SftBatch:
        return prepare_qwen_edit2_sft_batch(**kwargs)

    def prepare_dpo_batch(self, **kwargs: Any) -> QwenEdit2DpoBatch:
        return prepare_qwen_edit2_dpo_batch(**kwargs)

    def predict_velocity(
        self, transformer: Any, batch: QwenForwardBatch, **kwargs: Any
    ) -> torch.Tensor:
        del kwargs
        return predict_qwen(transformer, batch)
