"""Krea2 single-stream MMDiT components, multi-layer text encoding, and key remapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmtrainer.nn.flow import flow_sample, paired_flow_samples
from mmtrainer.plugins.base import BaseModelAdapter

SELECT_LAYERS: tuple[int, ...] = (9, 18, 27)
KREA2_LORA_TARGET_MODULES: tuple[str, ...] = (
    "qkv",
    "proj",
    "w1",
    "w2",
    "w3",
    "linear",
)


@dataclass(frozen=True)
class SingleMMDiTConfig:
    in_channels: int = 128
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 24
    mlp_ratio: float = 4.0
    context_in_dim: int = 7680


@dataclass(frozen=True)
class Krea2ForwardBatch:
    img_tokens: torch.Tensor
    txt_tokens: torch.Tensor
    timesteps: torch.Tensor
    target_token_count: int


@dataclass(frozen=True)
class Krea2SftBatch:
    forward: Krea2ForwardBatch
    target_velocity: torch.Tensor
    sigmas: torch.Tensor


@dataclass(frozen=True)
class Krea2DpoBatch:
    forward: Krea2ForwardBatch
    win_target_velocity: torch.Tensor
    lose_target_velocity: torch.Tensor
    sigmas: torch.Tensor


def pad_text_features(
    features: list[torch.Tensor],
    pad_token: torch.Tensor,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(feat.shape[0] for feat in features)
    if pad_to is not None:
        max_len = max(max_len, pad_to)
    batch_size = len(features)
    dim = features[0].shape[-1]
    padded = pad_token.view(1, 1, dim).expand(batch_size, max_len, dim).clone()
    mask = torch.zeros(
        batch_size, max_len, dtype=torch.bool, device=features[0].device
    )
    for idx, feat in enumerate(features):
        seq_len = feat.shape[0]
        padded[idx, :seq_len] = feat
        mask[idx, :seq_len] = True
    return padded, mask


def encode_krea_prompt(
    tokenizer: Any,
    text_encoder: Any,
    prompts: list[str],
    *,
    max_length: int = 512,
    select_layers: tuple[int, ...] = SELECT_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(text_encoder.device)
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    stacked = torch.cat([outputs.hidden_states[layer] for layer in select_layers], dim=-1)
    lengths = batch.attention_mask.sum(dim=1).tolist()
    unpadded = [stacked[i, : int(lengths[i])] for i in range(len(prompts))]
    pad_vec = stacked[0, -1]
    return pad_text_features(unpadded, pad_vec)


def pack_patch_tokens(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    reshaped = latents.view(
        batch_size, channels, h_patches, patch_size, w_patches, patch_size
    )
    permuted = reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
    return permuted.view(
        batch_size, h_patches * w_patches, channels * patch_size * patch_size
    )


def prepare_krea2_tokens(
    target_latents: torch.Tensor,
    cond_latents: torch.Tensor | None = None,
    *,
    patch_size: int = 2,
) -> tuple[torch.Tensor, int]:
    target_tokens = pack_patch_tokens(target_latents, patch_size=patch_size)
    token_count = int(target_tokens.shape[1])
    if cond_latents is None:
        return target_tokens, token_count
    cond_tokens = pack_patch_tokens(cond_latents, patch_size=patch_size)
    return torch.cat([target_tokens, cond_tokens], dim=1), token_count


def diffusers_to_native_key(key: str) -> str:
    remapped = key
    replacements = (
        ("context_embedder.", "txt_in."),
        ("x_embedder.", "img_in."),
        ("time_guidance_embed.timestep_embedder.", "time_in."),
        ("single_transformer_blocks.", "single_blocks."),
        (".attn.to_qkv_mlp_proj.", ".linear1."),
        (".attn.to_out.0.", ".linear2."),
        (".attn.norm_q.", ".norm.query_norm."),
        (".attn.norm_k.", ".norm.key_norm."),
        ("norm_out.linear.", "final_layer.adaLN_modulation.1."),
        ("proj_out.", "final_layer.linear."),
    )
    for src, dst in replacements:
        remapped = remapped.replace(src, dst)
    return remapped


class SingleStreamBlock(nn.Module):
    """Single-stream MMDiT block with parallel attention and SwiGLU MLP."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w1 = nn.Linear(hidden_size, self.mlp_hidden, bias=False)
        self.w2 = nn.Linear(self.mlp_hidden, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, self.mlp_hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mod_shift: torch.Tensor,
        mod_scale: torch.Tensor,
        mod_gate: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm(x) * (1.0 + mod_scale.unsqueeze(1)) + mod_shift.unsqueeze(1)
        bsz, seq_len, _ = normed.shape
        qkv = self.qkv(normed).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        attn_out = self.proj(attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))
        mlp_out = self.w2(F.silu(self.w1(normed)) * self.w3(normed))
        return x + mod_gate.unsqueeze(1) * (attn_out + mlp_out)


class SingleStreamDiT(nn.Module):
    """Lightweight single-stream MMDiT backbone for Krea2 training and tests."""

    def __init__(self, config: SingleMMDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.img_in = nn.Linear(config.in_channels, config.hidden_size, bias=False)
        self.txt_in = nn.Linear(config.context_in_dim, config.hidden_size, bias=False)
        self.time_in = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size * 3),
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.final_linear = nn.Linear(config.hidden_size, config.in_channels, bias=False)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        img_h = self.img_in(img_tokens)
        txt_h = self.txt_in(txt_tokens)
        seq = torch.cat([txt_h, img_h], dim=1)
        mod = self.time_in(timesteps.view(-1, 1).to(seq.dtype))
        shift, scale, gate = mod.chunk(3, dim=-1)
        for block in self.single_blocks:
            seq = block(seq, shift, scale, gate)
        img_out = seq[:, txt_h.shape[1] :, :]
        return self.final_linear(self.final_norm(img_out))


def prepare_krea2_sft_batch(
    target_latents: torch.Tensor,
    txt_tokens: torch.Tensor,
    *,
    cond_latents: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
    patch_size: int = 2,
) -> Krea2SftBatch:
    sample = flow_sample(target_latents, sigmas=sigmas)
    img_tokens, token_count = prepare_krea2_tokens(
        sample.noisy_latents, cond_latents, patch_size=patch_size
    )
    target_vel = pack_patch_tokens(sample.target_velocity, patch_size=patch_size)
    forward = Krea2ForwardBatch(
        img_tokens=img_tokens,
        txt_tokens=txt_tokens,
        timesteps=sample.sigmas,
        target_token_count=token_count,
    )
    return Krea2SftBatch(
        forward=forward,
        target_velocity=target_vel.float(),
        sigmas=sample.sigmas,
    )


def prepare_krea2_dpo_batch(
    win_latents: torch.Tensor,
    lose_latents: torch.Tensor,
    txt_tokens: torch.Tensor,
    *,
    cond_latents: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
    patch_size: int = 2,
) -> Krea2DpoBatch:
    win_sample, lose_sample = paired_flow_samples(
        win_latents, lose_latents, sigmas=sigmas
    )
    win_tokens, token_count = prepare_krea2_tokens(
        win_sample.noisy_latents, cond_latents, patch_size=patch_size
    )
    lose_tokens, _ = prepare_krea2_tokens(
        lose_sample.noisy_latents, cond_latents, patch_size=patch_size
    )
    forward = Krea2ForwardBatch(
        img_tokens=torch.cat([win_tokens, lose_tokens], dim=0),
        txt_tokens=txt_tokens.repeat(2, 1, 1),
        timesteps=win_sample.sigmas.repeat(2),
        target_token_count=token_count,
    )
    return Krea2DpoBatch(
        forward=forward,
        win_target_velocity=pack_patch_tokens(
            win_sample.target_velocity, patch_size=patch_size
        ).float(),
        lose_target_velocity=pack_patch_tokens(
            lose_sample.target_velocity, patch_size=patch_size
        ).float(),
        sigmas=win_sample.sigmas,
    )


def predict_krea2_velocity(
    transformer: Any, batch: Krea2ForwardBatch
) -> torch.Tensor:
    out = transformer(
        batch.img_tokens,
        batch.txt_tokens,
        batch.timesteps,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out[:, : batch.target_token_count, :].float()


class Krea2Adapter(BaseModelAdapter):
    """Composable model adapter for Krea2 single-stream MMDiT."""

    model_id = "krea2"
    default_lora_target_modules = KREA2_LORA_TARGET_MODULES

    def prepare_sft_batch(self, **kwargs: Any) -> Krea2SftBatch:
        return prepare_krea2_sft_batch(**kwargs)

    def prepare_dpo_batch(self, **kwargs: Any) -> Krea2DpoBatch:
        return prepare_krea2_dpo_batch(**kwargs)

    def predict_velocity(
        self, transformer: Any, batch: Krea2ForwardBatch, **kwargs: Any
    ) -> torch.Tensor:
        del kwargs
        return predict_krea2_velocity(transformer, batch)
