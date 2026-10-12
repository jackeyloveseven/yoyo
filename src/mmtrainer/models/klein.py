"""FLUX.2-klein model conditioning, token packing, and velocity prediction adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mmtrainer.nn.flow import flow_sample, paired_flow_samples
from mmtrainer.plugins.base import BaseModelAdapter

KLEIN_LORA_TARGET_MODULES: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff.linear_in",
    "ff.linear_out",
    "ff_context.linear_in",
    "ff_context.linear_out",
    "to_qkv_mlp_proj",
)


@dataclass(frozen=True)
class KleinForwardBatch:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    timestep: torch.Tensor
    img_ids: torch.Tensor
    txt_ids: torch.Tensor
    target_token_count: int


@dataclass(frozen=True)
class KleinSftBatch:
    forward: KleinForwardBatch
    target_velocity: torch.Tensor
    sigmas: torch.Tensor


@dataclass(frozen=True)
class KleinDpoBatch:
    forward: KleinForwardBatch
    win_target_velocity: torch.Tensor
    lose_target_velocity: torch.Tensor
    sigmas: torch.Tensor


def encode_klein_latents(
    pipeline: Any,
    vae: Any,
    images: torch.Tensor,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    encoded = vae.encode(images.to(dtype=weight_dtype)).latent_dist.mode()
    latents = pipeline._patchify_latents(encoded)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    return (latents - bn_mean) / bn_std


def encode_klein_reference_tokens(
    pipeline: Any,
    vae: Any,
    reference_images: torch.Tensor | list[torch.Tensor] | None,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if reference_images is None:
        return None, None
    streams = (
        [reference_images] if isinstance(reference_images, torch.Tensor) else list(reference_images)
    )
    if not streams:
        return None, None
    ref_latents = [encode_klein_latents(pipeline, vae, stream, weight_dtype) for stream in streams]
    batch_size = ref_latents[0].shape[0]
    per_sample_refs = [[stream[idx] for stream in ref_latents] for idx in range(batch_size)]
    image_ids = pipeline._prepare_image_ids(per_sample_refs).to(device=ref_latents[0].device)
    packed = torch.cat([pipeline._pack_latents(stream) for stream in ref_latents], dim=1)
    return packed, image_ids


def _concat_klein_ref_tokens(
    target_tokens: torch.Tensor,
    target_ids: torch.Tensor,
    ref_tokens: torch.Tensor | None,
    ref_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    token_count = int(target_tokens.shape[1])
    if ref_tokens is not None and ref_ids is not None:
        hidden_states = torch.cat([target_tokens, ref_tokens], dim=1)
        img_ids = torch.cat([target_ids, ref_ids], dim=1)
        return hidden_states, img_ids, token_count
    return target_tokens, target_ids, token_count


def prepare_klein_sft_batch(
    pipeline: Any,
    vae: Any,
    prompts: list[str],
    target_images: torch.Tensor,
    weight_dtype: torch.dtype,
    *,
    reference_images: torch.Tensor | list[torch.Tensor] | None = None,
    sigmas: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> KleinSftBatch:
    with torch.no_grad():
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompts,
            device=target_images.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        clean_latents = encode_klein_latents(pipeline, vae, target_images, weight_dtype)
        ref_tokens, ref_ids = encode_klein_reference_tokens(
            pipeline, vae, reference_images, weight_dtype
        )

    sample = flow_sample(clean_latents, sigmas=sigmas, noise=noise)
    packed_noisy = pipeline._pack_latents(sample.noisy_latents)
    packed_target = pipeline._pack_latents(sample.target_velocity)
    img_ids = pipeline._prepare_latent_ids(clean_latents).to(device=target_images.device)
    hidden_states, full_img_ids, token_count = _concat_klein_ref_tokens(
        packed_noisy, img_ids, ref_tokens, ref_ids
    )
    forward = KleinForwardBatch(
        hidden_states=hidden_states.to(dtype=weight_dtype),
        encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
        timestep=sample.sigmas.to(dtype=weight_dtype),
        img_ids=full_img_ids,
        txt_ids=text_ids,
        target_token_count=token_count,
    )
    return KleinSftBatch(
        forward=forward,
        target_velocity=packed_target.float(),
        sigmas=sample.sigmas,
    )


def prepare_klein_dpo_batch(
    pipeline: Any,
    vae: Any,
    prompts: list[str],
    win_images: torch.Tensor,
    lose_images: torch.Tensor,
    weight_dtype: torch.dtype,
    *,
    reference_images: torch.Tensor | list[torch.Tensor] | None = None,
    sigmas: torch.Tensor | None = None,
) -> KleinDpoBatch:
    with torch.no_grad():
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompts,
            device=win_images.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        win_latents = encode_klein_latents(pipeline, vae, win_images, weight_dtype)
        lose_latents = encode_klein_latents(pipeline, vae, lose_images, weight_dtype)
        ref_tokens, ref_ids = encode_klein_reference_tokens(
            pipeline, vae, reference_images, weight_dtype
        )

    win_sample, lose_sample = paired_flow_samples(win_latents, lose_latents, sigmas=sigmas)
    packed_noisy = torch.cat(
        [
            pipeline._pack_latents(win_sample.noisy_latents),
            pipeline._pack_latents(lose_sample.noisy_latents),
        ],
        dim=0,
    )
    img_ids = pipeline._prepare_latent_ids(win_latents).to(device=win_images.device).repeat(2, 1, 1)
    doubled_ref_tokens = ref_tokens.repeat(2, 1, 1) if ref_tokens is not None else None
    doubled_ref_ids = ref_ids.repeat(2, 1, 1) if ref_ids is not None else None
    hidden_states, full_img_ids, token_count = _concat_klein_ref_tokens(
        packed_noisy, img_ids, doubled_ref_tokens, doubled_ref_ids
    )
    forward = KleinForwardBatch(
        hidden_states=hidden_states.to(dtype=weight_dtype),
        encoder_hidden_states=prompt_embeds.repeat(2, 1, 1).to(dtype=weight_dtype),
        timestep=win_sample.sigmas.repeat(2).to(dtype=weight_dtype),
        img_ids=full_img_ids,
        txt_ids=text_ids.repeat(2, 1, 1),
        target_token_count=token_count,
    )
    return KleinDpoBatch(
        forward=forward,
        win_target_velocity=pipeline._pack_latents(win_sample.target_velocity).float(),
        lose_target_velocity=pipeline._pack_latents(lose_sample.target_velocity).float(),
        sigmas=win_sample.sigmas,
    )


def predict_klein(transformer: Any, batch: KleinForwardBatch) -> torch.Tensor:
    prediction = transformer(
        hidden_states=batch.hidden_states,
        encoder_hidden_states=batch.encoder_hidden_states,
        timestep=batch.timestep,
        img_ids=batch.img_ids,
        txt_ids=batch.txt_ids,
        guidance=None,
        return_dict=False,
    )[0]
    return prediction[:, : batch.target_token_count, :].float()


class KleinAdapter(BaseModelAdapter):
    """Composable model adapter for FLUX.2-klein T2I and image editing."""

    model_id = "klein"
    default_lora_target_modules = KLEIN_LORA_TARGET_MODULES

    def prepare_sft_batch(self, **kwargs: Any) -> KleinSftBatch:
        return prepare_klein_sft_batch(**kwargs)

    def prepare_dpo_batch(self, **kwargs: Any) -> KleinDpoBatch:
        return prepare_klein_dpo_batch(**kwargs)

    def predict_velocity(
        self, transformer: Any, batch: KleinForwardBatch, **kwargs: Any
    ) -> torch.Tensor:
        del kwargs
        return predict_klein(transformer, batch)
