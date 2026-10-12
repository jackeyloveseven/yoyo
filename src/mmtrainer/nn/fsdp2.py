"""FSDP2 configuration, FP32 master parameter guards, and state export utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

SUPPORTED_WORLD_SIZES = (1, 2, 4, 8, 16, 24, 32)
GPUS_PER_NODE = 8
MODEL_FAMILY_WRAP_CLASSES: dict[str, tuple[str, ...]] = {
    "qwen_image": ("QwenImageTransformerBlock",),
    "klein": ("Flux2TransformerBlock", "Flux2SingleTransformerBlock"),
    "krea2": ("SingleStreamBlock",),
    "firered": ("QwenImageTransformerBlock",),
}


def register_fsdp2_wrap_classes(model_family: str, wrap_classes: tuple[str, ...]) -> None:
    """Register transformer block class names for FSDP2 auto-wrapping."""
    if not model_family or not wrap_classes:
        raise ValueError("model_family and wrap_classes must be non-empty")
    MODEL_FAMILY_WRAP_CLASSES[model_family] = tuple(wrap_classes)


def _validate_topology_inputs(
    model_family: str,
    world_size: int,
    machine_rank: int,
    main_process_ip: str,
    main_process_port: int,
) -> int:
    if model_family not in MODEL_FAMILY_WRAP_CLASSES:
        raise ValueError(
            f"unknown model family {model_family!r}; choose from {sorted(MODEL_FAMILY_WRAP_CLASSES)}"
        )
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"world_size must be one of {SUPPORTED_WORLD_SIZES}")
    num_machines = max(1, math.ceil(world_size / GPUS_PER_NODE))
    if not 0 <= machine_rank < num_machines:
        raise ValueError(f"machine_rank must be in 0..{num_machines - 1}")
    if not main_process_ip or any(ch.isspace() for ch in main_process_ip):
        raise ValueError("main_process_ip must be a non-empty address")
    if not 1024 <= main_process_port <= 65535:
        raise ValueError("main_process_port must be in 1024..65535")
    return num_machines


def build_accelerate_config(
    *,
    model_family: str,
    world_size: int,
    machine_rank: int = 0,
    main_process_ip: str = "127.0.0.1",
    main_process_port: int = 29671,
) -> dict[str, Any]:
    """Build an Accelerate FSDP2 configuration dictionary."""
    num_machines = _validate_topology_inputs(
        model_family, world_size, machine_rank, main_process_ip, main_process_port
    )
    wrap_classes = ",".join(MODEL_FAMILY_WRAP_CLASSES[model_family])
    config: dict[str, Any] = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": "FSDP",
        "downcast_bf16": "no",
        "fsdp_config": {
            "fsdp_version": 2,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": wrap_classes,
            "fsdp_cpu_ram_efficient_loading": False,
            "fsdp_offload_params": False,
            "fsdp_reshard_after_forward": True,
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        },
        "machine_rank": machine_rank,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": num_machines,
        "num_processes": world_size,
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False,
    }
    if num_machines > 1:
        config["main_process_ip"] = main_process_ip
        config["main_process_port"] = main_process_port
    return config


def validate_accelerate_config(config: Mapping[str, Any], *, model_family: str) -> None:
    """Validate that an Accelerate configuration matches the FSDP2 contract."""
    expected = build_accelerate_config(
        model_family=model_family,
        world_size=int(config.get("num_processes", 0)),
        machine_rank=int(config.get("machine_rank", -1)),
        main_process_ip=str(config.get("main_process_ip", "127.0.0.1")),
        main_process_port=int(config.get("main_process_port", 29671)),
    )
    if dict(config) != expected:
        raise ValueError("Accelerate configuration differs from the FSDP2 contract")


def require_fsdp2_fp32_parameters(
    accelerator: Any,
    model: Any,
    *,
    trainer_name: str,
) -> dict[str, Any]:
    """Require FSDP2 and FP32 trainable parameters before ``accelerator.prepare``."""
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "fsdp_plugin", None)
    fsdp_version = getattr(plugin, "fsdp_version", None)
    if not bool(getattr(accelerator, "is_fsdp2", False)) or fsdp_version != 2:
        raise RuntimeError(f"{trainer_name} requires Accelerate FSDP2")
    if getattr(accelerator, "mixed_precision", None) != "bf16":
        raise RuntimeError(f"{trainer_name} requires bf16 mixed-precision compute")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"{trainer_name} has no trainable parameters")
    bad_dtypes = sorted({str(p.dtype) for p in trainable if p.dtype != torch.float32})
    if bad_dtypes:
        raise RuntimeError(
            f"{trainer_name} requires FP32 trainable parameters; found {bad_dtypes}"
        )
    return {
        "strategy": "fsdp2",
        "fsdp_version": 2,
        "optimizer_parameter_dtype": "float32",
        "compute_dtype": "bfloat16",
        "trainable_parameter_tensors": len(trainable),
    }


def effective_batch_size(
    accelerator: Any,
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return global batch size implied by micro-batch, grad accumulation, and world size."""
    if micro_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("micro_batch_size and gradient_accumulation_steps must be positive")
    world_size = int(getattr(accelerator, "num_processes", 0))
    if world_size <= 0:
        raise ValueError("Accelerator world size must be positive")
    configured = int(getattr(accelerator, "gradient_accumulation_steps", 0))
    if configured != gradient_accumulation_steps:
        raise RuntimeError(
            f"Accelerator grad accumulation mismatch: {configured} != {gradient_accumulation_steps}"
        )
    return micro_batch_size * gradient_accumulation_steps * world_size


def complete_accumulation_batches(
    *,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """Return usable rank-local microbatches and dropped tail count."""
    if batches_per_epoch <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batches_per_epoch and gradient_accumulation_steps must be positive")
    usable = (batches_per_epoch // gradient_accumulation_steps) * gradient_accumulation_steps
    if usable == 0:
        raise ValueError("fewer batches than one gradient accumulation window")
    return usable, batches_per_epoch - usable


def bf16_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a gathered state dict to contiguous BF16 CPU tensors for export."""
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for name, tensor in state_dict.items()
    }
