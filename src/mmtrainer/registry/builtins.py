"""Declarative built-in registry; adding an entry never changes engine code."""

from __future__ import annotations

from mmtrainer.registry.core import (
    LazyRegistry,
    PluginKind,
    RegistryEntry,
    RegistrySet,
)


def _entry(
    plugin_id: str,
    kind: PluginKind,
    target: str,
    *,
    capabilities: frozenset[str] = frozenset(),
    metadata: dict[str, object] | None = None,
) -> RegistryEntry:
    return RegistryEntry.create(
        plugin_id=plugin_id,
        kind=kind,
        import_path=target,
        capabilities=capabilities,
        metadata=metadata,
    )


def build_builtin_registries() -> RegistrySet:
    models = LazyRegistry(PluginKind.MODEL)
    model_rows = (
        ("firered", "FireRedPlugin", {"diffusion_alignment", "image_edit", "lora"}),
        (
            "sensenova",
            "SenseNovaPlugin",
            {"diffusion_alignment", "image_edit", "t2i", "lora"},
        ),
        ("klein", "KleinPlugin", {"image_edit", "image_training", "t2i"}),
        ("qwen_image", "QwenImagePlugin", {"image_edit", "image_training"}),
        ("qwen_image_edit_2x", "QwenImageEdit2Plugin", {"image_edit", "image_training"}),
        ("hidream", "HiDreamPlugin", {"image_edit", "image_training", "t2i"}),
        ("krea2", "Krea2Plugin", {"diffusion_alignment", "image_edit", "image_training"}),
    )
    for plugin_id, class_name, capabilities in model_rows:
        models.register(
            _entry(
                plugin_id,
                PluginKind.MODEL,
                f"mmtrainer.models.catalog:{class_name}",
                capabilities=frozenset(capabilities),
            )
        )

    objectives = LazyRegistry(PluginKind.OBJECTIVE)
    objective_rows = (
        ("diffusionnft", "DiffusionNFTPlugin", {"diffusion_alignment"}, True),
        ("flow_grpo", "FlowGRPOPlugin", {"diffusion_alignment"}, True),
        ("diffusion_opsd", "DiffusionOPSDPlugin", {"diffusion_alignment"}, True),
        (
            "edit_r1",
            "EditR1Plugin",
            {"diffusion_alignment", "image_edit"},
            True,
        ),
        ("flowmimic_opsd", "FlowMimicOPSDPlugin", {"diffusion_alignment"}, False),
        ("image_sft", "ImageSFTPlugin", {"image_training"}, False),
        ("image_dpo", "ImageDPOPlugin", {"image_training"}, False),
        ("image_opd", "ImageOPDPlugin", {"image_training"}, True),
        ("latent_consistency", "LatentConsistencyPlugin", {"image_training"}, False),
        ("on_policy_distill", "OnPolicyDistillPlugin", {"image_training"}, False),
    )
    for plugin_id, class_name, required, reward_required in objective_rows:
        objectives.register(
            _entry(
                plugin_id,
                PluginKind.OBJECTIVE,
                f"mmtrainer.algorithms.catalog:{class_name}",
                metadata={
                    "required_model_capabilities": frozenset(required),
                    "reward_required": reward_required,
                },
            )
        )

    rewards = LazyRegistry(PluginKind.REWARD)
    for plugin_id, class_name in (
        ("rational_rewards", "RationalRewardsPlugin"),
        ("hpsv3pp", "HPSv3Plugin"),
        ("pickscore", "PickScorePlugin"),
        ("edit_r1_mllm", "EditR1MLLMRewardPlugin"),
        ("professional_ocr", "ProfessionalOCRPlugin"),
        ("identity", "IdentityRewardPlugin"),
        ("none", "NoRewardPlugin"),
    ):
        rewards.register(
            _entry(
                plugin_id,
                PluginKind.REWARD,
                f"mmtrainer.rewards.catalog:{class_name}",
            )
        )

    datasets = LazyRegistry(PluginKind.DATASET)
    datasets.register(
        _entry(
            "manifest",
            PluginKind.DATASET,
            "mmtrainer.datasets.catalog:ManifestDatasetPlugin",
        )
    )

    checkpoints = LazyRegistry(PluginKind.CHECKPOINT)
    checkpoints.register(
        _entry(
            "v2_directory",
            PluginKind.CHECKPOINT,
            "mmtrainer.runtime.checkpoints:V2DirectoryCheckpointCodec",
        )
    )
    return RegistrySet(
        models=models,
        objectives=objectives,
        rewards=rewards,
        datasets=datasets,
        checkpoints=checkpoints,
    )
