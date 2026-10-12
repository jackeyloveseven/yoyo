from mmtrainer.plugins.base import CatalogObjectivePlugin


class DiffusionNFTPlugin(CatalogObjectivePlugin):
    plugin_id = "diffusionnft"
    required_model_capabilities = frozenset({"diffusion_alignment"})
    reward_required = True
    objective_import_path = "mmtrainer.algorithms.diffusion_alignment.nft:diffusion_nft_loss"


class FlowGRPOPlugin(CatalogObjectivePlugin):
    plugin_id = "flow_grpo"
    required_model_capabilities = frozenset({"diffusion_alignment"})
    reward_required = True
    objective_import_path = "mmtrainer.algorithms.flow_grpo.objectives:ppo_clipped_surrogate"


class DiffusionOPSDPlugin(CatalogObjectivePlugin):
    plugin_id = "diffusion_opsd"
    required_model_capabilities = frozenset({"diffusion_alignment"})
    reward_required = True
    objective_import_path = (
        "mmtrainer.algorithms.diffusion_opsd.objectives:diffusion_opsd_fitting_loss"
    )


class EditR1Plugin(CatalogObjectivePlugin):
    plugin_id = "edit_r1"
    required_model_capabilities = frozenset({"diffusion_alignment", "image_edit"})
    reward_required = True
    objective_import_path = "mmtrainer.algorithms.edit_r1.objectives:edit_r1_optimality"


class FlowMimicOPSDPlugin(CatalogObjectivePlugin):
    plugin_id = "flowmimic_opsd"
    required_model_capabilities = frozenset({"diffusion_alignment"})
    objective_import_path = (
        "mmtrainer.algorithms.flowmimic_opsd.objectives:flowmimic_clean_distribution_loss"
    )


class ImageSFTPlugin(CatalogObjectivePlugin):
    plugin_id = "image_sft"
    required_model_capabilities = frozenset({"image_training"})
    objective_import_path = "mmtrainer.algorithms.sft_dpo.objectives:flow_sft_loss"


class ImageDPOPlugin(CatalogObjectivePlugin):
    plugin_id = "image_dpo"
    required_model_capabilities = frozenset({"image_training"})
    objective_import_path = "mmtrainer.algorithms.sft_dpo.objectives:flow_dpo_loss"


class ImageOPDPlugin(CatalogObjectivePlugin):
    plugin_id = "image_opd"
    required_model_capabilities = frozenset({"image_training"})
    reward_required = True
    objective_import_path = "mmtrainer.algorithms.sft_dpo.objectives:flow_opd_ode_loss"
