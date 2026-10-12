from mmtrainer.plugins.base import CatalogModelPlugin


class FireRedPlugin(CatalogModelPlugin):
    plugin_id = "firered"
    capabilities = frozenset({"diffusion_alignment", "image_edit", "lora"})
    v1_policy_keys = ("policy_lora",)
    adapter_import_path = "mmtrainer.models.firered:FireRedAdapter"


class SenseNovaPlugin(CatalogModelPlugin):
    plugin_id = "sensenova"
    capabilities = frozenset({"diffusion_alignment", "image_edit", "t2i", "lora"})
    v1_policy_keys = ("policy_lora", "policy")
    adapter_import_path = "mmtrainer.models.sensenova:SenseNovaAdapter"


class KleinPlugin(CatalogModelPlugin):
    plugin_id = "klein"
    capabilities = frozenset({"image_edit", "image_training", "t2i"})
    v1_policy_keys = ("model", "state_dict")
    adapter_import_path = "mmtrainer.models.klein:KleinAdapter"


class QwenImagePlugin(CatalogModelPlugin):
    plugin_id = "qwen_image"
    capabilities = frozenset({"image_edit", "image_training"})
    v1_policy_keys = ("model", "state_dict")
    adapter_import_path = "mmtrainer.models.qwen_image:QwenImageAdapter"


class HiDreamPlugin(CatalogModelPlugin):
    plugin_id = "hidream"
    capabilities = frozenset({"image_edit", "image_training", "t2i"})
    v1_policy_keys = ("model", "state_dict")
    adapter_import_path = "mmtrainer.models.sensenova:HiDreamAdapter"


class Krea2Plugin(CatalogModelPlugin):
    plugin_id = "krea2"
    capabilities = frozenset({"diffusion_alignment", "image_edit", "image_training"})
    v1_policy_keys = ("model", "state_dict", "policy_lora")
    adapter_import_path = "mmtrainer.models.krea2:Krea2Adapter"
