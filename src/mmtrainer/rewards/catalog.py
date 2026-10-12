from mmtrainer.plugins.base import CatalogRewardBackend


class RationalRewardsPlugin(CatalogRewardBackend):
    plugin_id = "rational_rewards"
    scorer_import_path = "mmtrainer.rewards.protocols:CallableRewardScorer"


class HPSv3Plugin(CatalogRewardBackend):
    plugin_id = "hpsv3pp"
    scorer_import_path = "mmtrainer.rewards.protocols:CallableRewardScorer"


class PickScorePlugin(CatalogRewardBackend):
    plugin_id = "pickscore"
    scorer_import_path = "mmtrainer.rewards.protocols:CallableRewardScorer"


class EditR1MLLMRewardPlugin(CatalogRewardBackend):
    plugin_id = "edit_r1_mllm"
    scorer_import_path = "mmtrainer.rewards.protocols:CallableRewardScorer"


class ProfessionalOCRPlugin(CatalogRewardBackend):
    plugin_id = "professional_ocr"
    scorer_import_path = "mmtrainer.rewards.protocols:CallableRewardScorer"


class NoRewardPlugin(CatalogRewardBackend):
    plugin_id = "none"
    scorer_import_path = "mmtrainer.rewards.protocols:CallableRewardScorer"
