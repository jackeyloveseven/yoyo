from mmtrainer.plugins.base import CatalogDatasetPlugin


class ManifestDatasetPlugin(CatalogDatasetPlugin):
    plugin_id = "manifest"
    dataset_import_path = "mmtrainer.datasets.manifest:JsonlEditDataset"
