"""Training Code v2 public API.

Importing this package intentionally does not import torch or any model runtime.
"""

from mmtrainer.config import RunSpec, load_run_spec
from mmtrainer.registry import RegistrySet, build_builtin_registries
from mmtrainer.version import __version__

__all__ = [
    "RegistrySet",
    "RunSpec",
    "__version__",
    "build_builtin_registries",
    "load_run_spec",
]
