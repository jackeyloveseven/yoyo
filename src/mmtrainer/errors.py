"""Stable error taxonomy for CLI and engine boundaries."""


class MMTrainerError(Exception):
    """Base class for expected, user-actionable v2 failures."""


class RegistryError(MMTrainerError):
    """A plugin registration or resolution contract failed."""


class CapabilityError(MMTrainerError):
    """Selected model, objective, dataset, or reward capabilities disagree."""


class QualificationError(MMTrainerError):
    """A phase was requested before its qualification receipt existed."""


class CheckpointError(MMTrainerError):
    """Checkpoint bytes or lineage semantics are invalid."""
