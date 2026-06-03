"""Project-specific exception hierarchy."""


class HerGnnError(Exception):
    """Base exception for all her-gnn errors."""

    pass


class CompositionError(HerGnnError):
    """Raised when composition validation fails."""

    pass


class RelaxationError(HerGnnError):
    """Raised when structure relaxation fails."""

    pass


class ModelLoadError(HerGnnError):
    """Raised when a model checkpoint cannot be loaded."""

    pass


class FileFormatError(HerGnnError):
    """Raised when file format is invalid or unsupported."""

    pass


class SlabConstructionError(HerGnnError):
    """Raised when slab construction fails."""

    pass
