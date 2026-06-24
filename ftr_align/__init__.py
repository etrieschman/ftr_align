"""ftr_align -- FTR/DAM structural misalignment via support-function geometry."""

from .network import (
    NetworkModel,
    PhysicalNetwork,
    StackedSystem,
    align,
    build_stacked_system,
    compute_ptdf,
    embed,
)
from .solve import (
    DamInstance,
    DamResult,
    SupportData,
    SupportProblem,
    SupportSolution,
    clear_dam,
)
from .metrics import gap, ratio

__all__ = [
    "NetworkModel",
    "PhysicalNetwork",
    "StackedSystem",
    "align",
    "build_stacked_system",
    "compute_ptdf",
    "embed",
    "DamInstance",
    "DamResult",
    "SupportData",
    "SupportProblem",
    "SupportSolution",
    "clear_dam",
    "gap",
    "ratio",
]
