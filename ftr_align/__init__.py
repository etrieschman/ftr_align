"""ftr_align -- FTR/DAM structural misalignment via support-function geometry."""

from .network import (
    NetworkModel,
    PhysicalNetwork,
    StackedSystem,
    build_stacked_system,
    compute_ptdf,
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
    "build_stacked_system",
    "compute_ptdf",
    "DamInstance",
    "DamResult",
    "SupportData",
    "SupportProblem",
    "SupportSolution",
    "clear_dam",
    "gap",
    "ratio",
]
